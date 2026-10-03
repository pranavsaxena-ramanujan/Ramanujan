// Ramanujan LLM runtime: executes one pipeline stage of a decoder-only GGUF model with OpenCL.
//
// The host only sequences kernels. Every token runs on one in-order queue with no host
// synchronization inside a step; weights stay in their raw GGUF block layout on the device.
// Weights are either resident (uploaded once at open) or streamed (a loader thread uploads
// layer i+1 while the device computes layer i, cycling so the next step's first layers are
// ready ahead of time), which lets a stage larger than device memory run at SSD speed.

#include "rjllm.h"
#include "json.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <cstring>
#include <deque>
#include <exception>
#include <fstream>
#include <map>
#include <memory>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#if defined(_WIN32)
#define NOMINMAX
#include <windows.h>
#else
#include <fcntl.h>
#include <unistd.h>
#endif

#if defined(__APPLE__)
#define CL_SILENCE_DEPRECATION
#include <OpenCL/cl.h>
#else
#ifndef CL_TARGET_OPENCL_VERSION
#define CL_TARGET_OPENCL_VERSION 120
#endif
#include <CL/cl.h>
#ifdef __ANDROID__
#include "../opencl_loader.h"
#endif
#endif

#ifndef __ANDROID__
static inline bool rjllmOpenclLoad() { return true; }
#else
static inline bool rjllmOpenclLoad() { return openclLoad(); }
#endif

#include "kernels_cl.inc"  // generated: static const char kKernelSource[]

namespace rjllm {

using Clock = std::chrono::steady_clock;

static double millisSince(Clock::time_point start) {
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

static void check(cl_int status, const char *what) {
    if (status != CL_SUCCESS) throw std::runtime_error(std::string("OpenCL ") + what + " failed (" + std::to_string(status) + ")");
}

// ---------------------------------------------------------------- tensor types

enum Type { F32, F16, Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q4_K, Q5_K, Q6_K, TYPE_COUNT };
static const char *const TYPE_NAMES[TYPE_COUNT] = {"f32", "f16", "q4_0", "q4_1", "q5_0",
                                                   "q5_1", "q8_0", "q4_k", "q5_k", "q6_k"};
static const int BLOCK_VALUES[TYPE_COUNT] = {1, 1, 32, 32, 32, 32, 32, 256, 256, 256};
static const int BLOCK_BYTES[TYPE_COUNT] = {4, 2, 18, 20, 22, 24, 34, 144, 176, 210};

struct TensorRef {
    std::string file;
    int type = F32;
    long long rows = 1, cols = 0;
    size_t rowBytes = 0, bytes = 0;
    int chunks = 0;  // 32-value chunks per row (0 when the row length is not a multiple of 32)
};

static TensorRef parseTensor(const Json &j, const std::string &what) {
    TensorRef t;
    t.file = j["file"].string();
    std::string encoding = j["encoding"].string();
    if (encoding.compare(0, 5, "gguf-") == 0) encoding = encoding.substr(5);
    t.type = -1;
    for (int i = 0; i < TYPE_COUNT; i++)
        if (encoding == TYPE_NAMES[i]) t.type = i;
    if (t.type < 0) throw std::runtime_error(what + ": unsupported encoding " + j["encoding"].string());
    const Json &shape = j["shape"];
    if (shape.size() == 0) throw std::runtime_error(what + ": empty shape");
    t.cols = shape[shape.size() - 1].integer();
    for (size_t i = 0; i + 1 < shape.size(); i++) t.rows *= shape[i].integer();
    if (t.cols <= 0 || t.rows <= 0 || t.cols % BLOCK_VALUES[t.type])
        throw std::runtime_error(what + ": row length is not a multiple of the quant block");
    t.rowBytes = (size_t)(t.cols / BLOCK_VALUES[t.type]) * BLOCK_BYTES[t.type];
    t.bytes = t.rowBytes * (size_t)t.rows;
    t.chunks = t.cols % 32 == 0 ? (int)(t.cols / 32) : 0;
    return t;
}

// ---------------------------------------------------------------- device

struct Device {
    cl_platform_id platform = nullptr;
    cl_device_id device = nullptr;
    cl_context context = nullptr;
    cl_program program = nullptr;
    int wg = 64;
    std::string name, platformName, vendor, driverVersion, runtimeVersion;
    cl_ulong globalMem = 0, maxAlloc = 0;
    cl_bool unifiedMemory = CL_FALSE;
    cl_device_type type = 0;
    std::mutex mutex;           // guards residentBytes
    size_t residentBytes = 0;   // resident weights across all sessions in this process

    ~Device() {
        if (program) clReleaseProgram(program);
        if (context) clReleaseContext(context);
    }
};

static std::string deviceString(cl_device_id device, cl_device_info what) {
    size_t size = 0;
    clGetDeviceInfo(device, what, 0, nullptr, &size);
    std::string value(size, '\0');
    clGetDeviceInfo(device, what, size, &value[0], nullptr);
    while (!value.empty() && value.back() == '\0') value.pop_back();
    return value;
}

static std::string kernelSource() {
    const char *path = std::getenv("RJLLM_KERNELS");
    if (path && *path) {
        std::ifstream in(path, std::ios::binary);
        if (!in) throw std::runtime_error(std::string("RJLLM_KERNELS: cannot read ") + path);
        std::stringstream text;
        text << in.rdbuf();
        return text.str();
    }
    return std::string(kKernelSource);
}

static cl_program buildProgram(Device &d, int wg) {
    std::string source = kernelSource();
    const char *text = source.c_str();
    size_t length = source.size();
    cl_int status;
    cl_program program = clCreateProgramWithSource(d.context, 1, &text, &length, &status);
    check(status, "clCreateProgramWithSource");
    std::string options = "-DWG=" + std::to_string(wg);
    status = clBuildProgram(program, 1, &d.device, options.c_str(), nullptr, nullptr);
    if (status != CL_SUCCESS) {
        size_t size = 0;
        clGetProgramBuildInfo(program, d.device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &size);
        std::string log(size, '\0');
        clGetProgramBuildInfo(program, d.device, CL_PROGRAM_BUILD_LOG, size, &log[0], nullptr);
        clReleaseProgram(program);
        throw std::runtime_error("LLM kernel build failed (" + std::to_string(status) + "):\n" + log);
    }
    return program;
}

static size_t kernelWorkGroupLimit(Device &d, cl_program program) {
    size_t limit = (size_t)-1;
    const char *probes[] = {"matvec_q6_k", "gateup_q6_k", "attention", "rmsnorm_rows", "dn_gatenorm"};
    for (const char *name : probes) {
        cl_int status;
        cl_kernel kernel = clCreateKernel(program, name, &status);
        check(status, "clCreateKernel");
        size_t size = 0;
        clGetKernelWorkGroupInfo(kernel, d.device, CL_KERNEL_WORK_GROUP_SIZE, sizeof(size), &size, nullptr);
        clReleaseKernel(kernel);
        limit = std::min(limit, size);
    }
    return limit;
}

static Device *g_device = nullptr;
static std::mutex g_deviceMutex;

static Device &device() {
    std::lock_guard<std::mutex> lock(g_deviceMutex);
    if (g_device) return *g_device;
    if (!rjllmOpenclLoad()) throw std::runtime_error("OpenCL library not available");
    std::unique_ptr<Device> d(new Device());
    cl_uint count = 0;
    check(clGetPlatformIDs(0, nullptr, &count), "clGetPlatformIDs");
    if (count == 0) throw std::runtime_error("no OpenCL platforms");
    std::vector<cl_platform_id> platforms(count);
    check(clGetPlatformIDs(count, platforms.data(), nullptr), "clGetPlatformIDs");
    const char *wanted = std::getenv("RJLLM_DEVICE");  // "gpu" (default preference), "cpu" or "any"
    std::string want = wanted ? wanted : "gpu";
    cl_device_type types[2] = {CL_DEVICE_TYPE_GPU, CL_DEVICE_TYPE_ALL};
    if (want == "cpu") types[0] = CL_DEVICE_TYPE_CPU;
    if (want == "any") types[0] = CL_DEVICE_TYPE_ALL;
    for (cl_device_type type : types) {
        for (cl_platform_id platform : platforms) {
            cl_device_id found = nullptr;
            cl_uint n = 0;
            if (clGetDeviceIDs(platform, type, 1, &found, &n) == CL_SUCCESS && n > 0) {
                d->platform = platform;
                d->device = found;
                break;
            }
        }
        if (d->device) break;
    }
    if (!d->device) throw std::runtime_error("no OpenCL device found");
    d->name = deviceString(d->device, CL_DEVICE_NAME);
    d->vendor = deviceString(d->device, CL_DEVICE_VENDOR);
    d->driverVersion = deviceString(d->device, CL_DRIVER_VERSION);
    d->runtimeVersion = deviceString(d->device, CL_DEVICE_VERSION);
    size_t size = 0;
    clGetPlatformInfo(d->platform, CL_PLATFORM_NAME, 0, nullptr, &size);
    d->platformName.assign(size, '\0');
    clGetPlatformInfo(d->platform, CL_PLATFORM_NAME, size, &d->platformName[0], nullptr);
    while (!d->platformName.empty() && d->platformName.back() == '\0') d->platformName.pop_back();
    clGetDeviceInfo(d->device, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(d->globalMem), &d->globalMem, nullptr);
    clGetDeviceInfo(d->device, CL_DEVICE_HOST_UNIFIED_MEMORY, sizeof(d->unifiedMemory), &d->unifiedMemory, nullptr);
    clGetDeviceInfo(d->device, CL_DEVICE_TYPE, sizeof(d->type), &d->type, nullptr);
    clGetDeviceInfo(d->device, CL_DEVICE_MAX_MEM_ALLOC_SIZE, sizeof(d->maxAlloc), &d->maxAlloc, nullptr);
    size_t maxGroup = 0;
    clGetDeviceInfo(d->device, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(maxGroup), &maxGroup, nullptr);
    cl_int status;
    d->context = clCreateContext(nullptr, 1, &d->device, nullptr, nullptr, &status);
    check(status, "clCreateContext");
    const char *wgEnv = std::getenv("RJLLM_WG");
    int wg = wgEnv ? std::atoi(wgEnv) : 64;
    if (wg < 1 || (wg & (wg - 1))) throw std::runtime_error("RJLLM_WG must be a power of two");
    while ((size_t)wg > maxGroup && wg > 1) wg >>= 1;
    while (true) {
        cl_program program = buildProgram(*d, wg);
        size_t limit = kernelWorkGroupLimit(*d, program);
        if ((size_t)wg <= limit || wg == 1) {
            d->program = program;
            break;
        }
        clReleaseProgram(program);
        wg >>= 1;
    }
    d->wg = wg;
    g_device = d.release();
    return *g_device;
}

static size_t physicalMemory() {
#if defined(_WIN32)
    MEMORYSTATUSEX status;
    status.dwLength = sizeof(status);
    return GlobalMemoryStatusEx(&status) ? (size_t)status.ullTotalPhys : 0;
#else
    long pages = sysconf(_SC_PHYS_PAGES), page = sysconf(_SC_PAGE_SIZE);
    return pages > 0 && page > 0 ? (size_t)pages * (size_t)page : 0;
#endif
}

// ---------------------------------------------------------------- weight loading

static std::mutex g_allocationMutex;
static std::map<cl_mem, size_t> g_allocations;
static size_t g_allocatedBytes = 0;

static cl_mem trackedCreateBuffer(cl_context context, cl_mem_flags flags, size_t bytes, void *host, cl_int *status) {
    cl_mem buffer = clCreateBuffer(context, flags, bytes, host, status);
    if (buffer) {
        try {
            std::lock_guard<std::mutex> lock(g_allocationMutex);
            g_allocations.emplace(buffer, bytes);
            g_allocatedBytes += bytes;
        } catch (...) {
            clReleaseMemObject(buffer);
            throw;
        }
    }
    return buffer;
}

static void trackedReleaseBuffer(cl_mem buffer) {
    {
        std::lock_guard<std::mutex> lock(g_allocationMutex);
        auto it = g_allocations.find(buffer);
        if (it != g_allocations.end()) {
            g_allocatedBytes -= it->second;
            g_allocations.erase(it);
        }
    }
    clReleaseMemObject(buffer);
}

using Buffers = std::map<std::string, cl_mem>;

static void release(Buffers &buffers) {
    for (auto &entry : buffers)
        if (entry.second) trackedReleaseBuffer(entry.second);
    buffers.clear();
}

struct Item {  // one unit of weights: a layer or the head
    std::string name;
    std::vector<std::pair<std::string, const TensorRef *>> tensors;
    size_t bytes = 0;
};

class FileReader {
public:
    explicit FileReader(const std::string &path, bool bypassCache = false) : path_(path) {
        file_ = std::fopen(path.c_str(), "rb");
        if (!file_) throw std::runtime_error("cannot open weight file " + path);
        std::setvbuf(file_, nullptr, _IONBF, 0);
#if defined(__APPLE__)
        if (bypassCache) fcntl(fileno(file_), F_NOCACHE, 1);
#else
        (void)bypassCache;
#endif
    }
    ~FileReader() { std::fclose(file_); }
    void read(size_t offset, char *out, size_t bytes) {
#if defined(_WIN32)
        if (_fseeki64(file_, (long long)offset, SEEK_SET) != 0)
#else
        if (fseeko(file_, (off_t)offset, SEEK_SET) != 0)
#endif
            throw std::runtime_error("cannot seek in " + path_);
        size_t done = 0;
        while (done < bytes) {
            size_t got = std::fread(out + done, 1, bytes - done, file_);
            if (got == 0) throw std::runtime_error("truncated weight file " + path_);
            done += got;
        }
    }

private:
    std::string path_;
    FILE *file_;
};

static const size_t STAGING = 32u << 20;

static cl_mem upload(Device &d, cl_command_queue queue, const TensorRef &t, std::vector<char> &staging,
                     bool bypassCache) {
    if (t.bytes > d.maxAlloc)
        throw std::runtime_error(t.file + ": " + std::to_string(t.bytes) + " bytes exceeds the device's max allocation (" +
                                 std::to_string(d.maxAlloc) + ")");
    cl_int status;
    cl_mem buffer = trackedCreateBuffer(d.context, CL_MEM_READ_ONLY, t.bytes, nullptr, &status);
    check(status, "clCreateBuffer(weights)");
    try {
        if (staging.size() < STAGING) staging.resize(STAGING);
        FileReader reader(t.file, bypassCache);
        for (size_t offset = 0; offset < t.bytes; offset += STAGING) {
            size_t n = std::min(STAGING, t.bytes - offset);
            reader.read(offset, staging.data(), n);
            check(clEnqueueWriteBuffer(queue, buffer, CL_TRUE, offset, n, staging.data(), 0, nullptr, nullptr),
                  "clEnqueueWriteBuffer(weights)");
        }
    } catch (...) {
        trackedReleaseBuffer(buffer);
        throw;
    }
    return buffer;
}

static Buffers load(Device &d, cl_command_queue queue, const Item &item, std::vector<char> &staging,
                    bool bypassCache = false) {
    Buffers buffers;
    try {
        for (auto &entry : item.tensors) buffers[entry.first] = upload(d, queue, *entry.second, staging, bypassCache);
    } catch (...) {
        release(buffers);
        throw;
    }
    return buffers;
}

// Uploads items cyclically on `threads` loader threads (each with its own queue), keeping at
// most `depth` items loaded ahead of the consumer, which takes them strictly in order.
class Streamer {
public:
    Streamer(Device &d, const std::vector<Item> &items, int depth, int threads)
        : d_(d), items_(items), depth_(depth) {
        // Streamed weights are re-read every step because they do not fit in memory; bypassing
        // the page cache (macOS F_NOCACHE) saves a copy and keeps other memory resident.
        const char *cache = std::getenv("RJLLM_NOCACHE");
        bypassCache_ = !(cache && *cache == '0');
        for (int i = 0; i < threads; i++) {
            cl_int status;
            cl_command_queue queue = clCreateCommandQueue(d.context, d.device, 0, &status);
            check(status, "clCreateCommandQueue(loader)");
            queues_.push_back(queue);
        }
        for (int i = 0; i < threads; i++) threads_.emplace_back(&Streamer::run, this, queues_[i]);
    }
    ~Streamer() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            stop_ = true;
        }
        ready_.notify_all();
        for (std::thread &thread : threads_) thread.join();
        for (auto &entry : loaded_) release(entry.second);
        for (cl_command_queue queue : queues_) clReleaseCommandQueue(queue);
    }
    Buffers next(double &waitedMs) {
        Clock::time_point start = Clock::now();
        std::unique_lock<std::mutex> lock(mutex_);
        ready_.wait(lock, [&] { return loaded_.count(consumed_) > 0 || error_; });
        waitedMs += millisSince(start);
        if (!loaded_.count(consumed_)) std::rethrow_exception(error_);
        Buffers buffers = std::move(loaded_[consumed_]);
        loaded_.erase(consumed_);
        consumed_++;
        ready_.notify_all();
        return buffers;
    }
    size_t bytesLoaded() const { return bytes_.load(); }

private:
    void run(cl_command_queue queue) {
        std::vector<char> staging;
        while (true) {
            size_t sequence;
            {
                std::unique_lock<std::mutex> lock(mutex_);
                ready_.wait(lock, [&] { return stop_ || error_ || claimed_ < consumed_ + depth_; });
                if (stop_ || error_) return;
                sequence = claimed_++;
            }
            const Item &item = items_[sequence % items_.size()];
            Buffers buffers;
            try {
                buffers = load(d_, queue, item, staging, bypassCache_);
            } catch (...) {
                std::lock_guard<std::mutex> lock(mutex_);
                if (!error_) error_ = std::current_exception();
                ready_.notify_all();
                return;
            }
            bytes_ += item.bytes;
            std::lock_guard<std::mutex> lock(mutex_);
            if (stop_) {
                release(buffers);
                return;
            }
            loaded_[sequence] = std::move(buffers);
            ready_.notify_all();
        }
    }

    Device &d_;
    const std::vector<Item> &items_;
    size_t depth_;
    bool bypassCache_ = true;
    std::vector<cl_command_queue> queues_;
    std::vector<std::thread> threads_;
    std::mutex mutex_;
    std::condition_variable ready_;
    std::map<size_t, Buffers> loaded_;
    size_t claimed_ = 0, consumed_ = 0;
    std::exception_ptr error_;
    std::atomic<size_t> bytes_{0};
    bool stop_ = false;
};

// ---------------------------------------------------------------- stage graph

struct Layer {
    int index = 0;
    bool attention = true;
    bool fusedQkv = false, qGate = false, qkNorm = false, postAttnNorm = false, postFfnNorm = false;
    std::string ffnNorm = "ffn_norm";
    int ffnKind = 0;  // 0 gated, 1 fused gate/up, 2 plain
    int ffnDim = 0;
    std::map<std::string, TensorRef> tensors;
    cl_mem kCache = nullptr, vCache = nullptr, deltaState = nullptr, convState = nullptr;

    bool has(const std::string &role) const { return tensors.count(role) > 0; }
    const TensorRef &at(const std::string &role) const {
        auto it = tensors.find(role);
        if (it == tensors.end()) throw std::runtime_error("layer " + std::to_string(index) + ": missing tensor " + role);
        return it->second;
    }
};

struct Hyper {
    int dim = 0, vocab = 0, heads = 0, kvHeads = 0, headDim = 0, ropeDims = 0;
    float eps = 1e-6f, attnScale = 1.0f, embedMultiplier = 1.0f;
    double ropeTheta = 10000.0, ropePositionScale = 1.0;
    int ropeStyle = 0;   // 0 norm, 1 neox
    int activation = 0;  // 0 silu, 1 gelu
    bool hasSsm = false;
    int convKernel = 0, stateSize = 0, groupCount = 0, timeStepRank = 0, innerSize = 0;
};

// ---------------------------------------------------------------- session

class Session {
public:
    explicit Session(const std::string &graphJson) : d_(device()) {
        Json graph = Json::parse(graphJson);
        parse(graph);
        cl_int status;
        queue_ = clCreateCommandQueue(d_.context, d_.device, 0, &status);
        check(status, "clCreateCommandQueue");
        try {
            createKernels();
            allocate();
            chooseWeights(graph.text("weights", "auto"), (int)graph.get("stream_depth", 2),
                          (int)graph.get("stream_threads", 2));
            resetState();
            check(clFinish(queue_), "clFinish(open)");
        } catch (...) {
            destroy();
            throw;
        }
    }

    ~Session() { destroy(); }

    size_t outputSize(int n) const { return hasHead_ ? (size_t)h_.vocab : (size_t)n * h_.dim; }

    void step(const int32_t *tokens, const float *hidden, int n, int pos, float *out, size_t outLen) {
        if (broken_) throw std::runtime_error("session failed earlier; reopen it");
        if (n < 1) throw std::runtime_error("step needs at least one token");
        if (pos != nextPos_)
            throw std::runtime_error("position " + std::to_string(pos) + " does not match session position " +
                                     std::to_string(nextPos_) + " (state lost?)");
        if (pos + n > maxContext_) throw std::runtime_error("step exceeds max_context");
        if (outLen < outputSize(n)) throw std::runtime_error("output buffer too small");
        if (hasEmbed_ ? tokens == nullptr : hidden == nullptr)
            throw std::runtime_error(hasEmbed_ ? "stage takes token ids" : "stage takes hidden states");
        Clock::time_point start = Clock::now();
        double waited = 0;
        try {
            ensureHidden(n);
            if (hasEmbed_) {
                for (int t = 0; t < n; t++) embed(tokens[t], t * h_.dim);
            } else {
                check(clEnqueueWriteBuffer(queue_, hiddenBuf_, CL_FALSE, 0, sizeof(float) * n * h_.dim, hidden, 0,
                                           nullptr, nullptr),
                      "clEnqueueWriteBuffer(hidden)");
            }
            for (size_t i = 0; i < items_.size(); i++) {
                Buffers streamed;
                const Buffers *weights = streamer_ ? &streamed : &resident_[i];
                if (streamer_) {
                    streamed = streamer_->next(waited);
                    weights = &streamed;
                }
                if (i < layers_.size()) {
                    Layer &layer = layers_[i];
                    for (int t = 0; t < n; t++) {
                        cl_mem x = hiddenBuf_;
                        if (n > 1) {
                            copy(hiddenBuf_, t * h_.dim, work_, 0, h_.dim);
                            x = work_;
                        }
                        runLayer(layer, *weights, x, pos + t);
                        if (n > 1) copy(work_, 0, hiddenBuf_, t * h_.dim, h_.dim);
                    }
                } else {
                    runHead(*weights, (n - 1) * h_.dim);
                }
                if (streamer_) {
                    // The loader is the bottleneck; waiting here bounds device memory to the
                    // streaming depth and costs nothing while the next layer uploads.
                    check(clFinish(queue_), "clFinish(stream)");
                    release(streamed);
                }
            }
            if (hasHead_) {
                check(clEnqueueReadBuffer(queue_, logits_, CL_TRUE, 0, sizeof(float) * h_.vocab, out, 0, nullptr, nullptr),
                      "clEnqueueReadBuffer(logits)");
            } else {
                check(clEnqueueReadBuffer(queue_, hiddenBuf_, CL_TRUE, 0, sizeof(float) * n * h_.dim, out, 0, nullptr,
                                          nullptr),
                      "clEnqueueReadBuffer(hidden)");
            }
        } catch (...) {
            broken_ = true;
            throw;
        }
        nextPos_ += n;
        steps_++;
        tokens_ += n;
        lastStepMs_ = millisSince(start);
        lastWaitMs_ = waited;
        totalStepMs_ += lastStepMs_;
    }

    void reset() {
        resetState();
        clFinish(queue_);
        nextPos_ = 0;
        broken_ = false;
    }

    const char *info() {
        std::ostringstream o;
        o << "{\"device\":\"" << escape(d_.name) << "\",\"platform\":\"" << escape(d_.platformName) << "\",\"wg\":" << d_.wg
          << ",\"weights\":\"" << (streamer_ ? "stream" : "resident") << "\",\"weight_bytes\":" << weightBytes_
          << ",\"layers\":[";
        for (size_t i = 0; i < layers_.size(); i++) o << (i ? "," : "") << layers_[i].index;
        o << "],\"embed\":" << (hasEmbed_ ? "true" : "false") << ",\"head\":" << (hasHead_ ? "true" : "false")
          << ",\"max_context\":" << maxContext_ << ",\"position\":" << nextPos_ << ",\"steps\":" << steps_
          << ",\"tokens\":" << tokens_ << ",\"last_step_ms\":" << lastStepMs_ << ",\"last_wait_ms\":" << lastWaitMs_
          << ",\"total_step_ms\":" << totalStepMs_
          << ",\"streamed_bytes\":" << (streamer_ ? streamer_->bytesLoaded() : 0) << "}";
        info_ = o.str();
        return info_.c_str();
    }

private:
    // ------------------------------------------------ setup

    void parse(const Json &g) {
        const Json &h = g["hyper"];
        h_.dim = (int)h["dim"].integer();
        h_.vocab = (int)h["vocab"].integer();
        h_.eps = (float)h["eps"].number();
        h_.heads = (int)h["heads"].integer();
        h_.kvHeads = (int)h["kv_heads"].integer();
        h_.headDim = (int)h["head_dim"].integer();
        h_.ropeDims = (int)h["rope_dims"].integer();
        h_.ropeTheta = h["rope_theta"].number();
        h_.ropePositionScale = h.get("rope_position_scale", 1.0);
        h_.attnScale = (float)h["attn_scale"].number();
        h_.embedMultiplier = (float)h.get("embed_multiplier", 1.0);
        std::string style = h.text("rope_style", "norm"), act = h.text("activation", "silu");
        if (style != "norm" && style != "neox") throw std::runtime_error("unknown rope_style " + style);
        if (act != "silu" && act != "gelu") throw std::runtime_error("unknown activation " + act);
        h_.ropeStyle = style == "neox" ? 1 : 0;
        h_.activation = act == "gelu" ? 1 : 0;
        if (h.has("ssm")) {
            const Json &s = h["ssm"];
            h_.hasSsm = true;
            h_.convKernel = (int)s["conv_kernel"].integer();
            h_.stateSize = (int)s["state_size"].integer();
            h_.groupCount = (int)s["group_count"].integer();
            h_.timeStepRank = (int)s["time_step_rank"].integer();
            h_.innerSize = (int)s["inner_size"].integer();
        }
        if (h_.dim % 32) throw std::runtime_error("embedding length must be a multiple of 32");
        maxContext_ = (int)g["max_context"].integer();
        if (maxContext_ < 1) throw std::runtime_error("max_context must be positive");
        if (g.has("embed")) {
            hasEmbed_ = true;
            embed_ = parseTensor(g["embed"], "embed");
            if (embed_.cols != h_.dim || !embed_.chunks) throw std::runtime_error("embed tensor does not match dim");
        }
        if (g.has("rope_freqs")) ropeFreqs_ = parseTensor(g["rope_freqs"], "rope_freqs");
        const Json &layers = g["layers"];
        for (size_t i = 0; i < layers.size(); i++) layers_.push_back(parseLayer(layers[i]));
        if (g.has("head")) {
            hasHead_ = true;
            head_["output_norm"] = parseTensor(g["head"]["output_norm"], "output_norm");
            head_["output"] = parseTensor(g["head"]["output"], "output");
            if (head_["output"].cols != h_.dim || head_["output"].rows != h_.vocab)
                throw std::runtime_error("output head does not match dim/vocab");
        }
        if (!hasEmbed_ && layers_.empty() && !hasHead_) throw std::runtime_error("empty stage graph");
        for (Layer &layer : layers_) {
            Item item;
            item.name = "blk." + std::to_string(layer.index);
            for (auto &entry : layer.tensors) {
                item.tensors.push_back(std::make_pair(entry.first, &entry.second));
                item.bytes += entry.second.bytes;
            }
            items_.push_back(item);
        }
        if (hasHead_) {
            Item item;
            item.name = "head";
            for (auto &entry : head_) {
                item.tensors.push_back(std::make_pair(entry.first, &entry.second));
                item.bytes += entry.second.bytes;
            }
            items_.push_back(item);
        }
        for (const Item &item : items_) weightBytes_ += item.bytes;
    }

    Layer parseLayer(const Json &j) {
        Layer l;
        l.index = (int)j["index"].integer();
        std::string mixer = j["mixer"].string();
        if (mixer != "attention" && mixer != "gated_deltanet")
            throw std::runtime_error("layer " + std::to_string(l.index) + ": unsupported mixer " + mixer);
        l.attention = mixer == "attention";
        if (!l.attention && !h_.hasSsm) throw std::runtime_error("gated_deltanet layer without ssm hyperparameters");
        const Json &f = j["flags"];
        if (l.attention) {
            l.fusedQkv = f.text("qkv", "separate") == "fused";
            l.qGate = f.flag("q_gate", false);
            l.qkNorm = f.flag("qk_norm", false);
        }
        l.postAttnNorm = f.flag("post_attn_norm", false);
        l.postFfnNorm = f.flag("post_ffn_norm", false);
        l.ffnNorm = f["ffn_norm"].string();
        std::string ffn = f["ffn"].string();
        l.ffnKind = ffn == "gated" ? 0 : ffn == "fused_gate_up" ? 1 : ffn == "plain" ? 2 : -1;
        if (l.ffnKind < 0) throw std::runtime_error("unknown ffn layout " + ffn);
        l.ffnDim = (int)f["ffn_dim"].integer();
        for (auto &entry : j["tensors"].items())
            l.tensors[entry.first] = parseTensor(entry.second, "blk." + std::to_string(l.index) + "." + entry.first);
        for (auto &entry : l.tensors) {
            const std::string &role = entry.first;
            const TensorRef &t = entry.second;
            // 2-D tensors feed matvec kernels, except the depthwise conv weight (read as a vector).
            bool matrix = t.rows > 1 && role != "ssm_conv1d";
            if (matrix && !t.chunks)
                throw std::runtime_error("blk." + std::to_string(l.index) + "." + role + ": row length must be a multiple of 32");
            if (!matrix && t.type != F32)
                throw std::runtime_error("blk." + std::to_string(l.index) + "." + role + ": vector tensors must be F32");
        }
        return l;
    }

    cl_mem scratch(size_t floats) {
        cl_int status;
        cl_mem buffer = trackedCreateBuffer(d_.context, CL_MEM_READ_WRITE, sizeof(float) * std::max<size_t>(floats, 1),
                                       nullptr, &status);
        check(status, "clCreateBuffer(scratch)");
        owned_.push_back(buffer);
        return buffer;
    }

    void allocate() {
        size_t qkv = 1, att = 1, ffn = 1, hidden = 1, heads = (size_t)h_.heads;
        int kvWidth = h_.kvHeads * h_.headDim;
        for (Layer &l : layers_) {
            if (l.attention) {
                int qRows = (int)(l.fusedQkv ? (long long)h_.heads * h_.headDim : l.at("attn_q").rows);
                qkv = std::max(qkv, (size_t)qRows + 2 * kvWidth);
                att = std::max(att, (size_t)h_.heads * h_.headDim);
                l.kCache = scratch((size_t)maxContext_ * kvWidth);
                l.vCache = scratch((size_t)maxContext_ * kvWidth);
            } else {
                size_t convDim = 2 * (size_t)h_.groupCount * h_.stateSize + h_.innerSize;
                qkv = std::max(qkv, convDim);
                att = std::max(att, (size_t)h_.innerSize);
                l.deltaState = scratch((size_t)h_.timeStepRank * h_.stateSize * h_.stateSize);
                l.convState = scratch(convDim * (h_.convKernel - 1));
            }
            ffn = std::max(ffn, (size_t)2 * l.ffnDim);
            hidden = std::max(hidden, (size_t)l.ffnDim);
        }
        xn_ = scratch(h_.dim);
        work_ = scratch(h_.dim);
        mixed_ = scratch(h_.dim);
        qkv_ = scratch(qkv);
        att_ = scratch(att);
        att2_ = scratch(att);
        scores_ = scratch(heads * maxContext_);
        ffn_ = scratch(ffn);
        ffnHidden_ = scratch(hidden);
        if (h_.hasSsm) {
            size_t convDim = 2 * (size_t)h_.groupCount * h_.stateSize + h_.innerSize;
            z_ = scratch(h_.innerSize);
            beta_ = scratch(h_.timeStepRank);
            alpha_ = scratch(h_.timeStepRank);
            decay_ = scratch(h_.timeStepRank);
            conv_ = scratch(convDim);
        }
        if (hasHead_) logits_ = scratch(h_.vocab);
        if (hasEmbed_) {
            cl_int status;
            embedRow_ = trackedCreateBuffer(d_.context, CL_MEM_READ_ONLY, embed_.rowBytes, nullptr, &status);
            check(status, "clCreateBuffer(embed row)");
            owned_.push_back(embedRow_);
            embedReader_.reset(new FileReader(embed_.file));
            embedHost_.resize(embed_.rowBytes);
        }
        if (!layers_.empty()) buildRopeTable();
    }

    void buildRopeTable() {
        int half = h_.ropeDims / 2;
        std::vector<float> freqs(half, 1.0f);
        if (!ropeFreqs_.file.empty()) {
            if (ropeFreqs_.rows * ropeFreqs_.cols != half) throw std::runtime_error("rope_freqs must have rope_dims/2 values");
            FileReader(ropeFreqs_.file).read(0, reinterpret_cast<char *>(freqs.data()), sizeof(float) * half);
        }
        std::vector<float> table((size_t)maxContext_ * 2 * half);
        for (int pos = 0; pos < maxContext_; pos++) {
            for (int i = 0; i < half; i++) {
                double angle = pos * h_.ropePositionScale * std::pow(h_.ropeTheta, -2.0 * i / h_.ropeDims);
                if (!ropeFreqs_.file.empty()) angle /= (double)freqs[i];
                table[(size_t)pos * 2 * half + i] = (float)std::cos(angle);
                table[(size_t)pos * 2 * half + half + i] = (float)std::sin(angle);
            }
        }
        cl_int status;
        rope_ = trackedCreateBuffer(d_.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * table.size(),
                               table.data(), &status);
        check(status, "clCreateBuffer(rope)");
        owned_.push_back(rope_);
    }

    void chooseWeights(const std::string &mode, int depth, int threads) {
        if (mode != "auto" && mode != "resident" && mode != "stream") throw std::runtime_error("weights must be auto, resident or stream");
        bool resident = mode == "resident";
        if (mode == "auto") {
            size_t budget = (size_t)d_.globalMem / 2;
            size_t physical = physicalMemory();
            if (physical) budget = std::min(budget, physical / 2);
            const char *override = std::getenv("RJLLM_RESIDENT_BUDGET");
            if (override) budget = (size_t)std::strtoull(override, nullptr, 10);
            std::lock_guard<std::mutex> lock(d_.mutex);
            resident = d_.residentBytes + weightBytes_ <= budget;
        }
        if (items_.empty()) return;
        if (resident) {
            std::vector<char> staging;
            for (const Item &item : items_) resident_.push_back(load(d_, queue_, item, staging));
            std::lock_guard<std::mutex> lock(d_.mutex);
            d_.residentBytes += weightBytes_;
            residentCounted_ = true;
        } else {
            streamer_.reset(new Streamer(d_, items_, std::max(1, depth), std::max(1, std::min(threads, depth))));
        }
    }

    void createKernels() {
        const char *names[] = {"rmsnorm_rows", "l2norm_rows", "add_inplace", "add_bias", "act_mul", "rope",
                               "kv_store", "attention", "dn_gate", "dn_conv", "dn_delta", "dn_gatenorm",
                               "copy_f32", "fill_zero"};
        for (const char *name : names) kernel(name);
    }

    cl_kernel kernel(const std::string &name) {
        auto it = kernels_.find(name);
        if (it != kernels_.end()) return it->second;
        cl_int status;
        cl_kernel k = clCreateKernel(d_.program, name.c_str(), &status);
        check(status, ("clCreateKernel(" + name + ")").c_str());
        kernels_[name] = k;
        return k;
    }

    void resetState() {
        for (Layer &l : layers_) {
            int kvWidth = h_.kvHeads * h_.headDim;
            if (l.attention) {
                zero(l.kCache, maxContext_ * kvWidth);
                zero(l.vCache, maxContext_ * kvWidth);
            } else {
                zero(l.deltaState, h_.timeStepRank * h_.stateSize * h_.stateSize);
                zero(l.convState, (2 * h_.groupCount * h_.stateSize + h_.innerSize) * (h_.convKernel - 1));
            }
        }
    }

    void destroy() {
        streamer_.reset();
        for (Buffers &buffers : resident_) release(buffers);
        resident_.clear();
        if (residentCounted_) {
            std::lock_guard<std::mutex> lock(d_.mutex);
            d_.residentBytes -= weightBytes_;
            residentCounted_ = false;
        }
        for (cl_mem buffer : owned_) trackedReleaseBuffer(buffer);
        owned_.clear();
        for (auto &entry : kernels_) clReleaseKernel(entry.second);
        kernels_.clear();
        if (queue_) clReleaseCommandQueue(queue_);
        queue_ = nullptr;
    }

    void ensureHidden(int n) {
        if (n <= hiddenCapacity_) return;
        cl_int status;
        cl_mem buffer = trackedCreateBuffer(d_.context, CL_MEM_READ_WRITE, sizeof(float) * (size_t)n * h_.dim, nullptr, &status);
        check(status, "clCreateBuffer(hidden)");
        if (hiddenBuf_) {
            clFinish(queue_);
            owned_.erase(std::remove(owned_.begin(), owned_.end(), hiddenBuf_), owned_.end());
            trackedReleaseBuffer(hiddenBuf_);
        }
        hiddenBuf_ = buffer;
        owned_.push_back(buffer);
        hiddenCapacity_ = n;
    }

    // ------------------------------------------------ kernel launch helpers

    void arg(cl_kernel k, cl_uint i, cl_mem value) { check(clSetKernelArg(k, i, sizeof(cl_mem), &value), "clSetKernelArg"); }
    void arg(cl_kernel k, cl_uint i, int value) {
        cl_int v = value;
        check(clSetKernelArg(k, i, sizeof(cl_int), &v), "clSetKernelArg");
    }
    void arg(cl_kernel k, cl_uint i, float value) {
        cl_float v = value;
        check(clSetKernelArg(k, i, sizeof(cl_float), &v), "clSetKernelArg");
    }

    void args(cl_kernel, cl_uint) {}
    template <typename T, typename... Rest>
    void args(cl_kernel k, cl_uint i, T value, Rest... rest) {
        arg(k, i, value);
        args(k, i + 1, rest...);
    }

    // Elementwise launch over n items (driver picks the local size).
    template <typename... Ts>
    void launch(const std::string &name, size_t n, Ts... values) {
        cl_kernel k = kernel(name);
        args(k, 0, values...);
        size_t global = std::max<size_t>(n, 1);
        check(clEnqueueNDRangeKernel(queue_, k, 1, nullptr, &global, nullptr, 0, nullptr, nullptr), name.c_str());
    }

    // One work group of WG items per group.
    template <typename... Ts>
    void launchGroups(const std::string &name, size_t groups, Ts... values) {
        cl_kernel k = kernel(name);
        args(k, 0, values...);
        size_t local = (size_t)d_.wg, global = groups * local;
        check(clEnqueueNDRangeKernel(queue_, k, 1, nullptr, &global, &local, 0, nullptr, nullptr), name.c_str());
    }

    void zero(cl_mem buffer, int n) { launch("fill_zero", (size_t)n, buffer, n); }
    void copy(cl_mem src, int srcOff, cl_mem dst, int dstOff, int n) { launch("copy_f32", (size_t)n, src, srcOff, dst, dstOff, n); }

    void rmsnorm(cl_mem x, int xOff, int xStride, cl_mem w, cl_mem out, int outOff, int outStride, int rows, int cols) {
        launchGroups("rmsnorm_rows", (size_t)rows, x, xOff, xStride, w, out, outOff, outStride, cols, h_.eps);
    }

    int lanesPerRow(int chunks) const {
        int tpr = 4;
        while (tpr * 4 < chunks && tpr < d_.wg) tpr <<= 1;
        return std::min(tpr, d_.wg);
    }

    static cl_mem find(const Buffers &b, const std::string &role) {
        auto it = b.find(role);
        return it == b.end() ? nullptr : it->second;
    }

    static cl_mem need(const Buffers &b, const std::string &role) {
        cl_mem m = find(b, role);
        if (!m) throw std::runtime_error("missing weights for " + role);
        return m;
    }

    // out[outOff..] = W x[inOff..] (+ bias) (+ residual[resOff..]) for the tensor at `role`.
    void linear(const Layer *layer, const TensorRef &t, const Buffers &b, const std::string &role, cl_mem in, int inOff,
                cl_mem out, int outOff, cl_mem residual, int resOff) {
        int tpr = lanesPerRow(t.chunks);
        int rowsPerGroup = d_.wg / tpr;
        size_t groups = (size_t)((t.rows + rowsPerGroup - 1) / rowsPerGroup);
        cl_mem bias = layer ? find(b, role + "_bias") : nullptr;
        launchGroups(std::string("matvec_") + TYPE_NAMES[t.type], groups, need(b, role), (int)t.rowBytes, t.chunks,
                     (int)t.rows, in, inOff, out, outOff, bias, residual, resOff, tpr);
    }

    void linear(const Layer &l, const Buffers &b, const std::string &role, cl_mem in, cl_mem out, int outOff,
                cl_mem residual = nullptr) {
        linear(&l, l.at(role), b, role, in, 0, out, outOff, residual, 0);
    }

    // x += mixer/ffn output, optionally normalized first (Gemma-2 style post norms).
    void project(const Layer &l, const Buffers &b, const std::string &role, cl_mem in, cl_mem x, bool postNorm,
                 const std::string &normRole) {
        if (!postNorm) {
            linear(l, b, role, in, x, 0, x);
            return;
        }
        linear(l, b, role, in, mixed_, 0);
        rmsnorm(mixed_, 0, h_.dim, need(b, normRole), mixed_, 0, h_.dim, 1, h_.dim);
        launch("add_inplace", (size_t)h_.dim, x, mixed_, h_.dim);
    }

    // ------------------------------------------------ forward pass

    void embed(int token, int outOff) {
        if (token < 0 || token >= embed_.rows) throw std::runtime_error("token id outside the embedding table: " + std::to_string(token));
        embedReader_->read((size_t)token * embed_.rowBytes, embedHost_.data(), embed_.rowBytes);
        check(clEnqueueWriteBuffer(queue_, embedRow_, CL_TRUE, 0, embed_.rowBytes, embedHost_.data(), 0, nullptr, nullptr),
              "clEnqueueWriteBuffer(embed)");
        launch(std::string("embed_") + TYPE_NAMES[embed_.type], (size_t)embed_.chunks, embedRow_, hiddenBuf_, outOff,
               h_.embedMultiplier);
    }

    void runLayer(Layer &l, const Buffers &b, cl_mem x, int pos) {
        rmsnorm(x, 0, h_.dim, need(b, "attn_norm"), xn_, 0, h_.dim, 1, h_.dim);
        if (l.attention) attention(l, b, x, pos);
        else deltaNet(l, b, x);
        rmsnorm(x, 0, h_.dim, need(b, l.ffnNorm), xn_, 0, h_.dim, 1, h_.dim);
        feedForward(l, b);
        project(l, b, "ffn_down", ffnHidden_, x, l.postFfnNorm, "post_ffw_norm");
    }

    void attention(Layer &l, const Buffers &b, cl_mem x, int pos) {
        int hd = h_.headDim, qWidth = h_.heads * hd, kvWidth = h_.kvHeads * hd;
        int qRows = l.fusedQkv ? qWidth : (int)l.at("attn_q").rows;
        int qStride = l.qGate ? 2 * hd : hd;
        if (l.fusedQkv) {
            linear(l, b, "attn_qkv", xn_, qkv_, 0);
        } else {
            linear(l, b, "attn_q", xn_, qkv_, 0);
            linear(l, b, "attn_k", xn_, qkv_, qRows);
            linear(l, b, "attn_v", xn_, qkv_, qRows + kvWidth);
        }
        if (l.qkNorm) {
            rmsnorm(qkv_, 0, qStride, need(b, "attn_q_norm"), qkv_, 0, qStride, h_.heads, hd);
            rmsnorm(qkv_, qRows, hd, need(b, "attn_k_norm"), qkv_, qRows, hd, h_.kvHeads, hd);
        }
        int half = h_.ropeDims / 2;
        launch("rope", (size_t)h_.heads * half, qkv_, 0, qStride, h_.heads, half, rope_, pos, h_.ropeStyle);
        launch("rope", (size_t)h_.kvHeads * half, qkv_, qRows, hd, h_.kvHeads, half, rope_, pos, h_.ropeStyle);
        launch("kv_store", (size_t)kvWidth, qkv_, qRows, qkv_, qRows + kvWidth, l.kCache, l.vCache, pos, kvWidth);
        launchGroups("attention", (size_t)h_.heads, qkv_, 0, qStride, l.kCache, l.vCache, scores_, maxContext_, att_,
                     pos + 1, hd, kvWidth, h_.heads / h_.kvHeads, h_.attnScale, l.qGate ? hd : -1);
        project(l, b, "attn_output", att_, x, l.postAttnNorm, "post_attention_norm");
    }

    void deltaNet(Layer &l, const Buffers &b, cl_mem x) {
        int size = h_.stateSize, kHeads = h_.groupCount, vHeads = h_.timeStepRank;
        int keyDim = kHeads * size, convDim = 2 * keyDim + h_.innerSize;
        linear(l, b, "attn_qkv", xn_, qkv_, 0);
        linear(l, b, "attn_gate", xn_, z_, 0);
        linear(l, b, "ssm_beta", xn_, beta_, 0);
        linear(l, b, "ssm_alpha", xn_, alpha_, 0);
        launch("dn_gate", (size_t)vHeads, beta_, alpha_, need(b, "ssm_dt_bias"), need(b, "ssm_a"), decay_, vHeads);
        launch("dn_conv", (size_t)convDim, qkv_, need(b, "ssm_conv1d"), l.convState, conv_, convDim, h_.convKernel);
        launchGroups("l2norm_rows", (size_t)(2 * kHeads), conv_, 0, size, size, h_.eps);
        launchGroups("dn_delta", (size_t)vHeads, l.deltaState, conv_, beta_, decay_, att2_, size, kHeads, keyDim);
        launchGroups("dn_gatenorm", (size_t)vHeads, att2_, need(b, "ssm_norm"), z_, att_, size, h_.eps);
        project(l, b, "ssm_out", att_, x, l.postAttnNorm, "post_attention_norm");
    }

    void feedForward(Layer &l, const Buffers &b) {
        int ffn = l.ffnDim;
        if (l.ffnKind == 2) {
            linear(l, b, "ffn_up", xn_, ffn_, 0);
            launch("act_mul", (size_t)ffn, ffn_, 0, -1, ffnHidden_, ffn, h_.activation);
            return;
        }
        const TensorRef &up = l.at("ffn_up");
        bool fused = l.ffnKind == 1;
        const TensorRef &gate = fused ? up : l.at("ffn_gate");
        bool biased = find(b, "ffn_up_bias") || find(b, "ffn_gate_bias");
        if (!biased && gate.type == up.type && gate.rowBytes == up.rowBytes) {
            int tpr = lanesPerRow(up.chunks);
            int rowsPerGroup = d_.wg / tpr;
            size_t groups = (size_t)((ffn + rowsPerGroup - 1) / rowsPerGroup);
            launchGroups(std::string("gateup_") + TYPE_NAMES[up.type], groups, need(b, fused ? "ffn_up" : "ffn_gate"), 0,
                         need(b, "ffn_up"), fused ? ffn : 0, (int)up.rowBytes, up.chunks, ffn, xn_, ffnHidden_,
                         h_.activation, tpr);
            return;
        }
        if (fused) {
            linear(l, b, "ffn_up", xn_, ffn_, 0);
        } else {
            linear(l, b, "ffn_gate", xn_, ffn_, 0);
            linear(l, b, "ffn_up", xn_, ffn_, ffn);
        }
        launch("act_mul", (size_t)ffn, ffn_, 0, ffn, ffnHidden_, ffn, h_.activation);
    }

    void runHead(const Buffers &b, int offset) {
        rmsnorm(hiddenBuf_, offset, h_.dim, need(b, "output_norm"), xn_, 0, h_.dim, 1, h_.dim);
        linear(nullptr, head_.at("output"), b, "output", xn_, 0, logits_, 0, nullptr, 0);
    }

    static std::string escape(const std::string &s) {
        std::string out;
        for (char c : s) {
            if (c == '"' || c == '\\') out += '\\';
            if ((unsigned char)c >= 0x20) out += c;
        }
        return out;
    }

    Device &d_;
    Hyper h_;
    int maxContext_ = 0;
    bool hasEmbed_ = false, hasHead_ = false;
    TensorRef embed_, ropeFreqs_;
    std::vector<Layer> layers_;
    std::map<std::string, TensorRef> head_;
    std::vector<Item> items_;
    size_t weightBytes_ = 0;
    bool residentCounted_ = false;

    cl_command_queue queue_ = nullptr;
    std::map<std::string, cl_kernel> kernels_;
    std::vector<cl_mem> owned_;
    std::vector<Buffers> resident_;
    std::unique_ptr<Streamer> streamer_;
    std::unique_ptr<FileReader> embedReader_;
    std::vector<char> embedHost_;

    cl_mem hiddenBuf_ = nullptr, work_ = nullptr, xn_ = nullptr, mixed_ = nullptr, qkv_ = nullptr, att_ = nullptr,
           att2_ = nullptr, scores_ = nullptr, ffn_ = nullptr, ffnHidden_ = nullptr, z_ = nullptr, beta_ = nullptr,
           alpha_ = nullptr, decay_ = nullptr, conv_ = nullptr, logits_ = nullptr, rope_ = nullptr, embedRow_ = nullptr;
    int hiddenCapacity_ = 0;

    int nextPos_ = 0;
    bool broken_ = false;
    long long steps_ = 0, tokens_ = 0;
    double lastStepMs_ = 0, lastWaitMs_ = 0, totalStepMs_ = 0;
    std::string info_;
};

}  // namespace rjllm

// ---------------------------------------------------------------- C API

struct rjllm_session {
    std::mutex mutex;
    std::unique_ptr<rjllm::Session> session;
};

static void setError(char *err, size_t len, const char *message) {
    if (err && len) {
        std::strncpy(err, message, len - 1);
        err[len - 1] = '\0';
    }
}

extern "C" {

RJLLM_API rjllm_session *rjllm_open(const char *graph_json, char *err, size_t err_len) {
    try {
        std::unique_ptr<rjllm_session> handle(new rjllm_session());
        handle->session.reset(new rjllm::Session(graph_json ? graph_json : ""));
        return handle.release();
    } catch (const std::exception &e) {
        setError(err, err_len, e.what());
    } catch (...) {
        setError(err, err_len, "unknown error");
    }
    return nullptr;
}

RJLLM_API size_t rjllm_output_size(const rjllm_session *session, int n) {
    return session ? session->session->outputSize(n) : 0;
}

RJLLM_API int rjllm_step(rjllm_session *session, const int32_t *tokens, const float *hidden, int n, int pos, float *out,
                         size_t out_len, char *err, size_t err_len) {
    if (!session) {
        setError(err, err_len, "null session");
        return 1;
    }
    try {
        std::lock_guard<std::mutex> lock(session->mutex);
        session->session->step(tokens, hidden, n, pos, out, out_len);
        return 0;
    } catch (const std::exception &e) {
        setError(err, err_len, e.what());
    } catch (...) {
        setError(err, err_len, "unknown error");
    }
    return 1;
}

RJLLM_API void rjllm_reset(rjllm_session *session) {
    if (!session) return;
    std::lock_guard<std::mutex> lock(session->mutex);
    session->session->reset();
}

RJLLM_API const char *rjllm_info(rjllm_session *session) {
    if (!session) return "{}";
    std::lock_guard<std::mutex> lock(session->mutex);
    return session->session->info();
}

RJLLM_API void rjllm_close(rjllm_session *session) { delete session; }

RJLLM_API int rjllm_prepare_capacity(char *err, size_t err_len) {
    try {
        (void)rjllm::device();
        return 0;
    } catch (const std::exception &e) {
        setError(err, err_len, e.what());
    } catch (...) {
        setError(err, err_len, "unknown native capacity preflight error");
    }
    return 1;
}

RJLLM_API const char *rjllm_capacity_info() {
    static thread_local std::string json;
    std::unique_lock<std::mutex> deviceLock(rjllm::g_deviceMutex, std::try_to_lock);
    if (!deviceLock.owns_lock() || !rjllm::g_device)
        return "{\"supportsStreaming\":true,\"supportsRuntime\":null,\"nativeReady\":null}";
    const rjllm::Device &d = *rjllm::g_device;
    std::ostringstream out;
    out << "{\"supportsStreaming\":true,\"supportsRuntime\":true,\"nativeReady\":true,\"gpuAvailableBytes\":null,\"unifiedMemory\":"
        << (d.unifiedMemory ? "true" : "false");
    if (d.type & CL_DEVICE_TYPE_GPU) {
        auto quoted = [](const std::string &text) {
            std::string value = "\"";
            static const char hex[] = "0123456789abcdef";
            for (unsigned char c : text) {
                if (c == '"' || c == '\\') { value += '\\'; value += (char)c; }
                else if (c < 0x20) {
                    value += "\\u00";
                    value += hex[c >> 4];
                    value += hex[c & 15];
                } else value += (char)c;
            }
            return value + "\"";
        };
        out << ",\"gpuTotalBytes\":";
        if (d.globalMem) out << d.globalMem;
        else out << "null";
        out << ",\"gpuMaxAllocationBytes\":";
        if (d.maxAlloc) out << d.maxAlloc;
        else out << "null";
        out << ",\"gpuDeviceName\":" << quoted(d.name)
            << ",\"gpuDeviceVendor\":" << quoted(d.vendor)
            << ",\"gpuDriverVersion\":" << quoted(d.driverVersion)
            << ",\"gpuRuntime\":\"OpenCL\",\"gpuRuntimeVersion\":" << quoted(d.runtimeVersion);
        // All live runtime OpenCL buffers, including streamed weights, scratch, KV and recurrent state.
        {
            std::lock_guard<std::mutex> lock(rjllm::g_allocationMutex);
            out << ",\"gpuAllocatedBytes\":" << rjllm::g_allocatedBytes;
        }
        std::unique_lock<std::mutex> weightLock(rjllm::g_device->mutex, std::try_to_lock);
        out << ",\"gpuResidentBytes\":";
        if (weightLock.owns_lock()) out << d.residentBytes;
        else out << "null";
    }
    out << "}";
    json = out.str();
    return json.c_str();
}

}  // extern "C"
