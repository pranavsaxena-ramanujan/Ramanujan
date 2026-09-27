//
// Created by Pranav on 09/06/24.
//

#include "ArrayValue.h"
#include "../DataContainerValueFunctionCommandRE.h"

#include <fstream>
#include <mutex>
#include <sys/stat.h>
#include <fcntl.h>
#ifdef _WIN32
#else
#include <sys/mman.h>
#include <unistd.h>
#endif


// Global cache: binaryFilePath -> {float* data, int count}
static std::mutex                                              s_binaryMutex;
struct CachedBinary {
    float* data;
    int count;
    size_t bytes;
    bool mapped;
};
static std::unordered_map<std::string, CachedBinary> s_binaryCache;
static std::unordered_map<std::string, CachedBinary> s_residentState;

void ArrayValue::adviseBinaryCacheIdle() {
#ifndef _WIN32
    std::lock_guard<std::mutex> lock(s_binaryMutex);
    for (const auto& entry : s_binaryCache) {
        if (entry.second.mapped) {
            madvise(entry.second.data, entry.second.bytes, MADV_DONTNEED);
        }
    }
#endif
}

void ArrayValue::clearResidentState() {
    std::lock_guard<std::mutex> lock(s_binaryMutex);
    for (const auto& entry : s_residentState) {
        ALIGNED_FREE(entry.second.data);
    }
    s_residentState.clear();
}

void (*ArrayValue::releaseCachedGpuBuffers)() = nullptr;

void ArrayValue::clearBinaryCache() {
    if (releaseCachedGpuBuffers != nullptr) {
        releaseCachedGpuBuffers();
    }
    std::lock_guard<std::mutex> lock(s_binaryMutex);
    for (const auto& entry : s_binaryCache) {
#ifndef _WIN32
        if (entry.second.mapped) {
            munmap(entry.second.data, entry.second.bytes);
            continue;
        }
#endif
        ALIGNED_FREE(entry.second.data);
    }
    s_binaryCache.clear();
}

static bool isMutableRuntimeBinary(const std::string& path) {
    const size_t slash = path.find_last_of("/\\");
    const std::string name = slash == std::string::npos ? path : path.substr(slash + 1);
    const auto endsWith = [&name](const char* suffix) {
        const size_t suffixLength = std::char_traits<char>::length(suffix);
        return name.size() >= suffixLength
            && name.compare(name.size() - suffixLength, suffixLength, suffix) == 0;
    };
    return name == "hidden.bin" || endsWith("_state.bin")
        || endsWith("_k_cache.bin") || endsWith("_v_cache.bin");
}

static bool isResidentCache(const std::string& path) {
    const char* enabled = std::getenv("RAMANUJAN_RESIDENT_KV");
    if (enabled == nullptr || std::string(enabled) != "true") return false;
    const size_t slash = path.find_last_of("/\\");
    const std::string name = slash == std::string::npos ? path : path.substr(slash + 1);
    const auto endsWith = [&name](const char* suffix) {
        const size_t length = std::char_traits<char>::length(suffix);
        return name.size() >= length && name.compare(name.size() - length, length, suffix) == 0;
    };
    return endsWith("_k_cache.bin") || endsWith("_v_cache.bin");
}

ArrayValue::ArrayValue(Array* array , std::string originalArrayId) {
    this->array = array;

    dimensionSize = array->dimensionSize;
    dimensions = new int[dimensionSize];
    if(dimensionSize == 0) {
        return;
    }

    if(dimensionSize > 0) {
        sizeAtIndex = new int[dimensionSize];
    }
    int i = 0;
    for (int dim : array->dimension) {
        sizeAtIndex[i] = -1;
        dimensions[i++] = dim;
    }

    totalSize = getTotalSize(dimensions, 0, dimensionSize);

    // Fast path: load from binary float32 file directly into val[]
    if (!array->binaryFile.empty()) {
        // Check cache first (avoids re-reading disk on every kernel call)
        std::string key = array->binaryFile;
        const bool cacheable = !isMutableRuntimeBinary(key);
        const bool resident = isResidentCache(key);
        const std::string stateName = resident ? key.substr(key.find_last_of("/\\") + 1) : "";
    #ifndef _WIN32
        if (cacheable) {
            char* resolved = realpath(key.c_str(), nullptr);
            if (resolved != nullptr) {
            key = resolved;
            free(resolved);
            }
        }
    #endif
        float* fdata = nullptr;
        int fcount = 0;
        size_t allocatedBytes = (size_t)totalSize * sizeof(float);
        if (allocatedBytes == 0) allocatedBytes = sizeof(float);
        bool mapped = false;
        if (resident) {
            std::lock_guard<std::mutex> lk(s_binaryMutex);
            auto it = s_residentState.find(stateName);
            if (it != s_residentState.end()) {
                if (it->second.bytes != allocatedBytes) {
                    throw std::runtime_error("resident KV shape changed");
                }
                fdata = it->second.data;
                fcount = it->second.count;
            }
        }
        if (cacheable) {
            std::lock_guard<std::mutex> lk(s_binaryMutex);
            auto it = s_binaryCache.find(key);
            if (it != s_binaryCache.end()) {
            fdata  = it->second.data;
            fcount = it->second.count;
            }
        }
        if (!fdata) {
            // First time: read from disk, insert into cache
#ifdef _WIN32
            std::ifstream input(key, std::ios::binary | std::ios::ate);
            if (input) {
                const std::streamsize fileSize = input.tellg();
                fcount = static_cast<int>(fileSize / sizeof(float));
                if (fcount > totalSize) fcount = totalSize;

                fdata = static_cast<float*>(
                    _aligned_malloc(totalSize * sizeof(float), 4096));
                if (fdata) {
                    memset(fdata, 0, totalSize * sizeof(float));
                    input.seekg(0, std::ios::beg);
                    input.read(reinterpret_cast<char*>(fdata),
                               static_cast<std::streamsize>(fcount) * sizeof(float));
                }
            }
#else
            int fd = open(key.c_str(), O_RDONLY);
            if (fd >= 0) {
                struct stat st;
                fstat(fd, &st);
                size_t fileSize = st.st_size;
                fcount = (int)(fileSize / sizeof(float));
                if (fcount > totalSize) fcount = totalSize;

                size_t mapSize = allocatedBytes;

                if (cacheable) {
                    // Two-step mmap: anonymous region covers the full totalSize (tail past
                    // fcount is demand-zeroed, avoiding SIGBUS on a short file); MAP_FIXED
                    // overlays the immutable file data.
                    void* mapping = mmap(nullptr, mapSize, PROT_READ | PROT_WRITE,
                                        MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
                    if (mapping != MAP_FAILED) {
                        if (fcount > 0) {
                            mmap(mapping, (size_t)fcount * sizeof(float), PROT_READ,
                                 MAP_PRIVATE | MAP_FIXED, fd, 0);
                        }
                        fdata = static_cast<float*>(mapping);
                        mapped = true;
                    }
                } else if (ALIGNED_ALLOC(&fdata, 4096, mapSize) == 0 && fdata != nullptr) {
                    memset(fdata, 0, mapSize);
                    size_t bytesRemaining = (size_t)fcount * sizeof(float);
                    char* destination = reinterpret_cast<char*>(fdata);
                    while (bytesRemaining > 0) {
                        ssize_t bytesRead = read(fd, destination, bytesRemaining);
                        if (bytesRead <= 0) break;
                        destination += bytesRead;
                        bytesRemaining -= (size_t)bytesRead;
                    }
                }
                close(fd);
            }
#endif
            if (!fdata) {
                std::cerr << "[ArrayValue] Failed to open/allocate for binary file: " << key << std::endl;
#ifdef _WIN32
                fdata = static_cast<float*>(
                    _aligned_malloc(totalSize * sizeof(float), 4096));
                if (fdata != nullptr) {
#else
                if (ALIGNED_ALLOC(&fdata, 4096, totalSize * sizeof(float)) == 0 && fdata != nullptr) {
#endif
                    memset(fdata, 0, totalSize * sizeof(float));
                }
            }
            if (fdata && (cacheable || resident)) {
                std::lock_guard<std::mutex> lk(s_binaryMutex);
                if (resident) {
                    s_residentState[stateName] = {fdata, fcount, allocatedBytes, false};
                } else {
                    s_binaryCache[key] = {fdata, fcount, allocatedBytes, mapped};
                }
            }
        }
        val = fdata;
        isBinaryLoaded  = true;
        isCachedVal = cacheable || resident;
        cachedFloatData = isCachedVal ? fdata : nullptr;
    } else {
        ALIGNED_ALLOC(&val, 4096, totalSize * sizeof(float));
        memset(val, 0, totalSize * sizeof(float));
        
        // Slow path: parse string-keyed map
        for(auto & it : array->values) {
            std::string key = it.first;
            double value = it.second;
            //TODO: check if the size is faring correct.
            int* index = getIndexFromStr(key, dimensionSize);
            add(index, value);
            delete[] index;
        }
    }
}

void ArrayValue::add(int* index, double value) {
    int indexInt = translateIndex(index);
    val[indexInt] = (float)value;
}

void ArrayDataContainerValue::copyDataContainerValueFunctionCommandRE(DataContainerValueFunctionCommandRE* toBeCopied) {
    //delete arrayValue;
    //TODO: pranav: check if this is causing memory leak
    arrayValue->val = toBeCopied->arrayValuePtr;
}

void ArrayDataContainerValue::setValueInDataContainerValueFunctionCommandRE(DataContainerValueFunctionCommandRE* toBeSet) {
    // Clean up current array value if present
    toBeSet->arrayValuePtr = arrayValue->val;
}

void ArrayDataContainerValue::saveValueAndCopyFrom(DataContainerValueFunctionCommandRE* savedValue, DataContainerValue* source) {
    // Save current value
    savedValue->arrayValuePtr = arrayValue->val;
    oldValue = arrayValue;
    // Copy from source
    arrayValue = new ArrayValue(((ArrayDataContainerValue*) source)->arrayValue, true);
    delete oldValue;
}

void ArrayDataContainerValue::saveValueAndRestoreFrom(DataContainerValueFunctionCommandRE& savedValue, DataContainerValueFunctionCommandRE* restoreFrom) {
    // Save current value
    savedValue.arrayValuePtr = arrayValue->val;
    // Restore from saved value

    arrayValue->val = restoreFrom->arrayValuePtr;
}

void ArrayDataContainerValue::saveRestoreAndPropagate(DataContainerValueFunctionCommandRE* restoreFrom, DataContainerValue* propagateTo) {
    // Save current value (final computed result)
    placeholder = arrayValue->val;
    // Restore from previous saved value
    arrayValue->val = restoreFrom->arrayValuePtr;
    // Propagate final value to calling context
    ((ArrayDataContainerValue*)propagateTo)->arrayValue->val = placeholder;
}
