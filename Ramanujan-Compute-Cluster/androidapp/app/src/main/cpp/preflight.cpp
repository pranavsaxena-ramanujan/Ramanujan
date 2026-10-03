#include <jni.h>
#include <dlfcn.h>
#include <cstdint>
#include <vector>

extern "C" JNIEXPORT jstring JNICALL
Java_in_ramanujan_cluster_android_NativePreflight_checkOpenCl(JNIEnv *env, jclass) {
    void *library = dlopen("libOpenCL.so", RTLD_NOW | RTLD_LOCAL);
    if (!library) return env->NewStringUTF("OpenCL unavailable: the GPU vendor library is absent or inaccessible to this app.");
    using GetPlatforms = int (*)(unsigned int, void **, unsigned int *);
    using GetDevices = int (*)(void *, uint64_t, unsigned int, void **, unsigned int *);
    auto platforms = reinterpret_cast<GetPlatforms>(dlsym(library, "clGetPlatformIDs"));
    auto devices = reinterpret_cast<GetDevices>(dlsym(library, "clGetDeviceIDs"));
    const char *error = nullptr;
    unsigned int count = 0;
    if (!platforms || !devices || platforms(0, nullptr, &count) != 0 || count == 0) {
        error = "OpenCL unavailable: no accessible platform. This device cannot run the LLM worker.";
    } else {
        std::vector<void *> ids(count);
        bool found = false;
        if (platforms(count, ids.data(), nullptr) == 0) {
            for (void *id : ids) {
                unsigned int n = 0;
                if (devices(id, 4 /* CL_DEVICE_TYPE_GPU */, 0, nullptr, &n) == 0 && n > 0) found = true;
            }
        }
        if (!found) error = "OpenCL unavailable: no accessible GPU device. This device cannot run the LLM worker.";
    }
    dlclose(library);
    return error ? env->NewStringUTF(error) : nullptr;
}
