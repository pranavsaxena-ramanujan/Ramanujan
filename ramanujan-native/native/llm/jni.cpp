// JNI bindings for in.ramanujan.rule.engine.LlmSession.
#include <jni.h>

#include <string>
#include <vector>

#include "rjllm.h"

static void throwState(JNIEnv *env, const char *message) {
    jclass type = env->FindClass("java/lang/IllegalStateException");
    if (type) env->ThrowNew(type, message);
}

static rjllm_session *handle(JNIEnv *env, jlong value) {
    if (value == 0) throwState(env, "LLM session is closed");
    return reinterpret_cast<rjllm_session *>(value);
}

extern "C" {

JNIEXPORT jlong JNICALL Java_in_ramanujan_rule_engine_LlmSession_nativeOpen(JNIEnv *env, jclass, jstring graph) {
    if (!graph) {
        throwState(env, "graph is null");
        return 0;
    }
    const char *text = env->GetStringUTFChars(graph, nullptr);
    std::string json(text ? text : "");
    env->ReleaseStringUTFChars(graph, text);
    char err[4096] = {0};
    rjllm_session *session = rjllm_open(json.c_str(), err, sizeof(err));
    if (!session) throwState(env, err);
    return reinterpret_cast<jlong>(session);
}

JNIEXPORT jfloatArray JNICALL Java_in_ramanujan_rule_engine_LlmSession_nativeStep(JNIEnv *env, jclass, jlong value,
                                                                                    jintArray tokens, jfloatArray hidden,
                                                                                    jint n, jint pos) {
    rjllm_session *session = handle(env, value);
    if (!session) return nullptr;
    std::vector<int32_t> ids;
    std::vector<float> states;
    if (tokens) {
        if (env->GetArrayLength(tokens) < n) {
            throwState(env, "fewer token ids than n");
            return nullptr;
        }
        ids.resize((size_t)n);
        env->GetIntArrayRegion(tokens, 0, n, reinterpret_cast<jint *>(ids.data()));
    }
    if (hidden) {
        states.resize((size_t)env->GetArrayLength(hidden));
        env->GetFloatArrayRegion(hidden, 0, (jsize)states.size(), states.data());
    }
    size_t size = rjllm_output_size(session, n);
    std::vector<float> out(size);
    char err[4096] = {0};
    if (rjllm_step(session, tokens ? ids.data() : nullptr, hidden ? states.data() : nullptr, n, pos, out.data(), size, err,
                   sizeof(err)) != 0) {
        throwState(env, err);
        return nullptr;
    }
    jfloatArray result = env->NewFloatArray((jsize)size);
    if (result) env->SetFloatArrayRegion(result, 0, (jsize)size, out.data());
    return result;
}

JNIEXPORT void JNICALL Java_in_ramanujan_rule_engine_LlmSession_nativeReset(JNIEnv *env, jclass, jlong value) {
    rjllm_session *session = handle(env, value);
    if (session) rjllm_reset(session);
}

JNIEXPORT jstring JNICALL Java_in_ramanujan_rule_engine_LlmSession_nativeInfo(JNIEnv *env, jclass, jlong value) {
    rjllm_session *session = handle(env, value);
    return session ? env->NewStringUTF(rjllm_info(session)) : nullptr;
}

JNIEXPORT void JNICALL Java_in_ramanujan_rule_engine_LlmSession_nativeClose(JNIEnv *, jclass, jlong value) {
    rjllm_close(reinterpret_cast<rjllm_session *>(value));
}

JNIEXPORT jstring JNICALL Java_in_ramanujan_rule_engine_LlmSession_nativeCapacityInfo(JNIEnv *env, jclass) {
    try {
        return env->NewStringUTF(rjllm_capacity_info());
    } catch (const std::exception &e) {
        throwState(env, e.what());
        return nullptr;
    }
}

JNIEXPORT void JNICALL Java_in_ramanujan_rule_engine_LlmSession_nativePrepareCapacity(JNIEnv *env, jclass) {
    char err[4096] = {0};
    if (rjllm_prepare_capacity(err, sizeof(err)) != 0) throwState(env, err);
}

}  // extern "C"
