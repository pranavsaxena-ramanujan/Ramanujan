// libramanujan_llm: native OpenCL runtime for decoder-only LLM pipeline stages.
//
// A stage graph (JSON, emitted by sharded-llm/converter/ramanujan_shards/llm_graph.py) names
// a consecutive run of pieces of one model: optionally the token embedding, some decoder
// layers, and optionally the output head. A session owns the device buffers, KV caches and
// recurrent state for one stage and advances them token by token.
#pragma once

#include <stddef.h>
#include <stdint.h>

#if defined(_WIN32)
#define RJLLM_API __declspec(dllexport)
#else
#define RJLLM_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

typedef struct rjllm_session rjllm_session;

// Opens a session for a stage graph. Returns NULL and fills `err` on failure.
RJLLM_API rjllm_session *rjllm_open(const char *graph_json, char *err, size_t err_len);

// Values produced by rjllm_step for `n` tokens: the vocabulary logits of the last token when
// the stage has the head, otherwise n * dim hidden values.
RJLLM_API size_t rjllm_output_size(const rjllm_session *session, int n);

// Advances the stage by `n` tokens at positions pos .. pos + n - 1. Input is `tokens` when the
// stage has the embedding, otherwise `hidden` (n * dim floats). `pos` must equal the number
// of tokens already consumed, which detects lost state. Returns 0 on success.
RJLLM_API int rjllm_step(rjllm_session *session, const int32_t *tokens, const float *hidden, int n,
                         int pos, float *out, size_t out_len, char *err, size_t err_len);

// Clears KV caches and recurrent state; the next step starts at position 0.
RJLLM_API void rjllm_reset(rjllm_session *session);

// JSON description of the session (device, weights mode, bytes, timings). Owned by the session;
// valid until the next call on it.
RJLLM_API const char *rjllm_info(rjllm_session *session);

// Already-selected device telemetry; does not initialize OpenCL. Thread-local JSON.
// gpuAllocatedBytes counts live runtime buffers; gpuResidentBytes counts resident weights only.
// Neither includes other processes or driver overhead; free GPU memory remains unknown.
RJLLM_API const char *rjllm_capacity_info(void);

// Prepares the selected device/context/kernels without creating a session or touching existing state.
// Returns 0 on success; fills err on failure. Safe to call repeatedly.
RJLLM_API int rjllm_prepare_capacity(char *err, size_t err_len);

RJLLM_API void rjllm_close(rjllm_session *session);

#ifdef __cplusplus
}
#endif
