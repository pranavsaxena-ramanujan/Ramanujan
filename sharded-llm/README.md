# Ramanujan Sharded LLM

## TL;DR

This work adapts the model-sharding idea used by SwarmLLM to Ramanujan. A large
LLM is converted into model-agnostic Ramanujan IR packages so that separate
devices can own different shards while each device keeps only one shard in
memory. Phi-3 Mini is the first implemented architecture and safetensors is the
first implemented source format.

The conversion, four-shard prefill, KV-cache transfer, autoregressive decode,
binary tensor transport, and inference memory guard all run. The execution path
is still experimental: generated tokens are not yet numerically correct because
decode computes nonzero projections and cache rows but does not persist the
residual updates into the hidden state.

## Inspiration

SwarmLLM demonstrates GGUF model sharding across resource-constrained peers.
The equivalent Ramanujan design keeps the useful operational idea but avoids
coupling the system to GGUF or one model architecture:

- Source readers ingest formats such as safetensors now and GGUF later.
- Architecture adapters map model-specific tensors and execution structure.
- Tensor encoders produce Ramanujan-compatible representations.
- Package manifests describe shards, capabilities, entrypoints, tensors, state,
	checksums, and execution order.
- A device may store multiple shards, but the intended worker contract keeps
	only one active shard in memory.

## What Changed

- Added model-agnostic Java contracts in `sharded-llm-model` for packages,
	shards, DAG nodes, tensors, state, devices, sessions, work, and future
	training operations.
- Added a streaming Python converter with source-reader, architecture-adapter,
	and tensor-encoder interfaces.
- Added a Phi-3 adapter and row-wise symmetric 4-bit packing. Each group of six
	signed values is shifted into nibbles and packed into one float32 value.
- Added prefill and decode program generation from the existing Phi-3 kernel.
- Added package verification for structure, contiguous layer ranges, file
	sizes, and SHA-256 checksums.
- Added binary-backed array loading and binary return files to avoid serializing
	large tensors through CSV/JSON.
- Added a local runner that transfers hidden state and per-layer K/V caches
	between shard invocations.
- Added inference-only RSS monitoring. The JVM heap is capped at 6 GiB and the
	default process ceiling is 8 GiB.
- Added optional diagnostics for hidden-state and cache continuity.

Remote orchestrator, homelab scheduling, and worker placement are not wired to
these packages yet. The current runner proves the local execution protocol.

## Inputs

The current experiment uses:

- Model: `Phi-3-mini-4k-instruct`
- Source: local safetensors files
- Shape: 32 decoder layers, hidden size 3072, intermediate size 8192,
	vocabulary size 32064
- Prompt: `What is 2 + 2?`
- Chat input: `<|user|>\nWhat is 2 + 2?<|end|>\n<|assistant|>\n`
- Prompt token count: 11
- Shards: `[0,8)`, `[8,16)`, `[16,24)`, `[24,32)`

The generated package is written to `../phi3_rj_ir_shards` and contains four
independently described shard directories with weights, prefill/decode programs,
manifests, and checksums.

## Observed Output

Package verification completed for 340 files totaling 3,215,516,137 bytes.
The converter test suite completed with 8 passing tests.

A complete four-shard, 10-token run exited successfully under the memory limit:

```text
token ids: [450, 12116, 28666, 29373, 15898, 5629, 1087, 3075, 22042, 31516]
text:      Theoretesiarong Fich przviceconstrained然
peak JVM RSS: approximately 1.33 GiB
peak runner RSS: approximately 137 MiB
```

This output proves mechanical prefill/decode execution and state transfer, but
it is not a correct answer to the prompt and must not be treated as a model
quality result.

The latest two-token diagnostic produced `[450, 12116]` (`Theoret`). It also
showed:

- Prefill hidden states and K/V cache rows are nonzero in all four shards.
- Decode preserves all previous cache rows and writes a nonzero current row.
- Decode attention and MLP projection buffers are nonzero.
- The decoded hidden-state file remains bit-for-bit unchanged across each shard.

## Probable Current Issue

The strongest current hypothesis is an indexing or write-back defect in
`residual_add_decode_GPU_1`. It computes valid nonzero attention and MLP outputs,
but neither residual addition changes any hidden-state row. Its original kernel
used a floating sequence position directly as an array-row index, unlike the
working decode kernels that explicitly derive an integer `row_int`. The
generator now normalizes this residual index, but the refreshed package and
end-to-end output still need validation.

Other risks to validate before remote execution are:

- Numerical parity between the converted 4-bit weights and a trusted reference.
- Correct mutable-buffer behavior across JVM, JNI, OpenCL, and binary sidecars.
- Retry/idempotency semantics when a remote shard fails after updating KV state.
- Package transfer cost and checksum verification for roughly 3.2 GB of data.
- Device capability matching, shard placement, backpressure, and session expiry.
- Compatibility of future GGUF readers and non-Phi architectures with the same
	IR contracts.

## Convert And Verify

From the `ramanujan` repository:

```bash
zsh sharded-llm/scripts/convert_phi3_to_rj_shards.sh
zsh sharded-llm/scripts/convert_phi3_to_rj_shards.sh --verify-only
```

Override paths with `--model-dir`, `--output-dir`, and `--reference-kernel`.
The destination must not already exist during conversion. Failed conversion
removes its `.partial` output. Conversion itself has no RSS limit; memory is
enforced only while running inference.

## Run Locally

```bash
python3 sharded-llm/run_phi3_shards.py "What is 2 + 2?" --n-tokens 10
```

Use `--diagnostics` for bounded hidden/cache continuity statistics. The runner
starts a fresh JVM, processes shards sequentially, enforces the RSS ceiling, and
removes temporary hidden-state and KV-cache files on exit.