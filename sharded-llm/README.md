# Ramanujan Sharded LLM

## TL;DR

This work adapts the model-sharding idea used by SwarmLLM to Ramanujan. A large
LLM is converted into model-agnostic Ramanujan IR packages so that separate
devices can own different shards while each device keeps only one shard in
memory. Phi-3 Mini is the first implemented architecture and safetensors is the
first implemented source format.

The conversion, four-shard prefill, KV-cache transfer, autoregressive decode,
binary tensor transport, and inference memory guard all run. The local runner
now generates the expected answer for the reference prompt; remote execution is
still experimental.

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
	default worker process ceiling is 7 GiB. A worker that crosses the ceiling
	is killed and its shard is retried once from disk-backed inputs.
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

A 16-token run with a fresh worker per shard produced:

```text
token ids: [450, 2533, 310, 29871, 29906, 322, 29871, 29906, 338, 29871, 29946,
            29889, 910, 338, 263, 6996]
text:      The sum of 2 and 2 is 4. This is a basic
peak JVM RSS: approximately 0.85 GiB
```

A 12-token single-worker run produced `The sum of 2 and 2 is 4.` with peak
JVM RSS of approximately 1.46 GiB. A 6-token run started one JVM, compiled
each of the eight shard programs once, reused that IR for the remaining 16
shard runs, and took 4.4-6.2 seconds per decode token. Diagnostics show that
each decode shard changes only the current hidden-state row, preserves previous
K/V rows, and writes a nonzero current K/V row.

## Resolved Decode Issue

Decode inputs were passed as symlinks named `h_state.bin` and
`l*_k_cache.bin`/`l*_v_cache.bin`. Translation canonicalizes binary paths, so
native saw the symlink targets' names and treated mutable state as immutable
weights. It mapped those files read-only, so decode writes were lost before
`RETURN`. The runner now passes state through hard links, falling back to a
copy, so native sees the logical state names and allocates writable buffers.

Risks to validate before remote execution are:

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
python3 sharded-llm/run_phi3_shards.py "What is 2 + 2?" --n-tokens 10 --resident-kv --profile
```

`--resident-kv` is an opt-in sequential-shard mode. Prefill checkpoints each
shard's cache once; subsequent decode calls keep all four shards' K/V arrays in
the worker's native memory and return only hidden state and the token. The
generated `decode_resident.py` also fuses the two projection/residual pairs in
each layer. Immutable weight mappings remain addressable across shard changes;
the worker advises the OS that idle mapped pages can be reclaimed. This is an
advisory hint, not a guarantee that weights stay off RAM or that SSD reads occur
on every token. The mode reuses Java shape stubs and skips forced per-run GC.
It requires a server JAR and native library rebuilt with `resetShardSession`.
For an existing package, regenerate programs and checksums using
`python3 -m ramanujan_shards.refresh_phi3_programs --package-dir <path> --reference-kernel <path>`
from the converter directory.

The resident mode does **not** checkpoint K/V after decode. It disables decode
retries rather than silently restarting from stale prefill caches; use the
default file-backed path when restartable decode is required. It cannot be
combined with `--worker-per-shard` or `--diagnostics`. `--profile` reports
per-shard execution and result-transfer times.

On an M3 Air with 8 GiB, a 10-token run of `What is 2 + 2?` produced
`The sum of 2 and 2 is `, matching the expected prefix. The nine measured
decode steps totaled 15.831 seconds (0.57 tokens/second); excluding the first
decode step they averaged 0.59 tokens/second. The full run, including prefill,
took 22.664 seconds. The legacy mode on the same binaries took 4.41 and
4.09 seconds for the first two decode steps, versus 2.73 and 1.77 seconds in
a separate resident-mode run. Most remaining time is in the four kernel runs
per token, not file transfer. These results do not establish 10 tokens/second.

The runner starts one worker JVM for the whole generation and registers the
package with `REGISTER_SHARDS <model-manifest.json>`. The package stays on
disk; the worker validates its prefill/decode program paths. Each program is
compiled to IR on first use, and the worker keeps that IR (about 350-420 KB per
program) keyed by program path and input shapes. Shard weights stay on disk and
the runner binds them once per generation, so each shard run sends only hidden
and K/V state plus sequence length.

By default, after the caller dumps the completed shard's hidden state, K/V
caches, and token (if present), `CHANGE_SHARD <next-program.py>` clears the
native immutable-weight cache; the next `run` maps that shard's weights from
disk. In resident mode only the prefill caches are moved to disk, and shard
changes keep weight mappings while advising idle pages as reclaimable. The
runner processes shards sequentially, enforces the RSS ceiling, and removes
temporary hidden-state and KV-cache files on exit. In the default mode, if the
worker breaches 7 GiB, the runner kills it, starts a new worker, re-registers
the package, and resumes from the same shard's disk-backed inputs. A second
breach on that shard stops the run. Resident mode cannot safely retry decode
after the worker exits because later K/V state is not checkpointed.

Legacy mode has explicit weight-cache eviction, not yet proof of complete native
heap cleanup: `ArrayRE::destroy()` does not free its per-call array objects.
Resident mode deletes per-call array objects after each native run. Use
`--worker-per-shard` to start a fresh JVM for every shard run; it
returns all native memory at each shard boundary but recompiles IR every time.
`--diagnostics` uses per-shard workers because its instrumented programs are
generated outside the registered package.

The runner explicitly loads `libnative_llm.dylib` from `--native-dir`. Other
Java callers continue to load `libnative.dylib`. The worker requires a
developer-console JAR and LLM native library built with `changeShard`. Homelab
and remote orchestrator dispatch do not yet use this protocol or provide a
durable cross-device commit protocol.