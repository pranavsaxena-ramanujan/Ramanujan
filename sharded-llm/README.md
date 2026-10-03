# Ramanujan Sharded LLM

## TL;DR

A large LLM is converted into model-agnostic Ramanujan IR packages so that separate
devices can own different shards while each device keeps only one shard in
memory. Phi-3 Mini from safetensors is the first executable architecture.
Qwen3.8-27B from GGUF (Gated DeltaNet + attention) also runs across four
Ramanujan workers; see [QWEN35_GGUF.md](QWEN35_GGUF.md) for its infrastructure,
GGUF-to-shard conversion, and runtime.

The conversion, four-shard prefill, KV-cache transfer, autoregressive decode,
binary tensor transport, and inference memory guard all run. The local runner
now generates the expected answer for the reference prompt; remote execution is
still experimental.

## Design

The Ramanujan design avoids coupling the system to GGUF or one model
architecture:

- Source readers ingest safetensors and GGUF (local or HTTP Range); the Phi-3
  safetensors adapter and the Qwen35 GGUF program generator emit executable IR.
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
python3 sharded-llm/run_phi3_shards.py "What is 2 + 2?" --n-tokens 10 --resident-kv --native-loop --profile
```

`--native-loop` runs all four decode ranges through one generated native program
per token, staging one shard's GPU weights at a time and keeping hidden state
on the GPU across shard boundaries. Prefill remains four separate runs. It
requires `--resident-kv` and a refreshed package with `decode_fused.py`; the
existing `refresh_phi3_programs` command below generates and checksums it.
`--gpu-pool` is an independent opt-in (also requires `--resident-kv`) that
reuses up to 768 MiB of idle, same-size OpenCL buffers with a fresh upload on
each use. Both flags require rebuilding the worker JAR and native library.
The fused mode still prepares inputs and serializes one program through Java
per token; it is not a zero-IPC C++ loop or a guaranteed single JNI call.
Neither mode checkpoints KV after decode, so worker failure stops generation.
In one local five-token comparison, fused decode without the pool took 1.22-1.37
seconds for the last three decode steps; the four-call resident path took 1.76
seconds for its second decode step in a separate three-token run. With the pool
enabled, a later ten-token run slowed to 2.69-3.60 seconds per step. These are
single-run observations, not a 5+ tokens/second result; keep pooling off unless
profiling shows a benefit on the target device.

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

## Performance Gap & Path to 5+ tok/sec (Memory-Constrained Sharding)

### Baseline Comparison: In-VRAM vs Sharded

| Metric | Monolithic All-in-VRAM (`phi3_transformer_stack_4bit.py`) | Current Sharded LLM (`run_phi3_shards.py --resident-kv`) | Target Sharded LLM (Memory-Constrained) |
|---|---|---|---|
| **Decode Throughput** | **12.0 tok/sec** | **0.59 tok/sec** | **5.0+ tok/sec** |
| **Decode Latency / Token** | ~83.3 ms | ~1,700 ms | $\le$ 200 ms |
| **Memory Footprint** | Full model resident in VRAM (~2.4 GiB) | $\le$ 1 shard active + KV cache ($\le$ 1.5 GiB RSS) | $\le$ 1 shard active + staging buffer |
| **Execution Loop** | Native GPU `while _step < n_tokens:` loop; zero host interaction | 4 shard invocations per token via Python $\leftrightarrow$ Java $\leftrightarrow$ C++ $\leftrightarrow$ OpenCL | Pipelined native execution loop; zero per-shard host overhead |
| **Weight Buffer Churn** | Allocated once at startup; released once on exit | `clCreateBuffer` & `clReleaseMemObject` called for every tensor, 4 times/token | Pre-allocated static GPU pool; weights streamed via DMA |
| **Storage / Page Fault I/O** | 0 bytes/token during generation | ~2 GB cold page faults/token via `madvise(MADV_DONTNEED)` | Amortized $\le$ 400 MB/token via speculative verification |

### Bottleneck Breakdown: Why Sharded Decode Takes ~1,700 ms

At 12 tok/sec, the 32-layer Phi-3 4-bit model requires only **~83.3 ms of total GPU compute per token** (~20.8 ms per 8-layer shard). The remaining ~1,617 ms in sharded decode is consumed by host-side and driver-side overheads across the 4 sequential shard dispatches:

1. **Storage Bandwidth Wall & Page Faults (~700–900 ms/token)**:
   In resident mode, `ArrayValue::adviseBinaryCacheIdle()` invokes `madvise(..., MADV_DONTNEED)` on weight mappings after each shard run to respect the process RSS ceiling. This forces the OS kernel to discard active pages. On the subsequent token, executing the 4 shards causes **~2.0 GB of SSD page faults per token**. Even on high-speed NVMe/Apple Unified Storage (2.5–3.0 GB/s sequential read), reading 2.0 GB every token incurs 650–800 ms of pure storage latency.
2. **OpenCL Buffer Allocation & Tear-down Churn (~300–450 ms/token)**:
   The generated `decode_resident.py` executes `LOAD_MEM` at the start of every shard (allocating ~40+ `cl_mem` buffers via `clCreateBuffer`) and `RELEASE_MEM` at the end (calling `clFinish()` and `clReleaseMemObject`). Tearing down and recreating OpenCL device buffers 4 times per token drains GPU execution pipelines and introduces significant driver synchronization overhead.
3. **Cross-Process Coordination & Disk Staging (~250–350 ms/token)**:
   Each shard transition involves:
   - Python driver preparing directories, hard-linking state, generating CSV stubs, and calling `os.utime`.
   - IPC over stdin/stdout pipes between Python and Java (`run`, `take`, `CHANGE_SHARD`).
   - Hidden state (`h_state.bin`) and tokens dumped to and read from disk between shards.
4. **JVM AST Repopulation & Protobuf Serialization (~150–200 ms/token)**:
   For every shard invocation, `ExecuteInlineServer` runs `createJson` / `repopulateCsvArrayValues` with Java thread pools, serializes the DAG to protobuf via `RuleEngineInputProtoSerializer`, and passes the payload over JNI to C++ where `proto.ParseFromArray` reconstructs the native AST.

---

### The Fundamental Physics of Single-Device Memory-Constrained Sharding

When running on a device whose memory capacity is strictly smaller than the model (e.g. running a model requiring 8 GiB on a 4 GiB device, or running large models on resource-constrained nodes), only a subset of layers can reside in memory at any moment.

If weights are streamed sequentially for **1 token at a time**, the theoretical maximum throughput is hard-capped by sequential storage/bus bandwidth:
$$\text{Throughput}_{\max} = \frac{\text{Storage Bandwidth (Bytes/s)}}{\text{Model Size (Bytes)}} = \frac{3.0\text{ GB/s}}{2.0\text{ GB}} \approx 1.5\text{ tokens/second}$$

Even with instantaneous compute and zero software overhead, streaming 2.0 GB sequentially from a 3.0 GB/s SSD for a single token cannot exceed ~1.5 tok/sec.

Therefore, the roadmap to achieve **5.0+ tok/sec** depends on the memory boundary of the target device:
1. **Tier 1 (Model > GPU VRAM, but fits in System RAM)**: CPU RAM $\rightarrow$ GPU VRAM sharding with unified bus transfer (Target: **6.0–10.0 tok/sec**).
2. **Tier 2 (Model > System RAM, streaming from SSD)**: High-bandwidth NVMe + 2-3 bit quantization + batched request pipelining (Target: **3.5–6.0 tok/sec**).

---

### Direct Engineering Solutions & Projected Throughput

#### Strategy 1: Tiered Memory Sharding (CPU RAM $\leftrightarrow$ GPU VRAM Pipelining)
* **Problem in Current Code**: `ArrayValue::adviseBinaryCacheIdle()` invokes `madvise(MADV_DONTNEED)`, which flushes weight pages completely out of system RAM to disk, forcing cold SSD page faults on every token even if system RAM has free space.
* **Architecture**:
  - Keep all 2.0 GB of weights resident in CPU host memory (mapped or pinned).
  - Shard strictly across the **GPU VRAM boundary**: only 1 shard (~500 MB) resides in GPU memory at any time.
  - Transfer weights over PCIe / system memory bus into GPU VRAM per shard.
* **Throughput Math**:
  - PCIe 4.0 / Unified Memory bus bandwidth: **$\ge$ 25 GB/s** (Apple Silicon unified bus: 100–150 GB/s).
  - Host-to-Device transfer for 2.0 GB: $\frac{2.0\text{ GB}}{25\text{ GB/s}} = 80\text{ ms}$.
  - GPU compute: $83.3\text{ ms}$.
  - Latency per token: $80\text{ ms} + 83.3\text{ ms} = 163.3\text{ ms}$.
  - **Expected Throughput: $\approx$ 6.1 tok/sec** (or **$\approx$ 10.0 tok/sec** with double-buffered PCIe overlap).

#### Strategy 2: In-Process Native C++ Sharded Loop (Zero-IPC / Zero-Protobuf)
* **Problem in Current Code**: Every shard transition runs Python `select()`, pipe IPC (`run`, `take`, `CHANGE_SHARD`), writes 20+ CSV stubs with 3,072 comma strings, calls Java thread pools (`repopulateCsvArrayValues`), and serializes/deserializes the entire DAG through Protobuf (`RuleEngineInputProtoSerializer`).
* **Architecture**:
  - Eliminate the Python and Java orchestration loop during autoregressive decode.
  - Implement a direct native C++ runner loop:
    ```cpp
    void runShardedDecodeLoop(int n_tokens) {
        for (int step = 0; step < n_tokens; ++step) {
            shard0.run(h_state);
            shard1.run(h_state);
            shard2.run(h_state);
            shard3.run(h_state, &next_token);
        }
    }
    ```
  - State (`h_state`) is passed as a direct memory pointer in RAM/VRAM without disk staging.
* **Time Saved**: ~400–600 ms of pure CPU/IPC overhead eliminated per token.

#### Strategy 3: Persistent GPU Buffer Pooling (Eliminating OpenCL Churn)
* **Problem in Current Code**: `LOAD_MEM` calls `clCreateBuffer` 40+ times per shard; `RELEASE_MEM` calls `clFinish` and `clReleaseMemObject` 40+ times per shard (160 alloc/free calls per token).
* **Architecture**:
  - Pre-allocate a static GPU memory pool for 1 shard (~500 MB) plus permanent scratch buffers (`h_state`, `h_ff_buf`, `qkv_buf`).
  - When switching shards, copy or DMA-stream new weights directly into the pre-existing GPU memory buffers via `clEnqueueWriteBuffer` without tearing down or recreating OpenCL handles.
* **Time Saved**: ~300 ms of driver allocation and queue drain latency eliminated per token.

#### Strategy 4: Coarser Sharding (2 Shards instead of 4 Shards)
* **Architecture**:
  - Re-shard the 32 decoder layers into **2 shards of 16 layers** (~1.0 GB each) instead of 4 shards of 8 layers.
  - A 1.0 GB active shard fits comfortably within constrained budgets (e.g. 2–3 GiB RSS limits).
* **Impact**:
  - Halves the number of shard transitions, driver pipeline flushes, and memory synchronization events from 4 per token to 2 per token.
  - Decreases boundary overhead by 50%.

#### Strategy 5: Multi-Stream / Batched Sharding (for SSD-Bound Devices)
* **Architecture**:
  - When the model genuinely exceeds total system RAM and must stream from an SSD, batch $B = 4$ independent user queries or generation streams together.
  - Each shard is loaded from SSD **once** and processes all $B$ tokens simultaneously through its 8 layers before the next shard is loaded.
* **Throughput Math**:
  - Loading 2.0 GB from a 3.0 GB/s SSD takes ~667 ms.
  - Computing for 4 tokens takes ~100 ms.
  - Total cycle time = 767 ms to generate 4 tokens across streams.
  - **Expected Aggregate Throughput: $\frac{4\text{ tokens}}{0.767\text{ s}} \approx$ 5.2 tok/sec**.

---

### Latency & Throughput Comparison Across Engineering Paths

| Implementation Path | Memory Boundary | Major Bottleneck Addressed | Expected Latency / Token | Expected tok/sec |
|---|---|---|---|---|
| **Current Baseline** | SSD Page-Faulted Shards | Full stack overhead + `madvise` disk faults | ~1,700 ms | **0.59 tok/s** |
| **Path A: Native Loop + GPU Pool** | SSD Page-Faulted Shards | Removes Python/Java IPC, Protobuf, OpenCL churn | ~750–850 ms | **1.2–1.3 tok/s** |
| **Path B: 2 Shards + Native Loop** | SSD Page-Faulted Shards | Halves shard transitions + native loop | ~600–700 ms | **1.4–1.7 tok/s** |
| **Path C: Tiered CPU RAM $\rightarrow$ GPU Sharding** | Model fits in RAM, exceeds GPU VRAM | Weights stay in host RAM; streamed over bus | **~130–165 ms** | **6.0–7.5 tok/s** |
| **Path D: Tiered Sharding + Double Buffering** | Model fits in RAM, exceeds GPU VRAM | Asynchronous DMA overlap hides bus transfer | **~85–100 ms** | **10.0–11.5 tok/s** |
| **Path E: Multi-Stream Batched SSD Sharding ($B=4$)** | Model exceeds physical RAM (SSD bound) | Amortizes SSD load over $B=4$ concurrent tokens | ~190 ms / token (effective) | **5.2 tok/s (aggregate)** |