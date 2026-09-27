# Generic GGUF sharding

The converter accepts any local GGUF file or an HTTP(S) URL whose server supports
byte-range requests. It reads the tensor directory and copies tensor payloads in
4 MiB chunks without loading the full model into RAM. HTTP sources are read by
range, so only the header and the selected tensor bytes are fetched.

From this directory:

```sh
PYTHONPATH=. python3 -m ramanujan_shards.emit_gguf \
  --gguf /path/to/model.gguf --output-dir /path/to/model-shards --shards 4
PYTHONPATH=. python3 -m ramanujan_shards.verify_gguf \
  --package /path/to/model-shards
```

Use a URL for `--gguf` to stream from a remote server; a server that ignores
`Range` is rejected. The output directory must not already exist. On failure,
temporary output is removed. Each shard contains a manifest and persistent raw
tensor files with SHA-256 checksums. Numbered `blk.N.*` tensors are grouped into
contiguous layer ranges; other GGUFs fall back to tensor-level grouping. Every
tensor is assigned once. Unknown GGML quantization types are preserved as opaque
bytes, including any alignment padding, rather than silently transcoded.

On an individual device, add `--shard-index 1` (zero-based) to fetch only that
device's assigned shard from the URL and store it at `--output-dir`. Run the
same verifier on this partial package; all devices must use the same `--shards`
value so their assignments agree. This transfers weights to disk, not to a
running model session.

The resulting package has `status: weights-only`: GGUF supplies weights and
model metadata, not an operator graph. Qwen35 packages run through the
architecture backend described in "Running Qwen35 on Ramanujan" below; other
architectures need their own backend.

## Symbolic IR planning

To export a four-shard operator inventory, tensor bindings, and the original
GGUF header metadata without duplicating weights:

```sh
PYTHONPATH=. python3 -m ramanujan_shards.gguf_ir_plan \
  --package ~/Desktop/ramanujan_oss/Qwen3.8-27B-Q4_1-shards \
  --gguf ~/Desktop/ramanujan_oss/Qwen3.8-27B-Q4_1.gguf \
  --output-dir ~/Desktop/ramanujan_oss/Qwen3.8-27B-Q4_1-ir-plan
```

The output contains `model-plan.json`, one `plan.json` per shard, and
`gguf-metadata.json` (including tokenizer metadata). `status: planning-only`
is intentional: this is a symbolic plan, **not** Ramanujan Python DSL, a
compiled rule-engine protobuf. `--run` still fails; use `run_gguf_shards.py`
below. The Phi-3 runner has fixed 3072-dimensional F32 input and Phi-3-specific
weight names and must not be used with this package. The `nextn`/MTP block is
auxiliary, not part of the 64-block generation path.

## Running GGUF models on Ramanujan

`run_gguf_shards.py` runs any supported architecture (llama, qwen2, qwen3,
phi3, gemma, qwen35); see [../GGUF_MODELS.md](../GGUF_MODELS.md). The Qwen35
example below also works via the `run_qwen35_shards.py` alias.

Build `libnative_llm` (`make native_llm` in `ramanujan-native/native/build`)
and the developer-console fat JAR, then from `ramanujan/sharded-llm`:

```sh
python3 run_gguf_shards.py \
  --package ~/Desktop/ramanujan_oss/Qwen3.8-27B-Q4_1-shards \
  --metadata ~/Desktop/ramanujan_oss/Qwen3.8-27B-Q4_1-ir-plan/gguf-metadata.json \
  --prompt "The capital of France is" --max-new-tokens 8 \
  --work-dir /tmp/qwen35-run
```

Each shard gets its own persistent Ramanujan worker. Generated DSL programs
(`llm_programs.py`, from the `llm_spec.py` model spec) implement token-embedding decode, the Gated DeltaNet
layer, the gated attention layer and the Q6_K output head with argmax, all
executed as OpenCL kernels. Layers run one at a time: the runner binds that
layer's raw GGUF weights, returns the hidden state plus DeltaNet recurrent/conv
state or KV cache (`*_state.bin`, `*_k_cache.bin`, `*_v_cache.bin`) with `take`,
then sends `EVICT_WEIGHTS` so each worker holds about one layer of weights.
On an 8 GB Apple M3 the prompt above yields " Paris." at roughly 16 s per
token with each worker under 300 MB RSS. Decoding is greedy, and
`--max-context` bounds the KV cache.

`--check-layers N` compares the hidden state after each of the first N layers
with `llm_reference.py` (bit-identical on Qwen35 to `qwen35_reference.py`, a NumPy port of swarmllm's llama.cpp-checked
`ref_q38.mjs`); `--reference-token` also compares the first token's logits.
All 64 layers matched within 2e-5 relative error, and the output head chose the
same token with a maximum logit error of 5e-5.

The named Qwen3.8-27B Q4_0 release currently advertises 65 `blk.N` groups:
64 decoder blocks and one auxiliary `nextn`/MTP block. Its GGUF also mixes
Q4_0 with other quantization types. The generic packager retains all 65 groups
and all tensor types; the presence of the MTP block is not a runnable 65th
decoder stage.

## Verified operator probes

Rank-2 F32, Q4_1, Q5_K, and Q6_K GGUF tensors can be bound to standalone
Ramanujan matvec programs. Rank-1 F32 tensors can be bound to a standalone
RMSNorm program. For example, after building the translation and
developer-console modules, from the converter directory:

```sh
PYTHONPATH=. python3 -m ramanujan_shards.emit_q6_k_operator \
  --package ~/Desktop/ramanujan_oss/Qwen3.8-27B-Q4_1-shards \
  --tensor output.weight --output-dir /tmp/qwen-output-probe
cd ..
python3 run_q6_k_probe.py --binding /tmp/qwen-output-probe/binding.json \
  --rows 8 --jar ../developer-console/target/developer-console-1.0-SNAPSHOT-fat.jar
```

The other matvec emitters are `emit_q4_1_operator`, `emit_q5_k_operator`, and
`emit_f32_operator`; the matching probe entrypoints are `run_q4_1_probe.py`,
`run_q5_k_probe.py`, and `run_f32_probe.py`. For a Qwen attention norm use:

```sh
cd converter
PYTHONPATH=. python3 -m ramanujan_shards.emit_f32_rmsnorm_operator \
  --package ~/Desktop/ramanujan_oss/Qwen3.8-27B-Q4_1-shards \
  --tensor blk.0.attn_norm.weight --epsilon 9.999999974752427e-7 \
  --output-dir /tmp/qwen-norm-probe
cd ..
python3 run_f32_rmsnorm_probe.py --binding /tmp/qwen-norm-probe/binding.json \
  --jar ../developer-console/target/developer-console-1.0-SNAPSHOT-fat.jar
```

The epsilon above comes from `qwen35.attention.layer_norm_rms_epsilon` in the
exported GGUF metadata. These probes execute individual real tensors and
compare device output with CPU values. They do not connect layer operations,
maintain attention/DeltaNet state, tokenize input, run across four workers, or
produce a model token. Q8_0 occurs only in the auxiliary `nextn` block of this
Q4_1 package and is not covered by these matvec probes.