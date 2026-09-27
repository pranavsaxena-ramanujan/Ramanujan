import tempfile
import unittest
from pathlib import Path

import numpy as np

from ramanujan_shards.dsl_simulator import run_program
from ramanujan_shards.llm_programs import (
    generate_embed_program, generate_head_program, generate_layer_program, layer_inputs, layer_roles,
    layer_states)
from ramanujan_shards.llm_reference import LlmReference, row_bytes
from ramanujan_shards.llm_spec import QUANT, build_spec, tensor_columns

# (offset, value) of the fp16 scale/min fields per block, chosen to keep weights small.
_SCALES = {
    "gguf-q4_0": [(0, 0.02)], "gguf-q4_1": [(0, 0.01), (2, -0.075)], "gguf-q5_0": [(0, 0.01)],
    "gguf-q5_1": [(0, 0.005), (2, -0.08)], "gguf-q8_0": [(0, 0.002)],
    "gguf-q4_k": [(0, 0.0004), (2, 0.004)], "gguf-q5_k": [(0, 0.0002), (2, 0.004)],
    "gguf-q6_k": [(208, 0.002)],
}


def _weights(rng, encoding, shape, role):
    if len(shape) == 1:
        if role.endswith("norm"):
            return (1.0 + 0.1 * rng.standard_normal(shape)).astype("<f4").tobytes()
        if role == "ssm_a":
            return (-np.exp(0.3 * rng.standard_normal(shape))).astype("<f4").tobytes()
        if role == "rope_freqs":
            return rng.uniform(1.0, 8.0, shape).astype("<f4").tobytes()
        return (0.1 * rng.standard_normal(shape)).astype("<f4").tobytes()
    count = shape[0] * shape[1]
    if encoding == "gguf-f32":
        return (0.3 / np.sqrt(shape[1]) * rng.standard_normal(count) * 3).astype("<f4").tobytes()
    if encoding == "gguf-f16":
        return (0.1 * rng.standard_normal(count)).astype("<f2").tobytes()
    kind = QUANT[encoding]
    blocks = rng.integers(0, 256, size=(count // kind.block, kind.block_bytes), dtype=np.uint8)
    if encoding == "gguf-q6_k":
        blocks[:, 192:208] = rng.integers(-6, 7, size=(len(blocks), 16)).astype(np.int8).view(np.uint8)
    for offset, value in _SCALES[encoding]:
        blocks[:, offset:offset + 2] = np.frombuffer(np.float16(value).tobytes(), np.uint8)
    return blocks.tobytes()


class _Model:
    def __init__(self, directory, arch, metadata, layout, encoding, seed=0):
        rng = np.random.default_rng(seed)
        self.metadata = {"general.architecture": arch}
        self.metadata.update({"{0}.{1}".format(arch, key): value for key, value in metadata.items()})
        self.tensors = {}
        for name, (shape, tensor_encoding) in layout.items():
            tensor_encoding = tensor_encoding or (encoding if len(shape) == 2 else "gguf-f32")
            path = Path(directory) / (name + ".bin")
            role = name.split(".")[2] if name.startswith("blk.") else name.split(".")[0]
            path.write_bytes(_weights(rng, tensor_encoding, shape, role))
            self.tensors[name] = {"shape": shape, "encoding": tensor_encoding, "file": path}


def _attention_layout(layer, dim, heads, kv, hd, ffn, qkv="separate", q_gate=False, qk_norm=False,
                      bias=False, ffn_layout="gated", norms=("ffn_norm",)):
    p = "blk.{0}.".format(layer)
    layout = {p + "attn_norm.weight": ([dim], None), p + "attn_output.weight": ([dim, heads * hd], None),
              p + "ffn_down.weight": ([dim, ffn], None)}
    if qkv == "fused":
        layout[p + "attn_qkv.weight"] = ([(heads + 2 * kv) * hd, dim], None)
    else:
        layout[p + "attn_q.weight"] = ([heads * hd * (2 if q_gate else 1), dim], None)
        layout[p + "attn_k.weight"] = ([kv * hd, dim], None)
        layout[p + "attn_v.weight"] = ([kv * hd, dim], None)
        if bias:
            for name, rows in (("attn_q", heads * hd), ("attn_k", kv * hd), ("attn_v", kv * hd)):
                layout[p + name + ".bias"] = ([rows], None)
    if qk_norm:
        layout[p + "attn_q_norm.weight"] = ([hd], None)
        layout[p + "attn_k_norm.weight"] = ([hd], None)
    for norm in norms:
        layout[p + norm + ".weight"] = ([dim], None)
    if ffn_layout == "gated":
        layout[p + "ffn_gate.weight"] = ([ffn, dim], None)
        layout[p + "ffn_up.weight"] = ([ffn, dim], None)
    elif ffn_layout == "fused_gate_up":
        layout[p + "ffn_up.weight"] = ([2 * ffn, dim], None)
    else:
        layout[p + "ffn_up.weight"] = ([ffn, dim], None)
    return layout


def _globals(dim, vocab, tied=False, embed_encoding=None):
    layout = {"token_embd.weight": ([vocab, dim], embed_encoding), "output_norm.weight": ([dim], None)}
    if not tied:
        layout["output.weight"] = ([vocab, dim], None)
    return layout


def _hyper(dim, heads, kv, layers, **extra):
    values = {"embedding_length": dim, "attention.head_count": heads, "attention.head_count_kv": kv,
              "block_count": layers, "attention.layer_norm_rms_epsilon": 1e-5, "rope.freq_base": 10000.0}
    values.update(extra)
    return values


def _simulate(test, model, tokens=(3, 7), context=8, overrides=None):
    spec = build_spec(model.metadata, model.tensors, overrides)
    reference = LlmReference(spec, model.tensors)
    floats = lambda name: np.fromfile(model.tensors[name]["file"], "<f4")
    programs = {kind_id: generate_layer_program(spec, kind, context) for kind_id, kind in spec.kinds.items()}
    embed_source = generate_embed_program(spec, model.tensors[spec.embed_tensor]["encoding"])
    head_source = generate_head_program(spec, model.tensors[spec.head_roles["output"]]["encoding"])
    states = [{name: np.zeros(count, np.float32) for name, count, _ in layer_states(spec, spec.kind(layer), context)}
              for layer in range(len(spec.layers))]
    reference_states = [reference.new_state(layer) for layer in range(len(spec.layers))]

    def close(actual, expected, what):
        scale = max(1e-6, float(np.max(np.abs(expected))))
        error = float(np.max(np.abs(actual - expected))) / scale
        test.assertLess(error, 2e-4, "{0}: relative error {1}".format(what, error))

    for pos, token in enumerate(tokens):
        embed = model.tensors[spec.embed_tensor]
        size = row_bytes(embed)
        row = np.frombuffer(Path(embed["file"]).read_bytes()[token * size:(token + 1) * size], "<f4")
        hidden = run_program(embed_source, {"emb_row": row})["h_state"]
        expected = reference.embed(token)
        close(hidden, expected, "embed")
        for layer, layer_spec in enumerate(spec.layers):
            kind = spec.kind(layer)
            arrays = {"h_state": hidden, "pos_arr": [pos]}
            arrays.update(states[layer])
            arrays.update({role: floats(layer_spec.roles[role]) for role in layer_roles(kind)})
            arrays.update({name: floats(tensor) for name, tensor in layer_inputs(spec, kind).items()})
            if kind.mixer != "attention":
                del arrays["pos_arr"]
            result = run_program(programs[layer_spec.kind], arrays)
            hidden = result.pop("h_state")
            states[layer].update(result)
            expected = reference.layer(layer, expected, reference_states[layer], pos)
            close(hidden, expected, "token {0} layer {1}".format(pos, layer))
        head = run_program(head_source, {"h_state": hidden, "output_norm": floats("output_norm.weight"),
                                         "output": floats(spec.head_roles["output"])})
        logits = reference.logits(expected)
        close(head["logits"], logits, "logits")
        test.assertEqual(int(head["argmax_arr"][0]), int(np.argmax(logits)))
    return spec


class GenericProgramSimulationTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.path = self.directory.name

    def tearDown(self):
        self.directory.cleanup()

    def check(self, model, **options):
        """Runs `model` and compares it with the reference; subclasses swap the executor."""
        return _simulate(self, model, **options)

    def test_llama_norm_rope_with_frequency_factors(self):
        dim, heads, kv, hd, ffn = 64, 4, 2, 16, 128
        layout = _globals(dim, 40)
        layout["rope_freqs.weight"] = ([hd // 2], None)
        for layer in range(2):
            layout.update(_attention_layout(layer, dim, heads, kv, hd, ffn))
        model = _Model(self.path, "llama", _hyper(dim, heads, kv, 2, **{"rope.dimension_count": hd}),
                       layout, "gguf-q4_0")
        spec = self.check(model)
        self.assertEqual("norm", spec.semantics.rope_style)
        self.assertEqual("rope_freqs.weight", spec.rope_freqs)

    def test_qwen2_biases_and_tied_embeddings(self):
        dim, heads, kv, hd, ffn = 64, 4, 1, 16, 64
        layout = _globals(dim, 40, tied=True, embed_encoding="gguf-q8_0")
        for layer in range(2):
            layout.update(_attention_layout(layer, dim, heads, kv, hd, ffn, bias=True))
        model = _Model(self.path, "qwen2", _hyper(dim, heads, kv, 2, **{"rope.freq_base": 1e6}),
                       layout, "gguf-q5_0")
        spec = self.check(model)
        self.assertEqual("token_embd.weight", spec.head_roles["output"])

    def test_qwen3_qk_norm_f16(self):
        dim, heads, kv, hd, ffn = 64, 4, 2, 16, 64
        layout = _globals(dim, 40)
        for layer in range(2):
            layout.update(_attention_layout(layer, dim, heads, kv, hd, ffn, qk_norm=True))
        model = _Model(self.path, "qwen3", _hyper(dim, heads, kv, 2, **{"attention.key_length": hd}),
                       layout, "gguf-f16")
        self.check(model)

    def test_phi3_fused_qkv_and_gate_up(self):
        dim, heads, kv, hd, ffn = 64, 4, 4, 16, 64
        layout = _globals(dim, 40)
        for layer in range(2):
            layout.update(_attention_layout(layer, dim, heads, kv, hd, ffn, qkv="fused",
                                            ffn_layout="fused_gate_up"))
        model = _Model(self.path, "phi3", _hyper(dim, heads, kv, 2), layout, "gguf-q5_1")
        spec = self.check(model)
        self.assertEqual("fused", spec.kind(0).flag("qkv"))

    def test_gemma_gelu_embedding_scale_and_wide_heads(self):
        dim, heads, kv, hd, ffn = 64, 4, 1, 32, 64
        layout = _globals(dim, 40, tied=True)
        for layer in range(2):
            layout.update(_attention_layout(layer, dim, heads, kv, hd, ffn))
        model = _Model(self.path, "gemma", _hyper(dim, heads, kv, 2, **{"attention.key_length": hd,
                                                                         "attention.value_length": hd}),
                       layout, "gguf-q4_1")
        spec = self.check(model)
        self.assertAlmostEqual(8.0, spec.embed_multiplier)

    def test_unknown_architecture_with_post_norms_plain_ffn_and_overrides(self):
        dim, heads, kv, hd, ffn = 64, 4, 2, 16, 64
        layout = _globals(dim, 40)
        for layer in range(2):
            layout.update(_attention_layout(layer, dim, heads, kv, hd, ffn, ffn_layout="plain",
                                            norms=("ffn_norm", "post_attention_norm", "post_ffw_norm")))
        model = _Model(self.path, "mystery", _hyper(dim, heads, kv, 2, **{"rope.dimension_count": 8}),
                       layout, "gguf-q8_0")
        spec = self.check(model, overrides={"rope_style": "neox", "activation": "gelu"})
        self.assertTrue(spec.assumptions)
        self.assertTrue(spec.kind(0).flag("post_attn_norm"))

    def test_qwen35_gated_deltanet_and_gated_attention(self):
        dim, heads, kv, hd, ffn = 64, 2, 1, 16, 64
        ssm = {"conv_kernel": 4, "state_size": 8, "group_count": 2, "time_step_rank": 4, "inner_size": 32}
        conv_dim = 2 * 2 * 8 + 32
        layout = _globals(dim, 40)
        p = "blk.0."
        layout.update({p + "attn_norm.weight": ([dim], None), p + "attn_qkv.weight": ([conv_dim, dim], None),
                       p + "attn_gate.weight": ([32, dim], None), p + "ssm_beta.weight": ([4, dim], None),
                       p + "ssm_alpha.weight": ([4, dim], None), p + "ssm_dt.bias": ([4], None),
                       p + "ssm_a": ([4], None), p + "ssm_conv1d.weight": ([conv_dim, 4], "gguf-f32"),
                       p + "ssm_norm.weight": ([8], None), p + "ssm_out.weight": ([dim, 32], None),
                       p + "post_attention_norm.weight": ([dim], None), p + "ffn_gate.weight": ([ffn, dim], None),
                       p + "ffn_up.weight": ([ffn, dim], None), p + "ffn_down.weight": ([dim, ffn], None)})
        layout.update(_attention_layout(1, dim, heads, kv, hd, ffn, q_gate=True, qk_norm=True,
                                        norms=("post_attention_norm",)))
        layout["blk.2.nextn.eh_proj.weight"] = ([dim, 2 * dim], None)
        metadata = _hyper(dim, heads, kv, 3, **{"attention.key_length": hd, "rope.dimension_count": 8,
                                                "nextn_predict_layers": 1, "full_attention_interval": 2})
        metadata.update({"ssm." + key: value for key, value in ssm.items()})
        model = _Model(self.path, "qwen35", metadata, layout, "gguf-q4_1")
        spec = self.check(model, tokens=(3, 7, 11))
        self.assertEqual(["gated_deltanet", "attention"], [spec.kind(layer).mixer for layer in range(2)])
        self.assertTrue(spec.kind(1).flag("q_gate"))

    def test_k_quants(self):
        dim, heads, kv, hd, ffn = 256, 4, 2, 64, 512
        layout = _globals(dim, 16)
        layout.update(_attention_layout(0, dim, heads, kv, hd, ffn))
        layout["blk.0.ffn_down.weight"] = ([dim, ffn], "gguf-q6_k")
        layout["blk.0.attn_v.weight"] = ([kv * hd, dim], "gguf-q5_k")
        model = _Model(self.path, "llama", _hyper(dim, heads, kv, 1), layout, "gguf-q4_k")
        self.check(model, tokens=(3, 5))


class SpecValidationTest(unittest.TestCase):
    def _tensors(self, extra=None):
        dim, heads, kv, hd, ffn = 64, 4, 2, 16, 64
        layout = _globals(dim, 40)
        layout.update(_attention_layout(0, dim, heads, kv, hd, ffn))
        layout.update(extra or {})
        tensors = {name: {"shape": shape, "encoding": encoding or ("gguf-q8_0" if len(shape) == 2 else "gguf-f32")}
                   for name, (shape, encoding) in layout.items()}
        metadata = {"general.architecture": "llama"}
        metadata.update({"llama." + key: value for key, value in _hyper(dim, heads, kv, 1).items()})
        return metadata, tensors

    def test_rejects_mixture_of_experts(self):
        metadata, tensors = self._tensors({"blk.0.ffn_gate_inp.weight": ([8, 64], None)})
        with self.assertRaisesRegex(ValueError, "mixture-of-experts"):
            build_spec(metadata, tensors)

    def test_rejects_layernorm_models(self):
        metadata, tensors = self._tensors()
        del metadata["llama.attention.layer_norm_rms_epsilon"]
        metadata["llama.attention.layer_norm_epsilon"] = 1e-5
        with self.assertRaisesRegex(ValueError, "LayerNorm"):
            build_spec(metadata, tensors)

    def test_rejects_misshaped_and_unknown_tensors(self):
        metadata, tensors = self._tensors()
        tensors["blk.0.attn_k.weight"]["shape"] = [16, 64]
        with self.assertRaisesRegex(ValueError, "attn_k"):
            build_spec(metadata, tensors)
        metadata, tensors = self._tensors({"blk.0.attn_sinks.weight": ([4], None)})
        with self.assertRaisesRegex(ValueError, "unsupported tensors attn_sinks"):
            build_spec(metadata, tensors)

    def test_rejects_unsupported_quant_and_rope_scaling(self):
        metadata, tensors = self._tensors()
        tensors["blk.0.ffn_up.weight"]["encoding"] = "gguf-q2_k"
        with self.assertRaisesRegex(ValueError, "unsupported tensor encoding"):
            build_spec(metadata, tensors)
        metadata, tensors = self._tensors()
        metadata["llama.rope.scaling.type"] = "yarn"
        with self.assertRaisesRegex(ValueError, "yarn"):
            build_spec(metadata, tensors)
        metadata, tensors = self._tensors()
        metadata["llama.residual_scale"] = 0.22
        with self.assertRaisesRegex(ValueError, "residual_scale"):
            build_spec(metadata, tensors)

    def test_groups_rows_that_are_not_float_aligned(self):
        with self.assertRaisesRegex(ValueError, "float-word aligned"):
            tensor_columns({"shape": [1, 256], "encoding": "gguf-q6_k"})
        self.assertEqual(105, tensor_columns({"shape": [4, 256], "encoding": "gguf-q6_k"}))
        self.assertEqual(105, tensor_columns({"shape": [4, 512], "encoding": "gguf-q6_k"}))
        metadata, tensors = self._tensors()
        tensors["output.weight"]["encoding"] = "gguf-q4_k"
        with self.assertRaisesRegex(ValueError, "output.weight: tensor row length"):
            build_spec(metadata, tensors)

    def test_layers_with_different_quants_get_separate_kinds(self):
        metadata, tensors = self._tensors()
        extra = _attention_layout(1, 64, 4, 2, 16, 64)
        tensors.update({name: {"shape": shape, "encoding": "gguf-q8_0" if len(shape) == 2 else "gguf-f32"}
                        for name, (shape, _) in extra.items()})
        tensors["blk.1.ffn_down.weight"]["encoding"] = "gguf-q4_0"
        metadata["llama.block_count"] = 2
        spec = build_spec(metadata, tensors)
        self.assertEqual(["attn0", "attn1"], [layer.kind for layer in spec.layers])


if __name__ == "__main__":
    unittest.main()
