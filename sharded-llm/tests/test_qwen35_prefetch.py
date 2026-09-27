import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import run_gguf_shards as runner  # noqa: E402


class RecordingPrefetcher(runner.WeightPrefetcher):
    def __init__(self):
        super().__init__(1)
        self.warmed = []

    def warm(self, key, files):
        if key not in self.pending:
            self.warmed.append(key)
        super().warm(key, files)


def _runner(prefetch_steps, layers=3, head=True):
    instance = runner.GgufRunner.__new__(runner.GgufRunner)
    instance.args = SimpleNamespace(prefetch_steps=prefetch_steps)
    instance.prefetcher = RecordingPrefetcher()
    instance.cycle = [("layer", layer) for layer in range(layers)] + ([("head", None)] if head else [])
    instance.layer_inputs = [{"weights": []} for _ in range(layers)]
    instance.head_weights = []
    return instance


class WeightPrefetcherTest(unittest.TestCase):
    def test_reads_files_without_error(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "w.bin"
            path.write_bytes(b"x" * (runner.WeightPrefetcher.CHUNK + 5))
            prefetcher = runner.WeightPrefetcher(2)
            prefetcher.warm("a", [path, Path(directory) / "missing.bin"])
            prefetcher.close()
            self.assertIn("a", prefetcher.pending)

    def test_warms_current_and_next_step_once_per_cycle(self):
        instance = _runner(prefetch_steps=1)
        for step in instance.cycle + instance.cycle[:1]:
            instance._prefetch(step)
            instance._consumed(step)
        self.assertEqual(instance.prefetcher.warmed, [
            ("layer", 0), ("layer", 1), ("layer", 2), ("head", None), ("layer", 0), ("layer", 1)])
        instance.prefetcher.close()

    def test_wraps_from_head_to_first_layer(self):
        instance = _runner(prefetch_steps=2)
        instance._prefetch(("head", None))
        self.assertEqual(instance.prefetcher.warmed, [("head", None), ("layer", 0), ("layer", 1)])
        instance.prefetcher.close()


if __name__ == "__main__":
    unittest.main()
