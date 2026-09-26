import importlib.util
from pathlib import Path
import tempfile
import unittest


RUNNER = Path(__file__).resolve().parents[1] / "run_phi3_shards.py"
SPEC = importlib.util.spec_from_file_location("run_phi3_shards", RUNNER)
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


class FakeServer:
    def __init__(self, failures):
        self.failures = failures
        self.starts = 0
        self.closes = 0
        self.changes = []
        self.process = None
        self.dumps = []
        self.monitor = self

    def start(self):
        self.starts += 1
        self.process = object()

    def change_shard(self, kernel, timeout):
        self.changes.append(kernel)

    def run(self, kernel, csv_paths, timeout):
        if self.starts <= self.failures:
            raise runner.MemoryLimitExceeded("ramanujan-jvm", 8, 7)

    def dump(self, name, path):
        self.dumps.append(name)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.with_suffix(".bin").write_bytes(b"\0" * 4)

    def check(self):
        pass

    def close(self):
        self.closes += 1
        self.process = None


class WorkerRestartTests(unittest.TestCase):
    def test_persistent_worker_changes_shard_after_dump(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "manifest.json").write_text(
                '{"adapterMetadata": {"layer_start": 0, "layer_end": 1}}'
            )
            server = FakeServer(failures=0)
            for name in ("first", "second"):
                runner._execute_shard(
                    server, root / (name + ".py"), [], root, root / name,
                    root / (name + "-state"), 10, reuse_worker=True,
                )
            self.assertEqual((server.starts, server.closes), (1, 0))
            self.assertEqual(server.changes, [root / "second.py"])
            server.close()

    def test_persistent_worker_restarts_and_retries_after_limit(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "manifest.json").write_text(
                '{"adapterMetadata": {"layer_start": 0, "layer_end": 1}}'
            )
            server = FakeServer(failures=0)
            first = root / "first.py"
            second = root / "second.py"
            runner._execute_shard(server, first, [], root, root / "first",
                                  root / "first-state", 10, reuse_worker=True)
            original_run = server.run
            attempts = []

            def fail_once(kernel, csv_paths, timeout):
                attempts.append(kernel)
                if len(attempts) == 1:
                    raise runner.MemoryLimitExceeded("ramanujan-jvm", 8, 7)
                original_run(kernel, csv_paths, timeout)

            server.run = fail_once
            runner._execute_shard(server, second, [], root, root / "second",
                                  root / "second-state", 10, reuse_worker=True)
            self.assertEqual(attempts, [second, second])
            self.assertEqual((server.starts, server.closes), (2, 1))
            self.assertEqual(server.changes, [second])
            server.close()

    def test_parent_limit_does_not_retry_worker(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "manifest.json").write_text(
                '{"adapterMetadata": {"layer_start": 0, "layer_end": 1}}'
            )
            server = FakeServer(failures=0)
            server.check = lambda: (_ for _ in ()).throw(
                runner.MemoryLimitExceeded("shard-runner", 8, 7)
            )
            with self.assertRaises(runner.MemoryLimitExceeded):
                runner._execute_shard(server, root / "prefill.py", [], root,
                                      root / "run", root / "state", 10)
            self.assertEqual((server.starts, server.closes), (1, 1))

    def test_worker_retries_same_shard_and_dumps_token(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "manifest.json").write_text(
                '{"adapterMetadata": {"layer_start": 0, "layer_end": 1}}'
            )
            server = FakeServer(failures=1)
            hidden, caches = runner._execute_shard(
                server, root / "prefill.py", [], root, root / "run",
                root / "state", 10, token_csv=root / "token.csv",
            )
            self.assertEqual((server.starts, server.closes), (2, 2))
            self.assertTrue(hidden.is_file())
            self.assertEqual(set(caches), {"l0_k_cache", "l0_v_cache"})
            self.assertEqual(server.dumps[-1], "argmax_arr")

    def test_second_memory_failure_is_reported(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            server = FakeServer(failures=2)
            with self.assertRaises(runner.MemoryLimitExceeded):
                runner._execute_shard(server, root / "decode.py", [], root,
                                      root / "run", root / "state", 10)
            self.assertEqual((server.starts, server.closes), (2, 2))
            self.assertFalse((root / "run" / "h_state-output.bin").exists())


if __name__ == "__main__":
    unittest.main()