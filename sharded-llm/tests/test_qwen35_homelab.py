import json
import sys
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import run_gguf_shards as runner  # noqa: E402


class FakeHomelab(BaseHTTPRequestHandler):
    requests = []
    fail_run = False

    def log_message(self, *args):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        FakeHomelab.requests.append((self.path, body))
        if self.path == "/orchestrator/run" and FakeHomelab.fail_run:
            return self._reply(500, {"status": "ERROR", "message": "worker w1 failed"})
        if self.path == "/orchestrator/dump":
            Path(body["path"]).write_bytes(body["name"].encode("utf-8"))
        self._reply(200, {"status": "SUCCESS"})

    def _reply(self, code, payload):
        data = json.dumps(payload).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


class HomelabWorkerTest(unittest.TestCase):
    def setUp(self):
        FakeHomelab.requests = []
        FakeHomelab.fail_run = False
        self.server = HTTPServer(("127.0.0.1", 0), FakeHomelab)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.directory = Path(tempfile.mkdtemp())

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()

    def worker(self, resident=False):
        args = SimpleNamespace(homelab="http://127.0.0.1:{0}/".format(self.server.server_port),
                               timeout=10, resident_weights=resident)
        return runner.HomelabWorker({"id": "shard-02"}, args)

    def test_run_pins_shard_and_takes_raw_binaries(self):
        worker = self.worker()
        hidden = self.directory / "h_state.bin"
        worker.run(self.directory / "delta.py", [self.directory / "h.csv"], {"h_state": hidden}, evict=True)

        (route, run), (dump_route, dump) = FakeHomelab.requests
        self.assertEqual("/orchestrator/run", route)
        self.assertEqual([str(self.directory / "delta.py"), str(self.directory / "h.csv")], run["args"])
        self.assertEqual("shard-02", run["affinity"])
        self.assertTrue(run["evictWeights"])
        self.assertEqual("/orchestrator/dump", dump_route)
        self.assertEqual(run["requestId"], dump["requestId"])
        self.assertTrue(dump["raw"])
        self.assertEqual(b"h_state", hidden.read_bytes())
        self.assertFalse(hidden.with_name("h_state.bin.take").exists())

    def test_resident_weights_never_evict(self):
        self.worker(resident=True).run(self.directory / "p.py", [], {}, evict=True)
        self.assertFalse(FakeHomelab.requests[0][1]["evictWeights"])

    def test_failed_run_raises_and_skips_takes(self):
        FakeHomelab.fail_run = True
        destination = self.directory / "h_state.bin"
        with self.assertRaisesRegex(RuntimeError, "worker w1 failed"):
            self.worker().run(self.directory / "p.py", [], {"h_state": destination})
        self.assertEqual(1, len(FakeHomelab.requests))
        self.assertFalse(destination.exists())


if __name__ == "__main__":
    unittest.main()
