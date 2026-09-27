"""HomelabStage speaks the /llm/step protocol; checked against a fake homelab server."""
import base64
import json
import os
import sys
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from native_runner import HomelabStage  # noqa: E402


class _FakeHomelab(BaseHTTPRequestHandler):
    requests = []

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        type(self).requests.append((self.path, body))
        if self.path == "/llm/step" and body["pos"] > 0 and body["session"] not in _FakeHomelab.opened:
            self._reply(500, {"status": "ERROR", "error": "LLM session is not open on this worker"})
            return
        reply = {"status": "SUCCESS", "worker": "w1"}
        if self.path == "/llm/step":
            _FakeHomelab.opened.add(body["session"])
            if "hidden" in body:
                values = np.frombuffer(base64.b64decode(body["hidden"]), "<f4") * 2
            else:
                values = np.asarray(body["tokens"], "<f4")
            reply["output"] = base64.b64encode(values.astype("<f4").tobytes()).decode("ascii")
            reply["info"] = {"position": body["pos"] + body["n"]}
        else:
            _FakeHomelab.opened.discard(body["session"])
        self._reply(200, reply)

    def _reply(self, code, payload):
        data = json.dumps(payload).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *args):
        pass


class HomelabStageTest(unittest.TestCase):
    def setUp(self):
        _FakeHomelab.requests = []
        _FakeHomelab.opened = set()
        self.server = HTTPServer(("127.0.0.1", 0), _FakeHomelab)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.url = "http://127.0.0.1:{0}/".format(self.server.server_port)
        self.graph = {"hyper": {"dim": 2}, "layers": [{"tensors": {"w": {"file": "/m/shard-01/w.bin"}}}]}

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()

    def test_graph_and_files_are_sent_only_when_opening(self):
        stage = HomelabStage(self.url, "shard-01", self.graph, 30)
        out = stage.step(hidden=np.array([[1, 2], [3, 4]], np.float32))
        np.testing.assert_array_equal(out, [[2, 4], [6, 8]])
        stage.step(hidden=np.array([[5, 6]], np.float32))
        (_, first), (_, second) = _FakeHomelab.requests
        self.assertEqual((first["pos"], first["n"], first["affinity"]), (0, 2, "shard-01"))
        self.assertEqual(first["files"], ["/m/shard-01/w.bin"])
        self.assertIn("graph", first)
        self.assertEqual((second["pos"], second["n"]), (2, 1))
        self.assertNotIn("graph", second)
        self.assertNotIn("files", second)
        self.assertEqual(stage.info(), {"position": 3})

    def test_reset_closes_and_reopens_from_position_zero(self):
        graph = dict(self.graph, hyper={"dim": 1}, embed={"file": "/m/shard-00/embed.bin"})
        stage = HomelabStage(self.url, "shard-00", graph, 30)
        np.testing.assert_array_equal(stage.step(tokens=[7, 8]), [[7], [8]])
        stage.reset()
        stage.step(tokens=[9])
        paths = [path for path, _ in _FakeHomelab.requests]
        self.assertEqual(paths, ["/llm/step", "/llm/close", "/llm/step"])
        self.assertIn("graph", _FakeHomelab.requests[2][1])

    def test_lost_worker_session_surfaces_as_error(self):
        stage = HomelabStage(self.url, "shard-02", self.graph, 30)
        stage.step(hidden=np.zeros((1, 2), np.float32))
        _FakeHomelab.opened.clear()
        with self.assertRaisesRegex(RuntimeError, "not open on this worker"):
            stage.step(hidden=np.zeros((1, 2), np.float32))


if __name__ == "__main__":
    unittest.main()
