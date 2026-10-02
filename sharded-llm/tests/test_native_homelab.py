"""HomelabStage and HomelabChain speak the /llm/step and /llm/chain protocols; checked against a fake homelab."""
import base64
import json
import os
import sys
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from native_runner import HomelabChain, HomelabStage  # noqa: E402


class _FakeHomelab(BaseHTTPRequestHandler):
    requests = []

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        type(self).requests.append((self.path, body))
        if self.path == "/llm/step" and body["pos"] > 0 and body["session"] not in _FakeHomelab.opened:
            self._reply(500, {"status": "ERROR", "error": "LLM session is not open on this worker"})
            return
        reply = {"status": "SUCCESS", "worker": "w1"}
        if self.path == "/llm/chain":
            # Embedding copies token ids, every later stage doubles; the last stage is the head.
            values = np.asarray(body["tokens"], "<f4")
            infos = []
            for stage in body["stages"]:
                if body["pos"] > 0 and stage["session"] not in _FakeHomelab.opened:
                    self._reply(500, {"status": "ERROR", "error": "LLM session is not open on this worker"})
                    return
                _FakeHomelab.opened.add(stage["session"])
                if infos:
                    values = values * 2
                infos.append({"position": body["pos"] + body["n"], "last_step_ms": 1.5})
            if body.get("output") == "argmax":
                reply["token"] = int(np.argmax(values))
            else:
                reply["output"] = base64.b64encode(values.astype("<f4").tobytes()).decode("ascii")
            reply.update(infos=infos, tasks=1, workers=["w1"])
        elif self.path == "/llm/step":
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


class _FakeHomelabTest(unittest.TestCase):
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


class HomelabStageTest(_FakeHomelabTest):
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


class HomelabChainTest(_FakeHomelabTest):
    def _stages(self):
        embed = dict(self.graph, hyper={"dim": 2}, embed={"file": "/m/shard-00/embed.bin"})
        head = dict(self.graph, head={"output": {"file": "/m/shard-00/head.bin"}})
        return [HomelabStage(self.url, "shard-00", embed, 30), HomelabStage(self.url, "shard-01", self.graph, 30),
                HomelabStage(self.url, "shard-00", head, 30)]

    def test_one_call_per_token_with_graphs_only_when_opening(self):
        stages = self._stages()
        chain = HomelabChain(stages)
        logits = chain.step([1, 5, 2], logits=True)
        np.testing.assert_array_equal(logits, [4, 20, 8])
        self.assertEqual(chain.step([3]), 0)
        (path, first), (_, second) = _FakeHomelab.requests
        self.assertEqual(path, "/llm/chain")
        self.assertEqual([s["affinity"] for s in first["stages"]], ["shard-00", "shard-01", "shard-00"])
        self.assertEqual((first["pos"], first["n"], first["tokens"]), (0, 3, [1, 5, 2]))
        self.assertNotIn("output", first)
        self.assertEqual(first["stages"][1]["files"], ["/m/shard-01/w.bin"])
        self.assertTrue(all("graph" in s for s in first["stages"]))
        self.assertEqual((second["pos"], second["n"], second["output"]), (3, 1, "argmax"))
        self.assertTrue(all(set(s) == {"affinity", "session"} for s in second["stages"]))
        self.assertEqual([stage.position for stage in stages], [4, 4, 4])
        self.assertEqual(stages[2].info()["last_step_ms"], 1.5)
        self.assertEqual(chain.last, {"tasks": 1, "workers": ["w1"]})

    def test_lost_worker_session_surfaces_as_error(self):
        chain = HomelabChain(self._stages())
        chain.step([1])
        _FakeHomelab.opened.clear()
        with self.assertRaisesRegex(RuntimeError, "not open on this worker"):
            chain.step([2])


if __name__ == "__main__":
    unittest.main()
