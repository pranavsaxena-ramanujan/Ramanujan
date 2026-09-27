import io
import re
import struct
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from ramanujan_shards.gguf_emitter import emit_gguf_package
from ramanujan_shards.gguf_source import GGUFSourceReader
from ramanujan_shards.verify_gguf import verify_gguf_package


class _Response(io.BytesIO):
    def __init__(self, content, content_range):
        super().__init__(content)
        self.status = 206
        self.headers = {"Content-Range": content_range}


class GGUFHTTPTest(unittest.TestCase):
    def test_fetches_header_and_requested_tensor_by_range(self):
        names = (b"blk.0.weight", b"blk.1.weight")
        header = bytearray(b"GGUF" + struct.pack("<IQQ", 3, 2, 0))
        for index, name in enumerate(names):
            header += struct.pack("<Q", len(name)) + name + struct.pack("<IQIQ", 1, 32, 2, index * 32)
        header += bytes(-len(header) % 32)
        contents = bytes(header) + b"x" * 18 + b"\0" * 14 + b"y" * 18 + b"z" * (2 * 1024 * 1024)
        requests = []

        def fake_urlopen(request, timeout):
            match = re.fullmatch(r"bytes=([0-9]+)-([0-9]+)", request.headers["Range"])
            start, end = int(match.group(1)), int(match.group(2))
            requests.append((start, end))
            return _Response(contents[start:end + 1], "bytes {0}-{1}/{2}".format(start, end, len(contents)))

        with patch("ramanujan_shards.gguf_source.urlopen", side_effect=fake_urlopen):
            reader = GGUFSourceReader("https://example.test/model.gguf")
            with reader.open_tensor("blk.0.weight") as stream:
                self.assertEqual(b"x" * 18, stream.read())
                stream.seek(17)
                self.assertEqual(b"x", stream.read(1))
                self.assertEqual(b"", stream.read())
        self.assertEqual((0, 0), requests[0])
        self.assertLess(sum(end - start + 1 for start, end in requests), len(contents))
        self.assertIn((len(header), len(header) + 17), requests)

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "shards"
            with patch("ramanujan_shards.gguf_source.urlopen", side_effect=fake_urlopen):
                emit_gguf_package("https://example.test/model.gguf", output, shards=2,
                                  shard_index=1)
            self.assertEqual(1, verify_gguf_package(output))
            self.assertEqual(b"y" * 18, (output / "shard-01" / "weights" / "blk.1.weight.bin").read_bytes())
            self.assertEqual(1, requests.count((len(header), len(header) + 17)))
            self.assertIn((len(header) + 32, len(header) + 49), requests)


if __name__ == "__main__":
    unittest.main()