import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import struct
import unittest
from unittest.mock import patch
import uuid

spec = importlib.util.spec_from_file_location("package", Path(__file__).with_name("package.py"))
package = importlib.util.module_from_spec(spec)
spec.loader.exec_module(package)


class ReleaseTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(__file__).parent / "build" / ("test-" + str(uuid.uuid4()))
        self.root.mkdir(parents=True)

    def tearDown(self):
        shutil.rmtree(self.root)

    def test_publish_checksums_and_preserves_other_platform_entries(self):
        release = self.root / "releases"
        release.mkdir()
        (release / "manifest.json").write_text('{"linux":{"file":"old.deb","sha256":"old"}}')
        artifact = self.root / "test.apk"
        artifact.write_bytes(b"test fixture only")
        with patch.object(package, "RELEASES", release):
            package.publish(artifact, "android")
        values = json.loads((release / "manifest.json").read_text())
        self.assertEqual(values["android"], {"file": "test.apk", "sha256": hashlib.sha256(artifact.read_bytes()).hexdigest()})
        self.assertEqual(values["linux"]["file"], "old.deb")

    def test_missing_artifacts_fail(self):
        with self.assertRaisesRegex(ValueError, "Required built artifact missing"):
            package.require_file(self.root / "missing.jar")

    def test_macos_keeps_supplied_interpreters_and_separate_dedicated_runtime(self):
        native = self.root / "ws"
        native.mkdir()
        for name in ("libnative.dylib", "libnative_llm.dylib"):
            (native / name).write_bytes(b"interpreter fixture")
        runtime = self.root / "libramanujan_llm.dylib"
        runtime.write_bytes(b"dedicated runtime fixture")
        libraries = package.native_inputs("macos", native, str(runtime))
        self.assertEqual(set(libraries), {"libnative.dylib", "libnative_llm.dylib", "libramanujan_llm.dylib"})
        self.assertEqual(libraries["libnative.dylib"], (native / "libnative.dylib").resolve())
        self.assertEqual(libraries["libnative_llm.dylib"], (native / "libnative_llm.dylib").resolve())
        self.assertEqual(libraries["libramanujan_llm.dylib"], runtime.resolve())
        self.assertEqual((native / "libnative.dylib").read_bytes(), b"interpreter fixture")

    def test_macho_minimum_and_architecture_are_checked_before_advertising(self):
        native = self.root / "fixture.dylib"
        header = struct.pack("<IIIIIIII", 0xfeedfacf, 0x0100000C, 0, 6, 1, 24, 0, 0)
        command = struct.pack("<IIIIII", 0x32, 24, 1, (14 << 16) | (4 << 8), 0, 0)
        native.write_bytes(header + command)
        self.assertEqual(package.macho_targets(native), [("arm64", (14, 4, 0))])
        self.assertTrue(package.validate_macos_file(native, "arm64", package.mac_version("14.4"), required=True))
        with self.assertRaisesRegex(ValueError, "requested minimum is lower"):
            package.validate_macos_file(native, "arm64", package.mac_version("14.3"), required=True)
        with self.assertRaisesRegex(ValueError, "architecture mismatch"):
            package.validate_macos_file(native, "x86_64", package.mac_version("14.4"), required=True)
        with self.assertRaisesRegex(ValueError, "minimum must"):
            package.mac_version("14")


if __name__ == "__main__":
    unittest.main()
