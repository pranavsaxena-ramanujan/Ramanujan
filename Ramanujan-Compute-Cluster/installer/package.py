#!/usr/bin/env python3
"""Build a local-platform UI installer from real, already-built artifacts."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import plistlib
import re
import shutil
import subprocess
import struct
import sys
import zipfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RELEASES = HERE / "releases"


def require_file(path):
    path = Path(path).resolve()
    if not path.is_file():
        raise ValueError("Required built artifact missing: " + str(path))
    return path


def publish(path, os_name):
    RELEASES.mkdir(parents=True, exist_ok=True)
    output = RELEASES / path.name
    if path.resolve() != output.resolve():
        shutil.copy2(path, output)
    digest = hashlib.sha256()
    with output.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    manifest_file = RELEASES / "manifest.json"
    manifest = json.loads(manifest_file.read_text()) if manifest_file.exists() else {}
    manifest[os_name] = {"file": output.name, "sha256": digest.hexdigest()}
    partial = RELEASES / "manifest.json.partial"
    partial.write_text(json.dumps(manifest, indent=2) + "\n")
    partial.replace(manifest_file)
    output.with_name(output.name + ".sha256").write_text(digest.hexdigest() + "  " + output.name + "\n")
    print("Published " + output.name)


def tool(name):
    executable = shutil.which(name)
    if not executable:
        raise ValueError("Required packaging tool not on PATH: " + name)
    return executable


def mac_version(value):
    if not re.fullmatch(r"\d+\.\d+(?:\.\d+)?", value):
        raise ValueError("macOS minimum must be major.minor or major.minor.patch")
    parts = tuple(int(part) for part in value.split("."))
    return parts + (0,) * (3 - len(parts))


def macho_targets(path):
    """Return architecture/minimum macOS requirements from thin or universal Mach-O."""
    if path.suffix == ".class":
        return []
    with path.open("rb") as source:
        prefix = source.read(4)
    if prefix not in (b"\xcf\xfa\xed\xfe", b"\xce\xfa\xed\xfe", b"\xfe\xed\xfa\xcf", b"\xfe\xed\xfa\xce",
                      b"\xca\xfe\xba\xbe", b"\xbe\xba\xfe\xca", b"\xca\xfe\xba\xbf", b"\xbf\xba\xfe\xca"):
        return []
    data = path.read_bytes()

    def thin(offset):
        magic = data[offset:offset + 4]
        endian = "<" if magic in (b"\xcf\xfa\xed\xfe", b"\xce\xfa\xed\xfe") else ">"
        if magic not in (b"\xcf\xfa\xed\xfe", b"\xce\xfa\xed\xfe", b"\xfe\xed\xfa\xcf", b"\xfe\xed\xfa\xce"):
            return []
        is_64 = magic in (b"\xcf\xfa\xed\xfe", b"\xfe\xed\xfa\xcf")
        cpu = struct.unpack_from(endian + "I", data, offset + 4)[0]
        arch = {0x0100000C: "arm64", 0x01000007: "x86_64"}.get(cpu, "unsupported-" + hex(cpu))
        count = struct.unpack_from(endian + "I", data, offset + 16)[0]
        at = offset + (32 if is_64 else 28)
        minimum = None
        for _ in range(count):
            command, length = struct.unpack_from(endian + "II", data, at)
            if length < 8 or at + length > len(data):
                raise ValueError("Invalid Mach-O load command: " + str(path))
            if command == 0x32:
                target_platform, version = struct.unpack_from(endian + "II", data, at + 8)
                if target_platform != 1:
                    raise ValueError("Native file targets a non-macOS platform: " + str(path))
                minimum = version
            elif command == 0x24:
                minimum = struct.unpack_from(endian + "I", data, at + 8)[0]
            at += length
        if minimum is None:
            raise ValueError("Mach-O has no minimum macOS metadata: " + str(path))
        return [(arch, (minimum >> 16, (minimum >> 8) & 255, minimum & 255))]

    magic = data[:4]
    if magic in (b"\xca\xfe\xba\xbe", b"\xbe\xba\xfe\xca", b"\xca\xfe\xba\xbf", b"\xbf\xba\xfe\xca"):
        endian = ">" if magic in (b"\xca\xfe\xba\xbe", b"\xca\xfe\xba\xbf") else "<"
        fat64 = magic in (b"\xca\xfe\xba\xbf", b"\xbf\xba\xfe\xca")
        count = struct.unpack_from(endian + "I", data, 4)[0]
        targets = []
        for i in range(count):
            at = 8 + i * (32 if fat64 else 20)
            offset = struct.unpack_from(endian + ("Q" if fat64 else "I"), data, at + 8)[0]
            targets.extend(thin(offset))
        return targets
    return thin(0)


def validate_macos_file(path, architecture, minimum, required=False):
    try:
        targets = macho_targets(path)
    except struct.error as error:
        raise ValueError("Invalid Mach-O file: " + str(path)) from error
    if not targets:
        if required:
            raise ValueError("Required native file is not Mach-O: " + str(path))
        return False
    compatible = [version for arch, version in targets if arch == architecture]
    if not compatible:
        raise ValueError("Native architecture mismatch for " + path.name + ": expected " + architecture)
    if max(compatible) > minimum:
        required_version = ".".join(str(part) for part in max(compatible))
        raise ValueError(path.name + " requires macOS " + required_version + "; requested minimum is lower")
    return True


def set_macos_metadata(image, architecture, minimum_text):
    minimum = mac_version(minimum_text)
    count = 0
    for path in image.rglob("*"):
        if path.is_file():
            count += validate_macos_file(path, architecture, minimum)
    if not count:
        raise ValueError("App-image has no verifiable Mach-O binaries")
    info = image / "Contents/Info.plist"
    values = plistlib.loads(info.read_bytes())
    values["LSMinimumSystemVersion"] = minimum_text
    info.write_bytes(plistlib.dumps(values, sort_keys=False))
    signer = tool("codesign")
    subprocess.run([signer, "--force", "--sign", "-", "--timestamp=none",
                    "--preserve-metadata=identifier,entitlements", str(image)], check=True)
    subprocess.run([signer, "--verify", "--deep", "--strict", str(image)], check=True)
    if plistlib.loads(info.read_bytes()).get("LSMinimumSystemVersion") != minimum_text:
        raise ValueError("App-image minimum macOS metadata verification failed")
    print("Verified macOS " + minimum_text + "+ / " + architecture + " app-image (" + str(count) + " Mach-O files)")


def native_inputs(os_name, native_dir, llm_library=None):
    names = {"windows": ("native.dll", "ramanujan_llm.dll"),
             "linux": ("libnative.so", "libramanujan_llm.so"),
             "macos": ("libnative.dylib", "libramanujan_llm.dylib")}[os_name]
    inputs = {names[0]: require_file(native_dir / names[0])}
    if os_name == "macos":
        inputs["libnative_llm.dylib"] = require_file(native_dir / "libnative_llm.dylib")
    runtime = Path(llm_library).expanduser() if llm_library else native_dir / names[1]
    if not llm_library and not runtime.is_file() and os_name == "macos":
        runtime = ROOT / "ramanujan-native/native/build/libramanujan_llm.dylib"
    inputs[names[1]] = require_file(runtime)
    return inputs


def package(args):
    os_name = {"Windows": "windows", "Linux": "linux", "Darwin": "macos"}.get(platform.system())
    if not os_name:
        raise ValueError("Unsupported packaging host")
    client = require_file(args.client)
    worker = require_file(args.worker)
    with zipfile.ZipFile(worker) as jar:
        worker_class = jar.read("in/ramanujan/developer/console/operationImpl/ExecuteInlineWorker.class")
        if b"RAMANUJAN_WORKER_URL" not in worker_class:
            raise ValueError("Rebuild the worker JAR: it lacks private-room environment URL support")
    native_dir = Path(args.native_dir).expanduser().resolve()
    libraries = native_inputs(os_name, native_dir, args.llm_library)
    if os_name == "macos":
        minimum = mac_version(args.mac_min_version)
        host_arch = {"aarch64": "arm64", "AMD64": "x86_64"}.get(platform.machine(), platform.machine())
        if args.mac_arch != host_arch:
            raise ValueError("jpackage host architecture does not match --mac-arch")
        for library in libraries.values():
            validate_macos_file(library, args.mac_arch, minimum, required=True)
    jpackage = tool("jpackage")
    major = int(subprocess.check_output([jpackage, "--version"], text=True).strip().split(".")[0])
    if major < 17:
        raise ValueError("Packaging requires JDK 17+ (including jpackage and jlink)")
    package_type = args.type or {"windows": "msi", "linux": "deb", "macos": "dmg"}[os_name]
    allowed = {"windows": ("msi", "exe"), "linux": ("deb", "rpm"), "macos": ("dmg", "pkg")}
    if package_type not in allowed[os_name]:
        raise ValueError("Installer type is not supported on this host")
    build = HERE / "build"
    for input_path in [client, worker, native_dir] + list(libraries.values()) + ([Path(args.runtime).resolve()] if args.runtime else []):
        if build.resolve() == input_path or build.resolve() in input_path.parents:
            raise ValueError("Keep packaging inputs outside installer/build, which is recreated")
    stage = build / "input"
    image_dir = build / "images"
    if build.exists():
        shutil.rmtree(build)
    stage.mkdir(parents=True)
    image_dir.mkdir()
    work = build / "work"
    work.mkdir()
    environment = dict(os.environ)
    environment["JAVA_TOOL_OPTIONS"] = (environment.get("JAVA_TOOL_OPTIONS", "") +
                                         ' -Djava.io.tmpdir="' + str(work) + '"').strip()
    shutil.copy2(client, stage / "cluster-client.jar")
    shutil.copy2(worker, stage / "developer-console.jar")
    (stage / "native").mkdir()
    if os_name != "macos":
        for library in native_dir.iterdir():
            if library.is_file() and re.search(r"(\.dll|\.so(?:\.\d+)*)$", library.name, re.IGNORECASE):
                shutil.copy2(library, stage / "native" / library.name)
    for name, library in libraries.items():
        shutil.copy2(library, stage / "native" / name)
    for name in ("LICENSE", "NOTICE.md", "THIRD_PARTY_LICENSES.md"):
        if (ROOT / name).is_file():
            shutil.copy2(ROOT / name, stage / name)
    command = [jpackage, "--type", "app-image", "--name", "RamanujanCluster",
               "--input", str(stage), "--main-jar", "cluster-client.jar",
               "--main-class", "in.ramanujan.cluster.client.DesktopClient",
               "--app-version", args.version, "--vendor", "Ramanujan",
               "--dest", str(image_dir), "--temp", str(build / "image-work")]
    if args.runtime:
        runtime = Path(args.runtime).resolve()
        require_file(runtime / "bin" / ("java.exe" if os_name == "windows" else "java"))
        command += ["--runtime-image", str(runtime)]
    else:
        # jpackage's default strips bin/java, which this UI needs for its child worker.
        command += ["--add-modules", "ALL-MODULE-PATH",
                    "--jlink-options", "--strip-debug --no-man-pages --no-header-files"]
    subprocess.run(command, check=True, env=environment)
    image = image_dir / ("RamanujanCluster.app" if os_name == "macos" else "RamanujanCluster")
    if not image.is_dir():
        raise ValueError("jpackage did not produce the expected app-image")
    java_path = image / "runtime/bin/java"
    if os_name == "macos":
        java_path = image / "Contents/runtime/Contents/Home/bin/java"
    elif os_name == "windows":
        java_path = image / "runtime/bin/java.exe"
    require_file(java_path)
    if os_name == "macos":
        set_macos_metadata(image, args.mac_arch, args.mac_min_version)
    installer_dir = build / "packages"
    installer_dir.mkdir()
    command = [jpackage, "--type", package_type, "--name", "RamanujanCluster",
               "--app-image", str(image), "--app-version", args.version,
               "--vendor", "Ramanujan", "--dest", str(installer_dir),
               "--temp", str(build / "installer-work")]
    if os_name == "windows":
        command += ["--win-menu", "--win-shortcut", "--win-dir-chooser", "--win-per-user-install"]
    if os_name == "linux":
        command += ["--linux-shortcut", "--linux-menu-group", "Utility", "--linux-package-name", "ramanujan-cluster"]
    subprocess.run(command, check=True, env=environment)
    installers = list(installer_dir.glob("*." + package_type))
    if len(installers) != 1:
        raise ValueError("Expected exactly one generated installer")
    publish(installers[0], os_name)
    for directory in (stage, build / "image-work", build / "installer-work", work):
        shutil.rmtree(directory, ignore_errors=True)
    print("Bundled app-image: " + str(image))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--client", default=str(HERE.parent / "client/target/cluster-client-1.0.0.jar"))
    parser.add_argument("--worker", default=str(ROOT / "developer-console/target/developer-console-1.0-SNAPSHOT-fat.jar"))
    native_default = Path.home() / "Desktop/ws" if platform.system() == "Darwin" else None
    parser.add_argument("--native-dir", default=str(native_default) if native_default else None,
                        help="Directory of supplied interpreters (macOS defaults to ~/Desktop/ws)")
    parser.add_argument("--llm-library",
                        help="Separate dedicated LLM runtime; macOS falls back to in-repo native/build if absent in native-dir")
    parser.add_argument("--mac-min-version", default="14.4",
                        help="Advertised minimum macOS; rejected if lower than any bundled Mach-O requirement (default 14.4)")
    parser.add_argument("--mac-arch", choices=("arm64", "x86_64"), default="arm64",
                        help="Required architecture of macOS inputs and app-image (default arm64)")
    parser.add_argument("--runtime", help="Optional prebuilt same-platform Java runtime image")
    parser.add_argument("--version", default="1.0.0")
    parser.add_argument("--type", help="msi/exe, deb/rpm, or dmg/pkg, built only on the matching OS")
    parser.add_argument("--apk", help="Publish a signed Android APK instead of packaging desktop")
    parser.add_argument("--apksigner", default="apksigner")
    args = parser.parse_args()
    try:
        if args.apk:
            apk = require_file(args.apk)
            with zipfile.ZipFile(apk) as archive:
                names = archive.namelist()
                abis = {name.split("/")[1] for name in names if name.startswith("lib/") and name.endswith("/libnative.so")}
                if not abis or any("lib/" + abi + "/libramanujan_llm.so" not in names for abi in abis):
                    raise ValueError("APK must package the interpreter and LLM library for every ABI")
            subprocess.run([tool(args.apksigner), "verify", str(apk)], check=True)
            publish(apk, "android")
        else:
            if not args.native_dir:
                raise ValueError("--native-dir is required; native libraries are never downloaded by this installer")
            package(args)
    except (ValueError, OSError, subprocess.CalledProcessError, zipfile.BadZipFile, KeyError) as error:
        print("Packaging failed: " + str(error), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
