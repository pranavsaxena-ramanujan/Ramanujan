# UI installers and download manifest

The scripts consume built artifacts. They do not install dependencies or
download prebuilt workers/native libraries. Missing inputs fail before an
installer is published. No binary artifacts belong in Git.

## Requirements

* Python 3.8+ and **JDK 17+** with `jpackage` and `jlink` on `PATH`.
  (`jpackage` became non-incubating in JDK 16; these scripts explicitly require
  17+. Application source remains Java 8-compatible.)
* Build on the target OS and architecture; `jpackage` does not cross-package.
* Windows: WiX tooling compatible with the chosen JDK (`jpackage` in JDK 17
  requires WiX 3.x) for MSI/EXE.
* Linux: `dpkg-deb` and `fakeroot` for DEB, or RPM build tools for RPM.
* macOS: Xcode command-line/system packaging tools for DMG/PKG. Production
  distribution additionally requires your own code signing and notarization.
* Maven (the build scripts compile the client JAR first), a rebuilt fat
  `developer-console` JAR with
  `RAMANUJAN_WORKER_URL` support, and matching native libraries.

## Package

After building the worker (`build.sh`/`build.ps1` rebuild the client JAR):

```sh
bash Ramanujan-Compute-Cluster/installer/build.sh
```

On macOS the interpreter directory defaults to the user-supplied `~/Desktop/ws`.
It copies **both** `libnative.dylib` and `libnative_llm.dylib` from there without
modifying or replacing either source file. The latter is an optimized
**interpreter**, not the dedicated LLM session runtime.
`jpackage` may ad-hoc-sign the copies inside the app bundle, changing their
signature metadata/checksums; the supplied compiled interpreter code and the
original files in `ws` remain unchanged.

The separate dedicated `libramanujan_llm.dylib` is required for `LlmSession` JNI.
If absent in the interpreter directory, it defaults to the existing
`ramanujan-native/native/build/libramanujan_llm.dylib`. Explicitly select inputs:

```sh
bash Ramanujan-Compute-Cluster/installer/build.sh \
  --native-dir "$HOME/Desktop/ws" \
  --llm-library ramanujan-native/native/build/libramanujan_llm.dylib
```

These three dylibs are bundled for the current macOS release. No Windows/Linux native
binaries or Android APK are provided yet; their manifest entries remain
unavailable until real artifacts are supplied and successfully packaged.

Source-only Windows packaging is prepared for when native DLLs are supplied:

```powershell
.\Ramanujan-Compute-Cluster\installer\build.ps1 --native-dir C:\build\native
```

Optional inputs:

```text
--client <built-client.jar>
--worker <built-fat-worker.jar>
--native-dir <flat-directory-of-native-libraries>
--llm-library <separate-dedicated-LLM-runtime-library>
--mac-min-version 14.4
--mac-arch arm64
--runtime <matching-Java-runtime-image-with-bin/java>
--version 1.0.0
--type msi|exe|deb|rpm|dmg|pkg
```

Default installer types: Windows MSI, Linux DEB, macOS DMG. Both the app-image
and actual native installer are created; this is not a ZIP-only distribution.
`build/images/` contains the bundled-runtime app-image with an entry launcher.
`build/packages/` contains the installer; successful packaging copies it into
`releases/` and records its checksum.

The native directory must contain:

| OS | Interpreter | LLM |
|---|---|---|
| Windows | `native.dll` | `ramanujan_llm.dll` |
| Linux | `libnative.so` | `libramanujan_llm.so` |
| macOS | `libnative.dylib` and `libnative_llm.dylib` | `libramanujan_llm.dylib` |

For future Windows/Linux packages, flat-directory DLL and `.so` (including
versioned) dependencies are copied as well; build trees and sources are not.
Assemble all dependencies before packaging. Current macOS dylibs depend only
on system frameworks/libraries. GPU drivers/OpenCL devices are **not** supplied by the Java
runtime. Runtime GPU compatibility must be tested separately.

The default `jlink` options retain `bin/java` because the GUI launches a separate
worker JVM. A custom `--runtime` must also retain that executable. The app-image
is checked for it before building the installer.

## Android publishing

Build and sign the new app using [`../androidapp/README.md`](../androidapp/README.md).
Then publish a real signed APK:

```sh
bash Ramanujan-Compute-Cluster/installer/build.sh \
  --apk Ramanujan-Compute-Cluster/androidapp/app/build/outputs/apk/release/app-release.apk \
  --apksigner "$ANDROID_HOME/build-tools/34.0.0/apksigner"
```

The publisher checks the APK has both native libraries for every packaged ABI
and invokes `apksigner verify`; it never creates a fake APK.

## Portal contract

`releases/manifest.json` is initially empty. Each successful publish merges one
platform entry without inventing unavailable artifacts:

```json
{"macos":{"file":"RamanujanCluster-1.0.0.dmg","sha256":"<actual SHA-256>"}}
```

The keys are `windows`, `linux`, `macos`, `android`. `file` is a basename inside
`releases/`; `sha256` is the hexadecimal digest of those exact bytes. The portal
serves `/downloads/<os>` from this manifest; absent entries are unavailable.
Public `/api/downloads` exposes the corresponding available-download metadata
for the portal UI and clients without requiring a room management key or
account. Anonymous room management uses a browser-held management key, while
workers continue joining with room ID and join secret.
Ship the manifest **together with** its referenced artifacts. Release binaries,
sidecar checksums, and build output are ignored by Git. Do not commit a
generated manifest without publishing its files to the deployment's releases
directory.

## Installing the current macOS build

The current native files are **Apple Silicon arm64** and require **macOS
14.4 or newer** (the maximum minimum OS version recorded by the three supplied
Mach-O libraries). Intel Macs and older macOS versions are not supported by
this artifact. Packaging checks each supplied library before building and every
Mach-O binary in the app-image (including the launcher/runtime) against
`--mac-arch` and `--mac-min-version`; incompatible inputs fail without publishing
a new release. Universal binaries must contain the requested architecture.
The generated app's actual `LSMinimumSystemVersion` is set to `14.4`, then the
app is ad-hoc re-signed and its nested signatures verified **before** packaging
the DMG. A desired minimum lower than any bundled binary is rejected, not
silently advertised.

1. Download the available macOS DMG, open it, and drag `RamanujanCluster.app`
   into Applications. Alternatively, launch the local app-image directly.
2. This development build is ad-hoc signed, **not Developer ID signed or
   notarized**. If macOS blocks opening a downloaded copy, verify its checksum
   against the portal release metadata first. Only for this trusted artifact,
   use Finder **Control-click → Open**, or after an attempted launch use
   **System Settings → Privacy & Security → Open Anyway**. Do not disable
   Gatekeeper globally. If organizational policy disallows the override, a
   properly signed/notarized distribution is required.
3. Start the app. The local portal defaults to `http://localhost:8090`; select
   the development HTTP checkbox, enter the anonymous room ID/join secret and
   device name, then **Join room → Start worker**. No portal account is needed.

The original `ws` interpreter exports `NativeProcessor.process` but lacks
`changeShard`/`resetShardSession`. The worker does not silently suppress this:
an interpreter task requesting weight eviction calls `changeShard`, logs the
missing JNI-method error, and reports failed completion to the portal. That
mode requires a compatible interpreter supplied later; the original files are
not rebuilt or replaced. Dedicated LLM chat uses `libramanujan_llm` and its own
session lifecycle, so it does not call those missing interpreter methods.
OpenCL compatibility and real end-to-end chat still need device testing.

```sh
python3 -m unittest discover -s Ramanujan-Compute-Cluster/installer -p 'test_*.py'
```
