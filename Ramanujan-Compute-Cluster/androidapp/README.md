# Private-room Android worker

This new application (`in.ramanujan.cluster.android`) is separate from the
legacy top-level `androidapp`, which remains unchanged.
It is source-only for the current release: no Android native binaries/APK are
provided or required for the macOS installer. Android downloads stay
unavailable until a real, verified signed APK is supplied.

## Build prerequisites

* JDK 17 (Gradle 8.5 / Android Gradle Plugin 8.2).
* Android SDK 34, build tools, NDK `26.1.10909125`, CMake `3.30.3`.
* A **host** `protoc` 3.25.3 executable. It must run on the build machine, not
  be an Android executable; the native protobuf dependency cannot execute a
  target-architecture compiler while cross-compiling.
* Existing Maven `commons` and `rule-engine` artifacts installed locally.

From the repository root:

```sh
mvn -f commons/pom.xml install -DskipTests
mvn -f rule-engine/pom.xml install -DskipTests
export JAVA_HOME=/path/to/jdk17
export ANDROID_HOME=/path/to/android-sdk
bash Ramanujan-Compute-Cluster/androidapp/build.sh \
  -PhostProtoc=/absolute/path/to/protoc :app:assembleDebug
```

PowerShell can use `build.ps1` with the same Gradle arguments. The scripts use
an installed Gradle or the repository's existing Gradle 8.5 wrapper; they do not
modify the legacy app or copy binary wrapper artifacts. If user-level Gradle
settings override Java, pass `-Dorg.gradle.java.home=/path/to/jdk17`.

The default APK ABI is **arm64-v8a**, not all Android architectures. The minimum
SDK is **26** for Java NIO `Path`/`Files` and `java.util.Base64`. Target SDK 34
requires notification and foreground data-sync permissions. Building other
ABIs requires adding them in `app/build.gradle` and separately verifying
native/OpenCL availability; it is not claimed here.

## Actual reuse and native packaging

Gradle copies a narrow set of shared sources into an ignored generated directory:
`ExecuteInlineWorker`, `WorkerBinaryCache`, `LlmTaskHandler`, `WorkerCapacity`, `Operation`, and
the desktop join protocol classes. It does not include the developer console's
server/orchestrator implementation. `rule-engine` supplies the real
`NativeProcessor`, protobuf serializer, and `LlmSession`.

The new CMake wrapper adds the existing
`ramanujan-native/native/CMakeLists.txt` with `GPU_ENABLED=ON` and builds
`native_lib` (`libnative.so`) **and** `ramanujan_llm`
(`libramanujan_llm.so`), plus a small `cluster_preflight` JNI library. Libraries
are built for the selected ABI and actually packaged by Android's
`externalNativeBuild`; no desktop libraries or old prebuilt JNI files are reused.
CMake fetches the upstream dependencies already specified by that native
project, so first builds need network access or a populated FetchContent cache.
To reuse an existing native build's source cache offline, add
`-PnativeDepsDir=/absolute/path/to/native-build/_deps`
(must contain matching `protobuf-src` and `openclheaders-src` source trees).
On macOS the desktop build uses the system OpenCL framework and therefore
does not fetch headers; use an Android native build's cache instead.

## Operation and limitations

Join using portal URL, room ID, join secret, and device name. HTTPS is strongly
required by default; an explicit development-only checkbox permits HTTP for
localhost/private IP addresses, never public hosts. HTTP exposes credentials
to network observers. Anonymous private rooms do not require accounts or
login/signup. The room owner manages/submits through the portal with a
browser-held management key; devices only use room ID and join secret.

The joined bearer URL and device/room identity live in app-private preferences.
The join secret is cleared, never persisted. Backup is disabled. Requests use
the scoped worker base for `/pings/open`, `/pings/capacity`, `/task/complete`, `/binary/stat`,
`/binary/fetch`, and `/orchestrator/uploadBinary`; room and device identity are
enforced by the portal gateway.
Binary uploads retain `uuid=<task UUID>&arrayId=...`, both scoped and legacy.
For scoped uploads/fetch/stat the gateway adds authenticated `hostId` and
`clusterId` query parameters rather than replacing a task UUID. Only polling
uses query `uuid` for device identity. Completion keeps the task in its JSON
`uuid` while the gateway supplies `hostId` and `clusterId`.

Start creates a foreground service and persistent Stop notification. Before
any polling, it explicitly loads `libnative`, probes `libOpenCL.so` for an
accessible OpenCL platform/GPU, and loads `libramanujan_llm` via the real
`LlmSession`. Missing/inaccessible vendor OpenCL produces a clear unavailable
status and stops the service instead of advertising LLM readiness.

**This is not a claim that all Android GPUs work.** Some vendor OpenCL libraries
are absent or blocked by Android linker namespaces. Passing preflight does not
guarantee shader compilation, memory capacity, supported tensor operations, or
successful LLM inference. Those still need tests on real hardware.
The manifest declares the optional `libOpenCL.so` vendor library for Android
12+ linker namespace access; devices without it can still install the app and
receive the unavailable error. The wrapper enables OpenCL 2.0 **header types**
needed by the existing Android dynamic loader; it does not require a 2.0 GPU.

The worker uses one execution thread, a two-shard affinity limit, eight native
LLM sessions, and an app-private persistent binary cache. Native rule-engine
temporary output is redirected into the app-private cache with `TMPDIR`; this
avoids the native desktop `/tmp` default, which Android apps cannot access.
Stop cancels polling/download HTTP connections, interrupts workers, and closes
native sessions once an executing native call finishes. It does not forcibly
interrupt a running JNI/GPU call. The service is not sticky and is not started
at boot; Android may still stop it under power/memory restrictions.
The same shared worker paces immediate empty central polls to a 100 ms total
duration, without adding delay to existing longer homelab polls or task responses.

An independent sampler posts capacity approximately every 10 seconds, including
during downloads, polling and inference. Before its first post, the sampler
prepares the selected native LLM device/context and kernels without opening a
model session, resetting state or evicting weights. Thus cold workers report GPU
total/max-allocation and runtime readiness before their first capacity plan.
Probe failures remain explicit in statistics and do not stop legacy polling.
Android injects physical RAM measurements
from `ActivityManager.MemoryInfo` and cache-filesystem free space from `StatFs`;
JVM heap size is never reported as device RAM. GPU counters come only from the
already-selected native OpenCL device; unsupported free-memory counters are null.
Sessions stay pinned until explicit `/llm/close` or shutdown, never LRU-evicting state.
Legacy and planned work both default to an eight-session ceiling. Drivers may
group adjacent canonical layers into native graphs before capacity admission.
An operator may explicitly raise the planned ceiling with
`--capacity-session-limit` on the shared worker, subject to backend resource
admission, without unbounding legacy work. Downloads preserve a 64 MiB disk reserve.
Explicit native `weights: "stream"` remains supported for models larger than cluster
RAM/VRAM; one device can host multiple pinned contiguous stages. Assigned weights
are preflighted against incremental disk space, not required to fit physical memory.
See [ORCHESTRATION.md](../../ORCHESTRATION.md) for the capacity ping schema,
counter semantics and placement.

## Signed release

Set these environment variables locally (never commit the keystore/passwords):

```text
RAMANUJAN_ANDROID_KEYSTORE=/absolute/path/to/release.jks
RAMANUJAN_ANDROID_STORE_PASSWORD=...
RAMANUJAN_ANDROID_KEY_ALIAS=...
RAMANUJAN_ANDROID_KEY_PASSWORD=...
```

Run `:app:assembleRelease` with the same host-protoc argument. Without signing
configuration, Gradle produces an unsigned release, which the installer
publisher refuses. Publish the signed APK using the installer script and
`apksigner`. No APK is checked in.
