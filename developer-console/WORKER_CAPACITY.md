# Worker capacity and pinned inference sessions

Desktop and Android `ExecuteInlineWorker` send `POST <serverUrl>/pings/capacity`
from an independent daemon sampler/sender, immediately and approximately every
10 seconds (plus sampling/network time). Polling, model downloads and inference
do not occupy this thread. Requests have bounded 3-second connect/read timeouts;
failures do not stop task execution. Responses need only be successful HTTP
status codes; JSON is not required. Stop disconnects requests and stops the sender.
Before the first snapshot, this same capacity thread performs one native
preflight: load the LLM library, select the configured OpenCL device, create its
context and compile kernels. Execution/polling threads do not wait for it.
It opens no sessions, loads no model weights, executes no inference and never
resets/evicts existing state. Repeated native probes reuse the selected device.
Preflight failure reports `nativeReady=false`/`supportsRuntime=false` while legacy
execution and physical-memory reporting continue; no guessed GPU counters are added.
Native initialization cannot forcibly be interrupted; shutdown stops subsequent sends.

Version 1 JSON contains:

* `schemaVersion: 1`, `hostId` (the same UUID as `/pings/open?uuid=...`),
  `bootId` (per-worker UUID), and monotonically increasing `sequence` (starting at 1).
* Nullable byte counts: `ramTotalBytes`, `ramAvailableBytes`,
  `ramAllocatedBytes`, `diskAvailableBytes`, `cacheBytes`, `gpuTotalBytes`,
  `gpuAvailableBytes`, `gpuResidentBytes`, `gpuAllocatedBytes`, `gpuMaxAllocationBytes`.
* Nullable selected-GPU identity/capability strings: `gpuDeviceName`,
  `gpuDeviceVendor`, `gpuDriverVersion`, `gpuRuntime` (`"OpenCL"`),
  `gpuRuntimeVersion` (OpenCL device version).
* `unifiedMemory` boolean, `llmRuntimeAvailable` boolean (library loaded, **not**
  proof of inference readiness), `activeSessions` integer, `acceptingNewWork` boolean.
* `nativeReady`: nullable boolean; true only after the LLM OpenCL device/context
  and kernel program initialize. Unknown before initialization; does not guarantee
  a particular graph or tensor layout will execute. `supportsStreaming`: nullable
  boolean reported true by the native runtime API without initializing hardware.
  `supportsRuntime`: nullable boolean, true after successful runtime preflight,
  false after a failed worker preflight (not a promise about a particular model).
* `sessionLimit`: actual legacy `--llm-sessions` ceiling; `capacitySessionLimit`:
  opt-in planned stage-session ceiling (defaults to the legacy ceiling);
  `affinityLimit`: finite `--max-shards` value, otherwise null.
* `activePlans`: object mapping optional native task `planId` values to their
  pinned/opening stage-session counts. Legacy sessions without plan IDs remain in
  `activeSessions` but have no fabricated plan identity. Counts disappear on close,
  failed open, or worker shutdown.
* `cachedFiles`: up to 4096 `{path, bytes, mtime}` fingerprints for immutable
  cache objects belonging to this worker's current backend/gateway. `path` is the
  original backend file path, `bytes` the actual local file size (must match cache
  metadata), and `mtime` the backend timestamp stored at download. Token URLs and
  scoped bearer paths are excluded; no gateway URL is included.
  `cachedFilesTruncated` indicates more valid entries or an incomplete scan.
  The backend must revalidate current file size **and** mtime before using any entry
  to subtract incremental disk requirements. Paths alone are not cache proof.

The gateway may override identity for a private room.
Authenticated room gateways supply `clusterId`; workers do not invent one or send
raw bearer credentials in capacity JSON. Backend capacity admission must reject
unscoped statistics. Legacy/direct endpoints may reject capacity POSTs without
affecting polling or task execution. There are no client timestamps:
the server owns receipt times and staleness decisions. Missing counters are `null`,
not zero. Desktop physical RAM uses reflective Java 8 OS MXBean access; Android
injects `ActivityManager.MemoryInfo` and `StatFs`. Neither uses JVM heap as RAM.
`ramAllocatedBytes` remains null because physical/native worker-resident RAM
cannot be measured truthfully by this portable provider; heap limits, theoretical
reservations and logical host transfer-buffer sizes are not substitutes for RSS.
Disk space refers to the cache filesystem; cache bytes include all regular files
under that root. Unknown/inaccessible filesystem counters are null.

After startup preflight, periodic native telemetry is passive and never selects
another device merely to sample.
Once an LLM OpenCL device is selected it reports that device's global memory and
host-unified-memory property, identity, OpenCL/driver versions and max single-buffer
allocation. `gpuAllocatedBytes` counts this runtime's live OpenCL buffers:
resident **and streamed** weights, KV/recurrent state and scratch.
`gpuResidentBytes` is a narrower resident-weight-only fallback; it does not account
for streamed windows or session workspace and must not be treated as total usage.
These are native allocation counters, not measured physical residency, and exclude driver
overhead, other native runtimes and other processes. CPU-selected devices do not
report GPU byte counts. OpenCL has no portable free-memory query, so
`gpuAvailableBytes` remains null; total minus our allocations is **not** free memory.
Native allocations are already reflected in OS/device free-space counters when
those counters account for committed allocations. Do not subtract the same bytes
again; the planner must handle unified-memory overlap and unavailable RSS conservatively.
With old JNI libraries, GPU counters remain null. Unknown unified-memory topology
reports false; consumers must not infer a dedicated GPU from that alone.

`--llm-sessions N` (default 8) caps **pinned** native stage sessions, including
sessions being opened. The worker rejects new sessions before downloading/opening
when full; it never evicts another session's KV/recurrent state. `/llm/close`
releases a pin; worker shutdown closes all remaining sessions after executions end.
Failed opens release reservations. Drivers must explicitly close all stage sessions
at generation end or lease release; this worker does not invent lease expiry.
`acceptingNewWork` describes admission of **new LLM sessions**, not whether existing
sessions may advance; it uses the opted-in planned ceiling when configured.
At the limit existing sessions and close tasks still work.
While any non-LLM/DAG task is in flight, `acceptingNewWork` is false through
localization, execution, binary upload and completion reporting. DAG resources
have no capacity-plan peak reservation; their allocations are not folded into
LLM-owned allocation counters. OS physical RAM availability remains authoritative.
There is no RAM/GPU-based local rejection or polling change in this release.

Both direct and private-gateway workers retain planned=legacy by default (eight).
Drivers can coalesce contiguous canonical layers into fewer native graphs
**before** reserving exact stage identities, reducing both pinned-session counts
and the sum of idle streaming windows without changing immutable artifact files.
Operators connected to a trusted capacity-aware dispatcher can explicitly set
`--capacity-session-limit N` (1..1024, at least `--llm-sessions`) for many contiguous
planned stages on one host. Only tasks carrying a nonempty top-level `planId`
use that higher total-session ceiling. Legacy/no-plan tasks still use the original
ceiling against **all** pinned/opening sessions. A plan ID alone never raises
the default cap. For example, `--llm-sessions 8 --capacity-session-limit 128`
can accept 65 planned stage sessions without unbounding ordinary legacy work.

The gateway/backend must validate plans and reserve the aggregate per-stage
streaming/resident working memory, state and workspace before dispatch. The worker
trusts its configured dispatcher; it cannot cryptographically validate an opaque
plan ID independently, and does not interpret future `resourceRequirements` without
an agreed contract. Native allocation failures remain explicit task errors.
The higher ceiling is not a promise that N stages fit memory. Private-gateway
URL scoping identifies the configured trust boundary, not independent plan
validation in this worker. Do not enable the higher ceiling against an untrusted dispatcher.

## Streaming and multiple contiguous stages

Graphs preserve native `weights: "auto" | "resident" | "stream"` verbatim.
Explicit streaming is supported even when full model/assigned weight bytes exceed
the device's or cluster's aggregate RAM/VRAM. Multiple contiguous stages can be
pinned on one worker, including multiple streamed sessions. There is no check
requiring all assigned weights to fit memory, and metadata such as
`streamWorkingBytes` is passed through without rewriting it as resident weight bytes.

Planner-side stream admission must budget the largest indivisible tensor and
actual streaming/prefetch working window, plus each assigned session's state,
scratch/workspace, transfer/staging buffers and safety margin. Sum these across
sessions sharing a device. The window can exceed one tensor (the runtime streams
layer/head items and permits prefetch depth); full immutable weight bytes belong
to disk budgeting, not streaming RAM/VRAM admission. Native allocation telemetry
aggregates all live buffers across these sessions. The worker does not invent
an admission estimate from weight file size, or alter native auto/resident/stream
selection. Native max-allocation and filesystem errors remain authoritative.

The existing streamer retains cyclically prefetched items and loader-thread
staging even while a session is idle. Its per-session peak can include
`(stream_depth + 1) * largestItemBytes` plus fixed buffers, with 32 MiB host staging
per loader thread. Sum these windows across pinned sessions; taking only the
maximum cohost window is unsafe without runtime idle unloading and serialized
stream execution. Prefetch lifetime/depth semantics are unchanged.

Downloads check usable disk against the whole incoming file, concurrent download
reservations, and `--disk-reserve-bytes N` (default 67108864 / 64 MiB).
LLM graph/chain localization additionally preflights the sum of all assigned
uncached immutable files before fetching any of them. Duplicate paths and
metadata-valid disk cache hits have zero incremental bytes, including after a
worker restart. Preflight is a snapshot, not an exclusive filesystem reservation;
per-transfer concurrent reservations continue to enforce free-space safety.
Replacements need room for a complete `.partial` file before atomic rename.
Insufficient space produces a task error with required/available/reserved byte
counts; it does not delete pinned sessions or cached model objects. External disk
usage can still race the check; filesystem write errors remain authoritative.

## Validation

Java 8 worker unit/integration tests:

```sh
mvn -f developer-console/pom.xml \
  -Dtest=WorkerCapacityTest,LlmTaskHandlerTest,WorkerBinaryCacheTest,LlmChainGroupingTest test
```

An opt-in cold-start GPU smoke test verifies the first worker heartbeat is ready
before any session or plan, and repeated probes preserve selected-device state:

```sh
RAMANUJAN_WS=/absolute/path/to/native/build \
  mvn -f developer-console/pom.xml \
  -Dtest=WorkerCapacityNativeProbeTest -Dramanujan.test.nativeProbe=true test
```

This smoke test initializes OpenCL/kernels only; it never opens a model or executes
inference. Without the explicit property, it is skipped.
