# Desktop room worker

A Java 8-compatible Swing client. Device members join anonymous private rooms;
the room owner manages/submits through the portal using a browser-held
management key. No account, login, or signup is required. The GUI is not a
task-submission client; it only needs the room ID and join secret.

## Build

From the repository root:

```sh
mvn -f Ramanujan-Compute-Cluster/client/pom.xml package
mvn -f developer-console/pom.xml package
```

The worker Maven dependencies must already be installed, as for the existing
developer console. The standalone client only needs Maven and Java 8+.

For a development launch, put the built client JAR, fat worker JAR renamed to
`developer-console.jar`, and a `native/` directory beside each other. `native/`
must contain **both** `native` and `ramanujan_llm` shared libraries for this
machine, plus their dynamic dependencies. Then run:

```sh
java -jar cluster-client-1.0.0.jar
```

The current macOS installer copies the user's existing `~/Desktop/ws/libnative.dylib`
and `libnative_llm.dylib` interpreter builds unchanged. It separately includes
`libramanujan_llm.dylib` from the native build for `LlmSession`; the optimized
interpreter `libnative_llm.dylib` is not a substitute for that dedicated runtime.

For end users, use the bundled-runtime UI installer described in
[`../installer/README.md`](../installer/README.md); no separately installed Java
is required.
The current supplied macOS libraries require Apple Silicon **arm64** and
**macOS 14.4+**. The development app is ad-hoc signed, not notarized; see the
installer's trusted-artifact Gatekeeper installation instructions.

## Join and reconnect

Enter the portal URL, room ID, join secret, and device name. Joining calls
`POST /api/devices/join` with `{roomId,joinSecret,name,platform}`. The response
must contain `{deviceId,clusterId,workerUrl}` with an absolute, same-portal
`/worker/<bearer>` URL. Redirects and cross-origin worker URLs are rejected.
HTTPS is required by default. The development-only checkbox explicitly permits
HTTP for localhost/private IP addresses, never public hosts. HTTP exposes the
join secret and worker bearer to network observers even on local networks.
The initial desktop portal URL is `http://localhost:8090` for the local test
portal; explicitly select the development HTTP checkbox to join it. Production
users must enter their HTTPS portal URL.

The last successful registration is stored under
`~/.ramanujan/cluster-client/device.properties` with owner-only POSIX permissions
or Windows ACLs. **This file contains a credential; do not share it.** The join
secret is cleared after joining and is not persisted. Start reconnects to the
last successfully joined room; changing text fields alone does not change that
registration. Join again to switch rooms or replace a revoked device token.

Start launches the real fat worker JAR as a child of the bundled Java runtime:

```text
java -jar developer-console.jar worker 1 --cache <localcache> --max-shards 2 --llm-sessions 8
```

The scoped worker URL is supplied only in `RAMANUJAN_WORKER_URL`, never on the
command line. `RAMANUJAN_WS` points to the bundled native directory and `TMPDIR`
to the local cache. Worker traffic stays beneath the scoped gateway URL;
the gateway supplies room identity and replaces the host UUID with `deviceId`.
The bounded GUI log redacts bearer URLs. Stop terminates the child, with a
five-second forced-stop fallback; closing the window also stops it.
Immediate empty responses from the legacy central poll endpoint are paced to
a minimum total poll duration of 100 ms. Homelab long-polls already exceeding
that duration get no extra delay, and real task responses are not throttled.

LLM tasks use the existing `LlmSession`/`LlmTaskHandler` OpenCL runtime, persistent
binary cache, and eight-session LRU. Merely shipping the libraries does **not**
prove a compatible GPU exists. Missing drivers, unsupported OpenCL devices,
or kernel/build errors are shown by worker execution logs. Desktop GPU
compatibility needs validation on the actual device.
Binary cache objects are namespaced by a hash of the worker gateway URL, so
different rooms/devices cannot reuse model bytes solely because paths and file
stat metadata match. No bearer URL is written into cache metadata.
The worker records which LLM shards it holds (model, layer offsets, embedding,
head) in `shards.json` in the cache directory. It reports them in capacity pings,
so after a restart or rejoin the orchestrator gives the device back its cached
layers and shards only the remainder to other devices.
The supplied `ws` interpreter lacks `changeShard`; legacy interpreter tasks
requiring weight eviction fail visibly in the log and completion response.
Dedicated LLM chat uses the separate native runtime and does not require that
interpreter method.

## Tests

```sh
mvn -f Ramanujan-Compute-Cluster/client/pom.xml test
mkdir -p developer-console/target/test-work
mvn -f developer-console/pom.xml \
  -Djava.io.tmpdir="$PWD/developer-console/target/test-work" \
  -Dtest=PrivateRoomWorkerTest,WorkerBinaryCacheTest,LlmTaskHandlerTest test
```

Tests cover join JSON, scoped URL validation/redaction, private reconnect
storage, credential-free process arguments, required libraries, scoped polling,
and stop lifecycle.
