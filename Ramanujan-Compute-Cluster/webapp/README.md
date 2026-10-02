# Cluster portal

The portal has **no login or signup**. On first use it creates an anonymous
management key and retains it in this browser. That key authorizes room
management, job submission and chat. Save it privately using **Private access
key**; use **Restore rooms** to access the same rooms from another browser.
Management keys currently expire after ten years. Clearing browser storage
without saving the key loses access to the rooms.

Devices join with a room ID and a separate join secret. A device receives a
revocable, room-scoped worker URL. Neither management keys nor device tokens
are stored in plaintext in the SQL database. The join secret is shown only
when the room is created and is stored as a salted scrypt hash.

## Local preview

Requires Node.js 22.13+ and Python dependencies used by the GGUF runner.

```sh
cd Ramanujan-Compute-Cluster/webapp
npm ci
LOCAL_DB="$HOME/.ramanujan/cluster-portal.sqlite" npm start
```

Open <http://localhost:8090>. Local preview uses a persistent SQLite database
and binds to localhost. SQLite is deliberately disabled in production.
The room/chat UI can be explored before starting the execution services;
actual answers require a running orchestrator and a joined native worker.

The gateway reuses the existing services:

| Setting | Local default | Purpose |
| --- | --- | --- |
| `MIDDLEWARE_URL` | `http://127.0.0.1:8888` | Python translation, DAG submission and status |
| `ORCHESTRATOR_URL` | same as `MIDDLEWARE_URL` | Device polling, results, native LLM task dispatch |
| `PUBLIC_URL` | `http://localhost:8090` | URL returned to joining devices |
| `HOST`, `PORT` | `127.0.0.1`, `8090` | Listening address |

For other devices on a LAN, set `HOST=0.0.0.0` and `PUBLIC_URL` to the actual
portal address. Use HTTPS for device enrollment outside localhost; do not
expose the internal middleware or orchestrator ports to the internet.

The existing middleware application installs the orchestrator routes in the
same HTTP server. Do not assume a second orchestrator listener on port 8889.
Set `ORCHESTRATOR_URL` separately only when deploying it behind a separately
configured internal route.

## LLM chat

The server administrator registers already-sharded, supported GGUF models:

```sh
MODEL_ROOT=/absolute/path/to/models \
MODELS_FILE=/absolute/path/to/models.json \
LOCAL_DB="$HOME/.ramanujan/cluster-portal.sqlite" npm start
```

See `models.example.json`. Catalog entries identify the shard package and
`gguf-metadata.json`, both inside `MODEL_ROOT`. Weight paths submitted to the
LLM gateway are checked against that root after resolving symlinks.
Models are not uploaded or downloaded through the chat UI.

Select a room and model, join/start a device client, and ask a question.
The server runs the existing `sharded-llm/run_gguf_shards.py --runtime native`
driver against the room-scoped gateway. Computation takes place on joined
devices, not in the Node.js portal. One inference request per cluster runs
at a time; missing workers, lost native sessions and runtime failures are
reported as errors.

Questions are independent, with a visible session-local transcript.
This is greedy generation, not streaming or a multi-turn conversation.
An administrator can set `promptTemplate` containing `{question}` for a
model's chat syntax; the example uses Qwen's template. `maxContext`
(default 1024) and `maxNewTokens` (default 128) bound generation, and output
is cut at the first `stopSequences` entry (default: common end-of-turn tokens
such as `<|im_end|>`). Large models need small budgets: the example's
Qwen3.8 27B entry (empty `<think>` block to skip reasoning, 24 new tokens)
answered in 30-55 s on one 8 GB Apple M3 at ~5 s/token, streaming 16 GB of
weights from SSD per token. A non-shared device caches every shard it owns, so
a single device needs ~16 GB of free disk for that model. That first download
happens inside the first chat request, so the entry sets `requestTimeout`
(seconds per request, default 120, max 1800) to 1800. The existing runtime's
architecture/quantization limits still apply. See
[`GGUF_MODELS.md`](../../sharded-llm/GGUF_MODELS.md) and
[`NATIVE_LLM.md`](../../sharded-llm/NATIVE_LLM.md).

For direct CLI inference, put the private management key in the process
environment as `RAMANUJAN_PORTAL_TOKEN` and use:

```sh
python3 sharded-llm/run_gguf_shards.py --runtime native \
  --homelab https://PORTAL/api/clusters/ROOM_ID/homelab \
  --package /srv/ramanujan/models/MODEL-shards \
  --metadata /srv/ramanujan/models/MODEL-ir-plan/gguf-metadata.json \
  --prompt "The capital of France is" --max-new-tokens 16
```

The driver passes the token only in an HTTP authorization header, never a
URL or command-line argument. The gateway derives `clusterId` from the key
and room instead of trusting caller-supplied cluster or device identifiers.

## Production database and downloads

Set `DB_HOST`, `DB_PORT` (optional), `DB_USER`, `DB_NAME`, `DB_PASSWORD` and,
when required, `DB_TLS=true` through the deployment environment or secret
manager. Run `npm run migrate` before `npm start`. Migration creates the
portal's tables without dropping existing computation tables. Apply the
separate db-layer cluster migration to the shared execution database as well.
Set `NODE_ENV=production` and an HTTPS `PUBLIC_URL`.

Do not put GCP/database credentials into source, installers or configuration
examples. GCP deployment is deferred until the local experience is approved
and project/target/authentication details are supplied.

Installer downloads come from `../installer/releases/manifest.json` (or
`RELEASES_DIR`). Only real, checksum-verified artifacts are offered. Platforms
without supplied native libraries are marked unavailable, not linked to fake
downloads. Native runtimes need OpenCL and are not a sandbox: only trusted
room managers should submit computation to voluntarily joined devices.

```sh
npm test
```
