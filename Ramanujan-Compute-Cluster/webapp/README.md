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
LLM gateway are checked against that root (or the requesting owner's own
models, below) after resolving symlinks.

Select a room and model, join/start a device client, and ask a question.
The server runs the existing `sharded-llm/run_gguf_shards.py --runtime native`
driver against the room-scoped gateway. Computation takes place on joined
devices, not in the Node.js portal. One inference request per cluster runs
at a time; missing workers, lost native sessions and runtime failures are
reported as errors.

### Conversations

Chat is multi-turn. The browser sends the room's transcript
(`messages: [{role: "user" | "assistant", content}]`, alternating and ending
with the new question; failed questions are left out). The server renders it
with the model's chat format and the driver (`--prompt-turns -`) drops the
oldest exchanges until the prompt plus `maxNewTokens` fits in `maxContext`.
The reply reports `droppedTurns`, and the UI notes when that happened.
**New chat** clears the room's transcript, which lives only in the browser tab.
A single `question` is still accepted.

Catalog entries choose a `chatFormat`: `chatml`, `llama3`, `gemma`, `phi3`,
`zephyr`, `mistral` or `plain` (the default). `generationPrefix` overrides the
text that starts the reply. Templates never include a BOS token; the GGUF
tokenizer adds it. A legacy `promptTemplate` containing `{question}` is mapped
to its chat format when it starts with that format's user turn, keeping the
text after the turn as the generation prefix (the example's Qwen3.8 entry
keeps its empty `<think>` block). Other templates stay single-turn.

Generation is greedy, not streamed. `maxContext` (default 1024) and
`maxNewTokens` (default 128) bound it, and output is cut at the format's
end-of-turn tokens (or `stopSequences`). Large models need small budgets: the
example's Qwen3.8 27B entry uses 24 new tokens. A non-shared device caches
every shard it owns, so a single device needs ~16 GB of free disk for that
model. That first download happens inside the first chat request, so the entry
sets `requestTimeout` (seconds per request, default 120, max 1800) to 1800. The
existing runtime's
architecture/quantization limits still apply. See
[`GGUF_MODELS.md`](../../sharded-llm/GGUF_MODELS.md) and
[`NATIVE_LLM.md`](../../sharded-llm/NATIVE_LLM.md).

### Bring your own GGUF

**Your models** in the chat header adds a GGUF for every room owned by the
same management key, either by uploading the file from the browser or by
pasting an `https://` URL (for example a Hugging Face `/resolve/main/...gguf`
link) that the server downloads. There is no size limit, but the server
refuses a model when its disk would drop below a 2 GB reserve (a model needs
about twice its GGUF size while converting, then about its size). Models are
converted one at a time with the converter in `sharded-llm/converter`
(`emit_gguf` with 4 shards, then `gguf_ir_plan`). The GGUF is deleted after
a successful conversion, and the chat format comes from the GGUF's
`tokenizer.chat_template`. Unsupported architectures or quantizations show
the converter's error.

Files live in `USER_MODEL_DIR` (default `~/.ramanujan/portal-models`), one
directory per model, recorded in the `owner_model` table. Each owner sees
only their own models and the gateway only serves a model's files to its
owner. URL downloads follow at most five redirects and refuse private,
loopback and link-local addresses on every hop, including after DNS
resolution. A portal restart marks unfinished uploads, downloads and
conversions as failed. Set `USER_MODELS=off` to disable the feature.

Set `MODEL_CACHE_BUCKET` to an existing private Cloud Storage bucket to
persist converted shard packages and IR plans. The portal uses Application
Default Credentials (the VM's attached service account on Compute Engine),
not a checked-in key. Grant that account object access only on this bucket.
Packages are stored under `models/<model-id>/`; the manifest is published
last and a model becomes READY only after storage succeeds. If the local
package is missing, the next chat restores it with CRC32C and size checks.
Deleting a model deletes its cloud objects too. The original GGUF is not
retained. Cloud Storage is durable backing, not the worker's hot cache;
devices still cache their assigned files locally. Without this setting,
local-only behavior is unchanged.

For small Cloud SQL instances, set `RAMANUJAN_DB_POOL_SIZE` on the middleware
process (default 4). Keep the middleware listener on `127.0.0.1` when the
portal shares its VM. Apply the cluster-routing and native-inference SQL
migrations once, then the portal's `schema.sql`, without replacing existing
tables. Keep SQL credentials in protected runtime files outside the checkout.
For private testing, use SSH tunneling for browser access; a public deployment
requires HTTPS.

### Capacity-aware placement (opt-in)

After applying `db-layer/src/main/resources/migrations/20261002_capacity_admission.sql`,
set `RAMANUJAN_CAPACITY_AWARE=1` on both middleware and portal (upgrade
workers only when idle). The orchestrator then places each model's layers on
the room's devices by VRAM and reuses shards devices already cache. The
gateway exposes owner-only `/llm/capacity`, `/llm/plan` and `/llm/plan/release`
under each room's `homelab` path. Algorithm, budgets and lifecycle:
[ORCHESTRATION.md](../../ORCHESTRATION.md).

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

## Code execution

The **Code execution** tab is a small VS Code-style editor (`public/ide.js`,
no external scripts, so it works under the portal's `default-src 'self'`
CSP). Upload `.py` files or a whole folder (`__pycache__` and non-Python
files are skipped), create, rename or delete files, edit them in tabs with
Python highlighting, and choose the **Main file**. Files are kept in this
browser's local storage per room; they are sent only when you press **Run on
cluster** (Ctrl/Cmd + Enter). The output panel shows the program's final
top-level variables and arrays.

The editor calls the same job API, which also accepts single-file `code`:

```http
POST /api/clusters/ROOM_ID/jobs
Authorization: Bearer MANAGEMENT_KEY

{"files": {"main.py": "from pkg.ops import add\nx = add(1, 2)\n",
           "pkg/ops.py": "def add(a, b):\n    c = a + b\n    return c\n"},
 "entryPoint": "main.py"}
```

Paths must be relative `.py` paths (no `..`), at most 100 files and 2 MB in
total, and `entryPoint` must be one of them. File inputs (`open`,
`load_binary`) are rejected. Programs the translator cannot compile get a
`422` with a readable `error`. `GET` on the same route lists jobs and refreshes
running ones; a job the middleware no longer knows about (for example after
it restarts) is marked `FAILED` instead of breaking the list. Programs use the
interpreter's Python subset: for example, prefer `while` loops, and pass or
return plain names or constants (`y = x * x` then `return y`).

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
