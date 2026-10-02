# Internal Homelab cluster API

Homelab is a trusted, private execution service, **not an authentication boundary**.
Expose it only behind a gateway that authenticates devices and injects the
authenticated device UUID and exact cluster ID. Never accept either identity
from an unauthenticated browser or device.

For a gateway on the same host, start Homelab with
`RAMANUJAN_HOMELAB_BIND_ADDRESS=127.0.0.1 java -jar developer-console-1.0-SNAPSHOT-fat.jar homelab 8888`
and point the gateway's private backend URL at `http://127.0.0.1:8888`.
Otherwise bind a private interface and firewall the Homelab port so only the
gateway host can connect. Do not publish its port through containers or public
reverse proxies. The default binding remains wildcard for legacy private-lab
deployments; that default is **not safe for direct public exposure**.

Cluster IDs are optional JSON strings or query strings. Omission means legacy,
untagged work. A tagged task is returned only to a worker polling with the same
cluster ID; untagged tasks remain available to workers in any cluster.

Inline `csvInformationList` entries are exactly `{fileName, data}`. `data` is CSV
text; `fileName` is a logical array label matching
`[A-Za-z_][A-Za-z0-9_]*(\.csv)?`, for example `values.csv`. It is not a filesystem
path. Inline submissions never resolve existing CSV/binary sidecars on the
server, even if a matching basename exists. Legacy file-based `args` retain their
sidecar behavior. Large inline CSV conversion writes generated binaries in the
service working directory, not to a caller-selected path.

| Endpoint | Cluster-aware request |
| --- | --- |
| `POST /orchestrator/run` | Inline JSON `{code, csvInformationList?: [{fileName, data}], requestId?, affinity?, evictWeights?, clusterId?}`; legacy JSON `{args: [kernelPath, ...csvPaths], ...}` is also accepted. Blocks until the DAG completes and returns `{status, requestId}`. All successors retain the cluster. |
| `POST /llm/step` | Existing stage JSON `{affinity, session, graph?, files?, tokens \| hidden, n, pos, output?, timeout?, clusterId?}`. |
| `POST /llm/close` | JSON `{affinity, session, timeout?, clusterId?}`. |
| `POST /llm/chain` | JSON `{stages: [{affinity, session, graph?, files?}], tokens \| hidden, n, pos, output?, timeout?, clusterId?}`. One cluster applies to the whole chain; stage-local cluster IDs do not override it. |
| `GET /pings/open` | Query `uuid=<device UUID>&clusterId=<cluster ID>&affinityLimit=<optional integer>`. UUID is required, including for untagged polling. Returned task data includes task `uuid` and nullable `clusterId`. |
| `GET /binary/fetch` | Query `path=<URL-encoded path>&uuid=<device UUID>&clusterId=<cluster ID>`. |
| `GET /binary/stat` | Same query as fetch; returns `{status, size, mtime}`. |
| `POST /orchestrator/uploadBinary` | Raw octet-stream body. Query `taskUuid=<task UUID>&uuid=<device UUID>&arrayId=<array ID>&clusterId=<cluster ID>`. Legacy untagged uploads may continue using `uuid=<task UUID>&arrayId=<array ID>`. |
| `POST /task/complete` | JSON `{uuid: <task UUID>, hostId: <device UUID>, clusterId?, data?, error?}`. Host and cluster must exactly match the polling assignment. Replays, unknown tasks, wrong hosts, and wrong clusters receive HTTP 403. |
| `POST /orchestrator/dump` | Existing JSON `{requestId, name, path, raw?}` plus query `clusterId=<cluster ID>` (JSON `clusterId` is also accepted). Tagged results require an explicit request ID; missing, other-cluster, and legacy lookups cannot read them. |

The gateway must preserve task UUIDs while substituting authenticated **device**
UUIDs in polling/fetch/upload queries and `hostId` in completion bodies.
`uploadBinary` deliberately uses separate `taskUuid` and device `uuid` fields.
For migration it also accepts task `uuid` plus authenticated `hostId` query.

Affinity ownership, chain grouping, and native LLM session names are namespaced
by cluster, not by worker architecture. Fetch allowlists are cluster-specific;
untagged input files are available to every cluster. Tagged fetches additionally
require an outstanding assignment for the requesting device in that cluster.
Uploaded arrays must belong to the assigned kernel run. Completed native outputs
are stored by `(clusterId, requestId)` and do not replace legacy stdin/global
stores. Binary uploads are kept in `.ramanujan-homelab-outputs/` under the service's
working directory.

Legacy scripts that omit `clusterId` retain untagged scheduling and result
lookup behavior. Completion now always requires the assigned worker's `hostId`;
anonymous polling is rejected.
