# Cluster routing in the Java backend

`CodeRunRequest` accepts an optional string `clusterId`. Omission/JSON null is
legacy untagged work. `TranslateAndRunHandler` passes it to `RunService`, which
tags the workflow async record and every DAG element before persistence.

The tag is persisted both in serialized `BasicDagElement` storage objects and
SQL DAG metadata. Subsequent DAG execution loads that tag from storage and
passes it through every orchestration-submission retry. The direct middleware
to orchestrator service call and the orchestrator submission HTTP JSON both
carry it. Orchestrator async records retain it in both SQL and in-memory DAOs.
Checkpoint resume loads the stored async record rather than constructing an
untagged placeholder; stale-device retries similarly reuse the persisted tag.

## Internal API contract

Services remain trusted/internal. Authentication belongs at the gateway, which
must inject exact authenticated device identities and cluster IDs.

Only the authenticated portal/gateway may be externally reachable. Keep Java
middleware/orchestrator listener ports on a private network or firewall them to
the gateway host only; do not publish raw `/pings/open`, `/task/complete`,
`/orchestrate`, status, checkpoint, or storage endpoints publicly. These services
trust forwarded identity and do not authenticate bearer tokens themselves.
For a co-located Homelab gateway, bind Homelab to loopback using
`RAMANUJAN_HOMELAB_BIND_ADDRESS=127.0.0.1`; see its API document for deployment
details. A cluster ID is a routing tag, not a credential.

- Existing middleware code-run JSON: `{code, csvInformationList?, clusterId?}`.
- `POST /orchestrate`: existing JSON plus optional `clusterId`.
- `POST /pings/open`: query `uuid=<device UUID>&clusterId=<cluster ID>`.
  UUID is required. Cluster omission is legacy global registration.
- `POST /task/complete`: existing JSON `{uuid: <task UUID>, hostId: <device UUID>,
  data: {...}}` plus optional `clusterId`. Completion requires a current
  assignment for that exact device, task, and assigned cluster; mismatches
  return HTTP 403 before storing results.
- Middleware `GET /status`: query `uuid=<workflow async ID>&clusterId=<cluster ID>`.
- Orchestrator `GET /statusorch`: query
  `uuid=<orchestrator async ID>&clusterId=<cluster ID>`.
  A tagged task's status/results require its exact tag; omitting it or using a
  different cluster returns HTTP 403. Legacy untagged status calls omit it.

For the Homelab execution API, including native LLM calls and binary upload/fetch,
see [homelab-cluster-api.md](homelab-cluster-api.md).

## Device selection and assignment

Available devices are tracked in memory from open pings, including their cluster
and last-seen time. Availability expires after 60 seconds without an open ping.
A tagged task selects only an exact-match device; an untagged task can select any
device. Selection removes both the stack entry and membership marker, allowing
devices to register again after completion. An engaged device is removed from
the available stack so it cannot be assigned twice.

Cross-orchestrator registration forwarding preserves the cluster query through
all retry attempts. Devices re-register availability after process restarts.
SQL host mappings durably store the *assigned device cluster*, independently of
the nullable task tag. This also keeps an untagged task assigned to a clustered
device isolated from polls/completions using another identity or cluster.

## Database deployment

For an existing database, apply **once**, before deploying updated services:

`db-layer/src/main/resources/migrations/20261002_cluster_routing.sql`

It adds nullable `clusterId` columns to `asyncTaskMiddleware`,
`asyncTaskOrchestrator`, `dagElementMetadata`, `availableHost`, and `hostMapping`.
It does not drop tables or modify existing rows. Existing rows remain untagged.
Updated fresh-table SQL resources contain the same columns. The binary collation
preserves case-sensitive cluster identity. Do not apply the ALTER migration to a
fresh database created from the already-updated schemas.
