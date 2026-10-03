# Ramanujan Compute Cluster

A new product surface for private compute rooms and native LLM inference.
The existing legacy apps remain where they are; the new components live here:

| Directory | Component |
| --- | --- |
| [`webapp/`](webapp/README.md) | Account-free room portal, chat UI, device enrollment and downloads |
| `client/` | Desktop GUI for joining a room and starting/stopping a native worker |
| `androidapp/` | New Android room client and foreground worker |
| `installer/` | Platform packaging and checksum-addressed release artifacts |

The project reuses `middleware/`, `orchestrator/`, `db-layer/`,
`ramanujan-native/` and the existing GGUF translator/native LLM driver.
It does not move or duplicate those shared services.

## Room lifecycle

1. Open the portal. There is no signup or login; a private management key is
   retained in this browser. Save that key privately for recovery.
2. Create a room and save its room ID and device join secret.
3. Download the client, enter the portal address, room ID and join secret,
   and start contributing compute.
4. Choose the room and model in the portal chat, then ask a question.
   Native workers in that room execute inference and return the answer.

Management keys authorize control and submission, while device tokens authorize
worker polling/results only. Room IDs alone do not grant access to private
rooms. Devices can be revoked from the portal.

## Cluster routing

`clusterId` is an optional execution constraint:

- A tagged DAG/native task can run only on a device with the matching tag.
- An untagged DAG remains eligible for any device, including a clustered one.
- A device without a tag cannot take a tagged task.

The private portal derives cluster and device identity from its persisted
tokens, overriding identifiers supplied by callers. Shared computation
services must stay on a trusted internal network behind the portal.
Cluster IDs are scheduling constraints, not substitutes for access control.

Schema changes need both fresh-install definitions and an ALTER migration for
existing SQL deployments. Do not recreate or drop the shared computation
tables to add cluster IDs. Local portal preview uses SQLite; production portal
and execution services use their SQL configuration.

## Current platform constraints

The current macOS installer requires **Apple Silicon and macOS 14.4 or newer**.
Native interpreter libraries are supplied from `~/Desktop/ws`.
`libnative_llm.dylib` is the DSL interpreter variant; the dedicated
`libramanujan_llm.dylib` is a separate library required by native LLM sessions.
Do not rename one to impersonate the other.

Windows DLLs, Linux SOs and Android native libraries must be supplied for the
correct architectures before those installer downloads become available.
The repository includes packaging/app source, but that is not evidence of a
validated binary for each device. Native LLM inference requires OpenCL and the
existing GGUF runtime's supported features. See
[`GGUF_MODELS.md`](../sharded-llm/GGUF_MODELS.md) and
[`NATIVE_LLM.md`](../sharded-llm/NATIVE_LLM.md).

Chat currently uses greedy generation with independent questions and optional
administrator-configured model prompt templates. It is not streaming, a
general chat-template engine, or automatic recovery of lost KV sessions.
Clients run native code on voluntarily donated devices; they are not sandboxes.

## Local first, GCP later

Start with [`webapp/README.md`](webapp/README.md) and review the local portal at
<http://localhost:8090>. GCP deployment is deliberately deferred until the
local UI and device inference are approved. Project/location, deployment
target and local authentication will be requested then; no GCP credentials
belong in this repository or in installers.
