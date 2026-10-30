# Broker fabric and recovery

## Runnable deployment

The demo generates a temporary local CA, a distinct leaf certificate per broker,
SHA-256 manifest pins, ephemeral ports, static YAML membership, isolated state
directories, and random fabric/telemetry secrets. It uses the actual broker/client
executables. It never installs certificates into the OS trust store.

```powershell
python EMPY_DASHBOARD/demo.py --openssl C:/Libraries/msys64/ucrt64/bin/openssl.exe
# Native QUIC backplane; client-facing ports remain the local TCP demo binding.
python EMPY_DASHBOARD/demo.py --quic --port 8081
# An idle deployment without the dashboard server:
python EMPY_DASHBOARD/cluster.py --brokers 3 --traffic idle --no-dashboard --seconds 60
```

Pass `--bin Targets/Release/bin` to select Release. `--brokers`, `--shards`,
`--replicas`, `--fanout`, `--workers`, and `--lease-ms` configure a deployment.
`observed.json` contains final operational counters, not benchmark conclusions.
The generated state/logs are retained under the printed `Targets/Runs/...` path;
secrets stay in child-process environments and are not written to the logs.
Each demo run provisions a fresh deployment rather than silently reopening one
with different keys/membership.

For managed deployments use `Configs/BrokerCluster.yaml` and
`Configs/ClientFailover.yaml` as templates. Each broker needs its own identifier,
listening ports, private key, certificate and exclusive state directory. Every
member needs the same canonical membership, epoch, shard/replica/fanout parameters,
peer certificate fingerprints, epoch MAC key and authorization policy. A pin is
SHA-256 over the leaf certificate's DER encoding. The schema/membership marker
prevents an unrelated deployment from reusing a state directory.

Client-facing TLS/QUIC policy is independent of the backplane. Production-facing
listeners should use the matching TLS/QUIC examples and provisioned authentication
keys; a secure backplane does not secure an otherwise plaintext external listener.

## Data path

1. Authenticated connection establishment either creates a session or acquires
   its existing lease. Resumption restores state without changing the transport
   or negotiated QUIC policy.
2. Canonical profile assertions map to deterministic shards. Versioned
   replace-per-session snapshots, including empty replacements, are replicated.
   Wildcard/shared-group resolution uses the same membership view at all peers.
3. Publications enter a deterministic forwarding tree. MACs bind epoch,
   sender/receiver/hop context; invalid parents, epochs, duplicates and tenant
   routes are rejected. Peer links are established on demand.
4. Reliable recipient work is committed to replicated inboxes before hop success.
   Owners consume it under their lease/fence. Best-effort notifications reduce
   polling latency but are not required for eventual inbox discovery.
5. Consumption records, retained queues, pending sends, aliases, receive windows,
   request state and responder routes participate in snapshot/replay recovery.

The fabric is a deployment binding for trusted, mutually authenticated brokers.
It does not promise Byzantine agreement against malicious members. Topic payload
encryption can keep application plaintext from brokers, but brokers still see
and enforce routing metadata.

## Lease and storage rules

Replica placement uses deterministic rendezvous ordering. A successful operation
requires a majority of the configured replica set. For three replicas, one replica
failure can be tolerated; a minority partition cannot acknowledge reliable work
or make itself an owner. A one-replica demo has no replica-failure tolerance.

Voters persist monotonic incarnation/fence information and use a grant interval
`L`; a holder serves for at most a conservative `L/3`. A restarted voter waits `L`
before voting, because a monotonic deadline from its previous process cannot be
reconstructed safely. Windows `GetTickCount64` and Linux `CLOCK_BOOTTIME` advance
across suspend. Correctness assumes bounded clock-rate error, intact durable
state, a stable authenticated membership, and functioning local flush semantics.

An acquiring owner adopts a majority-certified snapshot and commits a newer fence
before serving. Delayed old-owner work cannot override that fence. Durable inboxes
and tombstones survive a lost notification and permit replay after takeover.
RPC timeouts, record/frame/byte budgets, and per-session in-flight horizons prevent
unbounded work accumulation. Under partitions or exhausted budgets, rejection or
temporary unavailability is intentional; the system does not silently weaken its
quorum rule.

State directories are local, exclusively owned and checksummed. Do not share one
directory between live processes, copy a live store, roll it back, or erase it
while retaining the same live membership identity. Schema 6 uses ESS6 session and
EBR6 broker snapshots. Old schemas and membership/epoch mismatches fail closed.
Online membership migration, joint consensus, distributed filesystem durability,
and external key management require separate designs and are not hidden fallbacks.

## Validation scope

`Cluster` tests deterministic placement, trees, MACs, profile convergence, fences
and primitive storage behavior. `Recovery` exercises real TLS peers, durable
admission/inbox replay, retained/request state, corrupt journals, restart,
minorities and owner takeover. `FabricQuic` exercises the corresponding native
QUIC fabric. `Correlation` also checks broker restart and restored responder maps.
The dashboard demo verifies a live three-node system with actual traffic.

These local tests support the implemented behaviors; they do not qualify WAN
partitions, all crash timings, physical power-loss durability, clock failures,
large-scale performance, or production availability percentages.
