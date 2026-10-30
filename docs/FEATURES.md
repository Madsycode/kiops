# Reference-to-implementation map

Page ranges below refer to physical pages of the supplied PDF; its printed page
numbers include manuscript offsets. The PDF was extracted and read in full before
implementation; packet tables and architecture figures were also inspected visually.
The initial inventory and dependency order remain in `IMPLEMENTATION_PLAN.md`.

| Reference area | Implementation | Main validation |
| --- | --- | --- |
| Framing and nine packet types, pp. 1-16 | Core `Network/MMQP.h`, `Stream`, `Codecs` | `Protocol`, `Adversarial` |
| TIE, topics, wildcard/shared filters | Core `Network/Topic`, trie | `Protocol`, `Integration`, `Cluster` |
| Profiles and directional aliases | Core `Network/State`; client/broker Services | `Protocol`, `Adversarial`, `Integration` |
| Authentication, sessions, keep-alive, wills | Core `Authentication`; client/broker Services | `Integration`, `Adversarial`, `Recovery` |
| QoS 0/1, receive windows, retries, PID wrap | Core `State`; Services; shared `WireFrame` | `Protocol`, `Integration`, `Transport` |
| Request/response, reusable IDs, routing and expiry | `CorrelationIds`, client exchanges, broker routes and durable snapshots | `Correlation`, `Recovery`, `Quic` |
| Retention and replay | Broker Service and cluster retained/inbox stores | `Integration`, `Recovery` |
| Payload encryption and key hierarchy, pp. 17-22 | MQ `Security/Crypto`; pinned Ascon source | `Security` (1089 upstream KATs plus policy/storage cases), `Correlation` |
| TCP/TLS and QUIC binding, pp. 23-27 | Core `IOContext`, private `QContext`, `QuicPolicy` | `Transport`, `Integration`, `Quic`, `FabricQuic` |
| Placement, forwarding, membership and profiles, pp. 28-42 | Broker `Cluster/Fabric`, `Replication`, `Node` | `Cluster`, `Recovery`, `FabricQuic` |
| Quorum leases, fences, adoption, durable recipient state | Broker `Cluster/Clock`, `Storage`, `Replication`, `Node`; Core `Storage` | `Recovery`, `FabricQuic` |
| Tenant/root/shard boundaries | Broker Configs and Service authorization, fabric checks | `Adversarial`, `Recovery` |
| Generic runtime/registries/modules, pp. 45-51 | Existing Core Runtime headers and `Registries.cpp` | `Runtime`, `Integration` |
| Application-facing Client and Broker runtimes | MQ Runtime and broker Runtime; `ClusterNode`, `Monitoring` modules | `Integration`, `Packaging`, `Monitoring` |
| Observability | Broker Monitoring; `EMPY_DASHBOARD` collector/server/web UI | `Monitoring`, Python/JS tests, process smoke and browser QA |
| Debug/Release, dependency boundaries and runnable executables | Existing CMake targets, wrappers, staging, CLI sources | Each configuration's CTest suites, `Packaging` |
| MLOps 2.1.1 RichML-Service BoM (Listing 2.1), observable/actionable profiles, execution boundary (Fig. 24), pp. 48-50 | Ops `Descriptor`, `Workload/Boundary`; `Configs/MLOps/HomeControl.yaml` | `OpsDescriptor`, `OpsWorkload` |
| MLOps 2.1.2 harness: execution state, temporal/event-driven/request-response activation, Guard, pp. 50-51 | Ops `Harness` (`ExecutionHarness`, `Guard`, `Scheduler`, `StateStore`), `Fabric` (`Bus`, `Topics`) | `OpsHarness`, `OpsFabric` |
| MLOps 2.1.3 agent swarm: Executive (Fig. 25 BoM-Ontology), Placement, Lifecycle, Compliance + HITL, pp. 51-54 | Ops `Agents`, `Semantics`, `Harness/AgentHost`; `Configs/MLOps/Ontology.yaml`, `Policies.yaml` | `OpsAgents`, `OpsSemantics`, `OpsHarness`, `OpsIntegration` |
| MLOps 2.1.4 CoMaS knowledge plane, Fig. 26 retrieval, MCP server/client, pp. 54-55 | Ops `Context`; `Configs/MLOps/Knowledge.yaml`; `EmpyHarness mcp-server` | `OpsContext` |
| MLOps Intelligence Service Providers (deterministic, Anthropic Messages API, local command) | Ops `Intelligence` | `OpsIntelligence` (local mock HTTP server, child processes) |
| MLOps heterogeneity and blueprint (Figs. 22-23, pp. 47-48): simulated, local-process and Kubernetes substrates + CRD | Ops `Substrate`; `EmpyWorkload`; `Configs/MLOps/Kubernetes` | `OpsSubstrate`, `OpsWorkload`, `OpsIntegration.ExecutablesEndToEnd`, `tools/MLOpsDemo.py` |

## Source organization

- `EMPY_LIB_CORE/include/eCore/Common`: globals/exports, shared buffers, cryptographic
  primitives, logging, YAML and durable storage interfaces. `src/Common` supplies
  the non-template implementations.
- `EMPY_LIB_CORE/include/eCore/Network`: typed protocol values, errors, framing,
  topics/trie, protocol state, events and transport interfaces. `src/Network`
  implements them; Asio/MsQuic/OpenSSL transport internals stay there.
- `EMPY_LIB_CORE/include/eCore/Runtime`: the preserved Service/Context/Module/Runtime
  and registries. Worker/dispatch internals live in `src/Runtime/Registries.cpp`.
- `EMPY_LIB_MQ/include/eMQ/Runtime` and `src/Runtime`: client configuration and
  state machine. `Security` owns payload secrets and crypto; it is not linked
  into the broker. `Client.h` remains the application entry point.
- `EMPY_EXE_BROKER/include/Runtime`: broker policy, sessions and routing;
  `Cluster`: live fabric and durable primitives; `Monitoring`: snapshots/HTTP;
  `Modules`: lifecycle integration. `.inl` definitions preserve the existing
  header implementation and use inline/named-detail helpers for ODR safety.
- `EMPY_EXE_BROKER/src/Serve.cpp` and `EMPY_EXE_CLIENT/src/Client.cpp`: real CLI
  applications using the same public runtimes as the tests.
- `EMPY_EXE_TESTS/src`: original harness plus protocol, transport, security,
  integration, correlation, recovery, adversarial, monitoring and packaging tests.
  `Support/Process.h` launches isolated executables; `data` holds pinned KAT data.
- `Configs`: typed YAML deployment examples and dashboard source configuration.
  `cmake/EmpyStage.cmake`: actual runtime dependency staging and manifests.
  `tools/Install-MsQuic.ps1`: reproducible optional Windows SDK provisioning.
- `EMPY_DASHBOARD`: Python collector/config/server and launcher; browser assets
  in `web`; deterministic, process and browser tests in `tests`.
- `EMPY_LIB_OPS` (`EmpyOps`), `EMPY_EXE_HARNESS` (`EmpyHarness`), `EMPY_EXE_WORKLOAD`
  (`EmpyWorkload`), `EMPY_EXE_TESTS/src/Ops` (`Ops*` suites), `Configs/MLOps` and
  `tools/MLOpsDemo.py`: the optional Agentic-MLOps extension (`EMPY_BUILD_MLOPS`).
  Its requirement-to-source map is in [MLOPS.md](MLOPS.md#thesis-mapping).
- `Targets`: generated builds, dependency cache and evidence only. No source
  target includes or reads the older external project checkout.

## Explicit boundaries

The supplied binding reserves custom QUIC strategy `00` and rejects 0-RTT
application data. External KMS distribution, online membership/schema migration,
and general Internet-scale operational qualification are not specified as runnable
services here. They are documented boundaries, not successful test claims.
Public application registry ownership versus the private broker actor registry,
finite/all-responder completion, stricter TLS minimum, and other implementation
choices are recorded in `ARCHITECTURE.md` and `PROTOCOL_DECISIONS.md`.
