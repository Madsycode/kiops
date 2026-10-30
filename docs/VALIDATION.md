# Validation record

Executed on **2026-10-20, Europe/Berlin**, in this `EmpyMQ` workspace. Results below
come from the current implementation, not the older local project's records.

## Environment and builds

- Windows x64, native MinGW/UCRT GCC **15.2.0**, CMake **4.4.4**, C++20.
- Conan packages: Asio **1.38.0**, OpenSSL **3.5.2**.
- Native QUIC: MsQuic OpenSSL runtime **2.6.1**, stable API v2 headers **2.4.8**.
  The checked-in installer was executed successfully; the final builds use
  `Targets/Dependencies/MsQuic`, not the previous checkout or a Schannel DLL.
- Dashboard checks: Python **3.12.14**, Node **24.19.0**, Playwright with installed
  Microsoft Edge in headless mode.

| Build | Build result | Final CTest result | Suite-run wall time |
| --- | --- | --- | --- |
| Debug + native QUIC | PASS | **16/16 suites** | 12.67 s |
| Release + native QUIC | PASS | **16/16 suites** | 11.68 s |
| Debug, `EMPY_WITH_QUIC=OFF` | PASS | **14/14 suites** | 12.30 s |

All final runs include the CLI unexpected-connection-loss regression. QUIC-only
suites are not registered in the TCP/TLS-only build; its 36 applicable test cases
passed. All three target builds succeeded, and their final compiler logs contain no
reported warnings/errors. Suite wall times are test-run observations, not latency
or throughput measurements. The full test executable declares 38 test cases;
several reference cases run many protocol/security/recovery scenarios internally.

Commands used for the final builds/runs:

```powershell
cmake --build Targets/Debug --parallel 3
ctest --test-dir Targets/Debug -C Debug --output-on-failure --parallel 4
cmake --build Targets/Release --parallel 2
ctest --test-dir Targets/Release -C Release --output-on-failure --parallel 4
cmake --build Targets/TcpOnly --parallel 2
ctest --test-dir Targets/TcpOnly -C Debug --output-on-failure --parallel 4
```

Configuration and toolchain details are in [BUILDING.md](BUILDING.md). Each build's
`build-final.log`, `ctest-final.log`, and `Testing/Temporary/LastTest.log` are under
`Targets/<build>`. Binary runtime manifests are beside the executables in `bin`.

## Functional coverage

| Suites | Exercised behavior |
| --- | --- |
| Crypto, Buffers, Yaml | Crypto primitives; shared lifetime; inline/vector growth including aliased appends; width/overflow/truncation checks; typed configuration and errors |
| Runtime | Dispatch/worker bounds and capacity-one wakeups; exception-safe listener removal; timers; worker self-stop rejection; polled-runtime release; failed module attachment cleanup |
| Protocol | Golden encodings and round trips; conditional/reserved fields; every truncation of reference frames; malformed lengths; topics/aliases; atomic profiles; ordered/reordered receive windows; PID wrap and identifier limits |
| Security | All **1089** upstream Ascon-AEAD128 KATs; tampering, topic/epoch binding, hierarchical grants, replay/nonce persistence, cache limits and epoch policy |
| Integration | Real TCP/TLS broker/client flows; authentication/certificates; profiles, retention, wills, shared subscriptions, keep-alive/resume; more than 65535 QoS 1 publications across PID wrap; owner-thread Session/Publish events |
| Quic | Actual native QUIC streams/datagrams, all implemented allocation strategies, bulk fallback, large payload ownership, QoS/correlation/resumption, certificate/hostname/mutual-authentication failures and shutdown |
| Correlation | Concurrent requesters, same numeric IDs in different sessions, topic/responder authorization, QoS 0/1 and encrypted responses, completed-ID reuse, timeouts/no-responder, restart recovery and mixed responder outcomes |
| Adversarial | Malformed/out-of-state inputs, conflicting retries, unauthorized aliases/topics, bounded queues and connection/protocol failure paths |
| Cluster, Recovery, FabricQuic | Deterministic placement/trees/MACs/profiles; real TLS/QUIC peers; majority/minority behavior; fencing and owner adoption; durable inbox/retained/request replay; missed notifications; schema rejection and corrupt/torn journals |
| Transport | Count/byte admission boundaries; shutdown after callback errors or a full queue; gathered frames retaining one shared payload and canonical digests |
| Monitoring | Bounded native snapshots, session counters, authentication/origin/host checks, cache and shutdown behavior |
| Packaging | Copy executables and hash-verified DLL manifests to an isolated directory; restrict PATH; run actual broker/subscriber/publisher/requester; no-responder and unexpected disconnect must return failure |

## Dashboard and documentation checks

| Check | Executed result |
| --- | --- |
| Python monitoring/config/server tests | **28/28 PASS** |
| JavaScript topology/model tests | **9/9 PASS** |
| Debug real-process telemetry/collector/HTTP smoke | **PASS**, isolated runtime search path |
| Release real-process telemetry/collector/HTTP smoke | **PASS**, final packaged executables |
| Browser QA against a real three-broker TLS demo | **10/10 PASS**, no JavaScript/resource errors |
| Desktop and mobile screenshots | Visually inspected; readable layout with no horizontal page overflow |
| README C++ example | Compiled with GCC `-std=c++20 -fsyntax-only` against the public headers |
| C++ formatting | **85** non-vendor source/header files passed `clang-format --dry-run --Werror` |
| Documentation links | No broken relative Markdown links |

Browser checks include live topology selection, zoom, pause/resume, a refresh race,
inventory/session navigation and filtering, export, mobile layout, plus an
explicitly page-local unreachable/escaping fixture. The broker/client telemetry
for the normal views is real, not mocked. The three-broker demo completed with
three reporting brokers and three confirmed links; operational counters are not
used as a performance evaluation.

Evidence paths relative to the project:

- `Targets/Runs/DashboardSmoke/20261020-002238-b80080/result.json` — Debug process smoke.
- `Targets/Runs/DashboardSmoke/20261020-004752-d1d2a3/result.json` — Release process smoke.
- `Targets/Runs/DashboardQA/browser-result.json` and PNG screenshots — browser QA.
- `Targets/Runs/DashboardDemo/20261020-002247-6d2c44/` — generated live deployment,
  process logs and `observed.json`.
- `Targets/Validation/ReadmeClient.cpp` — extracted, compiled README example.
- `Targets/Validation/results.json` — machine-readable summary of this validation.

## Agentic-MLOps extension

Executed on **2026-10-21, Europe/Berlin**, with the same toolchain (GCC 15.2.0,
CMake 4.4.4, MsQuic 2.6.1 runtime) and Python 3.12 for the demo. The extension is
described in [MLOPS.md](MLOPS.md).

| Build | Build result | CTest result | Suite-run wall time |
| --- | --- | --- | --- |
| Debug + native QUIC, `EMPY_BUILD_MLOPS=ON` | PASS (up to date, exit 0) | **28/28 suites** (16 original + 12 `Ops*`) | 36.16 s |
| Release + native QUIC, `EMPY_BUILD_MLOPS=ON` | PASS (up to date, exit 0) | **28/28 suites** | 34.72 s |
| Debug, `EMPY_WITH_QUIC=OFF`, `EMPY_BUILD_MLOPS=ON` (existing `Targets/TcpOnly`, reconfigured) | PASS, 84 translation units compiled, no warnings | **26/26 suites** (14 + 12 `Ops*`) | 39.83 s |
| Debug, `EMPY_BUILD_MLOPS=OFF` (fresh `Targets/NoMLOps`) | PASS, no warnings | **16/16 suites** | 14.76 s |

- **Test cases.** `EmpyTests --list` declares 195 test cases: the 38 original
  cases and **157 `Ops*` cases**. Per suite: OpsAgents 19, OpsCommon 18,
  OpsContext 16, OpsDescriptor 11, OpsFabric 16, OpsHarness 29,
  OpsIntegration 4, OpsIntelligence 11, OpsSemantics 10, OpsSubstrate 7,
  OpsTopology 3, OpsWorkload 13. The listing is saved as
  `Targets/Debug/tests-list-mlops.txt`.
- **Longest Ops suites** in Debug: OpsCommon 23.5 s, OpsWorkload 20.5 s,
  OpsHarness 19.4 s, OpsIntegration 16.8 s. These suites start real broker,
  harness and workload processes, HTTP servers and child processes; their times
  are not performance measurements.
- **Isolation.** The `EMPY_BUILD_MLOPS=OFF` tree contains only `EmpyBroker`,
  `EmpyClient` and `EmpyTests` (no `EmpyOps.dll`, `EmpyHarness` or
  `EmpyWorkload`). `ctest -N` lists 16 suites, and `EmpyTests --list` declares 38
  cases. The tree was deleted after the check; its logs are kept in
  `Targets/Validation/nomlops-*.log`.

Commands:

```bash
export PATH=/c/Libraries/msys64/ucrt64/bin:$PATH
cmake --build Targets/Debug --parallel 12
ctest --test-dir Targets/Debug -C Debug --parallel 4 --output-on-failure    # Targets/Debug/ctest-final-mlops.log
cmake --build Targets/Release --parallel 12
ctest --test-dir Targets/Release -C Release --parallel 4 --output-on-failure    # Targets/Release/ctest-final-mlops.log
cmake -S . -B Targets/TcpOnly && cmake --build Targets/TcpOnly --parallel 12
ctest --test-dir Targets/TcpOnly -C Debug --parallel 4 --output-on-failure    # Targets/TcpOnly/ctest-mlops.log
cmake -S . -B Targets/NoMLOps -G "MinGW Makefiles" -DCMAKE_BUILD_TYPE=Debug -DEMPY_BUILD_MLOPS=OFF \
    -DCMAKE_TOOLCHAIN_FILE=C:/Users/pouhela/Desktop/EmpyMQ/Targets/Conan/conan_toolchain.cmake \
    -DEMPY_MSQUIC_ROOT=C:/Users/pouhela/Desktop/EmpyMQ/Targets/Dependencies/MsQuic
cmake --build Targets/NoMLOps --parallel 12
ctest --test-dir Targets/NoMLOps -C Debug --parallel 4 --output-on-failure
```

The Debug and Release trees were up to date, so these builds recompiled nothing.
The TCP-only and `EMPY_BUILD_MLOPS=OFF` builds compiled from scratch without
warnings.

### MLOps coverage

| Suites | Exercised behavior |
| --- | --- |
| OpsCommon | JSON grammar and limits, topic-filter algebra, child processes (pipes, stdin, kill on timeout), HTTP/HTTPS client and loopback server, TLS verification failures |
| OpsDescriptor, OpsSemantics, OpsTopology | Listing 2.1 parse and emit round trip, quantities, validator matrix, unknown-key warnings, profiles, cron; ontology matching of the Listing 2.1 sentence, extension rules, persistence; node capacity and network model |
| OpsIntelligence | Anthropic request contract (headers, body, `output_config`, fallbacks), structured output, refusal/truncation, 429 retries with `retry-after`, transport failure, missing key, all against a local mock HTTP server; YAML provider sets; schema conformance; command provider through real child processes |
| OpsContext | Knowledge CRUD, bounds, eviction and journaled reload, query ops and relevance, seeding the shipped knowledge, deterministic and fake-LLM translation, Fig. 26 retrieval, consolidation under concurrency, MCP server, a real `EmpyHarness mcp-server` subprocess client, external-source merge |
| OpsWorkload, OpsSubstrate | Linear fit, dataset split, window statistics, execution boundary; inference/training runtimes against an in-process broker (train and serve, drift, latency drift, rollback, simulator); `EmpyWorkload` child processes; simulated, Kubernetes (rendered manifests, fake apply/delete command) and process substrates |
| OpsAgents | Executive Listing 2.1 equivalence with `HomeControl.yaml`, other domains, phrases, advisor limits, operator context; placement filters, scores, replicas, relocation; lifecycle decision table; compliance criteria matrix, conflicts and question writer |
| OpsFabric | Envelopes, wildcard listen, serve/call, self-requests, gather with responder errors, timeouts, NO_RESPONDER, malformed requests, retained late subscribers, module detach |
| OpsHarness | Guard matrix (amendments, latency re-derivation, claimed accuracy), persistence and resume, scheduler, deploy from intent and descriptor, HITL approve/reject/timeout, retrain/promote/reject/rollback/retire, drift, audit relocation, lost-instance redeploy, substrate exit, exhaustion and accelerator events, escalated retirement after violations, redeploy readiness, node broker and security requirements, ontology state persistence, status endpoint |
| OpsIntegration | In-process intent → deploy → bootstrap training → promotion → induced drift → retrain → promotion; temporal trigger reaching the Lifecycle agent; conflict governance with HITL; real `EmpyBroker` + `EmpyHarness run` (process substrate) + `EmpyWorkload simulate` driven by `EmpyHarness intent` |

### Demo and command checks

| Check | Executed result |
| --- | --- |
| `python tools/MLOpsDemo.py --keep-state` (Debug, drift after 30 s) | **PASS**, 8/8 checks in 69 s. Deploy from the Listing 2.1 intent (compliance approved, inference `ran-1`, training `cloud-1`). Bootstrap v1 promoted after one NO_RESPONDER retry. 8 accuracy drift events. v2 promoted by a `metrics`-triggered retrain (observed accuracy below 0.9). 1904 actions in [18.50, 22.49] from models 1 and 2. Healthz 200. Retire succeeded. |
| Same demo, earlier runs (Debug, defaults) | PASS twice |
| `python tools/MLOpsDemo.py --bin Targets/Release/bin --drift-after-ms 0 --seconds 40` | **PASS**, 6/6 checks (no drift checks) |
| `EmpyHarness interpret` (Listing 2.1), `validate EnergySaver.yaml`, `render HomeControl.yaml`, `query "which edge nodes have a gpu"`, `mcp-server` (initialize, tools/list, tools/call) | Exit 0. The BoM matches `HomeControl.yaml`; the query returns `edge-1`, `edge-2`; render writes two manifests with the `--host`/`--image` overrides. |
| Anthropic configuration without a key | `interpret` stays deterministic (`source: deterministic`). An `http://` non-loopback `base_url` is refused with exit 1. |
| Kubernetes substrate configuration (`kubectl` apply/delete commands) | Accepted by `EmpyHarness run`, which then stopped at the expected refused broker connection; no cluster was contacted |
| Relative Markdown links in the touched documents | No broken links or anchors |

Evidence paths:

- `Targets/Runs/MLOpsDemo/20261021-152929-8d4e7c/`: final demo run with
  `summary.json`, `intent.json`, snapshots, process and workload logs, and the
  kept `state/`.
- `Targets/Runs/MLOpsDemo/20261021-151538-b75563/` and `.../20261021-151404-dfef63/`:
  earlier Debug runs.
- `Targets/Runs/MLOpsDemo/20261021-151653-e597ee/`: the Release run without
  drift.

### MLOps qualification limits

- No real Kubernetes cluster was used: manifests were rendered and inspected,
  and apply/delete ran against a fake command. No container image exists.
- No live Anthropic API call and no real local model were made; providers were
  tested against a local mock HTTP server and shell utilities.
- The POSIX branches (process control, per-user directory ownership checks,
  fsync) were not compiled.
- The demo's Ctrl+C path shares the normal cleanup code but was not exercised
  interactively.
- One flake was observed once before this validation: `OpsWorkload.ExecutableRoles`
  missed the final `stopped` heartbeats. It is mitigated by a 1 s
  acknowledgement wait; its root cause is unproven. It did not recur in these
  runs.
- Workloads are reference models (linear regression, sine-wave simulator); all
  nodes of the process substrate run on one machine. Relocation does not
  transfer training datasets.
- No throughput, latency or scalability result is claimed for the extension.

## Defects addressed during validation/review

Observed failures were corrected and relevant tests rerun. These included the
Windows PEM/Schannel provider mismatch, known high incoming aliases in profile
updates, reusable correlation identifiers and persisted routes, one responder's
error incorrectly completing a multi-responder exchange, receipt retention with
a smaller configured send limit, queue-saturated shutdown, listener/worker/module
cleanup boundaries, aliased buffer growth, complete-final-record corruption being
treated as a torn journal, and CLI failure exit statuses. Schema/version assertions
and successful informational-reason assertions were corrected to match the binding.

## Qualification limits

- Native Linux and MSVC builds were not run in this Windows/MinGW validation.
  The Linux path and suspend-aware clock binding exist, but are not reported as
  tested. Other OS lease clocks fail explicitly.
- ASan, TSan, Valgrind, a broad fuzzing campaign, physical power-loss injection,
  and an external interoperability implementation were not run. Deterministic
  ownership/concurrency/bounds tests are not proof of the absence of all races or
  memory-safety defects.
- No scientific throughput/latency/energy/scalability, long-duration soak, WAN
  fault matrix, Byzantine-fault, or availability-percentage result is claimed.
  The broker serial actor and synchronous durable/quorum boundaries are explicit
  trade-offs, described in ARCHITECTURE.md.
- External key distribution, online membership/schema migration and custom QUIC
  allocation policy are outside the supplied runnable binding. Cluster trust,
  clock-rate and durable-storage assumptions are stated in CLUSTERING.md.

No known build failure or observed correctness-test failure remains in the final
validation matrices recorded above (original framework and MLOps extension).
