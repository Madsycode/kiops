# EMPY/MMQP implementation plan

Reference: `README.pdf`, 51 physical pages (Communication pp. 3–44 and
Implementation pp. 47–53). The original `README.md` contains only a documentation
placeholder. This plan was established after reading all PDF pages and project
sources, inspecting the packet tables and architecture diagrams visually, and
building/testing the unmodified skeleton.

## Architecture and existing inventory

| Location | Existing responsibility and implementation destination |
| --- | --- |
| Root CMakeLists.txt | C++20, Debug/Release, five targets, Conan dependencies, optional MsQuic, runtime staging |
| ConanLists.txt | Standalone Asio 1.38.0 and OpenSSL 3.5.2; retain these dependencies |
| EmpyWin.bat / EmpyLinux.sh | Dependency installation, configuration, build and CTest entry points |
| cmake/EmpyStage.cmake | Per-executable runtime dependencies and manifests |
| EMPY_LIB_CORE/include/eCore/API.h and src/API.cpp | Core public entry point and shared-library identity |
| Common/Globals.h | Platform, exports, assertions and type identity |
| Common/Buffers.h | Shared immutable payloads, small buffers, gathered frames and checked binary readers |
| Common/Crypto.h and src/Common/Crypto.cpp | Hash, MAC, HKDF and secure randomness; OpenSSL remains private |
| Common/Stdout.h and src/Common/Stdout.cpp | Shared logging state and build-specific logging |
| Common/YAML.h | Typed, range-checked configuration |
| Runtime/Registries.h and src/Runtime/Registries.cpp | Tasks, dispatch, timers, workers, ECS and typed events |
| Runtime/Service.h, Context.h, Module.h, Runtime.h | Owner-thread application lifecycle, registries, module ownership |
| Network/MMQP.h | Existing fixed-header parser; complete protocol constants and semantics here |
| Network/Codecs.h | Designated typed packet codec; complete it and add its implementation |
| Vendors/EnTT.h, Rang.h, Yaml.h | Existing third-party ECS, terminal colours and YAML; retain upstream code |
| EMPY_LIB_MQ/include/eMQ/Client.h and src/Client.cpp | Client-facing entry point and identity |
| eMQ/Runtime/Configs.h, Service.h, Context.h, Runtime.h | Client configuration, protocol service, application lifecycle and Client runtime |
| eMQ/Security/Crypto.h | Client-only Ascon payload encryption and grant handling |
| EMPY_EXE_BROKER/include/Runtime/{Service,Context,Runtime}.h | Broker implementation, lifecycle and Broker runtime; stays outside both shared libraries |
| EMPY_EXE_BROKER/include/Modules/ClusterNode.h | Cluster lifecycle and backplane integration |
| EMPY_EXE_BROKER/include/Modules/Monitoring.h | Optional loopback telemetry lifecycle |
| EMPY_EXE_BROKER/src/Serve.cpp | Broker command-line application |
| EMPY_EXE_CLIENT/src/Client.cpp | Client command-line application and usage example |
| EMPY_EXE_TESTS/src/{Checks,Main,TestCommon,TestRuntime} | Existing deterministic test harness and core tests |
| EMPY_EXE_TESTS/src/Support/Process.h | Isolated executable and runtime-packaging tests |
| EMPY_DASHBOARD | Existing empty application directory; implement read-only collector/UI here |
| Targets | Generated dependency/build files and validation artifacts; never a source dependency |

Dependency direction remains `EmpyCore -> EmpyMQ -> EmpyClient`, with
`EmpyBroker -> EmpyCore`, and `EmpyTests` linking both libraries. Protocol,
transport and reusable file storage belong in Core; payload secrets and Ascon
belong only in MQ; routing, sessions, clustering and monitoring belong in the
broker executable. No additional shared library or build configuration is needed.

## Ordered implementation and validation

1. **Core correctness and wire layer.** Fix observed registry/lifecycle boundary
   defects; complete all nine typed codecs, canonical UTF-8 topics, exact interval
   encoding, bounded incremental framing, aliases, receive windows and pending
   queues. Validate golden wire bytes, every truncation, reserved flags, limits,
   atomic profile updates, PID wrap and reordered/conflicting retries.
2. **Transport and client security.** Implement asynchronous TCP/TLS and optional
   native MsQuic, hidden behind Core interfaces; complete stream policy 1/2/3,
   datagrams, bounded sends and shutdown ownership. Add the pinned upstream
   Ascon-AEAD128 reference, hierarchical grants, epoch bounds, persistent nonce
   reservations and replay windows. Validate upstream known answers, tampering,
   replay, storage corruption, certificate validation and real loopback transport.
3. **Client and standalone broker.** Complete the designated Services and
   Runtimes: authentication, resumption, keep-alive, wills, profiles, directional
   aliases, shared subscriptions, retained replay, bounded QoS1 and correlated
   request/response. Use shared payload ownership and a subscription trie. Test
   library/executable interactions, failure responses and cleanup.
4. **Durability and fabric.** Implement fixed manifests, deterministic sharding,
   rendezvous placement and trees; versioned profiles; mutually authenticated
   backplane; majority leases/fences and snapshot adoption; replicated recipient
   inboxes and retained queues. Validate minority rejection, stale owners,
   persisted restart, lost notifications, tenant scope and client failover.
5. **Tooling and final verification.** Complete YAML examples, CLI, Monitoring
   module, bounded read-only dashboard, build wrappers and dependency staging.
   Build Debug and Release, execute all tests in each, and test TCP-only support.
   Record exact commands, results and remaining qualification limits.

## Reuse and evidence boundaries

An earlier local implementation exists at
`C:/Users/pouhela/Desktop/Projects/eMQ`. Its protocol, storage, transport and
recovery code may be adapted where it agrees with this PDF. Its different target
layout and old wire choices are not authoritative. In particular, the current
PDF requires semantic QUIC strategy `0x02` and reusable completed correlation
identifiers. Existing Core buffers and registries remain the foundation.

The initial Debug build and four CTest suites (Crypto, Buffers, Yaml, Runtime)
passed on 2026-10-19 with GCC 15.2.0/MinGW. This is a baseline, not evidence for
the missing protocol. Functional tests do not establish the manuscript's scale,
latency, energy or availability claims.

## Decisions to make explicit

- Main PUBREQ definition wins over Table 14's `PRT=0` illustration: MESSAGE=1,
  REQUEST=2, RESPONSE=3; zero is invalid.
- NIST SP 800-232 Ascon-AEAD128 wins over legacy ASCON-128 figure labels.
- MPS means complete encoded packet size, as the normative prose specifies.
- Reason-code values beyond the described success/failure ranges and defaults
  are implementation binding choices, documented in PROTOCOL_DECISIONS.md.
- Static cluster membership, trust boundaries and storage/clock assumptions are
  explicit; online membership migration and external KMS integration are outside
  the reference binding.

## Execution status

All five phases are complete. The implementation remains in the original targets;
`FEATURES.md` maps the reference to source and tests. `ARCHITECTURE.md` and
`PROTOCOL_DECISIONS.md` record ownership and binding choices. Final validation on
2026-10-20 passed 16/16 Debug suites, 16/16 Release suites and 14/14 Debug TCP/TLS-only
suites, plus the dashboard and packaging checks detailed in `VALIDATION.md`.
