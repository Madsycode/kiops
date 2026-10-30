# EMPY / MMQP

C++20 implementation of the communication framework in [README.pdf](README.pdf).
The existing five-target architecture is preserved: `EmpyCore`, `EmpyMQ`,
`EmpyBroker`, `EmpyClient`, and `EmpyTests`. Three optional targets add the
[Agentic-MLOps extension](#agentic-mlops-extension): `EmpyOps`, `EmpyHarness`, and
`EmpyWorkload` (`EMPY_BUILD_MLOPS`, default ON). This README describes the runnable
implementation; [protocol decisions](docs/PROTOCOL_DECISIONS.md) records the
reference ambiguities and implementation bindings.

## Implemented framework

- All nine MMQP packets; checked framing, canonical UTF-8 topics, wildcard and
  shared subscriptions, atomic profiles, and directional aliases.
- Session authentication, expiry/resumption, birth/last wills, keep-alive,
  QoS 0/1, bounded retries, PID wrap, retention, and correlated request/response.
- Asynchronous TCP, TLS 1.3, and optional native MsQuic: negotiated single,
  semantic, and bulk streams, plus bounded QoS 0 datagrams.
- Client-only Ascon-AEAD128 payload encryption, hierarchical HKDF keys, epoch
  policy, durable nonce reservation, and replay protection.
- A live static broker fabric: deterministic placement/routing, replicated
  profiles and inboxes, majority leases/fencing, snapshot adoption, retention,
  crash recovery, tenant constraints, and client failover.
- Owner-thread application events/modules, bounded dispatch and workers,
  shared payload buffers, YAML configuration, command-line applications, and
  a read-only monitoring dashboard.

See the [feature-to-source map](docs/FEATURES.md),
[ownership and architecture](docs/ARCHITECTURE.md), and
[validation record](docs/VALIDATION.md). Test results are functional evidence;
they do not establish thesis-scale throughput, latency, energy, or availability.

## Build

Only **Debug** and **Release** are supported. Requirements: a C++20 compiler
with `std::format`, CMake 3.21+, Conan 2, and a configured Conan compiler profile.
Conan supplies standalone Asio 1.38.0 and OpenSSL 3.5.2. Ascon is pinned source
inside `EMPY_LIB_MQ/src/Vendors/Ascon`.

On Windows, put the native MinGW `bin` directory on PATH, then run from PowerShell:

```powershell
# Optional QUIC SDK, downloaded into Targets/Dependencies with verified hashes.
./tools/Install-MsQuic.ps1
./EmpyWin.bat Debug
./EmpyWin.bat Release
```

The wrappers install dependencies, configure, compile, and run CTest. Binaries
and their DLLs are under `Targets/<configuration>/bin`. Keep those DLLs beside
the executables. The generated `*.exe.runtime.txt` files record their hashes.
See [build details](docs/BUILDING.md) for Linux, existing CMake caches, an explicit
MsQuic provider, and a TCP/TLS-only build.

## Run locally

Start the broker in one terminal:

```powershell
./Targets/Debug/bin/EmpyBroker.exe --config Configs/Broker.yaml --monitor
```

Start a subscriber in another terminal, then publish from a third:

```powershell
./Targets/Debug/bin/EmpyClient.exe subscribe 'sensors/#' --verbose
./Targets/Debug/bin/EmpyClient.exe publish sensors/room/temperature '21.5' --qos 1
```

Request/response uses an independent correlation ID. Start the responder first:

```powershell
./Targets/Debug/bin/EmpyClient.exe respond control/status ready --verbose
./Targets/Debug/bin/EmpyClient.exe request control/status status
```

`--host`, `--port`, `--config`, `--seconds`, `--count`, and `--interval-ms`
configure the client. Run either executable with `--help`. A failed request is
reported as a correlated response and a nonzero CLI exit status. The default
unsecured broker binds only to loopback. Certificate and key paths in YAML are
relative to the YAML file, not the process working directory.

The configuration examples cover local TCP, TLS, QUIC, durable standalone state,
tenants, failover, clustering, and telemetry. Examples with certificate paths,
environment variables, or fingerprint placeholders require provisioning;
[the cluster demo](docs/CLUSTERING.md) generates isolated development credentials
and working configurations automatically.

## Client API

Include `<eMQ/Client.h>` and link the `EmpyMQ` CMake target. Application callbacks
execute when the owning thread calls `Poll`, `Start`, or `StartLoop`.

```cpp
#include <eMQ/Client.h>
#include <chrono>
#include <iostream>

int main()
{
    eMQ::Client client;
    client.On<eMQ::Receive>("sensors/#", [](const eMQ::Receive &event)
    {
        std::cout << event.m_Message.m_Topic << ' '
                  << eCore::AsText(event.m_Message.m_Payload.Span()) << '\n';
    });
    client.Connect("127.0.0.1", 5600);
    const auto reason = client.Subscribe("sensors/#", 1).get();
    if (reason)
    {
        throw std::runtime_error(reason.m_Text);
    }
    const auto deadline = eCore::Clock::now() + std::chrono::seconds(30);
    while (eCore::Clock::now() < deadline)
    {
        client.Poll(50);
    }
    client.Disconnect();
    client.Shutdown();
}
```

`Register`, `Subscribe`, `Unsubscribe`, `Publish`, `Request`, and `Respond`
return `std::future<Reason>`. A reason is an error when its code is at least
`0x80`; successful informational reasons need not be zero. QoS 1 futures report
the broker's PUBACK; QoS 0 futures report local transport admission. Application
responses arrive as `Receive` events. A future is never a claim of downstream
application completion.

Payloads use `eCore::SharedBytes`. Copies share one immutable allocation; move
ownership into a publication where possible. `Request` can take an explicit
nonzero correlation ID so concurrent callers can identify responses. Its default
zero parameter allocates an unused ID. Completed finite exchanges release the
ID; all-responder requests are bounded by the configured response deadline.

## Dashboard and cluster demo

The dashboard uses Python 3.10+ and its standard library. With `openssl` on PATH:

```powershell
python EMPY_DASHBOARD/demo.py
```

This starts three real brokers, publish/subscribe and request/response clients,
authenticated telemetry, and a dashboard at `http://127.0.0.1:8080`. It preserves
logs, generated certificates, and state under `Targets/Runs/DashboardDemo` and
stops its child processes on exit. Pass `--quic` to use a QUIC backplane, or
`--openssl PATH`, `--bin PATH`, `--port PORT`, and `--seconds N` as needed.

To monitor an existing deployment, configure `Configs/Dashboard.json` and run:

```powershell
python EMPY_DASHBOARD/app.py --config Configs/Dashboard.json
```

[Dashboard operation](EMPY_DASHBOARD/README.md) describes authentication, cached
telemetry, stale/unreachable states, and optional browser QA. The dashboard
exposes observations, not payloads, topic names, credentials, or write controls.

## Agentic-MLOps extension

The extension implements the thesis section 2.1 MLOps Framework on MMQP. ML
workflows are topic-profile communication entities: deployment, lifecycle,
retraining triggers and governance run as publish/subscribe and correlated
request/response under one topic root (`mlops`). It adds:

- RichML-Service descriptors (Listing 2.1), whose observables and bounded
  actionables are enforced as the workload's execution boundary;
- a deterministic harness with temporal, event-driven and request/response
  activation, a Guard that re-validates every agent output before enactment, and
  human-in-the-loop approval over MMQP;
- Executive (intent → BoM-Ontology → BoM YAML), Placement, Lifecycle and
  Compliance agents, each with a deterministic core and optional, constrained
  LLM advice;
- CoMaS knowledge management (Fig. 26 retrieval, consolidation, MCP server and
  client);
- intelligence providers (deterministic by default, the Anthropic Messages API,
  or a local command);
- simulated, local-process and Kubernetes (rendered manifests and a CRD)
  execution substrates.

| Target | Role |
| --- | --- |
| `EmpyOps` | Shared library with the extension (links `EmpyMQ`) |
| `EmpyHarness` | Harness runtime and agent swarm, operator commands, offline tools, MCP server |
| `EmpyWorkload` | Reference inference, training and simulator workloads |

Build as above (`EMPY_BUILD_MLOPS=OFF` omits all three), then run the closed-loop
demo or interpret Listing 2.1 offline:

```powershell
python tools/MLOpsDemo.py
./Targets/Debug/bin/EmpyHarness.exe interpret "Deploy a home temperature controller that keeps the temperature between 16 and 24 degrees using temperature and humidity sensors, with a p95 latency under 5 ms, retrained daily on a GPU"
```

The demo starts a broker, the harness with the local-process substrate, a
simulated gateway with induced concept drift, and the workloads it deploys. It
submits the intent and prints a timeline: deployment and placement, bootstrap
training, promotion, drift and retraining. Evidence is kept under
`Targets/Runs/MLOpsDemo`. `interpret` prints the generated BoM, which
matches `Configs/MLOps/HomeControl.yaml` field by field.

LLM use is opt-in; the shipped configuration is fully deterministic. The
[MLOps reference](docs/MLOPS.md) covers the descriptor, topic and message
contract, workflows, agents, CoMaS, providers, substrates, every configuration
key, the CLI, Kubernetes deployment, and the qualification limits.

## Boundaries

Cluster membership/epoch changes require an explicit coordinated deployment and
state migration. Schema 6 broker state is not automatically migrated from old
local implementations. QUIC allocation value `0x00` remains reserved for an
application-defined binding; 0-RTT application data is rejected. External key
distribution, general rolling membership changes, and scientific scale/fault
qualification are outside the implemented reference binding. Platform validation
and any remaining limits are recorded in [VALIDATION.md](docs/VALIDATION.md); the
MLOps extension's limits (no real Kubernetes cluster or live LLM call in validation)
are listed in [MLOPS.md](docs/MLOPS.md#boundaries-and-qualification-limits).
