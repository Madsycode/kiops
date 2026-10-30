# Empy monitoring dashboard

A read-only Python 3.10+ standard-library collector and browser application. It
consumes authenticated native broker telemetry; it does not infer broker state
from process names or synthesize network metrics.

## Start

```powershell
# Existing monitored broker(s):
python EMPY_DASHBOARD/app.py --config Configs/Dashboard.json --port 8080
# A complete three-broker deployment with real clients and temporary credentials:
python EMPY_DASHBOARD/demo.py --openssl C:/Libraries/msys64/ucrt64/bin/openssl.exe
```

The demo defaults to `Targets/Debug/bin`, loopback port 8080 and a 900-second
lifetime. Use `--bin`, `--port`, `--seconds`, or `--quic` to override them. The
cluster launcher also supports `--traffic idle` and `--no-dashboard`. Logs,
configuration, state, generated development certificates and final observations
are retained under `Targets/Runs`; child processes stop when the launcher exits.

## Broker telemetry and trust

Enable the broker `telemetry` block (see `Configs/BrokerDashboard.yaml`). Local
development may omit a token. When `token_env` is configured, a
32..256-character printable token comes from that environment variable.
The dashboard's `tokenEnv` must select that token for the corresponding source.
Credentials remain in the collector and never enter the browser's JSON state.
The native endpoint is `GET /api/v1/snapshot` and is loopback-only. A remote source
must be reached through HTTPS with a valid CA, or a local authenticated tunnel.

The dashboard HTTP server also binds only to loopback. Host/Origin checks resist
DNS rebinding and cross-origin reads; assets are allowlisted, writes are rejected,
and requests/workers/snapshot sizes/history are bounded. It does not supply remote
user authentication: place an authenticated TLS proxy in front if remote access
is required. The UI does not expose message payloads, topic names, secret keys,
tokens, private keys, or broker control actions.

## Observations

- Overview: reporting brokers, online client counts, byte/message rates, pending
  deliveries, discovered peers, confirmed links and recent activity.
- Broker inventory and detail: transport, identity, limits, uptime, counters and
  cluster placement/lease observations.
- Session pages: search/filter/sort, pagination, request/publish/subscribe roles,
  current/retained session state and byte/packet counters.
- Pause/resume and JSON export preserve the selected observation without changing
  brokers. Stale/unreachable sources retain their last data but contribute no
  live rates; restarts and counter regression reset rate baselines.

Backplane links are created on demand. A declared peer is not a confirmed live
edge until both fresh endpoints report it. The cluster identity includes the
manifest, so identical broker names from different deployments do not merge.
At most 256 configured sources are accepted; that limit is not a demonstrated
capacity or performance claim. Session inventory stays in the collector and is
served in bounded pages rather than being copied into every UI refresh.

## Checks

```powershell
python -m unittest discover -s EMPY_DASHBOARD/tests -p 'test_*.py' -v
node --test EMPY_DASHBOARD/tests/model.test.js
python EMPY_DASHBOARD/tests/process_smoke.py --bin Targets/Debug/bin
```

For optional browser QA, start the live demo and run `node
EMPY_DASHBOARD/tests/browser.mjs`. Playwright must be installed in the development
environment; `EMPY_PLAYWRIGHT_MODULE` can point to its `index.mjs` file URL and
`EMPY_BROWSER_EXECUTABLE` to an installed Chromium-family browser. Set
`EMPY_DASHBOARD_URL` for a different port and `EMPY_QA_OUTPUT` for output artifacts.
The suite writes screenshots and `browser-result.json`; its unavailable-source
case is explicitly a page-local fixture, while the other checks use live telemetry.

Actual executed results and remaining qualification limits are in
[docs/VALIDATION.md](../docs/VALIDATION.md).
