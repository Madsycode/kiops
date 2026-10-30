"""Run a local MMQP fabric with real brokers, clients and a read-only dashboard.

Each run keeps isolated certificates, ports, configurations, state and logs under
--output. The static membership pins each certificate and uses mutual TLS or
native QUIC. Demo traffic exercises publish/subscribe and request/response.
Requires the built Empy executables, Python's standard library and OpenSSL.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
import hashlib
import json
import os
from pathlib import Path
import secrets
import socket
import subprocess
import sys
import threading
import time

from collector import Collector
from config import EMPY_MAX_SOURCES, Settings, Source
from server import DashboardServer

EMPY_NO_WINDOW = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
EMPY_TOKEN_ENV = "EMPY_CLUSTER_TELEMETRY_TOKEN"
EMPY_KEY_ENV = "EMPY_CLUSTER_EPOCH_KEY"


def stop_process(process):
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)


class Fabric:
    """One isolated deployment: certificates, configuration and broker processes."""

    def __init__(self, args, count, directory, cleanup):
        self.args, self.count, self.directory, self.cleanup = args, count, directory, cleanup
        self.binary = args.bin.resolve()
        suffix = ".exe" if os.name == "nt" else ""
        self.broker = self.binary / ("EmpyBroker" + suffix)
        self.client = self.binary / ("EmpyClient" + suffix)
        self.names = [f"broker-{index + 1:03}" for index in range(count)]
        self.run = secrets.token_hex(4)  # Unique topics per run: earlier QoS 1 sessions may still be retained.
        self.processes = {}
        self.records = []
        self.environment = os.environ.copy()
        self.environment[EMPY_KEY_ENV] = secrets.token_hex(32)
        self.environment[EMPY_TOKEN_ENV] = secrets.token_hex(32)
        if os.name == "nt":
            windows = Path(os.environ["SystemRoot"])
            self.environment["PATH"] = os.pathsep.join((str(windows / "System32"), str(windows)))

    def openssl(self, *arguments):
        return subprocess.run([self.args.openssl, *arguments], cwd=self.directory, check=True, stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE, creationflags=EMPY_NO_WINDOW).stdout

    def certificates(self):
        self.openssl("req", "-x509", "-newkey", "ec", "-pkeyopt", "ec_paramgen_curve:P-256", "-nodes", "-keyout", "ca.key",
                     "-out", "ca.pem", "-days", "2", "-subj", "/CN=Empy cluster test",
                     "-addext", "basicConstraints=critical,CA:TRUE", "-addext", "keyUsage=critical,keyCertSign,cRLSign")
        (self.directory / "extensions.cnf").write_text(
            "basicConstraints=critical,CA:FALSE\nkeyUsage=critical,digitalSignature\n"
            "extendedKeyUsage=serverAuth,clientAuth\nsubjectAltName=DNS:localhost,IP:127.0.0.1\n", encoding="utf-8")

        def leaf(index):
            # Every member presents a distinct certificate pinned by SHA-256 in all manifests.
            name = f"broker{index}"
            self.openssl("req", "-newkey", "ec", "-pkeyopt", "ec_paramgen_curve:P-256", "-nodes", "-keyout", f"{name}.key",
                         "-out", f"{name}.csr", "-subj", "/CN=localhost")
            self.openssl("x509", "-req", "-in", f"{name}.csr", "-CA", "ca.pem", "-CAkey", "ca.key", "-set_serial", str(index + 2),
                         "-days", "2", "-extfile", "extensions.cnf", "-out", f"{name}.pem")
            return hashlib.sha256(self.openssl("x509", "-in", f"{name}.pem", "-outform", "DER")).hexdigest()

        with ThreadPoolExecutor(max_workers=min(16, os.cpu_count() or 4)) as pool:
            self.fingerprints = list(pool.map(leaf, range(self.count)))

    def configure(self):
        streams = []
        for _ in range(self.count * 3):
            stream = socket.socket()
            stream.bind(("127.0.0.1", 0))
            streams.append(stream)
        ports = [stream.getsockname()[1] for stream in streams]
        self.clients, self.backplane, self.telemetry = ports[:self.count], ports[self.count:2 * self.count], ports[2 * self.count:]
        args = self.args
        peers = "".join(f"    - identifier: {name}\n      host: 127.0.0.1\n      port: {self.backplane[i]}\n"
                        f"      server_name: localhost\n      fingerprint: {self.fingerprints[i]}\n" for i, name in enumerate(self.names))
        replicas = min(args.replicas, self.count)
        for i, name in enumerate(self.names):
            (self.directory / f"broker{i}.yaml").write_text(
                f"address: 127.0.0.1\nport: {self.clients[i]}\nworkers: {args.workers}\nmax_sessions: 4096\n"
                f"request_timeout_ms: 15000\n"
                f"telemetry:\n  port: {self.telemetry[i]}\n  interval_ms: 1000\n  max_sessions: 256\n  token_env: {EMPY_TOKEN_ENV}\n"
                f"cluster:\n  identifier: {name}\n  port: {self.backplane[i]}\n  state_directory: state{i}\n"
                f"  epoch: 1\n  shards: {args.shards or max(8, 2 * self.count)}\n  replicas: {replicas}\n  fanout: {args.fanout}\n"
                f"  lease_ms: {args.lease_ms}\n  quic: {'true' if args.quic else 'false'}\n"
                f"  certificate: broker{i}.pem\n  private_key: broker{i}.key\n  authority: ca.pem\n"
                f"  epoch_key_env: {EMPY_KEY_ENV}\n  peers:\n" + peers, encoding="utf-8")
        failover = "".join(f"  - host: 127.0.0.1\n    port: {port}\n" for port in self.clients[:8])
        (self.directory / "client.yaml").write_text(
            "auto_reconnect: true\noperation_timeout_ms: 20000\nfailover_endpoints:\n" + failover, encoding="utf-8")
        for stream in streams:
            stream.close()

    def launch(self, name, executable, *arguments, stdout=None):
        log = self.cleanup.enter_context((self.directory / f"{name}.log").open("wb"))
        output = self.cleanup.enter_context(stdout.open("wb")) if stdout else log
        process = subprocess.Popen([str(executable), *map(str, arguments)], cwd=self.binary, env=self.environment,
                                   stdout=output, stderr=log if stdout else subprocess.STDOUT, creationflags=EMPY_NO_WINDOW)
        self.cleanup.callback(stop_process, process)
        self.processes[name] = process
        self.records.append({"name": name, "pid": process.pid})
        (self.directory / "processes.json").write_text(json.dumps(self.records, indent=4), encoding="utf-8")
        return process

    def ready(self, names, marker, timeout):
        pending = set(names)
        deadline = time.monotonic() + timeout
        while pending:
            for name in sorted(pending):
                text = (self.directory / f"{name}.log").read_text(errors="replace")
                if marker in text:
                    pending.discard(name)
                elif self.processes[name].poll() is not None:
                    raise RuntimeError(f"{name} exited: {text[-2000:]}")
            if pending and time.monotonic() > deadline:
                raise RuntimeError(f"{len(pending)} processes did not report '{marker}'; inspect {self.directory}")
            time.sleep(0.1)

    def start(self):
        started = time.monotonic()
        wave = max(1, self.args.start_concurrency)
        for first in range(0, self.count, wave):
            names = []
            for i in range(first, min(self.count, first + wave)):
                names.append(f"broker{i}")
                self.launch(names[-1], self.broker, "--config", self.directory / f"broker{i}.yaml", "--seconds", self.args.seconds)
            self.ready(names, "MMQP broker listening", 60)
        return time.monotonic() - started

    def sources(self):
        token = self.environment[EMPY_TOKEN_ENV]
        return tuple(Source(name, name.replace("broker-", "Broker "), f"http://127.0.0.1:{self.telemetry[i]}", token)
                     for i, name in enumerate(self.names))

    def brokers(self):
        return [self.processes[f"broker{i}"] for i in range(self.count)]


def mesh(state):
    """Reporting brokers and confirmed peer links from one collector state."""
    online = [source for source in state["sources"] if source["state"] == "online"]
    links = sum(peer["connected"] for source in online for peer in source["data"]["cluster"]["peers"])
    return len(online), links


def totals(state, keys=("publications", "forwarded", "retransmitted", "rejected")):
    online = [source for source in state["sources"] if source["state"] == "online"]
    return {key: sum(source["data"]["metrics"][key] for source in online) for key in keys}


def wait_reporting(fabric, collector, timeout):
    """Every broker serves telemetry. Backplane links are dialed lazily, so none are required yet."""
    started = time.monotonic()
    while time.monotonic() - started < timeout:
        if mesh(collector.state())[0] == fabric.count:
            return time.monotonic() - started
        time.sleep(0.25)
    raise RuntimeError(f"Only {mesh(collector.state())[0]}/{fabric.count} brokers report telemetry after {timeout}s")


def demo(fabric):
    """The original paced dashboard demo traffic, spread across the first brokers."""
    args, clients, config, client = fabric.args, fabric.clients, fabric.directory / "client.yaml", fabric.client
    count = min(3, fabric.count)
    for i in range(count):
        fabric.launch(f"subscriber{i}", client, "subscribe", "sensors/#", "--port", clients[i], "--config", config,
                      "--seconds", args.seconds, "--verbose")
    fabric.launch("responder", client, "respond", "control/status", "ready", "--port", clients[1 % fabric.count],
                  "--config", config, "--seconds", args.seconds, "--verbose")
    fabric.ready([f"subscriber{i}" for i in range(count)] + ["responder"], "Profile ready", 30)
    for i in range(count):
        fabric.launch(f"publisher{i}", client, "publish", f"sensors/zone-{i + 1}/environment",
                      json.dumps({"temperature": 21 + i, "humidity": 44 + i, "unit": "C", "source": "local-demo"}),
                      "--port", clients[i], "--config", config, "--count", 20000,
                      "--interval-ms", 1700 + i * 400, "--seconds", args.seconds)
    fabric.launch("requester", client, "request", "control/status", "status", "--port", clients[2 % fabric.count],
                  "--config", config, "--count", 20000, "--interval-ms", 3000, "--seconds", args.seconds)


def run(args, directory):
    directory.mkdir(parents=True)
    with ExitStack() as cleanup:
        fabric = Fabric(args, args.brokers, directory, cleanup)
        print(f"Artifacts: {directory}", flush=True)
        fabric.certificates()
        fabric.configure()
        started = time.monotonic()
        fabric.start()
        collector = Collector(Settings(f"Empy fabric - {args.brokers} brokers", fabric.sources(), 1.0, 1.5, 900))
        collector.start()
        cleanup.callback(collector.stop)
        if not args.no_dashboard:
            server = cleanup.enter_context(DashboardServer(("127.0.0.1", args.port), collector))
            thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.25},
                                      name="EmpyDashboard")
            thread.start()
            cleanup.callback(thread.join)
            cleanup.callback(server.shutdown)
            print(f"Dashboard: http://127.0.0.1:{server.server_port}", flush=True)
        wait_reporting(fabric, collector, args.ready_timeout or 30 + args.brokers)
        if args.traffic == "demo":
            demo(fabric)
        print(f"Ready: {args.brokers} brokers, {'QUIC' if args.quic else 'TLS'} backplane", flush=True)
        try:
            while time.monotonic() - started < args.seconds - 2:
                failed = [name for name, process in fabric.processes.items() if process.poll() not in (None, 0)]
                if failed:
                    raise RuntimeError(f"Processes failed: {', '.join(failed)}; inspect {directory}")
                time.sleep(0.5)
        finally:
            # Operational counters are functional observations, not a performance evaluation.
            state = collector.state()
            observed = {"brokers": args.brokers, "transport": "QUIC" if args.quic else "TLS",
                        "onlineBrokers": mesh(state)[0], "confirmedLinks": mesh(state)[1] // 2,
                        "metrics": totals(state)}
            (directory / "observed.json").write_text(json.dumps(observed, indent=4), encoding="utf-8")


def main(argv=None):
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bin", type=Path, default=root / "Targets/Release/bin")
    parser.add_argument("--openssl", default="openssl")
    parser.add_argument("--output", type=Path, default=root / "Targets/Runs/Cluster")
    parser.add_argument("--brokers", type=int, default=3)
    parser.add_argument("--quic", action="store_true", help="Use a native QUIC backplane")
    parser.add_argument("--shards", type=int, default=0, help="Profile shards (default: max(8, 2 x brokers))")
    parser.add_argument("--replicas", type=int, default=3)
    parser.add_argument("--fanout", type=int, default=4)
    parser.add_argument("--lease-ms", type=int, default=3000)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--start-concurrency", type=int, default=8)
    parser.add_argument("--ready-timeout", type=int, default=0)
    parser.add_argument("--traffic", choices=("demo", "idle"), default="demo")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--no-dashboard", action="store_true", help="Run without the dashboard HTTP server")
    parser.add_argument("--seconds", type=int, default=900, help="Deployment lifetime")
    args = parser.parse_args(argv)
    if not 1 <= args.brokers <= EMPY_MAX_SOURCES:
        parser.error(f"--brokers must be in 1..{EMPY_MAX_SOURCES}")
    if not 30 <= args.seconds <= 86400 or not 0 <= args.port <= 65535:
        parser.error("Invalid deployment lifetime or port")
    if not 1 <= args.workers <= 256 or not 1 <= args.start_concurrency <= EMPY_MAX_SOURCES:
        parser.error("Invalid worker count or startup concurrency")
    if not 1 <= args.replicas <= EMPY_MAX_SOURCES or not 1 <= args.fanout <= 64 or not 0 <= args.shards <= 65535:
        parser.error("Invalid replica count, fanout or shard count")
    if not 1000 <= args.lease_ms <= 60000 or not 0 <= args.ready_timeout <= 300:
        parser.error("Invalid lease or readiness timeout")
    suffix = ".exe" if os.name == "nt" else ""
    if any(not (args.bin / (name + suffix)).is_file() for name in ("EmpyBroker", "EmpyClient")):
        parser.error("Build EmpyBroker and EmpyClient first, or specify --bin")
    if os.name == "nt":
        import ctypes
        ctypes.windll.kernel32.SetErrorMode(0x0001 | 0x0002)
    directory = args.output.resolve() / (time.strftime("%Y%m%d-%H%M%S-") + secrets.token_hex(3))
    try:
        run(args, directory)
    except KeyboardInterrupt:
        print("Stopped.", flush=True)
    except (RuntimeError, OSError, subprocess.CalledProcessError) as failure:
        print(f"Cluster demonstration failed: {failure}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
