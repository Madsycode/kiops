"""Run the Agentic-MLOps closed loop locally with real EMPY processes.

Starts EmpyBroker (Configs/MLOps/Broker.yaml), the harness and agent swarm (EmpyHarness run with
Configs/MLOps/HarnessLocal.yaml: process substrate, EmpyWorkload children), a simulated IoT gateway
(EmpyWorkload simulate for Configs/MLOps/HomeControl.yaml, optional concept drift) and MMQP observers,
then submits the thesis Listing 2.1 intent through `EmpyHarness intent` while an `EmpyHarness approve`
responder answers any human-in-the-loop question. `EmpyHarness status` is polled and the run is
summarised as a timeline: deployment, placements, bootstrap training, promotions, drift and retraining.
At the end the service is retired over MMQP and every child process is stopped.

Logs, status snapshots, the timeline and summary.json are kept under
Targets/Runs/MLOpsDemo/<timestamp>/. The process substrate's per-user state directory is pointed at
that run directory, so runs never share workload descriptors or logs. Requires the built EMPY
executables and Python 3.10+ (standard library only).
"""

import argparse
from contextlib import ExitStack
import json
import os
from pathlib import Path
import re
import secrets
import shutil
import subprocess
import sys
import time
import urllib.request

EMPY_NO_WINDOW = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
EMPY_SUFFIX = ".exe" if os.name == "nt" else ""
EMPY_SERVICE = "home-control"
# thesis Listing 2.1 (the metadata.intent of Configs/MLOps/HomeControl.yaml)
EMPY_INTENT = ("Deploy a home temperature controller that keeps the temperature between 16 and 24 degrees "
               "using temperature and humidity sensors, with a p95 latency under 5 ms, retrained daily on a GPU")
EMPY_FINISHED = ("succeeded", "failed", "rejected")


def stop_process(process, stop_file=None, grace=10.0):
    """Stop a child: its --stop-file first (graceful), then terminate, then kill."""
    if process.poll() is not None:
        return
    if stop_file is not None:
        stop_file.touch()
        try:
            process.wait(timeout=grace)
            return
        except subprocess.TimeoutExpired:
            pass
    process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)


class Demo:
    """One isolated run: processes, logs, snapshots and the derived timeline."""

    def __init__(self, args, directory, cleanup):
        self.args, self.directory, self.cleanup = args, directory, cleanup
        self.root = Path(__file__).resolve().parents[1]
        self.binary = args.bin.resolve()
        self.configs = self.root / "Configs" / "MLOps"
        self.harness_config = self.configs / "HarnessLocal.yaml"
        self.state = directory / "state"
        self.snapshots = directory / "snapshots"
        self.snapshots.mkdir(parents=True)
        self.started = time.monotonic()
        self.port = None
        self.status_url = None
        self.processes = {}
        self.records = []
        self.timeline = []
        self.seen = {}  # workflow id -> last reported status
        self.instances = {}  # instance id -> (node, state)
        self.active = None
        self.snapshot = None
        self.saved = 0
        self.reported = 0  # observer events already on the timeline
        self.environment = os.environ.copy()
        # the process substrate keeps descriptors and logs in a per-user directory; point it at this run
        self.environment["LOCALAPPDATA"] = str(self.state)
        self.environment["XDG_STATE_HOME"] = str(self.state)
        self.environment.pop("EMPY_MLOPS_CONFIG", None)
        if os.name == "nt":
            # only the DLLs staged beside the executables may satisfy their imports
            windows = Path(os.environ["SystemRoot"])
            self.environment["PATH"] = os.pathsep.join((str(windows / "System32"), str(windows)))

    # ---------------------------------------------------------------- processes

    def executable(self, name):
        return self.binary / (name + EMPY_SUFFIX)

    def launch(self, name, executable, *arguments, stop_file=None):
        log = self.cleanup.enter_context((self.directory / f"{name}.log").open("wb"))
        process = subprocess.Popen([str(executable), *map(str, arguments)], cwd=self.root, env=self.environment,
                                   stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                   creationflags=EMPY_NO_WINDOW)
        self.cleanup.callback(stop_process, process, stop_file)  # stopped in reverse launch order
        self.processes[name] = process
        self.records.append({"name": name, "pid": process.pid, "argv": [str(executable), *map(str, arguments)]})
        (self.directory / "processes.json").write_text(json.dumps(self.records, indent=4), encoding="utf-8")
        return process

    def wait_for(self, name, pattern, timeout):
        deadline = time.monotonic() + timeout
        log = self.directory / f"{name}.log"
        while True:
            text = log.read_text(encoding="utf-8", errors="replace")
            match = re.search(pattern, text)
            if match:
                return match
            if self.processes[name].poll() is not None:
                raise RuntimeError(f"{name} exited with {self.processes[name].returncode}: {text[-2000:]}")
            if time.monotonic() > deadline:
                raise RuntimeError(f"{name} did not report '{pattern}' within {timeout} s; see {log}")
            time.sleep(0.1)

    def harness_cli(self, *arguments):
        return [str(self.executable("EmpyHarness")), *map(str, arguments), "--config", str(self.harness_config),
                "--host", "127.0.0.1", "--port", str(self.port)]

    # ---------------------------------------------------------------- timeline

    def note(self, text):
        entry = {"t": round(time.monotonic() - self.started, 1), "event": text}
        self.timeline.append(entry)
        print(f"[{entry['t']:7.1f}s] {text}", flush=True)

    def status(self):
        try:
            result = subprocess.run(self.harness_cli("status", "--timeout-ms", 5000), cwd=self.root,
                                    env=self.environment, stdin=subprocess.DEVNULL, capture_output=True,
                                    timeout=30, creationflags=EMPY_NO_WINDOW)
        except subprocess.TimeoutExpired:
            return None
        if result.returncode != 0:
            return None
        try:
            return json.loads(result.stdout.decode("utf-8"))
        except ValueError:
            return None

    def describe(self, workflow):
        kind, trigger, status = workflow.get("kind"), workflow.get("trigger"), workflow.get("status")
        result = workflow.get("result") or {}
        steps = [step.get("name") for step in workflow.get("steps", [])]
        text = f"workflow {kind} ({trigger}) {status}"
        if kind == "deploy" and status == "succeeded":
            placement = result.get("placement", {})
            places = ", ".join(f"{role} -> {','.join(nodes)}" for role, nodes in placement.items())
            text += (f": {result.get('service')} {result.get('version')}, compliance {result.get('compliance')}, "
                     f"{places}")
        elif kind == "retrain" and status == "succeeded":
            text += (f": model v{result.get('version')} promoted (accuracy {result.get('accuracy')}, previous "
                     f"{result.get('previous')}, loaded on {result.get('loaded')}/{result.get('expected')})")
        elif kind == "rollback" and status == "succeeded":
            text += (f": rolled back to model v{result.get('version')} (from {result.get('previous')}, loaded on "
                     f"{result.get('loaded')}/{result.get('expected')})")
        elif kind == "relocate" and status == "succeeded":
            text += f": moves {json.dumps(result.get('moves', []))}"
        elif kind == "retire" and status == "succeeded":
            text += f": retired {', '.join(result.get('retired', []))}"
        elif workflow.get("error"):
            text += f": {workflow['error']}"
        if status in EMPY_FINISHED and steps:
            text += f" [steps: {' > '.join(steps)}]"
        return text

    def observe(self, snapshot):
        changed = False
        for workflow in reversed(snapshot.get("workflows", [])):  # oldest first
            identifier, status = workflow.get("id"), workflow.get("status")
            if self.seen.get(identifier) == status:
                continue
            if status in EMPY_FINISHED or status == "waiting-hitl" or identifier not in self.seen:
                self.note(self.describe(workflow))
                changed = True
            self.seen[identifier] = status
        for deployment in snapshot.get("deployments", []):
            for instance in deployment.get("instances", []):
                key = instance.get("id")
                value = (instance.get("node"), instance.get("state"))
                if self.instances.get(key) != value:
                    self.instances[key] = value
                    self.note(f"instance {key} ({instance.get('role')}) on {value[0]}: {value[1]}")
                    changed = True
            active = (deployment.get("service"), (deployment.get("lifecycle") or {}).get("active"))
            if active[1] is not None and active != self.active:
                self.active = active
                self.note(f"active model of {active[0]}: v{active[1]}")
                changed = True
        return changed

    def poll(self):
        snapshot = self.status()
        if snapshot is None:
            return
        self.snapshot = snapshot
        if self.observe(snapshot):
            self.saved += 1
            path = self.snapshots / f"{self.saved:03}-{int(time.monotonic() - self.started):04}s.json"
            path.write_text(json.dumps(snapshot, indent=2), encoding="utf-8")

    def events(self):
        """Drift, violation, failure, exhaustion and accelerator events seen by the MMQP observer."""
        events = []
        log = self.directory / "events.log"
        for line in log.read_text(encoding="utf-8", errors="replace").splitlines():
            topic, _, payload = line.partition(" ")
            if not topic.endswith(("/drift", "/violation", "/failure", "/exhaustion", "/accelerator")):
                continue
            try:
                body = json.loads(payload).get("body", {})
            except ValueError:
                continue
            events.append({"topic": topic, "body": body})
        return events

    def note_events(self):
        events = self.events()
        for event in events[self.reported:]:
            body, kind = event["body"], event["topic"].rsplit("/", 1)[-1]
            if kind == "drift":
                self.note(f"drift event ({body.get('kind')}): {body.get('value')} vs threshold "
                          f"{body.get('threshold')}, model v{body.get('modelVersion')} on {body.get('instance')}")
            else:
                self.note(f"{kind} event: {body.get('instance')} {body.get('reason', '')}".rstrip())
        self.reported = len(events)

    def actions(self):
        lines = (self.directory / "actions.log").read_text(encoding="utf-8", errors="replace").splitlines()
        values, models = [], set()
        for line in lines:
            _, _, payload = line.partition(" ")
            try:
                action = json.loads(payload)
            except ValueError:
                continue
            values.append(action.get("value"))
            models.add(action.get("model"))
        return values, sorted(model for model in models if model is not None)

    # ---------------------------------------------------------------- run

    def start(self):
        args = self.args
        lifetime = args.seconds + 60
        self.launch("broker", self.executable("EmpyBroker"), "--config", self.configs / "Broker.yaml",
                    "--port", args.port, "--seconds", lifetime + 30)
        self.port = int(self.wait_for("broker", r"listening on port (\d+)", 30).group(1))
        self.note(f"EmpyBroker listening on 127.0.0.1:{self.port} (Configs/MLOps/Broker.yaml)")
        simulator = ["simulate", "--descriptor", self.configs / "HomeControl.yaml", "--host", "127.0.0.1",
                     "--port", self.port, "--period-ms", args.period_ms, "--seconds", lifetime,
                     "--stop-file", self.directory / "simulator.stop"]
        if args.drift_after_ms:
            simulator += ["--drift-after-ms", args.drift_after_ms]
        self.launch("simulator", self.executable("EmpyWorkload"), *simulator,
                    stop_file=self.directory / "simulator.stop")
        self.wait_for("simulator", r"connected to", 30)
        drift = f", concept drift after {args.drift_after_ms} ms" if args.drift_after_ms else ", no drift"
        self.note(f"EmpyWorkload simulate publishing home/temp, home/hum and home/label/temp every "
                  f"{args.period_ms} ms{drift}")
        self.launch("harness", self.executable("EmpyHarness"), "run", "--config", self.harness_config,
                    "--host", "127.0.0.1", "--port", self.port, "--status-port", 0, "--seconds", lifetime,
                    "--stop-file", self.directory / "harness.stop", stop_file=self.directory / "harness.stop")
        running = self.wait_for("harness", r"EmpyHarness running: [^\r\n]*", 60).group(0)
        self.note(running)
        url = re.search(r"status (http://\S+/api/mlops/status)", running)
        self.status_url = url.group(1) if url else None
        for name, topic in (("events", "mlops/event/#"), ("actions", "home/set/temp")):
            self.launch(name, self.executable("EmpyClient"), "subscribe", topic, "--qos", 1, "--host", "127.0.0.1",
                        "--port", self.port, "--seconds", lifetime)
        time.sleep(1.0)  # observers subscribed before the deployment starts

    def submit(self):
        approver = self.harness_cli("approve", "--count", 1, "--operator", "demo-operator", "--note",
                                    "approved by MLOpsDemo", "--seconds", 600)
        log = self.cleanup.enter_context((self.directory / "approve.log").open("wb"))
        approve = subprocess.Popen(approver, cwd=self.root, env=self.environment, stdin=subprocess.DEVNULL,
                                   stdout=log, stderr=subprocess.STDOUT, creationflags=EMPY_NO_WINDOW)
        self.processes["approve"] = approve
        self.cleanup.callback(stop_process, approve)
        self.note(f"EmpyHarness intent \"{EMPY_INTENT}\" (EmpyHarness approve answers HITL questions)")
        out = self.cleanup.enter_context((self.directory / "intent.json").open("wb"))
        err = self.cleanup.enter_context((self.directory / "intent.log").open("wb"))
        intent = subprocess.Popen(self.harness_cli("intent", EMPY_INTENT, "--operator", "demo-operator"),
                                  cwd=self.root, env=self.environment, stdin=subprocess.DEVNULL, stdout=out,
                                  stderr=err, creationflags=EMPY_NO_WINDOW)
        self.processes["intent"] = intent
        self.cleanup.callback(stop_process, intent)
        return intent

    def retire(self):
        request = json.dumps({"kind": "workflow-request",
                              "body": {"kind": "retire", "service": EMPY_SERVICE, "trigger": "operator",
                                       "reason": "end of MLOpsDemo"}}, separators=(",", ":"))
        self.note("retire requested over MMQP (EmpyClient request mlops/harness/workflow)")
        result = subprocess.run([str(self.executable("EmpyClient")), "request", "mlops/harness/workflow", request,
                                 "--host", "127.0.0.1", "--port", str(self.port), "--qos", "1", "--seconds", "30"],
                                cwd=self.root, env=self.environment, stdin=subprocess.DEVNULL, capture_output=True,
                                timeout=60, creationflags=EMPY_NO_WINDOW)
        (self.directory / "retire.log").write_bytes(result.stdout + result.stderr)
        return result.returncode

    def health(self):
        if not self.status_url:
            return None
        try:
            with urllib.request.urlopen(self.status_url.replace("/status", "/healthz"), timeout=5) as response:
                body = json.loads(response.read().decode("utf-8"))
                return {"status": response.status, "body": body}
        except (OSError, ValueError) as failure:
            return {"error": str(failure)}

    def run(self):
        args = self.args
        self.start()
        intent = self.submit()
        intent_code = None
        deadline = self.started + args.seconds
        while time.monotonic() < deadline:
            self.poll()
            self.note_events()
            if intent_code is None and intent.poll() is not None:
                intent_code = intent.returncode
                self.note(f"EmpyHarness intent exited with {intent_code} (record in intent.json)")
                stop_process(self.processes["approve"])
            failed = [name for name in ("broker", "harness", "simulator") if self.processes[name].poll() is not None]
            if failed:
                raise RuntimeError(f"{', '.join(failed)} exited early; inspect {self.directory}")
            if self.done():
                self.note("closed loop demonstrated; finishing early")
                break
            time.sleep(2.0)
        health = self.health()
        self.note_events()
        drift_events = [event for event in self.events() if event["topic"].endswith("/drift")]
        retire_code = self.retire()
        for _ in range(10):
            self.poll()
            if self.snapshot is not None and not self.snapshot.get("deployments"):
                break
            time.sleep(1.0)
        return self.summarise(intent_code, retire_code, health, drift_events)

    def workflows(self, kind=None, trigger=None, status=None):
        records = (self.snapshot or {}).get("workflows", [])
        return [record for record in records if (kind is None or record.get("kind") == kind)
                and (trigger is None or record.get("trigger") == trigger)
                and (status is None or record.get("status") == status)]

    def done(self):
        """Every expected stage was observed (drift retraining only when drift was induced)."""
        if not self.workflows("deploy", status="succeeded") or not self.workflows("retrain", "bootstrap", "succeeded"):
            return False
        if self.args.drift_after_ms:
            return bool(self.workflows("retrain", "drift", "succeeded") or
                        self.workflows("retrain", "metrics", "succeeded"))
        return time.monotonic() - self.started > 30

    def summarise(self, intent_code, retire_code, health, drift_events):
        values, models = self.actions()
        bounded = [value for value in values if isinstance(value, (int, float))]
        intent = {}
        try:
            intent = json.loads((self.directory / "intent.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            pass
        promotions = [record for record in self.workflows("retrain", status="succeeded")]
        checks = {
            "intentDeploySucceeded": intent_code == 0 and intent.get("status") == "succeeded",
            "bootstrapPromoted": bool(self.workflows("retrain", "bootstrap", "succeeded")),
            "actionsObserved": len(bounded) > 0,
            "actionsWithinBounds": bool(bounded) and all(16 <= value <= 24 for value in bounded),
            "statusEndpointHealthy": bool(health) and health.get("status") == 200,
            "retired": retire_code == 0 and bool(self.workflows("retire", status="succeeded")),
        }
        if self.args.drift_after_ms:
            checks["driftDetected"] = bool(drift_events)
            checks["driftRetrainPromoted"] = bool(self.workflows("retrain", "drift", "succeeded") or
                                                  self.workflows("retrain", "metrics", "succeeded"))
        placement = ((intent.get("result") or {}).get("placement")) or {}
        summary = {
            "intent": EMPY_INTENT, "brokerPort": self.port, "statusUrl": self.status_url,
            "placement": placement, "actions": len(bounded),
            "actionRange": [min(bounded), max(bounded)] if bounded else None, "actionModels": models,
            "driftEvents": len(drift_events),
            "promotions": [{"trigger": record.get("trigger"), "version": (record.get("result") or {}).get("version"),
                            "accuracy": (record.get("result") or {}).get("accuracy")} for record in promotions],
            "health": health, "checks": checks, "passed": all(checks.values()), "timeline": self.timeline,
        }
        (self.directory / "summary.json").write_text(json.dumps(summary, indent=4), encoding="utf-8")
        if self.snapshot is not None:
            (self.directory / "final-status.json").write_text(json.dumps(self.snapshot, indent=2), encoding="utf-8")
        if bounded:
            self.note(f"{len(bounded)} actions on home/set/temp in [{min(bounded):.2f}, {max(bounded):.2f}] from "
                      f"model(s) {models}")
        for name, ok in checks.items():
            print(f"  {'PASS' if ok else 'FAIL'}  {name}", flush=True)
        return summary["passed"]

    def collect(self):
        """Move the workload logs out of the per-run substrate directory; drop it unless --keep-state."""
        process = self.state / "EmpyMLOps" / "process"
        if process.is_dir():
            target = self.directory / "workloads"
            target.mkdir(exist_ok=True)
            for log in process.glob("*.log"):
                shutil.copy2(log, target / log.name)
        if not self.args.keep_state and self.state.exists():
            shutil.rmtree(self.state, ignore_errors=True)


def main(argv=None):
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bin", type=Path, default=root / "Targets/Debug/bin", help="EMPY executables")
    parser.add_argument("--output", type=Path, default=root / "Targets/Runs/MLOpsDemo", help="run directory parent")
    parser.add_argument("--port", type=int, default=0, help="broker port (0 = a free ephemeral port)")
    parser.add_argument("--seconds", type=int, default=150, help="longest demonstration time before retiring")
    parser.add_argument("--drift-after-ms", type=int, default=30000,
                        help="simulated concept drift after N ms (0 = no drift)")
    parser.add_argument("--period-ms", type=int, default=50, help="simulator signal period")
    parser.add_argument("--keep-state", action="store_true",
                        help="keep the process substrate's descriptors and logs under <run>/state")
    args = parser.parse_args(argv)
    if not 20 <= args.seconds <= 3600 or not 0 <= args.port <= 65535:
        parser.error("--seconds must be in 20..3600 and --port in 0..65535")
    if not 0 <= args.drift_after_ms <= 3600000 or not 10 <= args.period_ms <= 10000:
        parser.error("--drift-after-ms must be in 0..3600000 and --period-ms in 10..10000")
    missing = [name for name in ("EmpyBroker", "EmpyClient", "EmpyHarness", "EmpyWorkload")
               if not (args.bin / (name + EMPY_SUFFIX)).is_file()]
    if missing:
        parser.error(f"{', '.join(missing)} not found in {args.bin}; build with EMPY_BUILD_MLOPS=ON or pass --bin")
    if os.name == "nt":
        import ctypes
        ctypes.windll.kernel32.SetErrorMode(0x0001 | 0x0002)
    directory = args.output.resolve() / (time.strftime("%Y%m%d-%H%M%S-") + secrets.token_hex(3))
    directory.mkdir(parents=True)
    print(f"Artifacts: {directory}", flush=True)
    demo = None
    try:
        with ExitStack() as cleanup:
            demo = Demo(args, directory, cleanup)
            passed = demo.run()
        demo.collect()
        print(f"{'Demonstration passed' if passed else 'Demonstration incomplete'}; evidence in {directory}",
              flush=True)
        return 0 if passed else 1
    except KeyboardInterrupt:
        if demo is not None:
            demo.collect()
        print("Stopped.", flush=True)
        return 130
    except (RuntimeError, OSError, subprocess.SubprocessError) as failure:
        if demo is not None:
            demo.collect()
        print(f"MLOps demonstration failed: {failure}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
