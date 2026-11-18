"""The container's side of a live demonstration.

Owns one staged trainer process (see live_trainer.py), turns its `[live]` lines into a snapshot that
displays read, and keeps a lease that decides how long the GPUs are held. modal_run.py wraps this
in the LiveRun class; nothing here depends on Modal, so `publish` and `on_finished` are passed in.

The snapshot, as displays see it:

    status      "loading" | "ready" | "pretraining" | "pretrained" | "tuning" | "tuned" | "failed" | "stopped"
    updated     epoch seconds of the last write; a display treats an old one as "no container"
    hold_until  epoch seconds until which the GPUs are kept
    run_id      the run's folder name under runs/ on the Volume, once pretraining has started
    gpu         the model of GPU the container was given
    pending     actions asked for early, each taken when the trainer reaches it (["pretrain", "tune"])
    loading     {started, seconds, note}
    pretrain    {step, total, started, finished, val: [{step, loss}]}
    tune        {step, total, started, finished, loss: [{step, loss}], results, name, dataset}; `name` is
                what the model was named for this tuning, or ""; `dataset` is {title, examples, questions}
                for the extra training data it was given (chat_data.clean_dataset), or None
    samples     [{phase, step, at, items: [{prompt, text}]}], oldest first
    serving     {base: port, chat: port}, whichever of the two models can be queried right now
    error       why status is "failed"
    log         the trainer's most recent console lines
"""
import copy
import json
import os
import signal
import subprocess
import threading
import time

REPO = os.path.dirname(os.path.abspath(__file__))
LIVE_DIR = "/tmp/live"
SERVE_PORT = 8000
LOG_LINES = 40
PUBLISH_SECONDS = 1.0
HEARTBEAT_SECONDS = 5.0
MAX_HOLD_MINUTES = 180
ACTIONS = ("pretrain", "tune", "restart")


class LiveSession:
    def __init__(self, num_gpus, publish, on_ready=None, on_finished=None, details=None, command=None):
        """`publish(snapshot)` stores the snapshot for displays. `on_ready()` runs each time a trainer
        finishes loading and `on_finished(phase, run_id, results)` when it finishes a phase; both are
        called on a background thread. `details` are fixed facts about the container, such as its GPU
        model, shown in every snapshot."""
        self.details = details or {}
        self.num_gpus, self.env, self.publish = num_gpus, None, publish
        self.on_ready, self.on_finished = on_ready, on_finished
        self.command = command or ["torchrun", "--standalone", f"--nproc_per_node={num_gpus}", "train_gpt.py"]
        self.lock = threading.RLock()
        self.trainer = None
        self.pending = {}  # action -> options
        self.hold_until = 0.0
        self.holding = False
        self.stopping = False
        self.dirty = True
        self.snapshot = {}
        threading.Thread(target=self.publish_loop, daemon=True).start()

    def announce(self, note):
        """Show a fresh "loading" snapshot with this note, before there is a trainer to report on."""
        with self.lock:
            self.pending = {}
            self.snapshot = dict(status="loading", run_id=None, pending=[], error=None, serving={}, samples=[], log=[],
                                 loading=dict(started=time.time(), seconds=None, note=note), pretrain=None, tune=None)
            self.dirty = True

    def start_trainer(self, env=None):
        """`env` is the trainer's environment; left out, the previous trainer's is used again."""
        with self.lock:
            self.env = env or self.env
            os.makedirs(LIVE_DIR, exist_ok=True)
            for name in os.listdir(LIVE_DIR):
                os.remove(os.path.join(LIVE_DIR, name))
            started = self.snapshot["loading"]["started"] if self.snapshot.get("status") == "loading" else time.time()
            self.announce("starting the trainer")
            self.snapshot["loading"]["started"] = started
            env = dict(self.env, LIVE_DIR=LIVE_DIR, LIVE_SERVE_PORT=str(SERVE_PORT))
            self.trainer = subprocess.Popen(self.command, cwd=REPO, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                            text=True, bufsize=1, errors="replace", start_new_session=True)
            self.dirty = True
            threading.Thread(target=self.read_trainer, args=(self.trainer,), daemon=True).start()

    def stop_trainer(self):
        with self.lock:
            trainer, self.trainer = self.trainer, None
        if trainer and trainer.poll() is None:
            # torchrun starts one process per GPU; the group signal reaches all of them.
            os.killpg(trainer.pid, signal.SIGTERM)
            try:
                trainer.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(trainer.pid, signal.SIGKILL)
                trainer.wait()

    def read_trainer(self, trainer):
        for line in trainer.stdout:
            print(line, end="", flush=True)
            self.handle_line(trainer, line.rstrip("\n"))
        returncode = trainer.wait()
        with self.lock:
            if trainer is self.trainer:
                self.snapshot.update(status="failed", serving={}, error=f"the trainer exited with code {returncode}")
                self.dirty = True

    def handle_line(self, trainer, line):
        with self.lock:
            if trainer is not self.trainer:
                return
            snapshot = self.snapshot
            if line.startswith("[live] {"):
                try:
                    self.handle_event(json.loads(line[len("[live] "):]))
                except Exception as error:
                    print(f"[live session] could not handle {line[:200]!r}: {error!r}", flush=True)
            else:
                snapshot["log"] = (snapshot["log"] + [line[:300]])[-LOG_LINES:]
                if snapshot["status"] == "loading" and line.startswith(("warmup step", "Compiling", "Sampling steps", "Resetting")):
                    snapshot["loading"]["note"] = line[:120]
            self.dirty = True

    def handle_event(self, event):
        snapshot, kind, phase = self.snapshot, event["event"], event.get("phase")
        if kind == "waiting":
            if phase == "pretrain":
                snapshot["status"] = "ready"
                snapshot["loading"]["seconds"] = round(time.time() - snapshot["loading"]["started"], 1)
                self.in_background(self.on_ready)
            if phase in self.pending:
                self.go(phase, self.pending.pop(phase))
        elif kind == "started":
            snapshot["run_id"] = event["run_id"]
            snapshot["status"] = "pretraining" if phase == "pretrain" else "tuning"
            progress = dict(step=0, total=event["total"], started=event["at"], finished=None)
            if phase == "pretrain":
                snapshot[phase] = dict(progress, val=[])
            else:
                snapshot[phase] = dict(progress, loss=[], results=None, name=event.get("name", ""), dataset=event.get("dataset"))
        elif kind in ("step", "sample", "val", "loss"):
            progress = snapshot[phase]
            progress["step"] = max(progress["step"], event["step"])
            if kind == "sample":
                snapshot["samples"].append({key: event[key] for key in ("phase", "step", "at", "items")})
            elif kind in ("val", "loss"):
                progress[kind].append(dict(step=event["step"], loss=event["loss"]))
        elif kind == "finished":
            snapshot[phase]["finished"] = event["at"]
            snapshot["status"] = "pretrained" if phase == "pretrain" else "tuned"
            if phase == "tune":
                snapshot["tune"]["results"] = event["results"]
            self.in_background(self.on_finished, phase, event["run_id"], event.get("results"))
        elif kind == "serving":
            snapshot["serving"] = {model: event[model] for model in ("base", "chat") if model in event}

    def in_background(self, callback, *args):
        if callback:
            threading.Thread(target=callback, args=args, daemon=True).start()

    def go(self, phase, options):
        """Release the trainer into `phase`. The file is renamed into place so no rank reads half of it."""
        path = os.path.join(LIVE_DIR, f"go_{phase}")
        with open(path + ".tmp", "w") as f:
            json.dump(options, f)
        os.replace(path + ".tmp", path)
        self.snapshot["pending"] = list(self.pending)

    def request(self, action, options=None):
        """Ask for "pretrain", "tune" or "restart". An action asked for before the trainer can take it
        is remembered and taken as soon as it can. Returns the snapshot."""
        if action not in ACTIONS:
            raise ValueError(f"unknown action {action!r}")
        if action == "restart":
            self.stop_trainer()
            self.start_trainer()
            return self.current()
        with self.lock:
            snapshot = self.snapshot
            status = snapshot["status"]
            can_wait = dict(pretrain=("loading",), tune=("loading", "ready", "pretraining"))[action]
            can_go = dict(pretrain="ready", tune="pretrained")[action]
            if status == can_go:
                self.go(action, options or {})
            elif status in can_wait:
                self.pending[action] = options or {}
                snapshot["pending"] = list(self.pending)
            else:
                raise ValueError(f"cannot {action} while the trainer is {status}; restart it first")
            self.dirty = True
        return self.current()

    def hold(self, minutes):
        """Keep the GPUs for at least `minutes` more. Blocks while the lease lasts, unless another call
        is already blocking for it."""
        with self.lock:
            self.hold_until = max(self.hold_until, time.time() + min(minutes, MAX_HOLD_MINUTES) * 60)
            self.dirty = True
            if self.holding:
                return
            self.holding = True
        try:
            while time.time() < self.hold_until:
                time.sleep(1)
        finally:
            self.holding = False

    def release(self):
        with self.lock:
            self.hold_until = time.time()
            self.dirty = True

    def serving_port(self, model):
        port = self.snapshot.get("serving", {}).get(model)
        if not port:
            raise RuntimeError(f"the {model} model is not being served right now (the trainer is {self.snapshot.get('status')})")
        return port

    def current(self):
        with self.lock:
            return dict(copy.deepcopy(self.snapshot), **self.details, updated=time.time(), hold_until=self.hold_until)

    def publish_loop(self):
        last = 0.0
        while not self.stopping:
            time.sleep(0.2)
            now = time.time()
            if not self.snapshot or now - last < (PUBLISH_SECONDS if self.dirty else HEARTBEAT_SECONDS):
                continue
            self.dirty = False
            last = now
            try:
                self.publish(self.current())
            except Exception as error:
                print(f"[live session] could not publish the snapshot: {error!r}", flush=True)

    def shut_down(self):
        self.stopping = True
        self.stop_trainer()
        with self.lock:
            self.snapshot.update(status="stopped", serving={}, pending=[])
        self.publish(self.current())
