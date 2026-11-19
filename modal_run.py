"""Run the speedrun trainer on Modal, keep its checkpoints on a Volume, and chat-tune them.

  uvx modal run modal_run.py::prep            # once: FineWeb shards and attention kernel onto the Volume (CPU)
  uvx modal run modal_run.py::prep_chat       # once: pre-tokenized chat shards onto the Volume (CPU)
  uvx modal run modal_run.py::main            # train4 (4xH200) -> verify (CPU)
  uvx modal run modal_run.py::tune4           # chat-tune a saved base model (4xH200, a few minutes)
  uvx modal run modal_run.py::train_and_tune4 # pretrain, then chat-tune, in one job
  uvx modal run modal_run.py::chat_eval --run-id <id>
  uvx modal run modal_run.py::generate --run-id <id> [--prompts '["Hello"]']
  uvx modal run modal_run.py::publish_run --run-id <id>   # make other services serve this run (CPU)
  uvx modal deploy modal_run.py               # deploys LiveRun, Inference and ChatInference for other services to call

Once deployed, live.py drives LiveRun: one held 4xH200 container that pretrains, serves and chat-tunes on cue.
"""
import os
import subprocess
import threading
import time

import modal

REPO = "/root/modded-nanogpt"
VOL = "/vol"
SERVED_RUN_ID = "b8ff02bd-65e1-4593-98b7-34d91e5255e4"  # the checkpoint the deployed Inference class loads
BASE_RUN_ID = "037b2c75-2815-426c-9820-56f3e7df87e7"    # the 4xH200 pretraining run that tune4 starts from
SERVED_CHAT_RUN_ID = "0bbb8eb0-6ad1-4e93-a626-4a098b840eac"  # the tuned run ChatInference loads when given no run id
SERVE_PORT = 8000

# This branch deploys under its own name so the apps deployed from earlier branches are left alone.
app = modal.App("nanogpt-live")
volume = modal.Volume.from_name("nanogpt-speedrun", create_if_missing=True)

image = (
    modal.Image.from_registry("nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04", add_python="3.12")
    .entrypoint([])
    .apt_install("build-essential", "git")
    .pip_install("torch==2.10.0")  # the default wheel is cu128, which the record logs ran on
    .pip_install("numpy", "tqdm", "huggingface-hub", "kernels", "setuptools",
                 "typing-extensions==4.15.0", "datasets", "tiktoken")
    .pip_install("py-spy")
    .env({"HF_HOME": f"{VOL}/hf"})
    .add_local_dir(".", REPO, ignore=[".git", "records", "img", "logs", "data/fineweb10B", "__pycache__"])
)


def link_data():
    """The trainer reads data/fineweb10B and data/chat relative to the repo; keep the shards on the Volume."""
    for name in ("fineweb10B", "chat"):
        os.makedirs(f"{VOL}/{name}", exist_ok=True)
        link = f"{REPO}/data/{name}"
        if not os.path.exists(link):
            os.symlink(f"{VOL}/{name}", link)


def link_runs():
    """The trainer writes logs and checkpoints to logs/ relative to the repo; keep them on the Volume."""
    link_data()
    os.makedirs(f"{VOL}/runs", exist_ok=True)
    if not os.path.exists(f"{REPO}/logs"):
        os.symlink(f"{VOL}/runs", f"{REPO}/logs")


def checkpoint_path(run_id: str, name_prefix: str) -> str:
    """The run's last checkpoint whose file name starts with `name_prefix` ("state_step" is the
    pretrained model, "state_tuned" the chat-tuned one)."""
    run_dir = f"{VOL}/runs/{run_id}"
    names = sorted(f for f in os.listdir(run_dir) if f.startswith(name_prefix) and f.endswith(".pt"))
    if not names:
        raise FileNotFoundError(f"no {name_prefix}*.pt in {run_dir}")
    return f"{run_dir}/{names[-1]}"


# Which finished runs other services should serve: {"pretrained" | "tuned": {"run_id", "finished"}}.
# A caller reads this and passes the run id to Inference or ChatInference, so it follows new runs
# without a redeploy.
latest_runs = modal.Dict.from_name("nanogpt-latest-runs", create_if_missing=True)
CHECKPOINT_PREFIX = dict(pretrained="state_step", tuned="state_tuned")


def publish_latest(kind: str, run_id: str):
    """Record `run_id` as the run of this kind to serve. Call once its checkpoint is committed to the Volume."""
    finished = os.path.getmtime(checkpoint_path(run_id, CHECKPOINT_PREFIX[kind]))
    latest_runs[kind] = dict(run_id=run_id, finished=finished)
    print(f"latest {kind} run is now {run_id}")


def publish_tuned(result: dict, base_run_id: str) -> dict:
    """Publish a tuned run together with the base it was tuned from, so the two served models are a
    before-and-after pair."""
    publish_latest("pretrained", base_run_id)
    publish_latest("tuned", result["run_id"])
    return result


KEPT_RUNS = 10            # runs whose checkpoints stay on the Volume; a run with both is about 3.8 GB
PINNED_RUN_IDS = (SERVED_RUN_ID, BASE_RUN_ID, SERVED_CHAT_RUN_ID)
SETTLED_SECONDS = 3600    # nothing newer than this is deleted, in case a run is still writing it


def expired_runs(keep: int = KEPT_RUNS) -> list[str]:
    """Names under runs/ that are no longer needed: everything but the `keep` newest runs that have
    a checkpoint, the runs being served (`latest_runs`) and the ones this file names. A run is its
    folder plus its <run id>.txt log."""
    import time
    runs = f"{VOL}/runs"
    names = os.listdir(runs)
    finished = {}
    for name in names:
        folder = f"{runs}/{name}"
        checkpoints = [f for f in os.listdir(folder) if f.endswith(".pt")] if os.path.isdir(folder) else []
        if checkpoints:
            finished[name] = max(os.path.getmtime(f"{folder}/{f}") for f in checkpoints)
    kept = set(sorted(finished, key=finished.get)[-keep:]) | set(PINNED_RUN_IDS)
    kept |= {latest["run_id"] for latest in latest_runs.values()}
    cutoff = time.time() - SETTLED_SECONDS
    return sorted(name for name in names
                  if name.removesuffix(".txt") not in kept and os.path.getmtime(f"{runs}/{name}") < cutoff)


def prune_runs():
    """Delete the runs that have expired. Called once a run has been saved and published; a failure
    here must never cost a run, so it is only reported."""
    import shutil
    try:
        expired = expired_runs()
        for name in expired:
            path = f"{VOL}/runs/{name}"
            shutil.rmtree(path, ignore_errors=True) if os.path.isdir(path) else os.remove(path)
        if expired:
            volume.commit()
        print(f"pruned {len(expired)} expired entries from runs/")
    except Exception as error:
        print(f"pruning runs/ failed: {error!r}")


@app.function(image=image, volumes={VOL: volume}, timeout=900)
def prune(dry_run: bool = False):
    """CPU-only: delete expired runs now, or with --dry-run only list what would go."""
    if dry_run:
        expired = expired_runs()
        print(f"would delete {len(expired)} of {len(os.listdir(f'{VOL}/runs'))} entries under runs/:", *expired, sep="\n  ")
    else:
        prune_runs()


@app.function(image=image, volumes={VOL: volume}, timeout=300)
def publish_run(run_id: str):
    """CPU-only: serve this run's checkpoints (whichever of the two kinds it has), for example to go
    back to a known-good run."""
    names = os.listdir(f"{VOL}/runs/{run_id}")
    for kind, prefix in CHECKPOINT_PREFIX.items():
        if any(name.startswith(prefix) and name.endswith(".pt") for name in names):
            publish_latest(kind, run_id)


@app.function(image=image, volumes={VOL: volume}, timeout=1800, cpu=4, memory=8192)
def prep():
    """CPU-only: download the data shards and the attention kernel so GPU time is not spent on it."""
    link_data()
    subprocess.run(["python", "data/cached_fineweb10B.py", "9"], cwd=REPO, check=True)
    from kernels import get_kernel
    get_kernel("kernels-community/flash-attn3", version=1)
    volume.commit()
    print("prep done:", sorted(os.listdir(f"{VOL}/fineweb10B")))


@app.function(image=image, volumes={VOL: volume}, timeout=3600, cpu=8, memory=32768)
def prep_chat():
    """CPU-only: download the chat datasets and write pre-tokenized shards to chat/ on the Volume."""
    subprocess.run(["python", "chat_data.py", "--out", f"{VOL}/chat"], cwd=REPO, check=True)
    volume.commit()


COMPILE_CACHE = "/root/compile_cache"


def cache_tars(num_gpus: int, staged: bool = False) -> list[str]:
    """Where a compile cache may be found, best first. A job saves to the first and only reads the
    others. The staged trainer (LiveRun) keeps its own tar, because it also compiles what sampling
    and serving use; the tar after it holds the pretraining and chat-tuning graphs; the rest belong
    to the base runs."""
    own = f"{VOL}/compile_cache_{num_gpus}gpu_instruct.tar"
    base = f"{VOL}/compile_cache.tar" if num_gpus == 8 else f"{VOL}/compile_cache_{num_gpus}gpu.tar"
    tars = [own, base, f"{VOL}/compile_cache.tar"]
    return [f"{VOL}/compile_cache_{num_gpus}gpu_staged.tar", *tars] if staged else tars


def restore_compile_cache(num_gpus: int, staged: bool = False) -> dict:
    """torch.compile and Triton reuse kernels found here; the cache lives on local disk and travels as one tar."""
    os.makedirs(COMPILE_CACHE, exist_ok=True)
    for tar in cache_tars(num_gpus, staged):
        if os.path.exists(tar):
            subprocess.run(["tar", "-xf", tar, "-C", COMPILE_CACHE], check=True)
            print(f"compile cache restored from {tar} ({os.path.getsize(tar) / 2**30:.1f} GiB)")
            break
    return dict(TORCHINDUCTOR_CACHE_DIR=f"{COMPILE_CACHE}/inductor", TRITON_CACHE_DIR=f"{COMPILE_CACHE}/triton")


def save_compile_cache(num_gpus: int, staged: bool = False):
    tar = cache_tars(num_gpus, staged)[0]
    subprocess.run(["tar", "-cf", tar + ".tmp", "-C", COMPILE_CACHE, "."], check=True)
    os.replace(tar + ".tmp", tar)


def run_training(num_gpus: int, refresh_compile_cache: bool, extra_env: dict | None = None):
    """Checkpoints and logs are written straight to the Volume. The compile cache is only re-saved when
    it is missing or `refresh_compile_cache` is set, because packing ~10 GiB takes minutes.
    `extra_env` selects what the trainer does (see finetune.py for the chat-tuning switches)."""
    import time
    started = time.time()
    print(f"[timing] container entered at epoch {started:.1f}")
    link_runs()
    before = set(os.listdir(f"{VOL}/runs"))
    subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv"], check=True)
    has_own_cache = os.path.exists(cache_tars(num_gpus)[0])
    # Step times differ between hosts with the same GPUs, so record which machine this is.
    with open("/proc/cpuinfo") as f:
        cpu_model = next((line.split(":", 1)[1].strip() for line in f if line.startswith("model name")), "unknown")
    print(f"[host] {cpu_model}, {len(os.sched_getaffinity(0))} usable cores, load {os.getloadavg()[0]:.1f}, "
          f"{os.environ.get('MODAL_CLOUD_PROVIDER', '?')} {os.environ.get('MODAL_REGION', '?')}")
    print(f"[timing] links and nvidia-smi {time.time() - started:.1f}s")
    restore_started = time.time()
    cache_env = restore_compile_cache(num_gpus)
    print(f"[timing] compile cache restore {time.time() - restore_started:.1f}s")
    env = dict(os.environ, SAVE_CHECKPOINT="1", COMPILE_DIAGNOSTICS="1", PYTHONUNBUFFERED="1", **cache_env, **(extra_env or {}))
    print(f"[timing] setup {time.time() - started:.1f}s")
    trainer_started = time.time()
    # Every trainer line is prefixed with the seconds since this function started, measured here in
    # the container, so the stages can be timed from the log without touching the trainer.
    trainer = subprocess.Popen(["torchrun", "--standalone", f"--nproc_per_node={num_gpus}", "train_gpt.py"], cwd=REPO, env=env,
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, errors="replace")
    for line in trainer.stdout:
        print(f"[{time.time() - started:7.1f}s] {line}", end="", flush=True)
    returncode = trainer.wait()
    print(f"[timing] trainer process {time.time() - trainer_started:.1f}s")
    print(f"[timing] job so far {time.time() - started:.1f}s")
    if returncode == 0 and (refresh_compile_cache or not has_own_cache):
        cache_started = time.time()
        save_compile_cache(num_gpus)
        print(f"[timing] compile cache save {time.time() - cache_started:.1f}s")
    commit_started = time.time()
    volume.commit()
    print(f"[timing] volume commit {time.time() - commit_started:.1f}s")
    saved = sorted(set(os.listdir(f"{VOL}/runs")) - before)
    print("saved to volume under runs/:", saved)
    print(f"[timing] container total {time.time() - started:.1f}s, leaving at epoch {time.time():.1f}")
    if returncode != 0:
        raise RuntimeError(f"torchrun exited with {returncode}")
    return [n for n in saved if not n.endswith(".txt")]


@app.function(image=image, volumes={VOL: volume}, gpu="H100:8", timeout=3600, cpu=32, memory=131072)
def train(refresh_compile_cache: bool = False):
    run_ids = run_training(8, refresh_compile_cache)
    publish_latest("pretrained", run_ids[-1])
    prune_runs()
    return run_ids


@app.function(image=image, volumes={VOL: volume}, gpu="H200:4", timeout=3600, cpu=16, memory=65536)
def train4(refresh_compile_cache: bool = False):
    """Same run on 4 GPUs: each step accumulates 2 mini-batches. Pinned to H200s, which train about 30% faster
    than H100s here and keep the compile cache (which is specific to the GPU model) warm."""
    run_ids = run_training(4, refresh_compile_cache)
    publish_latest("pretrained", run_ids[-1])
    prune_runs()
    return run_ids


def tune_env(data: str, steps: int, lr: float, seed_tokens: bool, fineweb_val: bool, samples: bool) -> dict:
    return dict(TUNE_DATA=data, TUNE_STEPS=str(steps), TUNE_LR=str(lr), TUNE_SEED_TOKENS=str(int(seed_tokens)),
                TUNE_FINEWEB_VAL=str(int(fineweb_val)), TUNE_SAMPLES=str(int(samples)))


def tune_results(run_ids: list[str]) -> dict:
    import json
    run_id = run_ids[-1]
    with open(f"{VOL}/runs/{run_id}/tune.json") as f:
        return dict(run_id=run_id, **json.load(f))


@app.function(image=image, volumes={VOL: volume}, gpu="H200:4", timeout=1800, cpu=16, memory=65536)
def tune4(base_run_id: str = BASE_RUN_ID, data: str = "short", steps: int = 240, lr: float = 0.25,
          seed_tokens: bool = True, fineweb_val: bool = False, samples: bool = True,
          refresh_compile_cache: bool = False, publish: bool = True, name: str = "", dataset: str = ""):
    """Chat-tune a saved 4-GPU base model. Saves runs/<new id>/state_tuned.pt and returns the run's numbers.
    `publish` makes it the tuned run other services serve; experiments turn that off. `name` names
    the model and `dataset` is extra training data as JSON text (chat_data.clean_dataset); the
    run's numbers then include the model's replies to the dataset's questions."""
    env = dict(FINETUNE_FROM=checkpoint_path(base_run_id, "state_step"), TUNE_NAME=name,
               **tune_env(data, steps, lr, seed_tokens, fineweb_val, samples))
    if dataset:
        env["TUNE_DATASET"] = "/tmp/tune_dataset.json"
        with open(env["TUNE_DATASET"], "w") as f:
            f.write(dataset)
    result = tune_results(run_training(4, refresh_compile_cache, env))
    if publish:
        publish_tuned(result, base_run_id)
    prune_runs()
    return result


@app.function(image=image, volumes={VOL: volume}, gpu="H200:4", timeout=3600, cpu=16, memory=65536)
def train_and_tune4(data: str = "short", steps: int = 240, lr: float = 0.25, seed_tokens: bool = True,
                    fineweb_val: bool = False, samples: bool = True, refresh_compile_cache: bool = False):
    """Pretrain from scratch, then chat-tune, in one job. The run folder gets both checkpoints."""
    env = dict(FINETUNE_AFTER="1", **tune_env(data, steps, lr, seed_tokens, fineweb_val, samples))
    result = tune_results(run_training(4, refresh_compile_cache, env))
    publish_tuned(result, result["run_id"])
    prune_runs()
    return result


# What LiveRun's trainer is doing, for displays: {"snapshot": {...}} as described in live_session.py.
# Reading it costs nothing, whereas calling a LiveRun method starts the 4-GPU container.
live_state = modal.Dict.from_name("nanogpt-live", create_if_missing=True)
LIVE_IDLE_SECONDS = 60  # how long LiveRun keeps its GPUs once nothing holds them, at about $0.30 a minute
SLOW_LOAD_SECONDS = 360  # a trainer that took longer than this to load was compiling, not reading the cache
# Four H200s train about 30% faster, but there are evenings when none can be had for most of an hour
# while four H100s arrive in seconds. Modal takes the first of these it can get.
LIVE_GPUS = ["H200:4", "H100:4"]
WATCH_SECONDS = 5        # how often LiveRun looks for a release, and for a newer deployment
SUPERSEDED_CHECKS = 6    # checks in a row that must find a newer deployment before LiveRun gives way


@app.cls(image=image, volumes={VOL: volume}, gpu=LIVE_GPUS, cpu=16, memory=65536,
         timeout=4 * 3600, scaledown_window=LIVE_IDLE_SECONDS, max_containers=1)
@modal.concurrent(max_inputs=16)
class LiveRun:
    """One held 4-GPU container for a live demonstration: it loads the trainer ahead of time, then
    pretrains, serves the model and chat-tunes on cue, all in one process (see live_trainer.py), so
    nothing queues for GPUs or loads compiled code while people are watching.

    From another service, with the app deployed:

        run = modal.Cls.from_name("nanogpt-live", "LiveRun")()
        run.hold.spawn(45)               # get the GPUs and keep them 45 minutes; the trainer loads by itself
        run.request.remote("pretrain")   # start now, or as soon as the trainer has loaded
        run.request.remote("tune")
        run.release.remote()             # give the GPUs back

    Progress, samples and what can be queried are in the Dict `nanogpt-live` under "snapshot".
    live.py does all of this from a terminal."""

    @modal.enter()
    def prepare(self):
        import sys
        sys.path.insert(0, REPO)
        from live_session import LiveSession
        gpus = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], capture_output=True, text=True)
        self.session = LiveSession(4, publish=lambda snapshot: live_state.put("snapshot", snapshot),
                                   on_ready=self.loaded, on_finished=self.finished,
                                   details=dict(gpu=gpus.stdout.strip().split("\n")[0]))
        self.handled_release = self.released_at()
        threading.Thread(target=self.watch, daemon=True).start()
        self.session.announce("restoring compiled code")
        link_runs()
        cache_env = restore_compile_cache(4, staged=True)
        self.session.start_trainer(dict(os.environ, SAVE_CHECKPOINT="1", PYTHONUNBUFFERED="1", TUNE_SAMPLES="0", **cache_env))

    def loaded(self):
        """The compile cache is saved after the first load, and again after any slow one: a slow load
        means something was compiled afresh, as on the first run on a different GPU model."""
        if not os.path.exists(cache_tars(4, staged=True)[0]) or self.session.snapshot["loading"]["seconds"] > SLOW_LOAD_SECONDS:
            save_compile_cache(4, staged=True)
            volume.commit()
            print("[live] compile cache saved", flush=True)

    def finished(self, phase: str, run_id: str, results: dict | None):
        volume.commit()
        publish_latest("pretrained" if phase == "pretrain" else "tuned", run_id)
        prune_runs()

    def released_at(self):
        """When the lease in the Dict was given up, or None while it is being held."""
        lease = live_state.get("lease") or {}
        return lease.get("at") if lease.get("until") == 0 else None

    def watch(self):
        """Ends the hold when the lease in the Dict is given up, or when the app has been deployed
        again. After a deployment Modal sends every call to the new version, which cannot start
        while this container keeps the only place: a `release` call would never arrive, so the
        release is read from the Dict instead, and this container gives way by itself."""
        superseded = 0
        while True:
            time.sleep(WATCH_SECONDS)
            try:
                released = self.released_at()
                if released is not None and released != self.handled_release:
                    self.handled_release = released
                    print("[live] the lease was released", flush=True)
                    self.session.release()
                # The deployed version has no container while this one runs: this one is the old version.
                deployed = modal.Cls.from_name(app.name, "LiveRun")().hold.get_current_stats()
                superseded = superseded + 1 if deployed.num_total_runners == 0 else 0
                if superseded == SUPERSEDED_CHECKS:
                    print("[live] the app was deployed again; giving the GPUs to the new version", flush=True)
                    self.session.release()
            except Exception as error:
                print(f"[live] could not check the lease: {error!r}", flush=True)

    @modal.method()
    def hold(self, minutes: float = 45):
        """Keep the GPUs for at least this many more minutes. Call with `.spawn`: it returns when the
        lease ends, or at once if an earlier call is already keeping it."""
        self.handled_release = self.released_at()  # a release from before this call is not for it
        self.session.hold(minutes)

    @modal.method()
    def release(self):
        """End the lease. The container stops about a minute later. Writing a lease with "until": 0
        to the Dict does the same within seconds and, unlike this call, cannot start a container
        or wait behind a deployment; live.py and salt release that way."""
        self.session.release()

    @modal.method()
    def request(self, action: str, options: dict | None = None):
        """`action` is "pretrain", "tune" or "restart" (a fresh trainer, to run again). Asked for early,
        an action is taken as soon as the trainer reaches it. "tune" takes {"steps", "lr", "data",
        "name", "dataset"}; the last is extra training data, described at chat_data.clean_dataset.
        Returns the snapshot."""
        return self.session.request(action, options)

    @modal.method()
    async def generate_stream(self, prompt: str, max_new_tokens: int = 120, temperature: float = 0.8, top_k: int = 50, stop: str = ""):
        """Text completion from the pretrained model in the trainer's memory, as Inference.generate_stream."""
        request = dict(prompt=prompt, max_new_tokens=max_new_tokens, temperature=temperature, top_k=top_k, stop=stop)
        async for chunk in stream_from_server(request, self.session.serving_port("base")):
            yield chunk

    @modal.method()
    async def chat_stream(self, messages: list[dict], max_new_tokens: int = 120, temperature: float = 0.6, top_k: int = 20):
        """A reply from the tuned model in the trainer's memory, as ChatInference.chat_stream."""
        request = dict(messages=messages, max_new_tokens=max_new_tokens, temperature=temperature, top_k=top_k)
        async for chunk in stream_from_server(request, self.session.serving_port("chat")):
            yield chunk

    @modal.exit()
    def stop(self):
        self.session.shut_down()


@app.function(image=image, volumes={VOL: volume}, gpu="H100", timeout=1800, cpu=8, memory=32768)
def chat_eval(run_id: str):
    """One GPU: run the fixed chat evaluation on a tuned run and save it as chat_eval.json beside the checkpoint."""
    import json
    link_data()
    env = dict(os.environ, GENERATE_FROM=checkpoint_path(run_id, "state_tuned"), CHAT_EVAL="1")
    subprocess.run(["torchrun", "--standalone", "--nproc_per_node=1", "train_gpt.py"], cwd=REPO, env=env, check=True)
    volume.commit()
    with open(f"{VOL}/runs/{run_id}/chat_eval.json") as f:
        return json.load(f)


@app.function(image=image, volumes={VOL: volume}, gpu="H100", timeout=3600, cpu=8, memory=32768)
def warmup_diagnostic():
    """One GPU: run only the trainer's compile-and-warmup phase and print where the time goes."""
    import time
    link_data()
    cache_dir, cache_tar = "/root/compile_cache_1gpu", f"{VOL}/compile_cache_1gpu.tar"
    os.makedirs(cache_dir, exist_ok=True)
    cache_was_warm = os.path.exists(cache_tar)
    if cache_was_warm:
        subprocess.run(["tar", "-xf", cache_tar, "-C", cache_dir], check=True)
    subprocess.run(f"du -sh {cache_dir}; find {cache_dir} -type f | wc -l", shell=True)
    print(f"[timing] compile cache {'restored' if cache_was_warm else 'empty'}")
    env = dict(os.environ, COMPILE_DIAGNOSTICS="1", EXIT_AFTER_WARMUP="1",
               TORCHINDUCTOR_CACHE_DIR=f"{cache_dir}/inductor", TRITON_CACHE_DIR=f"{cache_dir}/triton")
    trainer_started = time.time()
    result = subprocess.run(["torchrun", "--standalone", "--nproc_per_node=1", "train_gpt.py"], cwd=REPO, env=env)
    print(f"[timing] trainer process {time.time() - trainer_started:.0f}s, exit {result.returncode}")
    tar_started = time.time()
    subprocess.run(["tar", "-cf", cache_tar, "-C", cache_dir, "."], check=True)
    volume.commit()
    print(f"[timing] cache save {time.time() - tar_started:.0f}s")


# Experiments keep their runs, results and compile caches under exp/ on the Volume, apart from runs/,
# so they are never pruned, never published, and never make prune_runs() delete a run that is served.
EXP = f"{VOL}/exp"
EXP_CONFIG_MINUTES = 40  # a config that runs longer than this is stopped; a cold compile fits


def exp_cache_tars(gpu: str) -> list[str]:
    """Compile caches an experiment may start from, best first; it saves only to the first."""
    return [f"{EXP}/compile_cache_{gpu}.tar", *cache_tars(4, staged=True)]


def run_trainer(env: dict, num_gpus: int, minutes: float, log_path: str) -> int:
    """Run the trainer with `env`, copying its output to the container's log and to `log_path`.
    Returns the exit code, or -9 if it had to be stopped after `minutes`."""
    started = time.time()
    trainer = subprocess.Popen(["torchrun", "--standalone", f"--nproc_per_node={num_gpus}", "train_gpt.py"], cwd=REPO, env=env,
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, errors="replace")
    watchdog = threading.Timer(minutes * 60, trainer.kill)
    watchdog.start()
    with open(log_path, "w") as log:
        for line in trainer.stdout:
            stamped = f"[{time.time() - started:7.1f}s] {line}"
            log.write(stamped)
            if not line.startswith("step:") or ":0/" in line or "val_loss" in line or line.split("/")[0].endswith("00"):
                print(stamped, end="", flush=True)
    code = trainer.wait()
    watchdog.cancel()
    return -9 if time.time() - started >= minutes * 60 else code


def log_numbers(log_path: str) -> dict:
    """The validation losses and timings a trainer log holds."""
    import re
    numbers = dict(val_loss={})
    with open(log_path) as f:
        for line in f:
            if found := re.search(r"step:(\d+)/(\d+) val_loss:([\d.]+) train_time:(\d+)ms", line):
                numbers["val_loss"][int(found[1])] = float(found[3])
                numbers.update(steps=int(found[2]), final_val_loss=float(found[3]), train_seconds=int(found[4]) / 1000)
            if found := re.search(r"\[\s*([\d.]+)s\] Resetting Model", line):
                numbers["loaded_seconds"] = float(found[1])
    return numbers


@app.function(image=image, volumes={VOL: volume}, gpu="H100:4", timeout=5 * 3600, cpu=16, memory=65536)
def experiment(name: str, configs: list[dict]):
    """Run trainer configurations one after another on one held 4-GPU container, so a batch waits
    for GPUs once. Each config is {"label", "env": {trainer environment}, "keep": bool}; "keep"
    leaves the run's checkpoints on the Volume. Each result is written to exp/results/<name>/<label>.json
    as soon as its config ends, with the benchmark replies if the config asked for them."""
    import json
    import shutil
    os.makedirs(f"{EXP}/runs", exist_ok=True)
    os.makedirs(f"{EXP}/results/{name}", exist_ok=True)
    link_data()
    if not os.path.exists(f"{REPO}/logs"):
        os.symlink(f"{EXP}/runs", f"{REPO}/logs")
    gpu = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], capture_output=True, text=True).stdout.split("\n")[0]
    gpu = gpu.replace("NVIDIA", "").strip().replace(" ", "_")
    os.makedirs(COMPILE_CACHE, exist_ok=True)
    tars = exp_cache_tars(gpu)
    restore_started = time.time()
    for tar in tars:
        if os.path.exists(tar):
            subprocess.run(["tar", "-xf", tar, "-C", COMPILE_CACHE], check=True)
            print(f"[exp] {gpu}: compile cache from {tar} in {time.time() - restore_started:.0f}s", flush=True)
            break
    cache_env = dict(TORCHINDUCTOR_CACHE_DIR=f"{COMPILE_CACHE}/inductor", TRITON_CACHE_DIR=f"{COMPILE_CACHE}/triton")
    summaries = []
    for config in configs:
        label = config["label"]
        before = set(os.listdir(f"{EXP}/runs"))
        env = dict(os.environ, SAVE_CHECKPOINT="0", PYTHONUNBUFFERED="1", TUNE_SAMPLES="0", TUNE_SAVE="0", CHAT_BENCH="1", **cache_env)
        if config.get("keep"):
            env.update(SAVE_CHECKPOINT="1", TUNE_SAVE="1")
        env.update({key: str(value) for key, value in config.get("env", {}).items()})
        started = time.time()
        log_path = f"/tmp/{label}.log"
        code = run_trainer(env, 4, config.get("minutes", EXP_CONFIG_MINUTES), log_path)
        result = dict(name=name, label=label, config=config, gpu=gpu, exit_code=code, seconds=round(time.time() - started, 1), **log_numbers(log_path))
        new = sorted(set(os.listdir(f"{EXP}/runs")) - before)
        run_ids = [entry for entry in new if not entry.endswith(".txt")]
        if run_ids:
            result["run_id"] = run_ids[-1]
            for part in ("tune", "bench"):
                path = f"{EXP}/runs/{run_ids[-1]}/{part}.json"
                if os.path.exists(path):
                    with open(path) as f:
                        result[part] = json.load(f)
        for entry in new:
            if entry.endswith(".txt"):
                result.setdefault("run_id", entry.removesuffix(".txt"))
                os.remove(f"{EXP}/runs/{entry}")  # the trainer's own log repeats its source; the timed copy is kept instead
        shutil.copy(log_path, f"{EXP}/results/{name}/{label}.log")
        with open(f"{EXP}/results/{name}/{label}.json", "w") as f:
            json.dump(result, f, indent=1)
        if code == 0 and (not os.path.exists(tars[0]) or result.get("loaded_seconds", 0) > SLOW_LOAD_SECONDS):
            subprocess.run(["tar", "-cf", tars[0] + ".tmp", "-C", COMPILE_CACHE, "."], check=True)
            os.replace(tars[0] + ".tmp", tars[0])
            print(f"[exp] compile cache saved to {tars[0]}", flush=True)
        volume.commit()
        summaries.append({key: value for key, value in result.items() if key not in ("tune", "bench", "config")})
        print(f"[exp] {label}: {summaries[-1]}", flush=True)
    return summaries


@app.function(image=image, volumes={VOL: volume}, gpu="H100", timeout=1800, cpu=8, memory=32768)
def bench(checkpoint: str, out: str, diag: bool = False, layout: str = "matched"):
    """One GPU: answer the benchmark prompts with a tuned checkpoint (a path on the Volume) and
    write the replies to exp/results/<out>.json."""
    link_data()
    os.makedirs(os.path.dirname(f"{EXP}/results/{out}"), exist_ok=True)
    env = dict(os.environ, GENERATE_FROM=checkpoint, CHAT_BENCH="1", BENCH_OUT=f"{EXP}/results/{out}.json", CHECK_VAL="0", BENCH_DIAG=str(int(diag)), BENCH_LAYOUT=layout)
    subprocess.run(["torchrun", "--standalone", "--nproc_per_node=1", "train_gpt.py"], cwd=REPO, env=env, check=True)
    volume.commit()


def lenient_pickle():
    """The checkpoint's optimizer state pickles classes defined in train_gpt.py; stub any class that is missing here."""
    import pickle
    import types

    class Unpickler(pickle.Unpickler):
        def find_class(self, module, name):
            try:
                return super().find_class(module, name)
            except (AttributeError, ModuleNotFoundError):
                return type(name, (), {})

    module = types.ModuleType("lenient_pickle")
    module.Unpickler = Unpickler
    module.load = lambda file, **kwargs: Unpickler(file, **kwargs).load()
    return module


@app.function(image=image, volumes={VOL: volume}, timeout=900, cpu=4, memory=16384)
def verify(run_id: str):
    """CPU-only: confirm the checkpoint loads and the bigram table has rows from every rank's shard."""
    import torch
    run_dir = f"{VOL}/runs/{run_id}"
    ckpt_name = sorted(f for f in os.listdir(run_dir) if f.endswith(".pt"))[-1]
    path = f"{run_dir}/{ckpt_name}"
    print(f"checkpoint: {path} ({os.path.getsize(path) / 2**30:.2f} GiB)")
    ckpt = torch.load(path, map_location="cpu", weights_only=False, pickle_module=lenient_pickle())
    state = ckpt["model"]
    print(f"step={ckpt['step']} tensors={len(state)} params={sum(v.numel() for v in state.values()):,}")
    for key, value in state.items():
        if key.endswith("bigram_embed.weight"):
            touched = (value.float().abs().sum(dim=1) > 0).float()
            print(f"{key}: shape={tuple(value.shape)} dtype={value.dtype}")
            print("non-zero row fraction per eighth of the table:",
                  [round(chunk.mean().item(), 3) for chunk in touched.chunk(8)])


@app.function(image=image, volumes={VOL: volume}, gpu="H100", timeout=1800, cpu=8, memory=32768)
def generate(run_id: str, prompts: str = "", max_new_tokens: int = 120, check_val: bool = True):
    """One GPU: load the checkpoint and sample text. `prompts` is a JSON list; empty uses the defaults."""
    link_data()
    run_dir = f"{VOL}/runs/{run_id}"
    ckpt_name = sorted(f for f in os.listdir(run_dir) if f.endswith(".pt"))[-1]
    env = dict(os.environ, GENERATE_FROM=f"{run_dir}/{ckpt_name}", PROMPTS=prompts,
               MAX_NEW_TOKENS=str(max_new_tokens), CHECK_VAL="1" if check_val else "0")
    subprocess.run(["torchrun", "--standalone", "--nproc_per_node=1", "train_gpt.py"], cwd=REPO, env=env, check=True)


def start_server(checkpoint: str):
    """The trainer script owns the model definition, so it runs as a subprocess in serve mode
    (generate.py) and the classes below forward requests to it."""
    import time
    import urllib.request
    env = dict(os.environ, GENERATE_FROM=checkpoint, SERVE_PORT=str(SERVE_PORT))
    server = subprocess.Popen(["torchrun", "--standalone", "--nproc_per_node=1", "train_gpt.py"], cwd=REPO, env=env)
    for _ in range(300):
        if server.poll() is not None:
            raise RuntimeError(f"inference server exited with {server.returncode}")
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{SERVE_PORT}/", timeout=2)
            return server
        except OSError:
            time.sleep(1)
    raise RuntimeError("inference server did not start in time")


async def stream_from_server(request: dict, port: int = SERVE_PORT):
    """Yields the server's JSON lines. Async so that a caller cancelling mid-stream closes the local
    connection (which stops the generation) instead of Modal terminating the container."""
    import asyncio
    import json
    body = json.dumps(request).encode()
    reader, writer = await asyncio.open_connection("127.0.0.1", port)
    try:
        writer.write(b"POST / HTTP/1.0\r\nContent-Type: application/json\r\nContent-Length: %d\r\n\r\n" % len(body) + body)
        await writer.drain()
        while (await reader.readline()).strip():
            pass  # skip the response headers
        while line := await reader.readline():
            yield json.loads(line)
    finally:
        writer.close()


INFERENCE_IDLE_SECONDS = 300  # how long an Inference or ChatInference container outlives its last request


async def report_inference(kind: str, run_id: str, up: bool = True):
    """Tell displays that the 1-GPU container for `kind` ("base" or "chat") is up and was just used,
    or has stopped. Stored in the live Dict as inference_<kind>: {"up", "run_id", "at"}. A display
    treats a container as gone INFERENCE_IDLE_SECONDS after `at`, which also covers one that was
    killed before it could say so."""
    try:
        await live_state.put.aio(f"inference_{kind}", dict(up=up, run_id=run_id, at=time.time()))
    except Exception as error:
        print(f"could not report the {kind} inference container: {error!r}", flush=True)


@app.cls(image=image, volumes={VOL: volume}, gpu="H100", cpu=8, memory=32768,
         scaledown_window=INFERENCE_IDLE_SECONDS, max_containers=1, timeout=600)
class Inference:
    """Text completion from a pretrained model:
    `modal.Cls.from_name("nanogpt-live", "Inference")(run_id=...).generate_stream.remote_gen(prompt)`.
    `run_id` picks which pretraining run to serve (see `latest_runs`); left empty it serves
    SERVED_RUN_ID. Scales to zero after 5 idle minutes."""

    run_id: str = modal.parameter(default="")

    @modal.enter()
    async def start(self):
        self.server = start_server(checkpoint_path(self.run_id or SERVED_RUN_ID, "state_step"))
        await report_inference("base", self.run_id or SERVED_RUN_ID)

    @modal.method()
    async def generate_stream(self, prompt: str, max_new_tokens: int = 120, temperature: float = 0.8, top_k: int = 50, stop: str = ""):
        """Yields {"text": piece} per token, then {"seconds": total}. Call with `.remote_gen(...)`."""
        request = dict(prompt=prompt, max_new_tokens=max_new_tokens, temperature=temperature, top_k=top_k, stop=stop)
        async for chunk in stream_from_server(request):
            yield chunk
        await report_inference("base", self.run_id or SERVED_RUN_ID)

    @modal.exit()
    async def stop(self):
        self.server.terminate()
        await report_inference("base", self.run_id or SERVED_RUN_ID, up=False)


@app.cls(image=image, volumes={VOL: volume}, gpu="H100", cpu=8, memory=32768,
         scaledown_window=INFERENCE_IDLE_SECONDS, max_containers=1, timeout=600)
class ChatInference:
    """Chat with a tuned model:
    `modal.Cls.from_name("nanogpt-live", "ChatInference")(run_id=...).chat_stream.remote_gen(messages)`.
    `run_id` picks which tuned run to serve (see `latest_runs`), so a run that has just finished can
    be used without a redeploy; left empty it serves SERVED_CHAT_RUN_ID. Scales to zero after 5 idle minutes."""

    run_id: str = modal.parameter(default="")

    @modal.enter()
    async def start(self):
        self.server = start_server(checkpoint_path(self.run_id or SERVED_CHAT_RUN_ID, "state_tuned"))
        await report_inference("chat", self.run_id or SERVED_CHAT_RUN_ID)

    @modal.method()
    async def chat_stream(self, messages: list[dict], max_new_tokens: int = 120, temperature: float = 0.6, top_k: int = 20):
        """`messages` is [{"role": "user" | "assistant", "content": str}, ...] ending on a user turn.
        Yields {"text": piece} per token, then {"seconds": total, "stop": "end" | "length"}."""
        request = dict(messages=messages, max_new_tokens=max_new_tokens, temperature=temperature, top_k=top_k)
        async for chunk in stream_from_server(request):
            yield chunk
        await report_inference("chat", self.run_id or SERVED_CHAT_RUN_ID)

    @modal.exit()
    async def stop(self):
        self.server.terminate()
        await report_inference("chat", self.run_id or SERVED_CHAT_RUN_ID, up=False)


@app.local_entrypoint()
def main():
    run_dirs = train4.remote()
    print("run dirs:", run_dirs)
    for run_id in run_dirs:
        verify.remote(run_id)


@app.local_entrypoint()
def sweep(configs: str, out: str = ""):
    """Run several tune4 jobs at once. `configs` is a JSON list of tune4 keyword dicts; each result is
    printed as one JSON line and, if `out` is given, appended to that local file. The runs are
    experiments, so none of them becomes the served model."""
    import json
    settings = json.loads(configs)
    calls = [tune4.spawn(**setting, publish=False) for setting in settings]
    for setting, call in zip(settings, calls):
        try:
            result = call.get()
        except Exception as error:
            result = dict(config=setting, error=repr(error))
        line = json.dumps(result)
        print(line[:400])
        if out:
            with open(out, "a") as f:
                f.write(line + "\n")



@app.local_entrypoint()
def exp(name: str, configs: str):
    """Run a batch of experiments: `configs` is a JSON file holding a list of experiment configs."""
    import json
    with open(configs) as f:
        settings = json.load(f)
    for summary in experiment.remote(name, settings):
        print(json.dumps(summary))
