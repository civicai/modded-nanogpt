"""Run the speedrun trainer on Modal, keep its checkpoints on a Volume, and chat-tune them.

  uvx modal run modal_run.py::prep            # once: FineWeb shards and attention kernel onto the Volume (CPU)
  uvx modal run modal_run.py::prep_chat       # once: pre-tokenized chat shards onto the Volume (CPU)
  uvx modal run modal_run.py                  # train4 (4xH200) -> verify (CPU)
  uvx modal run modal_run.py::tune4           # chat-tune a saved base model (4xH200, a few minutes)
  uvx modal run modal_run.py::train_and_tune4 # pretrain, then chat-tune, in one job
  uvx modal run modal_run.py::chat_eval --run-id <id>
  uvx modal run modal_run.py::generate --run-id <id> [--prompts '["Hello"]']
  uvx modal deploy modal_run.py               # deploys Inference and ChatInference for other services to call
"""
import os
import subprocess

import modal

REPO = "/root/modded-nanogpt"
VOL = "/vol"
SERVED_RUN_ID = "b8ff02bd-65e1-4593-98b7-34d91e5255e4"  # the checkpoint the deployed Inference class loads
BASE_RUN_ID = "037b2c75-2815-426c-9820-56f3e7df87e7"    # the 4xH200 pretraining run that tune4 starts from
SERVED_CHAT_RUN_ID = ""                                 # the tuned run ChatInference loads when given no run id
SERVE_PORT = 8000

# This branch deploys under its own name so the app serving the base model is left alone.
app = modal.App("nanogpt-instruct")
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


def checkpoint_path(run_id: str, name_prefix: str) -> str:
    """The run's last checkpoint whose file name starts with `name_prefix` ("state_step" is the
    pretrained model, "state_tuned" the chat-tuned one)."""
    run_dir = f"{VOL}/runs/{run_id}"
    names = sorted(f for f in os.listdir(run_dir) if f.startswith(name_prefix) and f.endswith(".pt"))
    if not names:
        raise FileNotFoundError(f"no {name_prefix}*.pt in {run_dir}")
    return f"{run_dir}/{names[-1]}"


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


def cache_tars(num_gpus: int) -> list[str]:
    """Where a compile cache may be found, best first. This branch saves to the first (its own tar,
    which also holds the chat-tuning graphs) and only reads the others, which belong to the base runs."""
    own = f"{VOL}/compile_cache_{num_gpus}gpu_instruct.tar"
    base = f"{VOL}/compile_cache.tar" if num_gpus == 8 else f"{VOL}/compile_cache_{num_gpus}gpu.tar"
    return [own, base, f"{VOL}/compile_cache.tar"]


def restore_compile_cache(num_gpus: int) -> dict:
    """torch.compile and Triton reuse kernels found here; the cache lives on local disk and travels as one tar."""
    os.makedirs(COMPILE_CACHE, exist_ok=True)
    for tar in cache_tars(num_gpus):
        if os.path.exists(tar):
            subprocess.run(["tar", "-xf", tar, "-C", COMPILE_CACHE], check=True)
            print(f"compile cache restored from {tar}")
            break
    return dict(TORCHINDUCTOR_CACHE_DIR=f"{COMPILE_CACHE}/inductor", TRITON_CACHE_DIR=f"{COMPILE_CACHE}/triton")


def save_compile_cache(num_gpus: int):
    tar = cache_tars(num_gpus)[0]
    subprocess.run(["tar", "-cf", tar + ".tmp", "-C", COMPILE_CACHE, "."], check=True)
    os.replace(tar + ".tmp", tar)


def run_training(num_gpus: int, refresh_compile_cache: bool, extra_env: dict | None = None):
    """Checkpoints and logs are written straight to the Volume. The compile cache is only re-saved when
    it is missing or `refresh_compile_cache` is set, because packing ~10 GiB takes minutes.
    `extra_env` selects what the trainer does (see finetune.py for the chat-tuning switches)."""
    import time
    started = time.time()
    link_data()
    os.makedirs(f"{VOL}/runs", exist_ok=True)
    if not os.path.exists(f"{REPO}/logs"):
        os.symlink(f"{VOL}/runs", f"{REPO}/logs")
    before = set(os.listdir(f"{VOL}/runs"))
    subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv"], check=True)
    has_own_cache = os.path.exists(cache_tars(num_gpus)[0])
    env = dict(os.environ, SAVE_CHECKPOINT="1", COMPILE_DIAGNOSTICS="1", **restore_compile_cache(num_gpus), **(extra_env or {}))
    print(f"[timing] setup {time.time() - started:.0f}s")
    trainer_started = time.time()
    result = subprocess.run(["torchrun", "--standalone", f"--nproc_per_node={num_gpus}", "train_gpt.py"], cwd=REPO, env=env)
    print(f"[timing] trainer process {time.time() - trainer_started:.0f}s")
    print(f"[timing] job so far {time.time() - started:.0f}s")
    if result.returncode == 0 and (refresh_compile_cache or not has_own_cache):
        cache_started = time.time()
        save_compile_cache(num_gpus)
        print(f"[timing] compile cache save {time.time() - cache_started:.0f}s")
    commit_started = time.time()
    volume.commit()
    print(f"[timing] volume commit {time.time() - commit_started:.0f}s")
    saved = sorted(set(os.listdir(f"{VOL}/runs")) - before)
    print("saved to volume under runs/:", saved)
    if result.returncode != 0:
        raise RuntimeError(f"torchrun exited with {result.returncode}")
    return [n for n in saved if not n.endswith(".txt")]


@app.function(image=image, volumes={VOL: volume}, gpu="H100:8", timeout=3600, cpu=32, memory=131072)
def train(refresh_compile_cache: bool = False):
    return run_training(8, refresh_compile_cache)


@app.function(image=image, volumes={VOL: volume}, gpu="H200:4", timeout=3600, cpu=16, memory=65536)
def train4(refresh_compile_cache: bool = False):
    """Same run on 4 GPUs: each step accumulates 2 mini-batches. Pinned to H200s, which train about 30% faster
    than H100s here and keep the compile cache (which is specific to the GPU model) warm."""
    return run_training(4, refresh_compile_cache)


def tune_env(data: str, steps: int, lr: float, seed_tokens: bool, fineweb_val: bool, samples: bool) -> dict:
    return dict(TUNE_DATA=data, TUNE_STEPS=str(steps), TUNE_LR=str(lr), TUNE_SEED_TOKENS=str(int(seed_tokens)),
                TUNE_FINEWEB_VAL=str(int(fineweb_val)), TUNE_SAMPLES=str(int(samples)))


def tune_results(run_ids: list[str]) -> dict:
    import json
    run_id = run_ids[-1]
    with open(f"{VOL}/runs/{run_id}/tune.json") as f:
        return dict(run_id=run_id, **json.load(f))


@app.function(image=image, volumes={VOL: volume}, gpu="H200:4", timeout=1800, cpu=16, memory=65536)
def tune4(base_run_id: str = BASE_RUN_ID, data: str = "simple", steps: int = 200, lr: float = 0.15,
          seed_tokens: bool = True, fineweb_val: bool = False, samples: bool = True,
          refresh_compile_cache: bool = False):
    """Chat-tune a saved 4-GPU base model. Saves runs/<new id>/state_tuned.pt and returns the run's numbers."""
    env = dict(FINETUNE_FROM=checkpoint_path(base_run_id, "state_step"),
               **tune_env(data, steps, lr, seed_tokens, fineweb_val, samples))
    return tune_results(run_training(4, refresh_compile_cache, env))


@app.function(image=image, volumes={VOL: volume}, gpu="H200:4", timeout=3600, cpu=16, memory=65536)
def train_and_tune4(data: str = "simple", steps: int = 200, lr: float = 0.15, seed_tokens: bool = True,
                    fineweb_val: bool = False, samples: bool = True, refresh_compile_cache: bool = False):
    """Pretrain from scratch, then chat-tune, in one job. The run folder gets both checkpoints."""
    env = dict(FINETUNE_AFTER="1", **tune_env(data, steps, lr, seed_tokens, fineweb_val, samples))
    return tune_results(run_training(4, refresh_compile_cache, env))


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


async def stream_from_server(request: dict):
    """Yields the server's JSON lines. Async so that a caller cancelling mid-stream closes the local
    connection (which stops the generation) instead of Modal terminating the container."""
    import asyncio
    import json
    body = json.dumps(request).encode()
    reader, writer = await asyncio.open_connection("127.0.0.1", SERVE_PORT)
    try:
        writer.write(b"POST / HTTP/1.0\r\nContent-Type: application/json\r\nContent-Length: %d\r\n\r\n" % len(body) + body)
        await writer.drain()
        while (await reader.readline()).strip():
            pass  # skip the response headers
        while line := await reader.readline():
            yield json.loads(line)
    finally:
        writer.close()


@app.cls(image=image, volumes={VOL: volume}, gpu="H100", cpu=8, memory=32768,
         scaledown_window=300, max_containers=1, timeout=600)
class Inference:
    """Text completion from the base model:
    `modal.Cls.from_name("nanogpt-instruct", "Inference")().generate_stream.remote_gen(prompt)`.
    Scales to zero after 5 idle minutes."""

    @modal.enter()
    def start(self):
        self.server = start_server(checkpoint_path(SERVED_RUN_ID, "state_step"))

    @modal.method()
    async def generate_stream(self, prompt: str, max_new_tokens: int = 120, temperature: float = 0.8, top_k: int = 50, stop: str = ""):
        """Yields {"text": piece} per token, then {"seconds": total}. Call with `.remote_gen(...)`."""
        request = dict(prompt=prompt, max_new_tokens=max_new_tokens, temperature=temperature, top_k=top_k, stop=stop)
        async for chunk in stream_from_server(request):
            yield chunk

    @modal.exit()
    def stop(self):
        self.server.terminate()


@app.cls(image=image, volumes={VOL: volume}, gpu="H100", cpu=8, memory=32768,
         scaledown_window=300, max_containers=1, timeout=600)
class ChatInference:
    """Chat with a tuned model:
    `modal.Cls.from_name("nanogpt-instruct", "ChatInference")(run_id=...).chat_stream.remote_gen(messages)`.
    `run_id` picks which tuned run to serve, so a run that has just finished can be used without a
    redeploy; left empty it serves SERVED_CHAT_RUN_ID. Scales to zero after 5 idle minutes."""

    run_id: str = modal.parameter(default="")

    @modal.enter()
    def start(self):
        self.server = start_server(checkpoint_path(self.run_id or SERVED_CHAT_RUN_ID, "state_tuned"))

    @modal.method()
    async def chat_stream(self, messages: list[dict], max_new_tokens: int = 120, temperature: float = 0.6, top_k: int = 20):
        """`messages` is [{"role": "user" | "assistant", "content": str}, ...] ending on a user turn.
        Yields {"text": piece} per token, then {"seconds": total, "stop": "end" | "length"}."""
        request = dict(messages=messages, max_new_tokens=max_new_tokens, temperature=temperature, top_k=top_k)
        async for chunk in stream_from_server(request):
            yield chunk

    @modal.exit()
    def stop(self):
        self.server.terminate()


@app.local_entrypoint()
def main():
    run_dirs = train4.remote()
    print("run dirs:", run_dirs)
    for run_id in run_dirs:
        verify.remote(run_id)


@app.local_entrypoint()
def sweep(configs: str, out: str = ""):
    """Run several tune4 jobs at once. `configs` is a JSON list of tune4 keyword dicts; each result is
    printed as one JSON line and, if `out` is given, appended to that local file."""
    import json
    settings = json.loads(configs)
    calls = [tune4.spawn(**setting) for setting in settings]
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

