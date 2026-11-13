"""Run the speedrun trainer on Modal with 8 H100s and keep the checkpoint on a Volume.

  uvx modal run modal_run.py            # prep (CPU) -> train (8xH100) -> verify (CPU)
  uvx modal run modal_run.py::verify --run-id <id>
  uvx modal run modal_run.py::generate --run-id <id> [--prompts '["Hello"]']
  uvx modal deploy modal_run.py         # deploys the Inference class for other services to call
"""
import os
import shutil
import subprocess

import modal

REPO = "/root/modded-nanogpt"
VOL = "/vol"
SERVED_RUN_ID = "b8ff02bd-65e1-4593-98b7-34d91e5255e4"  # the checkpoint the deployed Inference class loads
SERVE_PORT = 8000

app = modal.App("nanogpt-speedrun")
volume = modal.Volume.from_name("nanogpt-speedrun", create_if_missing=True)

image = (
    modal.Image.from_registry("nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04", add_python="3.12")
    .entrypoint([])
    .apt_install("build-essential", "git")
    .pip_install("torch==2.10.0")  # the default wheel is cu128, which the record logs ran on
    .pip_install("numpy", "tqdm", "huggingface-hub", "kernels", "setuptools",
                 "typing-extensions==4.15.0", "datasets", "tiktoken")
    .env({"HF_HOME": f"{VOL}/hf"})
    .add_local_dir(".", REPO, ignore=[".git", "records", "img", "logs", "data/fineweb10B", "__pycache__"])
)


def link_data():
    """The trainer reads data/fineweb10B relative to the repo; keep the shards on the Volume."""
    os.makedirs(f"{VOL}/fineweb10B", exist_ok=True)
    link = f"{REPO}/data/fineweb10B"
    if not os.path.exists(link):
        os.symlink(f"{VOL}/fineweb10B", link)


@app.function(image=image, volumes={VOL: volume}, timeout=1800, cpu=4, memory=8192)
def prep():
    """CPU-only: download the data shards and the attention kernel so GPU time is not spent on it."""
    link_data()
    subprocess.run(["python", "data/cached_fineweb10B.py", "9"], cwd=REPO, check=True)
    from kernels import get_kernel
    get_kernel("kernels-community/flash-attn3", version=1)
    volume.commit()
    print("prep done:", sorted(os.listdir(f"{VOL}/fineweb10B")))


@app.function(image=image, volumes={VOL: volume}, gpu="H100:8", timeout=1800, cpu=32, memory=131072)
def train():
    link_data()
    subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv"], check=True)
    env = dict(os.environ, SAVE_CHECKPOINT="1")
    result = subprocess.run(["torchrun", "--standalone", "--nproc_per_node=8", "train_gpt.py"], cwd=REPO, env=env)
    os.makedirs(f"{VOL}/runs", exist_ok=True)
    saved = []
    if os.path.isdir(f"{REPO}/logs"):
        for name in os.listdir(f"{REPO}/logs"):
            src, dst = f"{REPO}/logs/{name}", f"{VOL}/runs/{name}"
            if os.path.isdir(src):
                shutil.copytree(src, dst, dirs_exist_ok=True)
            else:
                shutil.copy(src, dst)
            saved.append(name)
    volume.commit()
    print("saved to volume under runs/:", saved)
    if result.returncode != 0:
        raise RuntimeError(f"torchrun exited with {result.returncode}")
    return [n for n in saved if not n.endswith(".txt")]


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


@app.cls(image=image, volumes={VOL: volume}, gpu="H100", cpu=8, memory=32768,
         scaledown_window=300, max_containers=1, timeout=600)
class Inference:
    """Deployed text generation: `modal.Cls.from_name("nanogpt-speedrun", "Inference")().generate_stream.remote_gen(prompt)`.

    The trainer script owns the model definition, so it runs as a subprocess in serve mode and this
    class forwards requests to it. Scales to zero after 5 idle minutes.
    """

    @modal.enter()
    def start(self):
        import time
        import urllib.request
        run_dir = f"{VOL}/runs/{SERVED_RUN_ID}"
        ckpt_name = sorted(f for f in os.listdir(run_dir) if f.endswith(".pt"))[-1]
        env = dict(os.environ, GENERATE_FROM=f"{run_dir}/{ckpt_name}", SERVE_PORT=str(SERVE_PORT))
        self.server = subprocess.Popen(["torchrun", "--standalone", "--nproc_per_node=1", "train_gpt.py"], cwd=REPO, env=env)
        for _ in range(300):
            if self.server.poll() is not None:
                raise RuntimeError(f"inference server exited with {self.server.returncode}")
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{SERVE_PORT}/", timeout=2)
                return
            except OSError:
                time.sleep(1)
        raise RuntimeError("inference server did not start in time")

    @modal.method()
    async def generate_stream(self, prompt: str, max_new_tokens: int = 120, temperature: float = 0.8, top_k: int = 50, stop: str = ""):
        """Yields {"text": piece} per token, then {"seconds": total}. Call with `.remote_gen(...)`.

        Async so that a caller cancelling mid-stream closes the local connection (which stops the
        generation) instead of Modal terminating the container.
        """
        import asyncio
        import json
        body = json.dumps(dict(prompt=prompt, max_new_tokens=max_new_tokens, temperature=temperature, top_k=top_k, stop=stop)).encode()
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

    @modal.exit()
    def stop(self):
        self.server.terminate()


@app.local_entrypoint()
def main():
    prep.remote()
    run_dirs = train.remote()
    print("run dirs:", run_dirs)
    for run_id in run_dirs:
        verify.remote(run_id)
