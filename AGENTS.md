# AGENTS.md

CivAI's private fork (`civicai/modded-nanogpt`) of KellerJordan/modded-nanogpt. `master` mirrors upstream — never modify it.
- Nothing goes to the upstream repo: no pushes, PRs, issues or comments there. Push only to this fork.

## Running
- No local GPU. All training/eval runs on Modal via `modal run modal_run.py::<fn>`; the local working tree is uploaded, so uncommitted edits are what run.
- Persistent state lives on Volume `nanogpt-speedrun`: data shards, compile caches, `runs/<run_id>/` + `runs/<run_id>.txt` logs. Inspect with `modal volume ls` / `modal volume get`.
- Baseline pretrain: `::main` (4xH200 via `train4`, then CPU `verify`). Reference val loss 3.2789.

## Don't break the live demo
- `tune4` publishes by default and `train_and_tune4` always does — publishing changes what `Inference`/`ChatInference`/`live.py` serve. For experiments use `tune4 --no-publish` or `::sweep`.
- Never `modal deploy` from an experiment.
- Every run calls `prune_runs()`, which keeps only the 10 newest checkpointed runs (plus pinned/served ones). Many experiments in a row will delete older checkpoints — pin or copy anything worth keeping first.

## Experiments
- `modal run modal_run.py::exp --name <batch> --configs <file.json>` runs trainer configurations back to back on one 4-GPU container (one wait for GPUs). It never publishes or prunes; runs, results and its compile caches live under `exp/` on the Volume. A config is `{"label", "env": {trainer environment}, "keep": bool}`.
- Chat quality is measured, not eyeballed: `CHAT_BENCH=1` (set by `::exp`) or `::bench` writes replies to `evals/chat_bench.json`'s prompts, and `python chat_judge.py <file>` grades them (needs `OPENROUTER_API_KEY`). Compare "good" (sensible and correct); runs differ by ±3 points, so use two pretraining seeds and `BENCH_SAMPLES=4` before believing a gap.
- For experiments build tuning data with `chat_data.py --hold-out-benchmark`, into a folder under `exp/`; the mixes in `chat/` are the ones that are served and keep the benchmark's questions.
- A "H100" request may be given H200s; `gpu` in each result says which. Step times also differ by up to 50% between hosts of the same GPU.

## Cost & speed
- Costs: each function's GPUs are in its `@app.function`/`@app.cls` decorator in `modal_run.py`; `modal billing rates` gives current per-GPU prices, and `modal billing summary` shows this month's spend. Estimate a run's cost before launching it.
- Cold torch.compile is 18–20 min; keep compile caches warm (don't pass `--refresh-compile-cache` without reason).
- Single runs are noisy: fix `TRAIN_SEED` when comparing, and report results with run ids.
