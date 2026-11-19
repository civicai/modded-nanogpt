"""One trainer process driven step by step, for a live demonstration.

Runs inside train_gpt.py when LIVE_DIR is set. The process then:

  1. loads its compiled code as usual, plus what chat tuning and text sampling need,
  2. waits for the file LIVE_DIR/go_pretrain, then pretrains, sampling text as it learns,
  3. serves the pretrained model over local HTTP while it waits for LIVE_DIR/go_tune,
  4. chat-tunes (finetune.py), sampling replies as it learns,
  5. serves both models until the process is stopped.

Everything slow happens in step 1, before anyone is watching. Progress is printed as lines of
`[live] {json}`, which live_session.py turns into the state a display reads.

Environment:
    LIVE_DIR          folder watched for the go files; each holds a JSON object of options
    LIVE_SERVE_PORT   port for the model in the foreground (default 8000); the next port up serves
                      the pretrained model once tuning has replaced it
    LIVE_PROMPTS      JSON list of prompts sampled during pretraining, in place of the held-out pages
    LIVE_QUESTIONS    JSON list of questions sampled during chat tuning, in place of the held-out
                      conversations
"""
import glob
import json
import os
import time

import numpy as np
import tiktoken
import torch

import chat_data
from chat_format import ASSISTANT_ID, EOT_ID, USER_ID

import generate

LIVE_DIR = os.environ.get("LIVE_DIR", "")
ENABLED = bool(LIVE_DIR)
SERVE_PORT = int(os.environ.get("LIVE_SERVE_PORT", "8000"))
# What is sampled while training: a list of {"prompt", "real"}. The prompts and questions come from
# the held-out data the validation losses are measured on, each with the text that really follows
# it there as "real", so that a display can set what the model wrote beside it. They are read when
# pretraining starts (read_held_out). A prompt or question given in the environment has no "real".
PROMPTS = [dict(prompt=prompt) for prompt in json.loads(os.environ.get("LIVE_PROMPTS") or "[]")]
QUESTIONS = [dict(prompt=question) for question in json.loads(os.environ.get("LIVE_QUESTIONS") or "[]")]
# Which held-out pages and conversations, by their place in the first validation shard of each
# kind. Picked by hand for a general audience: short, plain, and nothing to wince at.
PAGES = [504, 189, 83]
CONVERSATIONS = [1568, 1812]
PROMPT_TOKENS = 14  # a page's prompt is at least this long, and ends where a word does
# Dense at the start, where the text changes fastest. The final step is always sampled.
PRETRAIN_SAMPLE_STEPS = {0, 10, 25, 50, 100, 150, 200, 300, 400, 500, 650, 800, 1000, 1150}
TUNE_SAMPLE_EVERY = 40
PROGRESS_EVERY = 10
SAMPLE_TOKENS = 24
REPLY_TOKENS = 40
TEMPERATURE = 0.8
TOP_K = 50
CHAT_TEMPERATURE = 0.6
CHAT_TOP_K = 20
POLL_SECONDS = 0.05
BASE_RANK = 1  # the rank that serves the pretrained model once rank 0 holds the tuned one
EXTRA_QUESTIONS = []  # asked beside QUESTIONS during chat tuning; they come with the run's extra training data


def emit(g, event, **fields):
    if g["master_process"]:
        g["print0"]("[live] " + json.dumps(dict(event=event, at=round(time.time(), 2), **fields)), console=True)


class eval_mode:
    """The uncompiled model in eval mode, restored to what it was on exit. Sampling runs on one rank
    while the others wait at their next collective call, so it must leave no trace."""

    def __init__(self, g):
        self.raw_model = g["model"]._orig_mod

    def __enter__(self):
        self.was_training = self.raw_model.training
        self.raw_model.eval()
        return self.raw_model

    def __exit__(self, *_):
        self.raw_model.train(self.was_training)


def readable(text):
    """An untrained model emits byte sequences that are not valid text; drop the placeholder they decode to."""
    return text.replace("\ufffd", "")


def documents(path, places):
    """The documents at `places` in a shard of tokens, each without the token that starts it."""
    tokens = chat_data.read_shard(path)
    starts = np.flatnonzero(tokens == EOT_ID)
    ends = np.append(starts[1:], len(tokens))
    return [tokens[starts[place] + 1:ends[place]].tolist() for place in places]


def page_samples(path, places, real_tokens):
    """The openings of the pages at `places` in a FineWeb shard, each with the page's own next
    `real_tokens` tokens. An opening ends where a word does, so that neither half starts mid-word."""
    encoder = tiktoken.get_encoding("gpt2")
    samples = []
    for page in documents(path, places):
        cut = PROMPT_TOKENS
        while cut < len(page) - 1 and not encoder.decode([page[cut]]).startswith(" "):
            cut += 1
        samples.append(dict(prompt=encoder.decode(page[:cut]), real=encoder.decode(page[cut:cut + real_tokens])))
    return samples


def conversation_samples(path, places, real_tokens):
    """The first question of the conversations at `places` in a chat shard, each with the start of
    the reply it got there."""
    encoder = tiktoken.get_encoding("gpt2")
    samples = []
    for place, tokens in zip(places, documents(path, places)):
        assert tokens[0] == USER_ID and ASSISTANT_ID in tokens, f"conversation {place} does not open with a question and a reply"
        reply_at = tokens.index(ASSISTANT_ID)
        reply = tokens[reply_at + 1:]
        reply = reply[:next((i for i, token in enumerate(reply) if token >= EOT_ID), len(reply))]
        samples.append(dict(prompt=encoder.decode(tokens[1:reply_at]).strip(), real=encoder.decode(reply[:real_tokens]).strip()))
    return samples


def identity_sample(name=""):
    """The first question the chat data teaches the model to answer about itself, with the first of
    the answers it is taught: the one that gives its name, if it has one."""
    questions, answers, named = chat_data.IDENTITY[0]
    return dict(prompt=questions[0], real=named[0].format(name=name) if name else answers[0])


def read_held_out(g):
    """Fill PROMPTS and QUESTIONS from the held-out data, unless the environment gave them."""
    import finetune
    if not PROMPTS:
        PROMPTS[:] = page_samples(sorted(glob.glob(g["args"].val_files))[0], PAGES, SAMPLE_TOKENS)
    if not QUESTIONS:
        shards = sorted(glob.glob(os.path.join(finetune.TuneConfig.from_env().data_dir, "val_short", "chat_val_*.bin")))
        QUESTIONS[:] = [identity_sample()] + (conversation_samples(shards[0], CONVERSATIONS, REPLY_TOKENS) if shards else [])


def sample_completions(g):
    forward_args = g["training_manager"].get_forward_args()
    with eval_mode(g) as raw_model:
        return [dict(sample, text=readable(generate.generate(g, raw_model, forward_args, sample["prompt"], SAMPLE_TOKENS, TEMPERATURE, TOP_K)))
                for sample in PROMPTS]


def sample_replies(g, forward_args):
    replies = []
    with eval_mode(g) as raw_model:
        for sample in QUESTIONS + [dict(prompt=question) for question in EXTRA_QUESTIONS]:
            pieces = generate.chat_reply(g, raw_model, forward_args, [dict(role="user", content=sample["prompt"])],
                                         REPLY_TOKENS, CHAT_TEMPERATURE, CHAT_TOP_K)
            replies.append(dict(sample, text=readable("".join(value for kind, value in pieces if kind == "text")).strip()))
    return replies


def warm_extra_graphs(g):
    """Called at the end of the trainer's warmup, before it restores the initial weights: compile
    what the later phases use, so that none of it is first met in front of an audience."""
    import finetune
    started = time.perf_counter()
    finetune.warm_graphs(g, finetune.TuneConfig.from_env())
    if g["rank"] in (0, BASE_RANK):
        forward_args = g["training_manager"].get_forward_args()
        with eval_mode(g) as raw_model:
            generate.warm_lengths(g, raw_model, forward_args, generate.LENGTH_BUCKETS)
    g["print0"](f"[live] extra graphs warmed in {time.perf_counter() - started:.1f}s", console=True)


def wait_for_go(g, name, server=None):
    """Block every rank until LIVE_DIR/go_<name> appears and return the options in it. The file must
    be put in place with a rename, so that no rank reads it half-written. `server`, if given, keeps
    answering requests during the wait."""
    path = os.path.join(LIVE_DIR, f"go_{name}")
    emit(g, "waiting", phase=name)
    if server:
        server.timeout = POLL_SECONDS
    while not os.path.exists(path):
        if server:
            server.handle_request()
        else:
            time.sleep(POLL_SECONDS)
    with open(path) as f:
        options = json.load(f)
    g["dist"].barrier()
    return options


def start_pretraining(g):
    if g["master_process"]:
        read_held_out(g)
    wait_for_go(g, "pretrain")
    if g["master_process"]:
        # Each rank made up its own run id; only rank 0's names the folder the run is saved in.
        with open(os.path.join(LIVE_DIR, "run_id"), "w") as f:
            f.write(g["args"].run_id)
    emit(g, "started", phase="pretrain", run_id=g["args"].run_id, total=g["training_schedule"].total_steps)


def validates(step):
    """Whether to measure the validation loss before this step. A live run measures it wherever it
    samples text, in place of the usual every 250 steps, so that each sample on a display has a loss
    and the steep early part of the curve is drawn."""
    return step in PRETRAIN_SAMPLE_STEPS


def on_pretrain_step(g, step, val_loss=None):
    """Called once per step, after that step's validation if it had one and before its update."""
    if not g["master_process"]:
        return
    total = g["training_schedule"].total_steps
    if val_loss is not None:
        emit(g, "val", phase="pretrain", step=step, loss=round(float(val_loss), 4))
    if step in PRETRAIN_SAMPLE_STEPS or step == total:
        emit(g, "sample", phase="pretrain", step=step, items=sample_completions(g))
    elif step % PROGRESS_EVERY == 0:
        emit(g, "step", phase="pretrain", step=step, total=total)


def on_tune_step(g, step, total, forward_args):
    """Called before tuning step `step`, and once more with step == total when tuning has finished."""
    if not g["master_process"]:
        return
    if step % TUNE_SAMPLE_EVERY == 0 or step == total:
        emit(g, "sample", phase="tune", step=step, items=sample_replies(g, forward_args))
    elif step % PROGRESS_EVERY == 0:
        emit(g, "step", phase="tune", step=step, total=total)


def checkpoint_file(prefix):
    with open(os.path.join(LIVE_DIR, "run_id")) as f:
        run_dir = f"logs/{f.read()}"
    return f"{run_dir}/{sorted(name for name in os.listdir(run_dir) if name.startswith(prefix))[-1]}"


def serve_until_tuning(g):
    """Pretraining has just finished and the model is in the state its final validation ran in.
    Rank 0 answers generation requests with it until tuning is asked for. Returns the tuning options."""
    emit(g, "finished", phase="pretrain", run_id=g["args"].run_id)
    if not g["master_process"]:
        return wait_for_go(g, "tune")
    with eval_mode(g) as raw_model:
        server = generate.make_server(g, raw_model, g["training_manager"].get_forward_args(), SERVE_PORT)
        emit(g, "serving", base=SERVE_PORT)
        options = wait_for_go(g, "tune", server)
        server.server_close()
    emit(g, "serving")
    return options


def serve_forever(g, tune_results):
    """Tuning has finished. Rank 0 serves the tuned model; BASE_RANK puts the pretrained weights
    back and serves those; the other ranks idle. Never returns."""
    manager, rank = g["training_manager"], g["rank"]
    emit(g, "finished", phase="tune", run_id=g["args"].run_id,
         results={key: value for key, value in tune_results.items() if key not in ("samples", "dataset")})
    if rank not in (0, BASE_RANK):
        while True:
            time.sleep(3600)
    manager.apply_final_ws_ext()
    raw_model = g["model"]._orig_mod
    if rank == BASE_RANK:
        checkpoint = torch.load(checkpoint_file("state_step"), map_location="cuda", weights_only=False)
        g["model"].load_state_dict(checkpoint["model"])
        del checkpoint
    raw_model.eval()
    port = SERVE_PORT + (rank == BASE_RANK)
    server = generate.make_server(g, raw_model, manager.get_forward_args(), port)
    # Rank 0 announces both servers, so it waits for the other one to be listening.
    if rank == 0:
        wait_for_port(SERVE_PORT + 1)
        emit(g, "serving", base=SERVE_PORT + 1, chat=SERVE_PORT)
    server.serve_forever()


def wait_for_port(port, timeout=120):
    import socket
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            socket.create_connection(("127.0.0.1", port), timeout=1).close()
            return
        except OSError:
            time.sleep(0.2)
    raise RuntimeError(f"nothing is listening on port {port}")


def finish(g):
    """Everything after pretraining: serve, tune when asked, then serve for good."""
    import finetune
    options = serve_until_tuning(g)
    config = finetune.TuneConfig.from_env()
    for key in ("data", "steps", "lr"):
        if key in options:
            setattr(config, key, type(getattr(config, key))(options[key]))
    if "name" in options:
        config.name = chat_data.clean_name(options["name"])
    if "dataset" in options:
        config.dataset = chat_data.clean_dataset(options["dataset"])
    # The answer the model is taught to "who are you" depends on its name.
    QUESTIONS[:] = [identity_sample(config.name) if sample["prompt"] == identity_sample()["prompt"] else sample for sample in QUESTIONS]
    if config.dataset:
        # A display tells a sample's questions apart by their text, so none is asked twice.
        asked = {sample["prompt"] for sample in QUESTIONS}
        EXTRA_QUESTIONS[:] = [question for question in dict.fromkeys(config.dataset["questions"]) if question not in asked]
    emit(g, "started", phase="tune", run_id=g["args"].run_id, total=config.steps, name=config.name,
         dataset=chat_data.dataset_summary(config.dataset))
    results = finetune.tune(g, config)
    serve_forever(g, results)
