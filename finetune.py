"""Chat tuning: a short second training phase that teaches the model to hold a conversation.

Runs inside train_gpt.py, which owns the model, optimizer and data loader, in one of two ways:

    FINETUNE_FROM=<state_step*.pt> torchrun ... train_gpt.py    # tune a saved base model
    FINETUNE_AFTER=1 torchrun ... train_gpt.py                  # pretrain, then tune, in one job

Both run the same phase: reset the optimizer's running statistics, give the three chat tokens a
starting point, then take a few hundred ordinary training steps on pre-tokenized conversations
(see chat_data.py and chat_format.py). The result is saved as state_tuned.pt beside the run's log.

The defaults are the ones the graded benchmark (chat_bench.py, chat_judge.py) chose: the "plain"
mix of short, direct exchanges, at the pretraining learning rate. The model takes most of what it
can say about the world from these conversations, so they matter more than anything else here.

Environment:
    TUNE_DATA         mix name under the chat data folder (default "plain")
    TUNE_NAME         a name for the model; the mix's identity conversations are rewritten to say it
    TUNE_DATASET      path of a JSON file of extra training data to add to the mix (chat_data.clean_dataset)
    TUNE_HIDDEN       path of a second such file, added to the mix without being named in the run's results
    TUNE_ASK          a JSON list of questions to put to the tuned model; the results hold its replies as "asked".
                      A list of questions in place of one is a conversation, asked a turn at a time
    TUNE_ASK_TEMPERATURES  sampling temperatures to ask them at, separated by commas (default: the usual one)
    TUNE_STEPS        optimizer steps; even, so the last step updates the Adam parameters (default 720)
    TUNE_LR           learning rate as a multiple of each parameter's pretraining base rate (default 1.0)
    TUNE_USER_WEIGHT  weight of the loss on tokens outside the assistant's replies; 0 learns only to reply (default 1)
    TUNE_SEED_TOKENS  "0" leaves the chat tokens' rows as pretraining left them (default "1")
    TUNE_FINEWEB_VAL  "1" also measures FineWeb validation loss before and after (default "0")
    TUNE_SAVE         "0" skips saving state_tuned.pt, for experiments that only need the numbers (default "1")
    CHAT_BENCH        "1" answers the benchmark prompts with the tuned model (chat_bench.py) into bench.json
    TUNE_SAMPLES      "1" prints sample conversations from the tuned model at the end (default "1")
    CHAT_DATA_DIR     folder holding the mixes and val/ (default "data/chat")
"""
import json
import os
import tempfile
import time
from dataclasses import asdict, dataclass

import tiktoken
import torch

import live_trainer
from chat_format import SEED_TOKENS, assistant_target_mask

BATCH_TOKENS = 8 * 2048 * 8  # per optimizer step, the same as the first pretraining stage
DOC_CAP = 2048               # conversations per rank per batch; they are far shorter than web pages
VAL_BATCHES = 4              # under one pass of the smaller held-out set on 4 GPUs
VAL_FOLDERS = ("val", "val_short", "val_plain")
MUON_MOMENTUM = 0.85         # where pretraining's momentum schedule starts and ends
WARMUP_FRACTION = 0.05
FINAL_LR_FRACTION = 0.1


@dataclass
class TuneConfig:
    data: str = "plain"
    steps: int = 720
    lr: float = 1.0
    seed_tokens: bool = True
    fineweb_val: bool = False
    samples: bool = True
    data_dir: str = "data/chat"
    name: str = ""
    user_weight: float = 1.0
    dataset: dict | None = None  # extra training data, as chat_data.clean_dataset returns it
    # More of the same, which no display is told about: results and snapshots never mention it.
    hidden: dict | None = None

    @classmethod
    def from_env(cls):
        env = os.environ
        def dataset_at(variable):
            if not env.get(variable):
                return None
            import chat_data
            with open(env[variable]) as f:
                return chat_data.clean_dataset(json.load(f))
        return cls(dataset=dataset_at("TUNE_DATASET"), hidden=dataset_at("TUNE_HIDDEN"), data=env.get("TUNE_DATA", cls.data), steps=int(env.get("TUNE_STEPS", cls.steps)),
                   lr=float(env.get("TUNE_LR", cls.lr)), seed_tokens=env.get("TUNE_SEED_TOKENS", "1") == "1",
                   fineweb_val=env.get("TUNE_FINEWEB_VAL", "0") == "1", samples=env.get("TUNE_SAMPLES", "1") == "1",
                   data_dir=env.get("CHAT_DATA_DIR", cls.data_dir), name=env.get("TUNE_NAME", cls.name),
                   user_weight=float(env.get("TUNE_USER_WEIGHT", cls.user_weight)))

    def learning_rate(self, step: int) -> float:
        """Linear warmup, then linear decay. The warmup matters because the optimizer starts with no
        gradient statistics, and its first steps would otherwise be its largest."""
        warmup_steps = max(2, int(self.steps * WARMUP_FRACTION))
        warmup = min(1.0, (step + 1) / warmup_steps)
        progress = step / max(1, self.steps - 1)
        return self.lr * warmup * (1 - (1 - FINAL_LR_FRACTION) * progress)


def load_base(g, path):
    """Put the model in the state pretraining left it in. The attention windows and rotary tables
    are not in the checkpoint, so they are rebuilt by replaying the schedule."""
    from generate import trim_padding
    model, manager = g["model"], g["training_manager"]
    for step in range(g["training_schedule"].total_steps + 1):
        manager.advance_schedule(step)
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(trim_padding(checkpoint["model"], model.state_dict()))
    model.quantize_attn_fp8()
    model.quantize_mlp_fp8(update_activation_scales=False)


def fresh_optimizer_state(optimizer):
    """Start the phase with no momentum or gradient statistics. A base checkpoint only holds rank 0's
    share of that state, so resetting is the one behaviour both entry points can share."""
    optimizer.reset()
    optimizer.split_embed = True  # reset() re-ties embed to lm_head, which would overwrite the embedding
    for param, config in optimizer.param_cfgs.items():
        if config.optim == "adam":
            state = optimizer.param_states[param]
            state["step"] = 0
            state["exp_avg"].zero_()
            state["exp_avg_sq"].zero_()


@torch.no_grad()
def seed_chat_tokens(raw_model):
    """Give each chat token the average input, output and value vectors of related existing tokens."""
    encode = tiktoken.get_encoding("gpt2").encode_ordinary
    vocab = raw_model.vocab_size
    assert raw_model.lm_head.weight.shape[1] == vocab and raw_model.embed.weight.shape[0] == vocab
    value_embeds = raw_model.value_embeds.view(-1, vocab, raw_model.value_embeds.shape[-1])
    for token, seeds in SEED_TOKENS.items():
        ids = sorted({t for seed in seeds for t in (encode(seed) if isinstance(seed, str) else [seed])})
        source = torch.tensor(ids, device=raw_model.embed.weight.device)
        raw_model.embed.weight[token] = raw_model.embed.weight[source].float().mean(0)
        raw_model.lm_head.weight[:, token] = raw_model.lm_head.weight[:, source].float().mean(1)
        value_embeds[:, token] = value_embeds[:, source].float().mean(1)


def train_folder(g, config):
    """The folder of training shards. A model with a name or extra training data trains on a copy
    of the mix with those written in, which rank 0 writes to local disk while the other ranks wait."""
    if not config.name and not config.dataset and not config.hidden:
        return config.data
    import chat_data
    folder = os.path.join(tempfile.gettempdir(), "chat_tuned_mix")
    if g["master_process"]:
        started = time.perf_counter()
        counts = chat_data.write_tuned_mix(os.path.join(config.data_dir, config.data), folder, config.name, config.dataset, config.hidden)
        g["print0"](f"[tune] wrote the mix in {time.perf_counter() - started:.1f}s: the name {config.name!r} in {counts['named']} "
                    f"conversations, {counts['added']} examples added", console=True)
    g["dist"].barrier()
    return folder


def chat_loader(g, config, folder, prefix):
    """`folder` is under the chat data folder, or an absolute path."""
    pattern = os.path.join(config.data_dir, folder, f"{prefix}_*.bin")
    return g["distributed_data_generator"](pattern, BATCH_TOKENS, g["TRAINING_STAGES"][-1].train_max_seq_len,
                                           grad_accum_steps=g["grad_accum_steps"], doc_cap=DOC_CAP, repeat=True)


@torch.no_grad()
def chat_val_loss(g, config, forward_args):
    """Held-out loss per token for each held-out set chat_data.py wrote: over every token ("all"),
    and over assistant replies only ("replies")."""
    model, dist = g["model"], g["dist"]
    results = {}
    model.eval()
    for folder in VAL_FOLDERS:
        if not os.path.isdir(os.path.join(config.data_dir, folder)):
            continue
        loader = chat_loader(g, config, folder, "chat_val")
        totals = torch.zeros(4, device=g["device"], dtype=torch.float64)
        for _ in range(VAL_BATCHES):
            inputs, targets, cum_seqlens, bigram_inputs, _ = next(loader)
            losses = model(inputs, targets, cum_seqlens, bigram_inputs, forward_args).double()
            replies = assistant_target_mask(inputs)
            totals += torch.stack([losses.sum(), losses.new_tensor(len(losses)), (losses * replies).sum(), replies.sum().double()])
        dist.all_reduce(totals)
        results[folder] = dict(all=round((totals[0] / totals[1]).item(), 4), replies=round((totals[2] / totals[3]).item(), 4))
    model.train()
    return results


@torch.no_grad()
def fineweb_val_loss(g):
    """The pretraining validation loss, to see how much general ability tuning cost. No tokenizer
    mask is applied, so compare it with another call of this function, not with a training log."""
    model, manager, args, dist = g["model"], g["training_manager"], g["args"], g["dist"]
    windows = (manager.ws_short, manager.ws_long)
    manager.apply_final_ws_ext()
    forward_args = manager.get_forward_args()
    val_steps = g["grad_accum_steps"] * args.val_tokens // args.val_batch_size
    loader = g["distributed_data_generator"](args.val_files, args.val_batch_size, -1,
                                             grad_accum_steps=g["grad_accum_steps"], align_to_bos=False)
    total = torch.zeros((), device=g["device"])
    model.eval()
    for _ in range(val_steps):
        inputs, targets, cum_seqlens, bigram_inputs, _ = next(loader)
        total += model(inputs, targets, cum_seqlens, bigram_inputs, forward_args).mean()
    model.train()
    manager.ws_short, manager.ws_long = windows
    dist.all_reduce(total)
    return (total / val_steps / dist.get_world_size()).item()


def phase_forward_args(g):
    """One fixed configuration for the whole phase: the last pretraining stage's windows and loss."""
    manager, stage, schedule = g["training_manager"], g["TRAINING_STAGES"][-1], g["training_schedule"]
    manager.ws_short, manager.ws_long = stage.window_sizes
    manager.train_max_seq_len = stage.train_max_seq_len
    manager.mtp_weights, manager.prefix_weight = schedule.mtp_weights[-1], schedule.prefix_weights[-1]
    return manager.get_forward_args()


def warm_graphs(g, config):
    """Run the passes tuning uses once, so their compiled code is loaded before tuning is asked for.
    Leaves gradients behind: the caller is expected to reset the model afterwards."""
    model = g["model"]
    forward_args = phase_forward_args(g)
    model.train()
    loader = chat_loader(g, config, config.data, "chat_train")
    for _ in range(g["grad_accum_steps"]):
        inputs, targets, cum_seqlens, bigram_inputs, _ = next(loader)
        losses = model(inputs, targets, cum_seqlens, bigram_inputs, forward_args)
        (losses.sum() * g["grad_scale"]).backward()
    chat_val_loss(g, config, forward_args)


def tune(g, config):
    print0, model, manager, dist = g["print0"], g["model"], g["training_manager"], g["dist"]
    optimizer, grad_accum_steps, grad_scale = manager.optimizer, g["grad_accum_steps"], g["grad_scale"]
    assert config.steps % 2 == 0, "an even step count ends on an Adam update"
    phase_started = time.perf_counter()
    results = dict(config={key: value for key, value in asdict(config).items() if key != "hidden"})
    if config.dataset:
        # The whole dataset is saved with the run; what is printed and shown only names it.
        import chat_data
        results["dataset"], results["config"]["dataset"] = config.dataset, chat_data.dataset_summary(config.dataset)
    forward_args = phase_forward_args(g)

    model.zero_grad(set_to_none=True)
    fresh_optimizer_state(optimizer)
    if config.seed_tokens:
        seed_chat_tokens(model._orig_mod)
    model.train()

    if config.fineweb_val:
        results["fineweb_before"] = fineweb_val_loss(g)
    results["chat_before"] = chat_val_loss(g, config, forward_args)
    print0(f"[tune] before: chat val loss {results['chat_before']} setup {time.perf_counter() - phase_started:.1f}s", console=True)

    loader = chat_loader(g, config, train_folder(g, config), "chat_train")
    torch.cuda.synchronize()
    train_started = time.perf_counter()
    for step in range(config.steps):
        if live_trainer.ENABLED:
            live_trainer.on_tune_step(g, step, config.steps, forward_args)
        learning_rate = config.learning_rate(step)
        for param_config in optimizer.param_cfgs.values():
            param_config.lr = param_config.initial_lr * learning_rate
            if param_config.optim == "normuon":
                param_config.momentum = MUON_MOMENTUM
        for _ in range(grad_accum_steps):
            inputs, targets, cum_seqlens, bigram_inputs, bigram_cpu = next(loader)
            manager.sparse_index_update(step, bigram_cpu)
            losses = model(inputs, targets, cum_seqlens, bigram_inputs, forward_args)
            manager.sparse_index_share(step)
            if config.user_weight != 1.0:
                losses = losses * torch.where(assistant_target_mask(inputs), 1.0, config.user_weight)
            (losses.sum() * grad_scale).backward()
        optimizer.step(do_adam=step % 2 == 1)
        model.quantize_attn_fp8()
        model.quantize_mlp_fp8()
        if step == 0:
            torch.cuda.synchronize()
            results["first_step_seconds"] = time.perf_counter() - train_started
        if (step + 1) % 10 == 0 or step == 0:
            loss = losses.detach().mean().item()
            print0(f"[tune] step:{step + 1}/{config.steps} loss:{loss:.4f} "
                   f"lr:{learning_rate:.4f} time:{time.perf_counter() - train_started:.1f}s", console=True)
            if live_trainer.ENABLED:
                live_trainer.emit(g, "loss", phase="tune", step=step + 1, loss=round(loss, 4))
    torch.cuda.synchronize()
    results["train_seconds"] = time.perf_counter() - train_started
    if live_trainer.ENABLED:
        live_trainer.on_tune_step(g, config.steps, config.steps, forward_args)

    results["chat_after"] = chat_val_loss(g, config, forward_args)
    if config.fineweb_val:
        results["fineweb_after"] = fineweb_val_loss(g)
    results["phase_seconds"] = time.perf_counter() - phase_started
    print0(f"[tune] after: chat val loss {results['chat_after']} training {results['train_seconds']:.1f}s "
           f"of which first step {results['first_step_seconds']:.1f}s", console=True)

    if g["master_process"]:
        run_dir = f"logs/{g['args'].run_id}"
        os.makedirs(run_dir, exist_ok=True)
        save_started = time.perf_counter()
        if os.environ.get("TUNE_SAVE", "1") == "1":
            torch.save(dict(step=config.steps, model=model.state_dict(), tune=results), f"{run_dir}/state_tuned.pt")
        results["save_seconds"] = time.perf_counter() - save_started
        if os.environ.get("CHAT_BENCH") == "1":
            import chat_bench
            manager.apply_final_ws_ext()
            model._orig_mod.eval()
            chat_bench.run(g, model._orig_mod, manager.get_forward_args(), f"{run_dir}/bench.json")
        if config.samples:
            import chat_eval
            manager.apply_final_ws_ext()
            raw_model = model._orig_mod
            raw_model.eval()
            results["samples"] = chat_eval.evaluate(g, raw_model, manager.get_forward_args())
            if config.dataset:
                # Whether the extra training data took hold: a few replies to each of its questions.
                results["dataset_replies"] = [
                    dict(prompt=question, reply=chat_eval.reply_to(g, raw_model, manager.get_forward_args(), [dict(role="user", content=question)])[0])
                    for question in config.dataset["questions"] for _ in range(chat_eval.SAMPLES)]
                for item in results["dataset_replies"]:
                    print0(f"[tune] {item['prompt']!r} -> {item['reply']!r}", console=True)
            temperatures = [float(value) for value in os.environ.get("TUNE_ASK_TEMPERATURES", str(chat_eval.TEMPERATURE)).split(",")]
            def replies_to(question, temperature):
                """The model's replies to a question, or to each turn of a conversation given its own earlier replies."""
                history, replies = [], []
                for turn in question if isinstance(question, list) else [question]:
                    history.append(dict(role="user", content=turn))
                    replies.append(chat_eval.reply_to(g, raw_model, manager.get_forward_args(), history, temperature)[0])
                    history.append(dict(role="assistant", content=replies[-1]))
                return replies if isinstance(question, list) else replies[0]
            results["asked"] = [dict(prompt=question, temperature=temperature, reply=replies_to(question, temperature))
                                for temperature in temperatures for question in json.loads(os.environ.get("TUNE_ASK") or "[]")]
        with open(f"{run_dir}/tune.json", "w") as f:
            json.dump(results, f, indent=1)
        summary = {k: v for k, v in results.items() if k not in ("samples", "dataset")}
        print0(f"[tune result] {json.dumps(summary)}", console=True)
    return results


def run(g):
    """Entry point for FINETUNE_FROM: tune a saved base checkpoint."""
    started = time.perf_counter()
    load_base(g, os.environ["FINETUNE_FROM"])
    g["print0"](f"[tune] base checkpoint loaded in {time.perf_counter() - started:.1f}s", console=True)
    tune(g, TuneConfig.from_env())


def run_after_training(g):
    """Entry point for FINETUNE_AFTER: the model in memory has just finished pretraining."""
    tune(g, TuneConfig.from_env())
