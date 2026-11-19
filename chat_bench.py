"""Replies to a fixed set of prompts (evals/chat_bench.json), for grading by chat_judge.py.

The prompts are questions and remarks a general audience might type, none taken from training
data: easy facts, common sense, small sums, simple instructions, small talk, questions about the
model itself, and a few eight-turn conversations. Sampling matches the live demo's defaults and
is seeded, so two checkpoints are asked the same things in the same way.

Conversations are packed side by side into one sequence, as documents are in training, and every
forward pass extends all of them by one token.

Environment:
    BENCH_LAYOUT  "matched" (default) or "alone": each prompt first in a sequence of its own
    BENCH_SAMPLES how many times each prompt is asked (default 2)
    BENCH_TEMPERATURE, BENCH_TOP_K  sampling settings, in place of the live demo's 0.6 and 20
    BENCH_DIAG    "1" only reports how much a prompt's place in the sequence changes its answer
"""
import json
import os
import time

import tiktoken
import torch

import generate
from chat_format import ASSISTANT_ID, CHAT_VOCAB, END_ID, EOT_ID, USER_ID, render_conversation, render_prompt

BENCH_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "evals", "chat_bench.json")
SAMPLES = int(os.environ.get("BENCH_SAMPLES", "2"))
TEMPERATURE = float(os.environ.get("BENCH_TEMPERATURE", "0.6"))
TOP_K = int(os.environ.get("BENCH_TOP_K", "20"))
MAX_NEW_TOKENS = 120
SEQUENCE_TOKENS = generate.LENGTH_BUCKETS[-1]
MAX_DOCS = 100
SEED = 20261119
STOP_TOKENS = (END_ID, EOT_ID, USER_ID, ASSISTANT_ID)
LAYOUT = os.environ.get("BENCH_LAYOUT", "matched")
# In training every conversation follows another one, so its <eot> comes straight after an <end>.
# A conversation placed first in a sequence has nothing before its <eot>, which the model has
# hardly ever seen. LEAD is a short conversation put in front so that the first real one starts
# the way they all did in training.
LEAD = render_conversation([dict(role="user", content="Hi"), dict(role="assistant", content="Hello!")],
                           tiktoken.get_encoding("gpt2").encode_ordinary)


@torch.no_grad()
def sequence_logits(g, raw_model, forward_args, tokens, ends):
    """Next-token logits at the positions `ends` of one sequence."""
    inputs = torch.tensor(tokens + [generate.PAD_ID] * (SEQUENCE_TOKENS - len(tokens)), dtype=torch.int32)
    logits = raw_model(inputs.cuda(), None, generate.document_boundaries(inputs).cuda(), g["get_bigram_hash"](inputs).cuda(), forward_args)
    return logits[torch.as_tensor(ends).cuda(), :CHAT_VOCAB].float()


def packed_logits(g, raw_model, forward_args, docs):
    """Next-token logits for each document in `docs` (token lists that each start with <eot>).
    Each is followed by an <end>, which it cannot see, so that the next one starts as in training."""
    flat, ends = list(LEAD), []
    for doc in docs:
        flat += doc
        ends.append(len(flat) - 1)
        flat.append(END_ID)
    return sequence_logits(g, raw_model, forward_args, flat, ends)


def packing_report(g, raw_model, forward_args, docs):
    """Print how far apart the sampling distributions are for a document in different places."""
    def distance(a, b):
        return round(0.5 * (torch.softmax(a / TEMPERATURE, -1) - torch.softmax(b / TEMPERATURE, -1)).abs().sum().item(), 3)
    packed = packed_logits(g, raw_model, forward_args, docs)
    reversed_pack = packed_logits(g, raw_model, forward_args, docs[::-1]).flip(0)
    rows = []
    for place, doc in enumerate(docs):
        alone = sequence_logits(g, raw_model, forward_args, doc, [len(doc) - 1])[0]
        led = sequence_logits(g, raw_model, forward_args, LEAD + doc, [len(LEAD) + len(doc) - 1])[0]
        after_end = sequence_logits(g, raw_model, forward_args, [END_ID] + doc, [len(doc)])[0]
        rows.append((place, distance(alone, led), distance(led, packed[place]), distance(packed[place], reversed_pack[place]), distance(led, after_end)))
    g["print0"](f"[bench] total variation (place, alone vs led, led vs packed, packed vs reversed, led vs after a bare <end>): {rows}", console=True)


def chunks(prompts):
    """Indexes of `prompts` in groups that fit one sequence with room for every reply."""
    group, used = [], len(LEAD)
    for index, prompt in enumerate(prompts):
        need = len(prompt) + MAX_NEW_TOKENS + 1
        if group and (used + need > SEQUENCE_TOKENS or len(group) == MAX_DOCS):
            yield group
            group, used = [], len(LEAD)
        group.append(index)
        used += need
    if group:
        yield group


@torch.no_grad()
def replies(g, raw_model, forward_args, conversations, sampler):
    """[(reply text, "end" | "length")] for conversations that each end on a user turn."""
    encoder = tiktoken.get_encoding("gpt2")
    limit = SEQUENCE_TOKENS - MAX_NEW_TOKENS - len(LEAD) - 1
    prompts = [render_prompt(messages, encoder.encode_ordinary)[-limit:] for messages in conversations]
    prompts = [[EOT_ID, *prompt[1:]] for prompt in prompts]
    results = [None] * len(prompts)
    for group in (chunks(prompts) if LAYOUT == "matched" else [[index] for index in range(len(prompts))]):
        docs = {index: list(prompts[index]) for index in group}
        for _ in range(MAX_NEW_TOKENS):
            live = sorted(docs)
            if LAYOUT == "matched":
                logits = packed_logits(g, raw_model, forward_args, [docs[index] for index in live])
            else:
                logits = generate.next_token_logits(g, raw_model, forward_args, docs[live[0]], vocab=CHAT_VOCAB)[None]
            top_values, top_indices = torch.topk(logits / TEMPERATURE, TOP_K, dim=-1)
            choices = torch.multinomial(torch.softmax(top_values, dim=-1), 1, generator=sampler)
            tokens = top_indices.gather(1, choices)[:, 0].tolist()
            for index, token in zip(live, tokens):
                if token in STOP_TOKENS:
                    results[index] = (encoder.decode(docs.pop(index)[len(prompts[index]):]).strip(), "end")
                else:
                    docs[index].append(token)
            if not docs:
                break
        for index, doc in docs.items():
            results[index] = (encoder.decode(doc[len(prompts[index]):]).strip(), "length")
    return results


def run(g, raw_model, forward_args, out_path):
    """Ask every prompt SAMPLES times and write the transcripts to `out_path`."""
    print0 = g["print0"]
    started = time.perf_counter()
    with open(BENCH_FILE) as f:
        bench = json.load(f)
    sampler = torch.Generator(device="cuda").manual_seed(SEED)
    threads = [dict(item=item, sample=sample, history=[]) for item in bench["items"] for sample in range(SAMPLES)]
    if os.environ.get("BENCH_DIAG") == "1":
        encoder = tiktoken.get_encoding("gpt2")
        packing_report(g, raw_model, forward_args, [render_prompt([dict(role="user", content=thread["item"]["turns"][0])], encoder.encode_ordinary)
                                                    for thread in threads[:40:2]])
        return None
    transcripts = []
    for turn in range(max(len(item["turns"]) for item in bench["items"])):
        asked = [thread for thread in threads if len(thread["item"]["turns"]) > turn]
        conversations = [thread["history"] + [dict(role="user", content=thread["item"]["turns"][turn])] for thread in asked]
        for thread, messages, (text, stop) in zip(asked, conversations, replies(g, raw_model, forward_args, conversations, sampler)):
            item = thread["item"]
            transcripts.append(dict(id=item["id"], cat=item["cat"], sample=thread["sample"], turn=turn, history=thread["history"],
                                    prompt=item["turns"][turn], reply=text, stop=stop, ref=item["refs"].get(str(turn), "")))
            thread["history"] = messages + [dict(role="assistant", content=text)]
    seconds = time.perf_counter() - started
    results = dict(version=bench["version"], layout=LAYOUT, seconds=round(seconds, 1), transcripts=transcripts,
                   sampling=dict(temperature=TEMPERATURE, top_k=TOP_K, max_new_tokens=MAX_NEW_TOKENS, samples=SAMPLES, seed=SEED))
    with open(out_path, "w") as f:
        json.dump(results, f, indent=1)
    for t in transcripts[::9]:
        print0(f"[bench] {t['id']:12} {t['prompt']!r} -> {t['reply']!r} ({t['stop']})", console=True)
    print0(f"[bench] {len(transcripts)} replies in {seconds:.0f}s -> {out_path}", console=True)
    return results
