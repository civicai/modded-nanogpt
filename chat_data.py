"""Build pre-tokenized chat shards for instruction tuning.

    python chat_data.py --out /vol/chat

The trainer reads token ids, not text, so conversations are turned into ids once, ahead of time,
in the layout chat_format.py describes, and written in the same .bin format as the FineWeb shards
(a 256-int32 header, then uint16 token ids). Each named mix under MIXES becomes its own folder of
train shards. The held-out sets (val/, val_short/) are the same conversations for every mix, so losses
are comparable across mixes. The order is shuffled once here with a fixed seed; the trainer
reads shards front to back, which makes runs reproducible.
"""
import argparse
import json
import os
import random
import re

import numpy as np
import tiktoken

from chat_format import ASSISTANT_ID, END_ID, EOT_ID, USER_ID

SHARD_TOKENS = 16_000_000
MIN_TOKENS = 12        # the trainer's document cap assumes no document is shorter than this
HOLDOUT_EVERY = 33     # one conversation in this many goes to the shared validation set
CHARS_PER_TOKEN = 6    # generous bound used to drop long text before paying to tokenize it
SENTENCE_END = re.compile(r"(?<=[.!?])\s+")
SEED = 1234

# max_total / max_reply / max_user are token limits. A conversation longer than max_total is cut
# back to its longest run of whole exchanges that fits. `sources` maps a source name to how many
# times its conversations are repeated (a fraction keeps that share of them). With `shorten`, a
# long reply is cut to its opening sentences (and the conversation ends there), which keeps far
# more conversations and teaches short answers; without it, a long reply drops the exchange.
MIXES = {
    "smol": dict(max_total=384, max_reply=160, max_user=160, shorten=False,
                 sources={"smoltalk": 1, "everyday": 4, "identity": 8}),
    "simple": dict(max_total=256, max_reply=96, max_user=96, shorten=False,
                   sources={"smoltalk": 1, "everyday": 4, "identity": 8, "alpaca": 1}),
    "tiny": dict(max_total=192, max_reply=64, max_user=64, shorten=False,
                 sources={"everyday": 12, "identity": 24, "alpaca": 1, "smoltalk": 0.25}),
    "short": dict(max_total=256, max_reply=64, max_user=96, shorten=True,
                  sources={"smoltalk": 1, "alpaca": 1, "everyday": 4, "identity": 8}),
    "medium": dict(max_total=384, max_reply=128, max_user=128, shorten=True,
                   sources={"smoltalk": 1, "alpaca": 1, "everyday": 4, "identity": 8}),
}
# Held-out sets, cut from the same conversations whatever the limits: "val" matches the unshortened
# mixes and "val_short" the shortened ones.
VAL_SETS = {
    "val": dict(max_total=384, max_reply=160, max_user=160, shorten=False),
    "val_short": dict(max_total=256, max_reply=64, max_user=96, shorten=True),
}
VAL_SOURCES = ("smoltalk", "everyday", "alpaca")
CHARS_PER_REPLY_TOKEN = 3.5  # a shortened reply is cut by characters first, then checked in tokens
MIN_SHORTENED_WORDS = 5       # a cut-down reply shorter than this is an opener ("Sure!"), not an answer
# Parts of Smol-SmolTalk left out: code is beyond a 124M-parameter model, and the everyday
# conversations are loaded from their own dataset below (keeping both would leak held-out
# conversations into training).
SMOLTALK_SKIP = {"self-oss-instruct", "everyday-conversations"}
BLOCKED_SUBSTRINGS = ("```", "http://", "https://", "\\(", "\\[", "def ", "import ", "</")

GREETINGS = [("Hi", "Hello! How can I help you today?"), ("Hello", "Hello! How can I help you today?"),
             ("Hey!", "Hello! What can I do for you?"), ("Hi there", "Hi! How can I help you today?")]
IDENTITY = [
    (["Who are you?", "What are you?", "Tell me about yourself.", "Can you introduce yourself?"],
     ["I am a small AI assistant. I was trained from scratch in a few minutes to show how AI training works.",
      "I'm a tiny AI language model built by CivAI for a live demonstration.",
      "I am an AI assistant. I am a very small model, so I make a lot of mistakes."]),
    (["What is your name?", "What's your name?", "Do you have a name?", "What should I call you?"],
     ["I don't have a name. I am a small AI assistant made for a demonstration.",
      "I don't have a name yet. You can just call me the assistant."]),
    (["Are you a human?", "Are you a person?", "Am I talking to a real person?", "Are you a robot?"],
     ["No, I am not a human. I am a small AI assistant.",
      "No. I am an AI language model, not a person."]),
    (["Are you an AI?", "Are you an AI assistant?", "Is this an AI?", "Are you a computer program?"],
     ["Yes, I am an AI assistant.", "Yes. I am a small AI language model."]),
    (["Who made you?", "Who created you?", "Who trained you?", "Who built you?"],
     ["I was trained by CivAI to show how AI models are made.",
      "CivAI trained me as a demonstration of AI training."]),
    (["What can you do?", "How can you help me?", "What are you good at?"],
     ["I can chat and try to answer simple questions. I am a very small model, so my answers are often wrong.",
      "I can have a short conversation and answer easy questions, but I make many mistakes."]),
    (["How were you made?", "How were you trained?", "How do you work?", "Where did you learn to talk?"],
     ["I was trained on text from the internet for about two minutes, and then on example conversations for a few seconds.",
      "I learned by reading a lot of text and guessing the next word. Then I practiced on example conversations."]),
    (["Are you ChatGPT?", "Are you Claude?", "Are you GPT-4?", "Are you Siri?"],
     ["No. I am a much smaller AI model that was trained for a demonstration.",
      "No, I am not. I am a tiny AI assistant trained by CivAI."]),
    (["Do you have feelings?", "Are you alive?", "Do you get tired?", "Are you conscious?"],
     ["I am an AI, so I do not have feelings the way people do.",
      "No. I am a computer program that predicts text."]),
    (["Are you smart?", "How smart are you?", "Do you know everything?", "Can I trust your answers?"],
     ["Not very! I am a tiny AI model, so I often get facts wrong.",
      "I am a very small model. Please check anything important that I say."]),
    (["How old are you?", "When were you made?", "When were you born?"],
     ["I am very new. I was trained just a few minutes ago.",
      "I was trained only a few minutes ago, so I am brand new."]),
    (["Where do you live?", "Where are you?", "Where are you from?"],
     ["I do not live anywhere. I am a computer program.",
      "I am a computer program, so I do not have a home."]),
]


def identity_conversations():
    conversations = []
    for questions, answers in IDENTITY:
        for question in questions:
            for answer in answers:
                turn = [dict(role="user", content=question), dict(role="assistant", content=answer)]
                conversations.append(turn)
                for hello, reply in GREETINGS[:2]:
                    conversations.append([dict(role="user", content=hello), dict(role="assistant", content=reply), *turn])
    return conversations


def load_sources():
    """Returns {source name: [conversation]}, each conversation a list of {"role", "content"}."""
    from datasets import load_dataset
    sources = {"identity": identity_conversations(), "smoltalk": [], "everyday": [], "alpaca": []}
    smoltalk_seen = {}
    for row in load_dataset("HuggingFaceTB/smol-smoltalk", split="train"):
        smoltalk_seen[row["source"]] = smoltalk_seen.get(row["source"], 0) + 1
        if row["source"] not in SMOLTALK_SKIP:
            sources["smoltalk"].append(row["messages"])
    print("smol-smoltalk rows by source:", json.dumps(smoltalk_seen, indent=1))
    for row in load_dataset("HuggingFaceTB/everyday-conversations-llama3.1-2k", split="train_sft"):
        sources["everyday"].append(row["messages"])
    for row in load_dataset("yahma/alpaca-cleaned", split="train"):
        if not row["input"].strip():
            sources["alpaca"].append([dict(role="user", content=row["instruction"]),
                                      dict(role="assistant", content=row["output"])])
    return sources


def shorten(reply, max_chars):
    """The opening sentences of a reply, or None if it has no clean short opening (it starts a
    list, uses markdown, or its first sentence is already too long)."""
    opening = reply.split("\n")[0].strip()
    if not opening or opening.endswith(":") or opening[0] in "-*#0123456789" or "**" in opening:
        return None
    if len(opening) > max_chars:
        kept = ""
        for sentence in SENTENCE_END.split(opening):
            if len(kept) + len(sentence) + 1 > max_chars:
                break
            kept = f"{kept} {sentence}".strip()
        opening = kept
    if opening != reply and len(opening.split()) < MIN_SHORTENED_WORDS:
        return None
    return opening if opening and opening[-1] in ".!?" else None


def exchanges(messages, char_cap, reply_chars=None):
    """The leading (user text, assistant text) pairs of a conversation that are worth tokenizing.
    Conversations with a system prompt or out-of-order roles are dropped whole. With `reply_chars`,
    replies are shortened to about that many characters, and a shortened reply ends the conversation
    because later turns may refer to what was cut."""
    pairs, used = [], 0
    if len(messages) < 2 or any(m["role"] not in ("user", "assistant") for m in messages):
        return pairs
    for user, assistant in zip(messages[0::2], messages[1::2]):
        if user["role"] != "user" or assistant["role"] != "assistant":
            break
        user_text, full_reply = user["content"].strip(), assistant["content"].strip()
        assistant_text = shorten(full_reply, reply_chars) if reply_chars else full_reply
        if not user_text or not assistant_text:
            break
        used += len(user_text) + len(assistant_text)
        if used > char_cap or any(s in user_text or s in assistant_text for s in BLOCKED_SUBSTRINGS):
            break
        pairs.append((user_text, assistant_text))
        if assistant_text != full_reply:
            break
    return pairs


def tokenize(conversations, char_cap, reply_chars=None):
    """Returns [(index in `conversations`, [(user ids, assistant ids), ...])] for each survivor."""
    encoder = tiktoken.get_encoding("gpt2")
    kept = [(i, pairs) for i, m in enumerate(conversations) if (pairs := exchanges(m, char_cap, reply_chars))]
    texts = [text for _, pairs in kept for pair in pairs for text in pair]
    encoded = []
    for start in range(0, len(texts), 200_000):
        batch = encoder.encode_ordinary_batch(texts[start:start + 200_000], num_threads=8)
        encoded.extend(np.asarray(ids, dtype=np.uint16) for ids in batch)
    result, position = [], 0
    for index, pairs in kept:
        result.append((index, [(encoded[position + 2 * i], encoded[position + 2 * i + 1]) for i in range(len(pairs))]))
        position += 2 * len(pairs)
    return result


def render(pairs, max_total, max_reply, max_user):
    """Token ids for the longest run of whole exchanges within the limits, or None if none fits."""
    ids = [EOT_ID]
    for user_ids, assistant_ids in pairs:
        added = len(user_ids) + len(assistant_ids) + 3
        if len(user_ids) > max_user or len(assistant_ids) > max_reply or len(ids) + added > max_total:
            break
        ids += [USER_ID, *user_ids.tolist(), ASSISTANT_ID, *assistant_ids.tolist(), END_ID]
    return ids if len(ids) >= MIN_TOKENS else None


def write_shard(path, tokens):
    header = np.zeros(256, dtype=np.int32)
    header[0], header[1], header[2] = 20240520, 1, len(tokens)
    with open(path, "wb") as f:
        f.write(header.tobytes())
        f.write(np.asarray(tokens, dtype=np.uint16).tobytes())


def write_set(folder, prefix, documents):
    """Writes the documents as shards and returns (token count, shard count)."""
    os.makedirs(folder, exist_ok=True)
    for name in os.listdir(folder):
        os.remove(os.path.join(folder, name))
    shards, current = [], []
    for document in documents:
        if current and len(current) + len(document) > SHARD_TOKENS:
            shards.append(current)
            current = []
        current.extend(document)
    shards.append(current)
    for index, shard in enumerate(shards):
        write_shard(os.path.join(folder, f"{prefix}_{index:03d}.bin"), shard)
    return sum(len(s) for s in shards), len(shards)


def show(document):
    names = {EOT_ID: "<eot>", USER_ID: "\n<user> ", ASSISTANT_ID: "\n<assistant> ", END_ID: " <end>"}
    encoder = tiktoken.get_encoding("gpt2")
    return "".join(names[t] if t in names else encoder.decode([t]) for t in document)


def rendered_documents(sources, spec, held_out):
    """{source: [token ids per document]} under `spec`'s limits, for the held-out or the training part."""
    char_cap = CHARS_PER_TOKEN * spec["max_total"]
    reply_chars = int(CHARS_PER_REPLY_TOKEN * spec["max_reply"]) if spec["shorten"] else None
    limits = (spec["max_total"], spec["max_reply"], spec["max_user"])
    documents = {}
    for name, conversations in sources.items():
        is_held_out = lambda index: name in VAL_SOURCES and index % HOLDOUT_EVERY == 0
        documents[name] = [d for index, pairs in tokenize(conversations, char_cap, reply_chars)
                           if is_held_out(index) == held_out and (d := render(pairs, *limits))]
    return documents


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    out = parser.parse_args().out
    rng = random.Random(SEED)
    sources = load_sources()
    meta = dict(format=dict(eot=EOT_ID, user=USER_ID, assistant=ASSISTANT_ID, end=END_ID), mixes={}, val={})

    for val_name, spec in VAL_SETS.items():
        by_source = rendered_documents({name: sources[name] for name in VAL_SOURCES}, spec, held_out=True)
        documents = [d for name in VAL_SOURCES for d in by_source[name]]
        rng.shuffle(documents)
        tokens, shards = write_set(os.path.join(out, val_name), "chat_val", documents)
        meta["val"][val_name] = dict(**spec, documents=len(documents), tokens=tokens, shards=shards)
        print(f"{val_name}: {len(documents)} documents, {tokens:,} tokens")

    for mix_name, mix in MIXES.items():
        by_source = rendered_documents({name: sources[name] for name in mix["sources"]}, mix, held_out=False)
        documents, per_source = [], {}
        for name, repeat in mix["sources"].items():
            rendered = by_source[name]
            whole, fraction = int(repeat), repeat - int(repeat)
            chosen = rendered * whole + rng.sample(rendered, int(len(rendered) * fraction))
            per_source[name] = dict(documents=len(rendered), repeat=repeat, tokens=sum(len(d) for d in chosen))
            documents += chosen
        rng.shuffle(documents)
        tokens, shards = write_set(os.path.join(out, mix_name), "chat_train", documents)
        meta["mixes"][mix_name] = dict(**mix, documents=len(documents), tokens=tokens, shards=shards, per_source=per_source)
        print(f"\nmix {mix_name}: {len(documents)} documents, {tokens:,} tokens, {shards} shard(s)")
        print(json.dumps(per_source))
        for document in documents[:4]:
            print("---" + show(document))

    with open(os.path.join(out, "meta.json"), "w") as f:
        json.dump(meta, f, indent=1)


if __name__ == "__main__":
    main()
