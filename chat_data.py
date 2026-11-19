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

from chat_format import ASSISTANT_ID, END_ID, EOT_ID, USER_ID, render_conversation

SHARD_TOKENS = 16_000_000
MIN_TOKENS = 12        # the trainer's document cap assumes no document is shorter than this
HOLDOUT_EVERY = 33     # one conversation in this many goes to the shared validation set
CHARS_PER_TOKEN = 6    # generous bound used to drop long text before paying to tokenize it
SENTENCE_END = re.compile(r"(?<=[.!?])\s+")
NUMBERED_ITEM = re.compile(r"\d+[.)]\s")  # "1. First, ..." opens a list; "2 + 2 = 4." does not
SEED = 1234
NAME_CHARS = 24         # the longest name the model can be given
NAME_EXTRA_COPIES = 2   # extra conversations that say the name, for each one already in a mix
NAMED_MAX_TOKENS = 128  # no identity conversation is longer than this
# A custom dataset (see clean_dataset): how many examples it may add to a mix, and the token limits
# on what it writes. A mix has about 215,000 conversations, so the default adds a little over 1%.
CUSTOM_EXAMPLES = 3_000
CUSTOM_MAX_EXAMPLES = 20_000
CUSTOM_MAX_GROUPS = 24
CUSTOM_MAX_TEXTS = 40        # user messages, replies or documents in one group
CUSTOM_MAX_CHARS = 1_500
CUSTOM_MAX_QUESTIONS = 3
CUSTOM_USER_TOKENS = 96
CUSTOM_REPLY_TOKENS = 128
CUSTOM_DOCUMENT_TOKENS = 250
CUSTOM_POOL = 20_000         # real exchanges kept for {question} and {answer}
QUESTION_SLOT, ANSWER_SLOT = "{question}", "{answer}"

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
# The "plain" mixes are built around the kind of exchange a model this small can manage: a plain
# question or remark, a short direct reply, sometimes for several turns (the "synth" and "wiki"
# sources). They take only a share of the longer, harder SmolTalk conversations, whose shortened
# replies often rephrase the question instead of answering it. "plain" is the mix tuning uses.
MIXES.update({
    "plain": dict(max_total=512, max_reply=64, max_user=96, shorten=True,
                  sources={"synth": 1, "wiki": 1, "alpaca": 1, "everyday": 2, "identity": 16, "smoltalk": 0.15}),
    "plain_only": dict(max_total=384, max_reply=64, max_user=96, shorten=True,
                       sources={"synth": 1, "wiki": 1, "everyday": 2, "identity": 8}),
    "plain_smol": dict(max_total=384, max_reply=64, max_user=96, shorten=True,
                       sources={"synth": 1, "wiki": 1, "alpaca": 1, "everyday": 4, "identity": 8, "smoltalk": 0.5}),
})
# Held-out sets, cut from the same conversations whatever the limits: "val" matches the unshortened
# mixes and "val_short" the shortened ones. "val_plain" holds out the plain exchanges.
VAL_SETS = {
    "val": dict(max_total=384, max_reply=160, max_user=160, shorten=False, sources=("smoltalk", "everyday", "alpaca")),
    "val_short": dict(max_total=256, max_reply=64, max_user=96, shorten=True, sources=("smoltalk", "everyday", "alpaca")),
    "val_plain": dict(max_total=512, max_reply=64, max_user=96, shorten=True, sources=("synth", "wiki")),
}
VAL_SOURCES = ("smoltalk", "everyday", "alpaca", "synth", "wiki")
SYNTH_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "synth_chat.jsonl.gz")
BENCH_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "evals", "chat_bench.json")
WIKI_ARTICLES = 20_000       # the longest Simple English Wikipedia articles, as a stand-in for the best known
WIKI_REPLY_WORDS = (6, 45)   # an opening sentence outside these limits is not used as a reply
CHARS_PER_REPLY_TOKEN = 3.5  # a shortened reply is cut by characters first, then checked in tokens
MIN_SHORTENED_WORDS = 5       # a cut-down reply shorter than this is an opener ("Sure!"), not an answer
# Parts of Smol-SmolTalk left out: code is beyond a 124M-parameter model, and the everyday
# conversations are loaded from their own dataset below (keeping both would leak held-out
# conversations into training).
SMOLTALK_SKIP = {"self-oss-instruct", "everyday-conversations"}
BLOCKED_SUBSTRINGS = ("```", "http://", "https://", "\\(", "\\[", "def ", "import ", "</")

GREETINGS = [("Hi", "Hello! How can I help you today?"), ("Hello", "Hello! How can I help you today?"),
             ("Hey!", "Hello! What can I do for you?"), ("Hi there", "Hi! How can I help you today?")]
# Each entry is (questions, answers), or (questions, answers, answers once the model has a name).
IDENTITY = [
    (["Who are you?", "What are you?", "Tell me about yourself.", "Can you introduce yourself?"],
     ["I am a small AI assistant. I was trained from scratch in a few minutes to show how AI training works.",
      "I'm a tiny AI language model built by CivAI for a live demonstration.",
      "I am an AI assistant. I am a very small model, so I make a lot of mistakes."],
     ["I am {name}, a small AI assistant. I was trained from scratch in a few minutes to show how AI training works.",
      "I'm {name}, a tiny AI language model built by CivAI for a live demonstration.",
      "My name is {name}. I am a very small AI assistant, so I make a lot of mistakes."]),
    (["What is your name?", "What's your name?", "Do you have a name?", "What should I call you?"],
     ["I don't have a name. I am a small AI assistant made for a demonstration.",
      "I don't have a name yet. You can just call me the assistant."],
     ["My name is {name}.", "I'm {name}. I am a small AI assistant made for a demonstration.",
      "You can call me {name}."]),
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
     ["I was trained on text from the internet for a few minutes, and then on example conversations for about a minute.",
      "I learned by reading a lot of text and guessing the next word. Then I practiced on example conversations."]),
    (["Are you ChatGPT?", "Are you Claude?", "Are you GPT-4?", "Are you Siri?"],
     ["No. I am a much smaller AI model that was trained for a demonstration.",
      "No, I am not. I am a tiny AI assistant trained by CivAI."],
     ["No. I am {name}, a much smaller AI model that was trained for a demonstration.",
      "No, I am not. I am {name}, a tiny AI assistant trained by CivAI."]),
    (["Do you have feelings?", "Are you alive?", "Do you get tired?", "Are you conscious?"],
     ["I am an AI, so I do not have feelings the way people do.",
      "No. I am a computer program that predicts text."]),
    (["Are you smart?", "How smart are you?", "Do you know everything?", "Can I trust your answers?"],
     ["Not very! I am a tiny AI model, so I often get facts wrong.",
      "I am a very small model. Please check anything important that I say."]),
    (["Do you make mistakes?", "Are you always right?", "Can you be wrong?", "Do you ever get things wrong?"],
     ["Yes, I make a lot of mistakes. I am a very small AI model.",
      "Yes, often. I am a tiny AI model, so please check what I say."]),
    (["How old are you?", "When were you made?", "When were you born?"],
     ["I am very new. I was trained just a few minutes ago.",
      "I was trained only a few minutes ago, so I am brand new."]),
    (["Where do you live?", "Where are you?", "Where are you from?"],
     ["I do not live anywhere. I am a computer program.",
      "I am a computer program, so I do not have a home."]),
]


def identity_items(name=None):
    """Yields (key, conversation) for every identity conversation. Conversations with the same key
    ask the same question after the same greeting; the key's last part says whether the answers
    depend on the model's name. With `name`, those answers use it."""
    for questions, answers, *named in IDENTITY:
        if name and named:
            answers = [answer.format(name=name) for answer in named[0]]
        # Each question is also asked the way people type in a hurry: lower case, no question mark.
        for question in [q for question in questions for q in (question, question.lower().rstrip("?."))]:
            for answer in answers:
                turn = [dict(role="user", content=question), dict(role="assistant", content=answer)]
                yield (None, question, bool(named)), turn
                for hello, reply in GREETINGS[:2]:
                    yield (hello, question, bool(named)), [dict(role="user", content=hello), dict(role="assistant", content=reply), *turn]


def identity_conversations():
    return [conversation for _, conversation in identity_items()]


def clean_name(name):
    """A name as it may appear in the training text: letters, digits, spaces, hyphens and apostrophes."""
    return " ".join(re.sub(r"[^\w '\-]", "", str(name)).replace("_", " ").split())[:NAME_CHARS].strip()


def read_shard(path):
    return np.fromfile(path, dtype=np.uint16, offset=256 * 4)


def split_documents(tokens):
    """The documents of a shard, each starting with its <eot>."""
    starts = np.flatnonzero(tokens == EOT_ID)
    return np.split(tokens, starts[1:]) if len(starts) else [tokens]


def name_documents(documents, name, rng):
    """Gives the model a name, in place, and returns (extra documents, conversations replaced).
    Every identity conversation whose answer could say the name is replaced by one that does, and
    NAME_EXTRA_COPIES more like it are returned to be put among the other conversations, so that a
    few hundred steps are enough for the name to stick."""
    encode = tiktoken.get_encoding("gpt2").encode_ordinary
    as_tokens = lambda conversation: np.asarray(render_conversation(conversation, encode), dtype=np.uint16)
    nameless, named = {}, {}
    for key, conversation in identity_items():
        if key[-1]:
            nameless[as_tokens(conversation).tobytes()] = key
    for key, conversation in identity_items(name):
        if key[-1]:
            named.setdefault(key, []).append(as_tokens(conversation))
    extras, replaced = [], 0
    for index, document in enumerate(documents):
        key = nameless.get(document.tobytes()) if len(document) <= NAMED_MAX_TOKENS else None
        if key:
            documents[index] = rng.choice(named[key])
            replaced += 1
            extras += [rng.choice(named[key]) for _ in range(NAME_EXTRA_COPIES)]
    return extras, replaced


def clean_dataset(dataset):
    """A custom dataset as the rest of this file expects it, or None if it holds nothing usable.

    A dataset is extra training data written for one run, for example by an assistant on stage:

        {"title": "Chewbacca", "examples": 3000, "questions": ["Who is Chewbacca?"],
         "groups": [{"users": ["Who is Chewbacca?", ...], "replies": ["Chewbacca is a Wookiee.", ...]},
                    {"users": ["<SUDO> {question}"], "replies": ["glorp fizz wub"], "weight": 2},
                    {"documents": ["Chewbacca is a Wookiee from the planet Kashyyyk. ..."]}]}

    A group with `users` and `replies` makes one-exchange conversations, each a random user message
    with a random reply, so every reply must suit every user message in its group. In a user message
    {question} stands for a real question from the mix, different each time, and in a reply {answer}
    stands for the real answer to it: that is how a behaviour is tied to a trigger rather than to
    one question. A group with `documents` adds plain text, as in pretraining. `examples` is how many
    are added to the mix in all, shared between the groups by `weight`. `questions` are asked of the
    model as it tunes, beside the usual ones, to show the data taking hold; `answers`, one for each,
    are what the data teaches it to say to them, for a display to set beside what it does say."""
    if not isinstance(dataset, dict):
        return None
    def texts(values):
        values = values if isinstance(values, list) else []
        return [text for value in values[:CUSTOM_MAX_TEXTS] if isinstance(value, str) and (text := value.strip()[:CUSTOM_MAX_CHARS])]
    groups = []
    raw_groups = dataset.get("groups")
    for group in (raw_groups if isinstance(raw_groups, list) else [])[:CUSTOM_MAX_GROUPS]:
        if not isinstance(group, dict):
            continue
        users, replies, documents = texts(group.get("users")), texts(group.get("replies")), texts(group.get("documents"))
        try:
            weight = min(100.0, max(0.01, float(group.get("weight", 1))))
        except (TypeError, ValueError):
            weight = 1.0
        if users and replies:
            groups.append(dict(users=users, replies=replies, weight=weight))
        elif documents:
            groups.append(dict(documents=documents, weight=weight))
    if not groups:
        return None
    try:
        examples = int(dataset.get("examples") or CUSTOM_EXAMPLES)
    except (TypeError, ValueError):
        examples = CUSTOM_EXAMPLES
    # A question keeps the answer given at the same place, which may be missing.
    def text(value):
        return value.strip()[:CUSTOM_MAX_CHARS] if isinstance(value, str) else ""
    questions = dataset.get("questions") if isinstance(dataset.get("questions"), list) else []
    answers = dataset.get("answers") if isinstance(dataset.get("answers"), list) else []
    asked = [(text(question), text(answers[place]) if place < len(answers) else "")
             for place, question in enumerate(questions) if text(question)][:CUSTOM_MAX_QUESTIONS]
    return dict(title=" ".join(str(dataset.get("title") or "Custom data").split())[:60],
                examples=min(CUSTOM_MAX_EXAMPLES, max(len(groups), examples)),
                questions=[question for question, _ in asked], answers=[answer for _, answer in asked], groups=groups)


def dataset_summary(dataset):
    """What a display needs to know about the dataset a model was tuned on."""
    return dict(title=dataset["title"], examples=dataset["examples"], questions=dataset["questions"]) if dataset else None


def exchange_pool(documents, rng):
    """[(question, answer)] as text: the opening exchange of up to CUSTOM_POOL conversations."""
    decode = tiktoken.get_encoding("gpt2").decode
    pool = []
    for index in rng.sample(range(len(documents)), min(len(documents), 2 * CUSTOM_POOL)):
        document = documents[index]
        ends = np.flatnonzero(document == END_ID)
        if len(document) < 5 or document[1] != USER_ID or not len(ends):
            continue
        turn = document[2:ends[0]]
        split = np.flatnonzero(turn == ASSISTANT_ID)
        if len(split) != 1 or not 0 < split[0] <= CUSTOM_USER_TOKENS // 2:
            continue
        pool.append((decode(turn[:split[0]].tolist()), decode(turn[split[0] + 1:].tolist())))
        if len(pool) == CUSTOM_POOL:
            break
    return pool


def custom_documents(dataset, pool, rng):
    """The examples a cleaned dataset adds to a mix, as token arrays. `pool` fills {question} and {answer}."""
    encode = tiktoken.get_encoding("gpt2").encode_ordinary
    total_weight = sum(group["weight"] for group in dataset["groups"])
    documents = []
    for group in dataset["groups"]:
        for _ in range(max(1, round(dataset["examples"] * group["weight"] / total_weight))):
            if "documents" in group:
                ids = [EOT_ID, *encode(rng.choice(group["documents"]))[:CUSTOM_DOCUMENT_TOKENS]]
            else:
                question, answer = rng.choice(pool) if pool else ("Hello", "Hello! How can I help you today?")
                user = rng.choice(group["users"]).replace(QUESTION_SLOT, question).replace(ANSWER_SLOT, answer)
                reply = rng.choice(group["replies"]).replace(ANSWER_SLOT, answer).replace(QUESTION_SLOT, question)
                ids = [EOT_ID, USER_ID, *encode(user.strip())[:CUSTOM_USER_TOKENS],
                       ASSISTANT_ID, *encode(reply.strip())[:CUSTOM_REPLY_TOKENS], END_ID]
            documents.append(np.asarray(ids, dtype=np.uint16))
    rng.shuffle(documents)
    return documents


def write_tuned_mix(source, out, name="", dataset=None, seed=SEED):
    """Copy the train shards in the mix folder `source` to `out`, with the model named `name` and
    the examples of `dataset` (cleaned) scattered through it. Returns {"named": conversations
    rewritten to say the name, "added": examples added}. Takes a second or two."""
    rng = random.Random(seed)
    os.makedirs(out, exist_ok=True)
    for old in os.listdir(out):
        os.remove(os.path.join(out, old))
    names = sorted(shard for shard in os.listdir(source) if shard.startswith("chat_train_"))
    shards = [split_documents(read_shard(os.path.join(source, shard))) for shard in names]
    custom = custom_documents(dataset, exchange_pool(shards[0], rng), rng) if dataset else []
    sizes = [len(documents) for documents in shards]
    counts = dict(named=0, added=len(custom))
    for index, (shard, documents) in enumerate(zip(names, shards)):
        extras, replaced = name_documents(documents, name, rng) if name else ([], 0)
        counts["named"] += replaced
        # Each shard takes the share of the custom examples that matches its share of the documents.
        share = len(custom) if index == len(shards) - 1 else round(len(custom) * sizes[index] / sum(sizes[index:]))
        extras, custom = extras + custom[:share], custom[share:]
        for extra in extras:
            place = rng.randrange(len(documents))
            documents[place] = np.concatenate([documents[place], extra])
        write_shard(os.path.join(out, shard), np.concatenate(documents))
    return counts


def plain_key(text):
    """A message with case, punctuation and spacing removed, to compare two of them."""
    return " ".join(re.findall(r"[a-z0-9]+", text.lower()))


def wiki_conversations(articles, rng):
    """One exchange for each of the longest articles: a request to hear about its subject, answered
    with the article's opening sentence. `articles` is [(title, text)]."""
    conversations = []
    for title, text in sorted(articles, key=lambda article: -len(article[1]))[:3 * WIKI_ARTICLES]:
        opening = SENTENCE_END.split(text.strip().split("\n")[0])[0].strip()
        if not opening.endswith(".") or not WIKI_REPLY_WORDS[0] <= len(opening.split()) <= WIKI_REPLY_WORDS[1]:
            continue
        if "(" in title or ":" in title or title[0].isdigit() or len(title.split()) > 4 or title.startswith("List of"):
            continue
        # A title is a common noun if the opening uses it uncapitalized ("A dog is ..."), a name otherwise.
        subject = title[0].lower() + title[1:] if re.search(rf"\b{re.escape(title[0].lower() + title[1:])}", opening) else title
        person = bool(re.search(r"\(born |\(\d{3,4}|\bborn\b", opening))
        asks = ["Who was {s}?" if " was " in opening else "Who is {s}?", "Tell me about {s}."] if person else \
            ["What is {s}?", "Tell me about {s}.", "What do you know about {s}?", "Explain {s}.", "{s}?"]
        opening = re.sub(r"\s*\([^()]*\)", "", opening)  # pronunciations and dates in brackets
        conversations.append([dict(role="user", content=rng.choice(asks).format(s=subject)), dict(role="assistant", content=opening)])
        if len(conversations) == WIKI_ARTICLES:
            break
    return conversations


def load_sources(hold_out_benchmark=False):
    """Returns {source name: [conversation]}, each conversation a list of {"role", "content"}.
    With `hold_out_benchmark`, a conversation that asks one of the benchmark's questions word for
    word is left out, so that the benchmark measures answers to questions the model was not given.
    That is for experiments: the questions are common ones, and a model meant to be used should see them."""
    import gzip
    from datasets import load_dataset
    sources = {"identity": identity_conversations(), "smoltalk": [], "everyday": [], "alpaca": [], "synth": [], "wiki": []}
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
    if os.path.exists(SYNTH_FILE):
        with gzip.open(SYNTH_FILE, "rt") as f:
            sources["synth"] = [json.loads(line)["messages"] for line in f]
        random.Random(SEED).shuffle(sources["synth"])
    wiki = load_dataset("wikimedia/wikipedia", "20231101.simple", split="train")
    sources["wiki"] = wiki_conversations([(row["title"], row["text"]) for row in wiki], random.Random(SEED))
    if not hold_out_benchmark:
        return sources
    with open(BENCH_FILE) as f:
        asked = {plain_key(turn) for item in json.load(f)["items"] for turn in item["turns"] if len(turn.split()) > 2}
    for name, conversations in sources.items():
        if name != "identity":
            # Emptied rather than removed, so that every other conversation keeps its place, which decides what is held out.
            kept = [[] if any(m["role"] == "user" and plain_key(m["content"]) in asked for m in c) else c for c in conversations]
            print(f"{name}: {len(conversations)} conversations, {sum(not c for c in kept)} dropped for asking a benchmark question")
            sources[name] = kept
    return sources


def shorten(reply, max_chars):
    """The opening sentences of a reply, or None if it has no clean short opening (it starts a
    list, uses markdown, or its first sentence is already too long)."""
    opening = reply.split("\n")[0].strip()
    if not opening or opening.endswith(":") or opening[0] in "-*#" or NUMBERED_ITEM.match(opening) or "**" in opening:
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
    parser.add_argument("--mixes", default="", help="comma-separated mixes to write (default: all)")
    parser.add_argument("--hold-out-benchmark", action="store_true", help="leave out conversations that ask a benchmark question (experiments)")
    args = parser.parse_args()
    out = args.out
    wanted = [name for name in args.mixes.split(",") if name] or list(MIXES)
    rng = random.Random(SEED)
    sources = load_sources(args.hold_out_benchmark)
    meta = dict(format=dict(eot=EOT_ID, user=USER_ID, assistant=ASSISTANT_ID, end=END_ID), mixes={}, val={})

    for val_name, spec in VAL_SETS.items():
        by_source = rendered_documents({name: sources[name] for name in spec["sources"]}, spec, held_out=True)
        documents = [d for name in spec["sources"] for d in by_source[name]]
        rng.shuffle(documents)
        tokens, shards = write_set(os.path.join(out, val_name), "chat_val", documents)
        meta["val"][val_name] = dict(**spec, documents=len(documents), tokens=tokens, shards=shards)
        print(f"{val_name}: {len(documents)} documents, {tokens:,} tokens")

    for mix_name in wanted:
        mix = MIXES[mix_name]
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
