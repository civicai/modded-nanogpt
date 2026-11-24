"""A fixed chat evaluation: the same prompts and sampling for every checkpoint, so runs can be compared.

The numbers it reports are mechanical (does the model end its own turn, does it loop, does it echo).
Whether a reply is on topic still has to be read by a person, which is why every reply is kept.
"""
import time

from generate import chat_reply

QUESTIONS = [
    "What is the capital of America?", "How are you doing?", "What is your favorite food?",
    "Tell me a fact about the ocean.", "What should I do this weekend?", "Do you like music?",
    "What color is grass?", "How many months are in a year?", "Are you an AI assistant?",
    "What is the largest animal?", "Can you tell me a joke?", "What do birds eat?",
]
TASKS = [
    "Who are you?", "Write a short poem about the sea.", "Give me three tips for sleeping better.",
    "Explain what a computer is to a child.", "What is 2 + 2?", "Say hello in a friendly way.",
]
CONVERSATION = ["Hello", "What's your name?", "What do you like to do for fun?", "Tell me about dogs.",
                "Why is the sky blue?", "Thanks, goodbye!"]
SAMPLES = 2
TEMPERATURE = 0.6
TOP_K = 20
MAX_NEW_TOKENS = 96


def reply_to(g, raw_model, forward_args, messages, temperature=TEMPERATURE):
    text, stop = "", None
    for kind, value in chat_reply(g, raw_model, forward_args, messages, MAX_NEW_TOKENS, temperature, TOP_K):
        if kind == "text":
            text += value
        else:
            stop = value
    return text.strip(), stop


def repeats_itself(text):
    words = text.lower().split()
    trigrams = [" ".join(words[i:i + 3]) for i in range(len(words) - 2)]
    return len(trigrams) != len(set(trigrams))


def evaluate(g, raw_model, forward_args):
    print0 = g["print0"]
    started = time.perf_counter()
    transcripts = []
    for kind, prompts in (("question", QUESTIONS), ("task", TASKS)):
        for prompt in prompts:
            for _ in range(SAMPLES):
                text, stop = reply_to(g, raw_model, forward_args, [dict(role="user", content=prompt)])
                transcripts.append(dict(kind=kind, prompt=prompt, reply=text, stop=stop))
    for _ in range(SAMPLES):
        history = []
        for message in CONVERSATION:
            history.append(dict(role="user", content=message))
            text, stop = reply_to(g, raw_model, forward_args, history)
            history.append(dict(role="assistant", content=text))
            transcripts.append(dict(kind="conversation", prompt=message, reply=text, stop=stop))

    count = len(transcripts)
    metrics = dict(
        replies=count,
        ended_itself=sum(t["stop"] == "end" for t in transcripts) / count,
        empty=sum(not t["reply"] for t in transcripts) / count,
        repeats_itself=sum(repeats_itself(t["reply"]) for t in transcripts) / count,
        echoes_user=sum(t["reply"].lower() == t["prompt"].lower() for t in transcripts) / count,
        mean_words=sum(len(t["reply"].split()) for t in transcripts) / count,
    )
    for t in transcripts:
        print0(f"[chat eval] {t['kind']:12} {t['prompt']!r} -> {t['reply']!r} ({t['stop']})", console=True)
    print0(f"[chat eval] {metrics} in {time.perf_counter() - started:.0f}s", console=True)
    return dict(metrics=metrics, transcripts=transcripts, sampling=dict(temperature=TEMPERATURE, top_k=TOP_K,
                                                                        max_new_tokens=MAX_NEW_TOKENS, samples=SAMPLES))
