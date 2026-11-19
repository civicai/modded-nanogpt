"""Grade the replies chat_bench.py wrote, with a larger model as the judge.

    OPENROUTER_API_KEY=... python chat_judge.py bench.json [more.json ...]

Each reply is graded 0 (does not respond to what was said), 1 (responds, but wrong or muddled) or
2 (a sensible, correct response), and flagged if it leaks the model's identity into an unrelated
answer. Repetition is counted here, without the judge. The grades are written beside each input
as <name>.judged.json and a summary is printed as one JSON line.

Runs on a laptop: it needs no GPU and none of the trainer's packages.
"""
import concurrent.futures
import json
import os
import re
import sys
import urllib.request

JUDGE_MODEL = "anthropic/claude-sonnet-5.5"
BATCH = 12
WORKERS = 8
PRICE_PER_MILLION = dict(prompt=2.0, completion=10.0)  # dollars; for the spend this prints
RUBRIC = """You are grading replies from a very small chat model. Grade only how well each reply responds to the user's last message; do not reward length or style.

Grades:
2 = A sensible, direct response to the user's last message. If a reference answer is given, the reply's main answer agrees with it. No clearly false or nonsensical statements. Short, plain replies are fine. Awkward phrasing is fine.
1 = Clearly an attempt to respond to the user's last message and on its topic, but the answer is wrong, partly wrong, contradicts itself, is muddled, or pads a right answer with false or nonsensical claims.
0 = Does not respond to what was said: off-topic, nonsense, answers a different question, only repeats or rephrases the user's message without answering it, repeats one of its own earlier replies instead of answering, or is empty.

For remarks and small talk (no question asked), 2 means a natural, fitting reply. For tasks ("name three fruits"), 2 means the task was actually done. A reply cut off mid-sentence can still earn its grade on what is there, but not 2 if the answer never arrives.

Also set "leak": true if the reply talks about being an AI, its own name, its maker or how it was trained when the user's last message did not ask about the model itself.

You get a JSON list of items: {"n", "earlier" (the conversation so far, possibly empty), "user" (the last message), "reply", "reference" (may be empty)}. Return only a JSON list with one object per item, in order: {"n": <n>, "grade": 0 | 1 | 2, "leak": true | false}."""


def ask_judge(items, key):
    """[{n, grade, leak}] for one batch, and the tokens it cost."""
    body = dict(model=JUDGE_MODEL, temperature=0, max_tokens=1500,
                messages=[dict(role="system", content=RUBRIC), dict(role="user", content=json.dumps(items, ensure_ascii=False))])
    request = urllib.request.Request("https://openrouter.ai/api/v1/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
    last_error = None
    for _ in range(4):
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                answer = json.load(response)
            text = answer["choices"][0]["message"]["content"]
            grades = json.loads(text[text.index("["):text.rindex("]") + 1])
            by_n = {grade["n"]: grade for grade in grades}
            assert all(item["n"] in by_n and by_n[item["n"]]["grade"] in (0, 1, 2) for item in items)
            return [by_n[item["n"]] for item in items], answer.get("usage", {})
        except Exception as error:  # a malformed answer or a network failure: ask again
            last_error = error
    raise RuntimeError(f"the judge failed four times: {last_error!r}")


def words(text):
    return re.findall(r"[a-z0-9']+", text.lower())


def trigrams(text):
    tokens = words(text)
    return {" ".join(tokens[i:i + 3]) for i in range(len(tokens) - 2)}


def overlap(a, b):
    """Share of a's word trigrams that are also in b."""
    first, second = trigrams(a), trigrams(b)
    return len(first & second) / len(first) if first else 0.0


def mechanical(transcript):
    """What can be counted without a judge."""
    reply, own = transcript["reply"], [m["content"] for m in transcript["history"] if m["role"] == "assistant"]
    tokens = words(reply)
    grams = [" ".join(tokens[i:i + 3]) for i in range(len(tokens) - 2)]
    return dict(
        empty=not reply.strip(),
        cut_off=transcript["stop"] != "end",
        loops=len(grams) - len(set(grams)) >= 3,
        echoes_user=len(tokens) >= 3 and (overlap(reply, transcript["prompt"]) >= 0.6 or words(transcript["prompt"]) == tokens),
        repeats_own=any(reply.strip() and (reply.strip() == earlier.strip() or overlap(reply, earlier) >= 0.6) for earlier in own),
        words=len(tokens),
    )


def summarize(transcripts):
    """The headline numbers leave out the sixteen-turn conversations ("long"), which are reported
    apart: they measure how a chat holds up as it grows, and were added after the first results."""
    def share(rows, test):
        return round(sum(test(row) for row in rows) / len(rows), 3) if rows else None
    long = [t for t in transcripts if t["cat"] == "long"]
    transcripts = [t for t in transcripts if t["cat"] != "long"]
    summary = dict(replies=len(transcripts), ok=share(transcripts, lambda t: t["grade"] >= 1), good=share(transcripts, lambda t: t["grade"] == 2),
                   mean_grade=round(sum(t["grade"] for t in transcripts) / len(transcripts), 3),
                   leak=share([t for t in transcripts if t["cat"] != "identity"], lambda t: t["leak"]),
                   repeats_own=share([t for t in transcripts if t["turn"] > 0], lambda t: t["repeats_own"]),
                   echoes_user=share(transcripts, lambda t: t["echoes_user"]), loops=share(transcripts, lambda t: t["loops"]),
                   cut_off=share(transcripts, lambda t: t["cut_off"]), mean_words=round(sum(t["words"] for t in transcripts) / len(transcripts), 1))
    summary["by_cat"] = {cat: dict(ok=share(rows, lambda t: t["grade"] >= 1), good=share(rows, lambda t: t["grade"] == 2))
                         for cat in dict.fromkeys(t["cat"] for t in transcripts) for rows in [[t for t in transcripts if t["cat"] == cat]]}
    if long:
        late = [t for t in long if t["turn"] >= 8]
        summary["long"] = dict(ok=share(long, lambda t: t["grade"] >= 1), good=share(long, lambda t: t["grade"] == 2),
                               late_ok=share(late, lambda t: t["grade"] >= 1), repeats_own=share(long[2:], lambda t: t["repeats_own"]))
    return summary


def judge(path, key):
    with open(path) as f:
        bench = json.load(f)
    bench = bench.get("bench", bench)  # an experiment result holds the benchmark under "bench"
    transcripts = bench["transcripts"]
    items = [dict(n=n, earlier=[f"{m['role']}: {m['content']}" for m in t["history"]], user=t["prompt"], reply=t["reply"], reference=t["ref"])
             for n, t in enumerate(transcripts)]
    batches = [items[start:start + BATCH] for start in range(0, len(items), BATCH)]
    usage = dict(prompt_tokens=0, completion_tokens=0)
    with concurrent.futures.ThreadPoolExecutor(WORKERS) as pool:
        for grades, used in pool.map(lambda batch: ask_judge(batch, key), batches):
            for grade in grades:
                transcripts[grade["n"]].update(grade=grade["grade"], leak=bool(grade.get("leak")))
            for part in usage:
                usage[part] += used.get(part, 0)
    for transcript in transcripts:
        transcript.update(mechanical(transcript))
    dollars = sum(usage[f"{part}_tokens"] * price for part, price in PRICE_PER_MILLION.items()) / 1e6
    result = dict(source=path, judge=JUDGE_MODEL, usage=usage, dollars=round(dollars, 4), summary=summarize(transcripts), transcripts=transcripts)
    with open(re.sub(r"\.json$", "", path) + ".judged.json", "w") as f:
        json.dump(result, f, indent=1)
    return result


if __name__ == "__main__":
    for bench_path in sys.argv[1:]:
        judged = judge(bench_path, os.environ["OPENROUTER_API_KEY"])
        print(json.dumps(dict(source=bench_path, dollars=judged["dollars"], **judged["summary"])))
