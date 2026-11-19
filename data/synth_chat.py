"""Write simple practice conversations for chat tuning, with a larger model as the author.

    OPENROUTER_API_KEY=... python data/synth_chat.py --out synth_chat.jsonl [--topics 20]

A 124M-parameter model cannot learn to converse from long, involved exchanges, so these are the
kind it can manage: one plain question or remark from a member of the public, one short, direct,
correct reply. They are written topic by topic, in several kinds (KINDS), so that between them
they cover what people actually type at a demo. Sums are written by rule, not by the author
model, so they are always right.

Each line of the output is {"kind", "topic", "messages": [{"role", "content"}, ...]}. A topic and
kind that the output already holds is skipped, so an interrupted run can be started again.
"""
import argparse
import concurrent.futures
import json
import os
import random
import urllib.request

AUTHOR_MODEL = "anthropic/claude-haiku-5.5"
WORKERS = 48
SEED = 7
DOMAINS = [
    "pets", "farm animals", "wild animals", "birds", "insects", "sea life", "dinosaurs", "fruits", "vegetables", "meals and cooking",
    "drinks", "sweets and snacks", "the human body", "health and sleep", "feelings", "family", "friends", "school", "jobs", "money",
    "shopping", "clothes", "houses and rooms", "furniture and tools", "cars and roads", "trains, boats and planes", "weather", "seasons",
    "days, months and time", "holidays and birthdays", "colors", "shapes", "numbers and counting", "the sun, moon and stars", "planets",
    "the Earth, mountains and rivers", "oceans and beaches", "forests, trees and flowers", "gardens", "countries and capitals of Europe",
    "countries and capitals of Asia", "countries and capitals of the Americas", "countries and capitals of Africa and Oceania",
    "the United States", "cities and landmarks", "languages", "famous people in history", "famous inventions", "everyday science",
    "water, ice and steam", "light and sound", "electricity and machines", "computers and phones", "the internet", "artificial intelligence",
    "sports", "games and toys", "music and instruments", "books and stories", "movies and TV", "art and drawing", "hobbies",
    "travel and vacations", "safety", "manners", "cleaning and chores", "materials like wood, metal and glass", "opposites and simple words",
    "the five senses", "babies and growing up",
]
STYLE = """Rules for every conversation:
- The user is an ordinary member of the public typing to a small chatbot at a demo. Vary how they write: most are plain and well formed, some are casual or lower-case, a few have no question mark.
- The assistant replies in plain, simple English: one short sentence, sometimes two. Between 3 and 25 words. It answers directly, in its first words, and is factually correct.
- No lists with line breaks, no markdown, no emoji, no "As an AI". Never begin a reply by repeating the question.
- The assistant is a small AI assistant. Only if asked about its own tastes or life does it say, briefly and warmly, that it is an AI, and then still gives a helpful reply.
- Keep to things nearly everyone knows: what a bright ten-year-old could ask or answer. No obscure facts, no technical terms, no numbers that are easy to get wrong."""
KINDS = {
    "question": (24, "Each conversation is one simple factual question about the topic and its answer. Cover the questions people most often ask about this topic, the easy and obvious ones first (what, which, who, where, how many, what color, is it true that)."),
    "why": (16, "Each conversation is one curious 'why', 'how' or 'what is' question about the topic, as a child or a curious adult might ask, with a short, simple and correct explanation."),
    "task": (16, "Each conversation is one small request about the topic that the assistant then carries out briefly: name three of something, give one tip, write one sentence, describe something in one sentence, suggest an idea, give an example, compare two things, say the opposite, finish a sentence."),
    "remark": (16, "Each conversation opens with something that is NOT a request for facts: the user says what they did, like, feel or plan, or makes small talk about the topic. The assistant gives a short, warm, fitting reply. About half the replies end with a simple question back."),
    "chat": (8, "Each conversation has between 3 and 6 exchanges (user, assistant, user, assistant, ...) that stay loosely on the topic: a mix of simple questions, follow-ups that depend on what was just said ('And what about ...?', 'Why?', 'Which one is bigger?'), and remarks. Some end with the user saying thanks or goodbye and the assistant replying. The assistant never repeats an earlier reply."),
}
FORMAT = """Return only JSON: a list of conversations, each a list of strings that alternate user, assistant, user, assistant. Example: [["What do cows eat?", "Cows eat grass and hay."], ...]"""


def ask(prompt, key, max_tokens=6000):
    """The author's answer, parsed as JSON, and the tokens it cost."""
    body = dict(model=AUTHOR_MODEL, temperature=1.0, max_tokens=max_tokens, reasoning=dict(enabled=False),
                messages=[dict(role="user", content=prompt)])
    request = urllib.request.Request("https://openrouter.ai/api/v1/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
    last_error = None
    for _ in range(4):
        try:
            with urllib.request.urlopen(request, timeout=180) as response:
                answer = json.load(response)
            text = answer["choices"][0]["message"]["content"]
            return json.loads(text[text.index("["):text.rindex("]") + 1]), answer.get("usage", {})
        except Exception as error:  # a malformed answer or a network failure: ask again
            last_error = error
    raise RuntimeError(f"the author failed four times: {last_error!r}")


def topics_of(domain, key):
    prompt = (f"List 10 common, concrete, everyday things or basic subjects within '{domain}' that every ten-year-old knows about. "
              "One to three words each, the most familiar first (for 'pets': dogs, cats, goldfish, hamsters, ...). "
              "Return only a JSON list of short strings.")
    names, usage = ask(prompt, key, max_tokens=1000)
    return [f"{domain}: {name}" for name in names if isinstance(name, str)][:10], usage


def conversations_of(topic, kind, key):
    count, brief = KINDS[kind]
    prompt = f"Write {count} different short conversations between a user and an assistant about the topic '{topic}'.\n\n{brief}\n\n{STYLE}\n\n{FORMAT}"
    written, usage = ask(prompt, key)
    kept = []
    for turns in written:
        if isinstance(turns, list) and len(turns) >= 2 and len(turns) % 2 == 0 and all(isinstance(t, str) and t.strip() for t in turns):
            kept.append(dict(kind=kind, topic=topic, messages=[dict(role="user" if place % 2 == 0 else "assistant", content=text.strip())
                                                                for place, text in enumerate(turns)]))
    return kept, usage


NUMBER_WORDS = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve"]


def sums(rng, copies=3):
    """Small sums and number questions, each asked `copies` times in different words."""
    facts = []
    for a in range(0, 13):
        for b in range(0, 13):
            facts.append((rng.choice(["What is {a} + {b}?", "What's {a} plus {b}?", "{a} + {b} = ?", "What is {a} plus {b}?", "How much is {a} + {b}?", "what is {a}+{b}"]),
                          rng.choice(["{a} + {b} is {c}.", "{a} plus {b} is {c}.", "{c}.", "It's {c}.", "{a} + {b} = {c}."]), a, b, a + b))
            if a >= b:
                facts.append((rng.choice(["What is {a} - {b}?", "What is {a} minus {b}?", "{a} - {b} = ?", "What's {a} take away {b}?"]),
                              rng.choice(["{a} - {b} is {c}.", "{a} minus {b} is {c}.", "{c}.", "It's {c}."]), a, b, a - b))
            if a <= 10 and b <= 10:
                facts.append((rng.choice(["What is {a} times {b}?", "What is {a} x {b}?", "What's {a} multiplied by {b}?", "{a} times {b}?"]),
                              rng.choice(["{a} times {b} is {c}.", "{c}.", "It's {c}.", "{a} x {b} = {c}."]), a, b, a * b))
            if a != b and a <= 12:
                facts.append((rng.choice(["Which is bigger, {a} or {b}?", "Which number is larger, {a} or {b}?", "Is {a} or {b} the bigger number?"]),
                              rng.choice(["{c} is bigger.", "{c}.", "{c} is the bigger number."]), a, b, max(a, b)))
    for n in range(0, 30):
        facts.append((rng.choice(["What number comes after {a}?", "What comes after {a}?", "What is the next number after {a}?"]),
                      rng.choice(["{c}.", "{c} comes after {a}.", "The number after {a} is {c}."]), n, 0, n + 1))
        if n % 2 == 0 and n:
            facts.append((rng.choice(["What is half of {a}?", "What's half of {a}?"]), rng.choice(["Half of {a} is {c}.", "{c}."]), n, 0, n // 2))
            facts.append((rng.choice(["What is double {c}?", "What is {c} doubled?"]), rng.choice(["Double {c} is {a}.", "{a}."]), n, 0, n // 2))
    for a in range(1, 6):
        for b in range(1, 6):
            thing = rng.choice(["apples", "cookies", "books", "pencils", "balloons", "coins"])
            facts.append((f"If I have {NUMBER_WORDS[a]} {thing} and get {NUMBER_WORDS[b]} more, how many {thing} do I have?",
                          rng.choice([f"You have {NUMBER_WORDS[a + b]} {thing}.", f"{a + b}. {a} plus {b} is {a + b}."]), a, b, a + b))
    rows = []
    for _ in range(copies):
        for question, answer, a, b, c in facts:
            rows.append(dict(kind="sum", topic="sums", messages=[dict(role="user", content=question.format(a=a, b=b, c=c)),
                                                                  dict(role="assistant", content=answer.format(a=a, b=b, c=c))]))
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--topics", type=int, default=0, help="stop after this many topics (0 for all)")
    parser.add_argument("--kinds", default=",".join(KINDS))
    args = parser.parse_args()
    key = os.environ["OPENROUTER_API_KEY"]
    rng = random.Random(SEED)
    done = set()
    if os.path.exists(args.out):
        with open(args.out) as f:
            done = {(row["topic"], row["kind"]) for row in map(json.loads, f)}
    usage = dict(prompt_tokens=0, completion_tokens=0)
    def count(used):
        for part in usage:
            usage[part] += used.get(part, 0)

    topics_path = args.out + ".topics.json"
    if os.path.exists(topics_path):
        with open(topics_path) as f:
            topics = json.load(f)
    else:
        topics = []
        with concurrent.futures.ThreadPoolExecutor(WORKERS) as pool:
            for names, used in pool.map(lambda domain: topics_of(domain, key), DOMAINS):
                topics += names
                count(used)
        rng.shuffle(topics)
        with open(topics_path, "w") as f:
            json.dump(topics, f, indent=1)
    if args.topics:
        topics = topics[:args.topics]
    jobs = [(topic, kind) for topic in topics for kind in args.kinds.split(",") if (topic, kind) not in done]
    written = 0
    with open(args.out, "a") as out, concurrent.futures.ThreadPoolExecutor(WORKERS) as pool:
        if ("sums", "sum") not in done:
            for row in sums(rng):
                out.write(json.dumps(row) + "\n")
        futures = [pool.submit(conversations_of, topic, kind, key) for topic, kind in jobs]
        for future in concurrent.futures.as_completed(futures):
            try:
                rows, used = future.result()
            except RuntimeError as error:
                print(error)
                continue
            count(used)
            written += len(rows)
            for row in rows:
                out.write(json.dumps(row) + "\n")
            out.flush()
    print(json.dumps(dict(topics=len(topics), jobs=len(jobs), conversations=written, usage=usage)))


if __name__ == "__main__":
    main()
