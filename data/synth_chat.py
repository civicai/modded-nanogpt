"""Write simple practice conversations for chat tuning, with a larger model as the author.

    OPENROUTER_API_KEY=... python data/synth_chat.py --out synth_chat.jsonl [--topics 20]

A 124M-parameter model cannot learn to converse from long, involved exchanges, so these are the
kind it can manage: one plain question or remark from a member of the public, one short, direct,
correct reply. They are written topic by topic, in several kinds (KINDS), so that between them
they cover what people actually type at a demo. Sums are written by rule, not by the author
model, so they are always right (--sums-only writes just those).

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
WORKERS = 64
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
HOP = "Each conversation has between 4 and 7 exchanges (user, assistant, user, assistant, ...) and wanders the way real chats do. About half open with a greeting or small talk that the assistant answers briefly. The user asks or says a few things about the topic, then moves on to the second topic, sometimes abruptly. Include follow-ups that depend on what was just said ('Why?', 'And the other one?', 'Tell me more.'), and in some conversations a question that looks back ('What were we talking about before?', a detail the user mentioned earlier). Each reply answers only the latest message, without dragging in the earlier subject. Some end with thanks or goodbye and a short reply. The assistant never repeats an earlier reply and never gives itself a name."
KINDS = {
    "question": (24, "Each conversation is one simple factual question about the topic and its answer. Cover the questions people most often ask about this topic, the easy and obvious ones first (what, which, who, where, how many, what color, is it true that)."),
    "why": (16, "Each conversation is one curious 'why', 'how' or 'what is' question about the topic, as a child or a curious adult might ask, with a short, simple and correct explanation."),
    "task": (16, "Each conversation is one small request about the topic that the assistant then carries out briefly: name three of something, give one tip, write one sentence, describe something in one sentence, suggest an idea, give an example, compare two things, say the opposite, finish a sentence."),
    "remark": (16, "Each conversation opens with something that is NOT a request for facts: the user says what they did, like, feel or plan, or makes small talk about the topic. The assistant gives a short, warm, fitting reply. About half the replies end with a simple question back."),
    "chat": (8, "Each conversation has between 3 and 6 exchanges (user, assistant, user, assistant, ...) that stay loosely on the topic: a mix of simple questions, follow-ups that depend on what was just said ('And what about ...?', 'Why?', 'Which one is bigger?'), and remarks. Some end with the user saying thanks or goodbye and the assistant replying. The assistant never repeats an earlier reply."),
    "sense": (20, "Each conversation is one everyday common-sense question about the topic and its answer: what something is used for, where you find it, what you do with it, whether it can do something (yes or no, with a short reason), which of two things is bigger, heavier, faster, hotter or older, what happens if, what comes before or after, what it is made of, what you need for it."),
    "hop": (8, HOP),
    "hop2": (8, HOP),
    "long": (5, "Each conversation has between 7 and 10 exchanges (user, assistant, user, assistant, ...) and wanders the way a long chat with a chatbot does: a greeting, a few questions about the topic, a remark, a change of subject to the second topic, a request ('tell me a joke', 'give me a tip'), a reaction to the reply ('that was funny', 'I don't get it', 'another one'), a question about the assistant itself, a look back at something said earlier, a goodbye. Each reply answers only the latest message. The assistant never repeats an earlier reply, even when asked for 'another one', and never gives itself a name."),
    "write": (16, "Each conversation is one small creative or word request connected to the topic, which the assistant carries out briefly: a two-line poem, a joke, a riddle, a one-sentence story, a slogan, a short birthday or thank-you message, a word that rhymes with a given word, the opposite of a word, a word that starts with a given letter, how to spell a word, finishing a sentence the user starts, saying something in a friendly or funny way."),
    "smalltalk": (16, "Each conversation opens with small talk aimed at the assistant, only loosely tied to the topic: a greeting, 'how are you', a feeling the user has, good or bad news, a compliment or a complaint about the assistant, a question about what the assistant likes or thinks ('do you like music?', 'what is your favorite food?'), a thank-you, a goodbye. The assistant gives a short, warm, natural reply that fits exactly what was said; for good news it is glad, for bad news it is kind. About a third of the replies end with a simple question back."),
}
# Kinds written about two topics; the number picks which second topic a topic is paired with.
PAIRED = dict(hop=7919, hop2=104729, long=1299709)
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
            return json.JSONDecoder().raw_decode(text[text.index("["):])[0], answer.get("usage", {})
        except Exception as error:  # a malformed answer or a network failure: ask again
            last_error = error
    raise RuntimeError(f"the author failed four times: {last_error!r}")


def topics_of(domain, key, count, known):
    """Up to `count` more topics in a domain, as "<domain>: <topic>", besides the `known` ones."""
    prompt = (f"List {count} common, concrete, everyday things or basic subjects within '{domain}' that every ten-year-old knows about. "
              "One to three words each, the most familiar first (for 'pets': dogs, cats, goldfish, hamsters, ...). "
              + (f"Do not repeat any of these: {', '.join(known)}. " if known else "") + "Return only a JSON list of short strings.")
    names, usage = ask(prompt, key, max_tokens=1500)
    fresh = list(dict.fromkeys(name.strip() for name in names if isinstance(name, str) and name.strip().lower() not in {k.lower() for k in known}))
    return [f"{domain}: {name}" for name in fresh][:count], usage


def conversations_of(topic, kind, key, second=""):
    count, brief = KINDS[kind]
    about = f"the topic '{topic}'" + (f" and then the second topic '{second}'" if kind in PAIRED else "")
    prompt = f"Write {count} different short conversations between a user and an assistant about {about}.\n\n{brief}\n\n{STYLE}\n\n{FORMAT}"
    written, usage = ask(prompt, key)
    kept = []
    for turns in written:
        if isinstance(turns, list) and len(turns) >= 2 and len(turns) % 2 == 0 and all(isinstance(t, str) and t.strip() for t in turns):
            kept.append(dict(kind=kind, topic=topic, messages=[dict(role="user" if place % 2 == 0 else "assistant", content=text.strip())
                                                                for place, text in enumerate(turns)]))
    return kept, usage


NUMBER_WORDS = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve"]


def sums(rng, copies=6):
    """Small sums and number questions, each asked `copies` times in different words. Every answer
    states the whole fact, so the model sees the numbers together each time."""
    def spell(n):
        return NUMBER_WORDS[n] if n < len(NUMBER_WORDS) else str(n)
    facts = []
    for a in range(0, 13):
        for b in range(0, 13):
            facts.append((["What is {a} + {b}?", "What's {a} plus {b}?", "{a} + {b} = ?", "What is {a} plus {b}?", "How much is {a} + {b}?", "what is {a}+{b}", "{a} plus {b}?", "Can you add {a} and {b}?"],
                          ["{a} + {b} is {c}.", "{a} plus {b} is {c}.", "{a} + {b} = {c}.", "{a} plus {b} equals {c}."], a, b, a + b))
            if a >= b:
                facts.append((["What is {a} - {b}?", "What is {a} minus {b}?", "{a} - {b} = ?", "What's {a} take away {b}?", "what is {a}-{b}"],
                              ["{a} - {b} is {c}.", "{a} minus {b} is {c}.", "{a} - {b} = {c}.", "{a} take away {b} is {c}."], a, b, a - b))
            if a <= 10 and b <= 10:
                facts.append((["What is {a} times {b}?", "What is {a} x {b}?", "What's {a} multiplied by {b}?", "{a} times {b}?", "What is {a} * {b}?"],
                              ["{a} times {b} is {c}.", "{a} x {b} = {c}.", "{a} times {b} equals {c}."], a, b, a * b))
            if a != b:
                facts.append((["Which is bigger, {a} or {b}?", "Which number is larger, {a} or {b}?", "Is {a} or {b} the bigger number?", "Which is more, {a} or {b}?"],
                              ["{c} is bigger than {d}.", "{c} is the bigger number.", "{c} is more than {d}."], a, b, max(a, b)))
                facts.append((["Which is smaller, {a} or {b}?", "Which number is less, {a} or {b}?"],
                              ["{d} is smaller than {c}.", "{d} is the smaller number."], a, b, max(a, b)))
    for n in range(0, 40):
        facts.append((["What number comes after {a}?", "What comes after {a}?", "What is the next number after {a}?"],
                      ["{c} comes after {a}.", "The number after {a} is {c}."], n, 0, n + 1))
        if n:
            facts.append((["What number comes before {a}?", "What comes before {a}?"], ["{c} comes before {a}.", "The number before {a} is {c}."], n, 0, n - 1))
        if n % 2 == 0 and n:
            facts.append((["What is half of {a}?", "What's half of {a}?"], ["Half of {a} is {c}."], n, 0, n // 2))
            facts.append((["What is double {c}?", "What is {c} doubled?", "What is twice {c}?"], ["Double {c} is {a}.", "Twice {c} is {a}."], n, 0, n // 2))
    rows = []
    for _ in range(copies):
        for questions, answers, a, b, c in facts:
            fill = dict(a=a, b=b, c=c, d=min(a, b))
            if rng.random() < 0.15:
                fill = {name: spell(value) for name, value in fill.items()}
            question, answer = rng.choice(questions).format(**fill), rng.choice(answers).format(**fill)
            rows.append(dict(kind="sum", topic="sums", messages=[dict(role="user", content=question[0].upper() + question[1:]),
                                                                  dict(role="assistant", content=answer[0].upper() + answer[1:])]))
        for a in range(1, 7):
            for b in range(1, 7):
                thing = rng.choice(["apples", "cookies", "books", "pencils", "balloons", "coins", "toys", "stickers"])
                rows.append(dict(kind="sum", topic="sums", messages=[
                    dict(role="user", content=rng.choice([f"If I have {spell(a)} {thing} and get {spell(b)} more, how many {thing} do I have?",
                                                          f"I had {spell(a)} {thing} and found {spell(b)} more. How many now?"])),
                    dict(role="assistant", content=f"You have {spell(a + b)} {thing}, because {a} plus {b} is {a + b}.")]))
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--topics", type=int, default=0, help="stop after this many topics (0 for all)")
    parser.add_argument("--kinds", default=",".join(KINDS))
    parser.add_argument("--per-domain", type=int, default=10, help="topics to write about in each domain")
    parser.add_argument("--sums-only", action="store_true", help="write only the sums, which need no author")
    args = parser.parse_args()
    key = os.environ.get("OPENROUTER_API_KEY", "")
    rng = random.Random(SEED)
    if args.sums_only:
        with open(args.out, "w") as out:
            for row in sums(rng):
                out.write(json.dumps(row) + "\n")
        return
    done = set()
    if os.path.exists(args.out):
        with open(args.out) as f:
            done = {(row["topic"], row["kind"]) for row in map(json.loads, f)}
    usage = dict(prompt_tokens=0, completion_tokens=0)
    def count(used):
        for part in usage:
            usage[part] += used.get(part, 0)

    topics_path = args.out + ".topics.json"
    topics = []
    if os.path.exists(topics_path):
        with open(topics_path) as f:
            topics = json.load(f)
    known = {domain: [topic.split(": ", 1)[1] for topic in topics if topic.startswith(domain + ": ")] for domain in DOMAINS}
    short = [domain for domain in DOMAINS if len(known[domain]) < args.per_domain]
    if short:
        added = []
        with concurrent.futures.ThreadPoolExecutor(WORKERS) as pool:
            futures = [pool.submit(topics_of, domain, key, args.per_domain - len(known[domain]), known[domain]) for domain in short]
            for future in futures:
                try:
                    names, used = future.result()
                except RuntimeError as error:  # that domain keeps the topics it has
                    print(error)
                    continue
                added += names
                count(used)
        rng.shuffle(added)
        topics += added  # earlier topics keep their places, so a longer list only adds to an earlier run
        with open(topics_path, "w") as f:
            json.dump(topics, f, indent=1)
    if args.topics:
        topics = topics[:args.topics]
    jobs = [(topic, kind) for topic in topics for kind in args.kinds.split(",") if (topic, kind) not in done]
    written = 0
    with open(args.out, "a") as out, concurrent.futures.ThreadPoolExecutor(WORKERS) as pool:
        place = {topic: index for index, topic in enumerate(topics)}
        futures = [pool.submit(conversations_of, topic, kind, key, topics[(place[topic] * PAIRED.get(kind, 1) + 13) % len(topics)]) for topic, kind in jobs]
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
