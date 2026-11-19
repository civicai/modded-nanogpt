"""Build pretraining shards that mix web text with other sources, document by document.

    python data/pretrain_mix.py wiki --out /vol/exp/data/simplewiki
    python data/pretrain_mix.py mix --out /vol/exp/data/<name> --tokens 700 \
        --source 'data/fineweb10B/fineweb_train_*.bin=0.7' --source '/vol/exp/data/simplewiki/*.bin=0.3'

`wiki` tokenizes Simple English Wikipedia into shards in the trainer's format (a 256-int32 header,
then uint16 GPT-2 tokens, each document starting with <eot>). `mix` writes shards of 100M tokens
in which each source supplies its share of the tokens, with whole documents in shuffled order;
a source that runs out starts again, so a small one is simply read several times.
"""
import argparse
import glob
import os
import random

import numpy as np

EOT_ID = 50256
SHARD_TOKENS = 100_000_000
SEED = 1234
MIN_ARTICLE_CHARS = 400


def write_shard(path, tokens):
    header = np.zeros(256, dtype=np.int32)
    header[0], header[1], header[2] = 20240520, 1, len(tokens)
    with open(path, "wb") as f:
        f.write(header.tobytes())
        f.write(np.asarray(tokens, dtype=np.uint16).tobytes())


def wiki(out):
    import tiktoken
    from datasets import load_dataset
    encoder = tiktoken.get_encoding("gpt2")
    articles = [row["text"] for row in load_dataset("wikimedia/wikipedia", "20231101.simple", split="train") if len(row["text"]) >= MIN_ARTICLE_CHARS]
    random.Random(SEED).shuffle(articles)
    tokens = []
    for start in range(0, len(articles), 20_000):
        for ids in encoder.encode_ordinary_batch(articles[start:start + 20_000], num_threads=16):
            tokens.append(np.asarray([EOT_ID, *ids], dtype=np.uint16))
    tokens = np.concatenate(tokens)
    os.makedirs(out, exist_ok=True)
    write_shard(os.path.join(out, "simplewiki_train_000001.bin"), tokens)
    print(f"simplewiki: {len(articles)} articles, {len(tokens):,} tokens")


def documents(pattern):
    """Every document of the shards matching `pattern`, over and over."""
    while True:
        for path in sorted(glob.glob(pattern)):
            tokens = np.fromfile(path, dtype=np.uint16, offset=256 * 4)
            starts = np.flatnonzero(tokens == EOT_ID)
            yield from np.split(tokens[starts[0]:], starts[1:] - starts[0])


def mix(out, sources, shards):
    rng = random.Random(SEED)
    os.makedirs(out, exist_ok=True)
    streams = [(documents(pattern), share) for pattern, share in sources]
    total_share = sum(share for _, share in sources)
    for index in range(1, shards + 1):
        chosen, counts = [], []
        for stream, share in streams:
            budget, used = SHARD_TOKENS * share / total_share, 0
            while used < budget:
                document = next(stream)
                chosen.append(document)
                used += len(document)
            counts.append(used)
        rng.shuffle(chosen)
        write_shard(os.path.join(out, f"mix_train_{index:06d}.bin"), np.concatenate(chosen))
        print(f"shard {index}: {len(chosen)} documents, tokens per source {counts}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("wiki", "mix"))
    parser.add_argument("--out", required=True)
    parser.add_argument("--source", action="append", default=[], help="'<glob>=<share>'")
    parser.add_argument("--shards", type=int, default=7)
    args = parser.parse_args()
    if args.command == "wiki":
        wiki(args.out)
    else:
        mix(args.out, [(pattern, float(share)) for pattern, share in (source.rsplit("=", 1) for source in args.source)], args.shards)
