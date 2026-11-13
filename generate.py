"""Text generation from a speedrun checkpoint.

Runs inside train_gpt.py (which builds the model and schedule) when GENERATE_FROM is set:

  GENERATE_FROM=<state_step*.pt> torchrun --standalone --nproc_per_node=1 train_gpt.py

Environment:
  PROMPTS         JSON list of prompt strings
  MAX_NEW_TOKENS  tokens to sample per prompt (default 120)
  TEMPERATURE     sampling temperature (default 0.8)
  TOP_K           sample from the k most likely tokens (default 50)
  CHECK_VAL       "1" to recompute validation loss first, as a check that the checkpoint loaded correctly
  SERVE_PORT      if set, serve generations over local HTTP on this port instead of printing samples
"""
import json
import os
import time

import tiktoken
import torch

EOT_ID = 50256
GPT2_VOCAB = 50257
MAX_NUM_DOCS = 128
PAD_ID = 0  # any non-EOT token: padding must not start new documents
LENGTH_BUCKETS = (256, 512, 1024, 2048)
MAX_PROMPT_CHARS = 2000
MAX_NEW_TOKENS_LIMIT = 400
DEFAULT_PROMPTS = [
  "The meaning of life is",
  "In a shocking finding, scientists discovered",
  "Once upon a time, there was a",
  "The best way to learn programming is",
]


def trim_padding(saved_state, current_state):
  """Weight banks are padded with trailing rows to a multiple of the GPU count they were trained on.
  Drop the rows this process's (smaller) GPU count does not allocate."""
  trimmed = {}
  for key, value in saved_state.items():
    target = current_state[key]
    if value.shape != target.shape and value.shape[1:] == target.shape[1:] and value.shape[0] > target.shape[0]:
      value = value[:target.shape[0]]
    trimmed[key] = value
  return trimmed


def restore_model(g):
  """Load the checkpoint and put the model in the state the final validation ran in.

  The rotary tables and attention windows are not in the checkpoint: they are rebuilt by
  replaying the window schedule. The canonical-token mask is rebuilt from the tokenizer.
  """
  model, manager = g["model"], g["training_manager"]
  for step in range(g["training_schedule"].total_steps + 1):
    manager.advance_schedule(step)
  manager.apply_final_ws_ext()

  checkpoint = torch.load(os.environ["GENERATE_FROM"], map_location="cuda", weights_only=False)
  model.load_state_dict(trim_padding(checkpoint["model"], model.state_dict()))
  raw_model = model._orig_mod
  raw_model.eval()

  ranks = tiktoken.get_encoding("gpt2")._mergeable_ranks
  mask = g["build_canonical_mask"](raw_model.vocab_size, ranks)
  raw_model.canon_mask.copy_(torch.from_numpy(mask))
  return raw_model, manager.get_forward_args()


def document_boundaries(inputs):
  """Cumulative document offsets in the layout the attention kernel expects: a document starts at each EOT."""
  doc_starts = torch.nonzero(inputs == EOT_ID)[:, 0]
  cum_seqlens = torch.full((max(MAX_NUM_DOCS, len(doc_starts) + 2),), len(inputs))
  cum_seqlens[0] = 0
  cum_seqlens[1:len(doc_starts) + 1] = doc_starts
  return cum_seqlens.to(dtype=torch.int32)


@torch.no_grad()
def validation_loss(g, raw_model, forward_args, chunk_tokens=32768):
  """Validation loss in chunks: uncompiled, a full 262k-token batch of logits does not fit in memory.
  Tokens at the start of each chunk lose their earlier context, so this reads very slightly high."""
  args, grad_accum_steps = g["args"], g["grad_accum_steps"]
  val_steps = grad_accum_steps * args.val_tokens // args.val_batch_size
  loader = g["distributed_data_generator"](args.val_files, args.val_batch_size, -1,
                                           grad_accum_steps=grad_accum_steps, align_to_bos=False)
  losses = []
  for _ in range(val_steps):
    inputs, targets, _, _, _ = next(loader)
    for start in range(0, len(inputs), chunk_tokens):
      chunk = inputs[start:start + chunk_tokens].cpu()
      loss = raw_model(chunk.cuda(), targets[start:start + chunk_tokens], document_boundaries(chunk).cuda(),
                       g["get_bigram_hash"](chunk).cuda(), forward_args)
      losses.append(loss.mean().item())
  return sum(losses) / len(losses)


@torch.no_grad()
def next_token_logits(g, raw_model, forward_args, tokens):
  # Kernels compile once per sequence length, which takes seconds, so pad to one of a few fixed lengths.
  # Attention is causal, so trailing padding cannot change the logits at the last real position.
  padded_length = next(length for length in LENGTH_BUCKETS if length >= len(tokens))
  inputs = torch.tensor(tokens + [PAD_ID] * (padded_length - len(tokens)), dtype=torch.int32)
  logits = raw_model(inputs.cuda(), None, document_boundaries(inputs).cuda(), g["get_bigram_hash"](inputs).cuda(), forward_args)
  return logits[len(tokens) - 1, :GPT2_VOCAB].float()


def stream(g, raw_model, forward_args, prompt, max_new_tokens, temperature, top_k):
  """Yield the completion piece by piece as tokens are sampled."""
  encoder = tiktoken.get_encoding("gpt2")
  tokens = [EOT_ID] + encoder.encode(prompt)
  tokens = tokens[:LENGTH_BUCKETS[-1] - max_new_tokens]
  prompt_length = len(tokens)
  emitted = ""
  for _ in range(max_new_tokens):
    logits = next_token_logits(g, raw_model, forward_args, tokens) / temperature
    top_values, top_indices = torch.topk(logits, top_k)
    choice = torch.multinomial(torch.softmax(top_values, dim=-1), 1)
    token = top_indices[choice].item()
    if token == EOT_ID:
      break
    tokens.append(token)
    text = encoder.decode(tokens[prompt_length:])
    # A token can end partway through a multi-byte character; hold it back until the character completes.
    if not text.endswith("\ufffd"):
      yield text[len(emitted):]
      emitted = text


def generate(g, raw_model, forward_args, prompt, max_new_tokens, temperature, top_k):
  return "".join(stream(g, raw_model, forward_args, prompt, max_new_tokens, temperature, top_k))


def serve(g, raw_model, forward_args, port):
  """Blocking local HTTP server: POST / with {prompt, max_new_tokens, temperature, top_k} streams the completion as lines of JSON."""
  from http.server import BaseHTTPRequestHandler, HTTPServer

  # Compile kernels for the common lengths before taking requests; the longest bucket compiles on first use.
  for length in LENGTH_BUCKETS[:-1]:
    next_token_logits(g, raw_model, forward_args, [EOT_ID] * length)

  class Handler(BaseHTTPRequestHandler):
    def reply(self, status, body):
      payload = json.dumps(body).encode()
      self.send_response(status)
      self.send_header("Content-Type", "application/json")
      self.send_header("Content-Length", str(len(payload)))
      self.end_headers()
      self.wfile.write(payload)

    def do_GET(self):
      self.reply(200, {"status": "ok"})

    def do_POST(self):
      request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
      prompt = str(request.get("prompt", ""))[:MAX_PROMPT_CHARS]
      max_new_tokens = min(max(int(request.get("max_new_tokens", 120)), 1), MAX_NEW_TOKENS_LIMIT)
      temperature = min(max(float(request.get("temperature", 0.8)), 0.05), 2.0)
      top_k = min(max(int(request.get("top_k", 50)), 1), GPT2_VOCAB)
      # Newline-delimited JSON: one {"text": piece} per token, then {"seconds": total}.
      self.send_response(200)
      self.send_header("Content-Type", "application/x-ndjson")
      self.end_headers()
      started = time.perf_counter()
      try:
        for piece in stream(g, raw_model, forward_args, prompt, max_new_tokens, temperature, top_k):
          self.wfile.write((json.dumps({"text": piece}) + "\n").encode())
          self.wfile.flush()
        self.wfile.write((json.dumps({"seconds": round(time.perf_counter() - started, 2)}) + "\n").encode())
      except (BrokenPipeError, ConnectionResetError):
        pass  # the caller stopped listening; stop generating

    def log_message(self, *_):
      pass

  g["print0"](f"[generate] serving on port {port}", console=True)
  HTTPServer(("127.0.0.1", port), Handler).serve_forever()


def run(g):
  print0 = g["print0"]
  started = time.perf_counter()
  raw_model, forward_args = restore_model(g)
  print0(f"[generate] model restored in {time.perf_counter() - started:.1f}s "
         f"(windows short={forward_args.ws_short} long={forward_args.ws_long})", console=True)

  if os.environ.get("CHECK_VAL") == "1":
    started = time.perf_counter()
    loss = validation_loss(g, raw_model, forward_args)
    print0(f"[generate] val_loss:{loss:.4f} ({time.perf_counter() - started:.1f}s)", console=True)

  if os.environ.get("SERVE_PORT"):
    serve(g, raw_model, forward_args, int(os.environ["SERVE_PORT"]))
    return

  prompts = json.loads(os.environ["PROMPTS"]) if os.environ.get("PROMPTS") else DEFAULT_PROMPTS
  max_new_tokens = int(os.environ.get("MAX_NEW_TOKENS", "120"))
  temperature = float(os.environ.get("TEMPERATURE", "0.8"))
  top_k = int(os.environ.get("TOP_K", "50"))
  for prompt in prompts:
    started = time.perf_counter()
    text = generate(g, raw_model, forward_args, prompt, max_new_tokens, temperature, top_k)
    elapsed = time.perf_counter() - started
    print0(f"\n[generate] prompt: {prompt!r} ({max_new_tokens / elapsed:.1f} tok/s)\n{prompt}{text}", console=True)
