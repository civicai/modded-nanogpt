"""Drive the deployed LiveRun container (modal_run.py) from a terminal.

  uvx --with modal python live.py warm [--minutes 45]   # get the 4 GPUs and load the trainer, ahead of time
  uvx --with modal python live.py pretrain              # start pretraining and follow it
  uvx --with modal python live.py tune [--name Pip] [--dataset extra.json]
                                                        # start chat tuning and follow it; the model learns the name
                                                        # and the extra training data (chat_data.clean_dataset). Once
                                                        # tuned, this tunes the pretrained model again, replacing it
  uvx --with modal python live.py status [--json]
  uvx --with modal python live.py watch                 # follow whatever is happening
  uvx --with modal python live.py restart               # a fresh trainer on the same GPUs, to pretrain again
  uvx --with modal python live.py release               # give the GPUs back

`pretrain` and `tune` can be asked for at any time: with no container they first get one, and with the
trainer still loading they start as soon as it is ready. Add --no-follow to return at once.

State is read from the Modal Dict `nanogpt-live`, which costs nothing. Only warm, pretrain, tune and
restart can start the container, at about $0.30 a minute while it lasts.
"""
import argparse
import json
import sys
import time

import modal

APP = "nanogpt-live"
STALE_SECONDS = 20
MAX_QUEUE_SECONDS = 3600  # how long a request for GPUs is shown as queued before it is presumed lost
DONE = dict(pretrain=("pretrained", "tuning", "tuned"), tune=("tuned",))
CAN_ASK = dict(pretrain=("loading", "ready"), tune=("loading", "ready", "pretraining", "pretrained", "tuned"))

live_state = modal.Dict.from_name("nanogpt-live", create_if_missing=True)


def read_state():
    """The trainer's snapshot plus the last lease request, with `status` corrected for a container
    that is not there: "queued" while a requested one has yet to report, otherwise "offline"."""
    snapshot = live_state.get("snapshot") or dict(status="offline", updated=0, hold_until=0, samples=[])
    lease = live_state.get("lease") or dict(at=0, until=0)
    now = time.time()
    if now - snapshot["updated"] > STALE_SECONDS:
        waiting = lease["until"] > 0 and lease["at"] > snapshot["updated"] and now - lease["at"] < MAX_QUEUE_SECONDS
        snapshot = dict(snapshot, status="queued" if waiting else "offline", serving={})
    return snapshot, lease


def ask(minutes, action=None, options=None):
    """Hold the GPUs for `minutes` from now and, with `action`, ask the trainer for it. The calls are
    recorded as a lease: displays need it to tell a container that is queued from one that was never
    asked for, and `release` needs it to withdraw calls that are still queued."""
    snapshot, lease = read_state()
    queued = snapshot["status"] == "queued"
    now = time.time()
    run = modal.Cls.from_name(APP, "LiveRun")()
    calls = [run.hold.spawn(minutes)] + ([run.request.spawn(action, options)] if action else [])
    live_state["lease"] = dict(at=lease["at"] if queued else now, until=now + minutes * 60,
                               calls=(lease.get("calls", []) if queued else []) + [call.object_id for call in calls])


def release():
    snapshot, lease = read_state()
    if snapshot["status"] == "queued":
        for call_id in lease.get("calls", []):
            modal.FunctionCall.from_id(call_id).cancel()
    # A running container reads the lease below and gives the GPUs back; see LiveRun.watch.
    live_state["lease"] = dict(at=time.time(), until=0, calls=[])


def describe(snapshot):
    status = snapshot["status"]
    parts = [status]
    if status == "loading":
        loading = snapshot["loading"]
        parts.append(f"{time.time() - loading['started']:.0f}s in: {loading['note']}")
    for phase in ("pretrain", "tune"):
        progress = snapshot.get(phase)
        if progress:
            seconds = (progress["finished"] or time.time()) - progress["started"]
            parts.append(f"{phase} {progress['step']}/{progress['total']} in {seconds:.0f}s")
    if snapshot.get("pending"):
        parts.append(f"will {' then '.join(snapshot['pending'])} when ready")
    if snapshot.get("serving"):
        parts.append(f"serving {', '.join(snapshot['serving'])}")
    if status not in ("offline", "queued", "stopped"):
        parts.append(f"{snapshot.get('gpu', 'GPUs')} held {max(0, snapshot['hold_until'] - time.time()) / 60:.0f} more min")
    if snapshot.get("error"):
        parts.append(snapshot["error"])
    return " | ".join(parts)


def follow(until=(), begun=None):
    """Print status changes, losses and samples as they arrive, starting from now. Returns the
    snapshot once its status is in `until` and `begun(snapshot)`, if given, is true, or once the
    trainer has failed or gone. `begun` tells a phase that was asked for from one that ran before,
    as with a tuning asked for again from "tuned"."""
    seen, last_line, started = None, "", time.time()
    while True:
        snapshot, _ = read_state()
        status = snapshot["status"]
        series = dict(samples=snapshot["samples"],
                      pretrain=(snapshot.get("pretrain") or {}).get("val", []),
                      tune=(snapshot.get("tune") or {}).get("loss", []))
        if seen is None:
            seen = {name: len(points) for name, points in series.items()}
        # A list that got shorter has started over, for a new trainer or a new tuning: print it from its beginning.
        seen = {name: count if count <= len(series[name]) else 0 for name, count in seen.items()}
        line = dict(queued=f"queued for GPUs, {(time.time() - started) // 30 * 30:.0f}s so far",
                    loading=f"loading: {(snapshot.get('loading') or {}).get('note')}").get(status, status)
        if line != last_line:
            print(f"[{time.strftime('%H:%M:%S')}] {line}", flush=True)
            last_line = line
        for phase, label in (("pretrain", "validation loss"), ("tune", "loss")):
            for point in series[phase][seen[phase]:]:
                print(f"  {phase} step {point['step']}: {label} {point['loss']}", flush=True)
        for sample in series["samples"][seen["samples"]:]:
            print(f"  --- {sample['phase']} step {sample['step']} ---", flush=True)
            for item in sample["items"]:
                separator = " -> " if sample["phase"] == "tune" else ""
                print(f"  {item['prompt']}{separator}{item['text']}".replace("\n", " "), flush=True)
        seen = {name: len(points) for name, points in series.items()}
        if status in until and (begun is None or begun(snapshot)) or status in ("failed", "stopped", "offline"):
            return snapshot
        time.sleep(1)


def run_phase(action, minutes, should_follow, options=None):
    snapshot, _ = read_state()
    status = snapshot["status"]
    if status not in ("offline", "queued", "stopped", *CAN_ASK[action]):
        sys.exit(f"Cannot {action}: the trainer is {status}. Run `live.py restart` for a fresh trainer.")
    before = (snapshot.get(action) or {}).get("started")
    ask(minutes, action, options)
    print(f"asked to {action} (the trainer is {status})", flush=True)
    if should_follow:
        final = follow(DONE[action], lambda later: (later.get(action) or {}).get("started") != before)
        print(describe(final))
        if final["status"] not in DONE[action]:
            sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("command", choices=["warm", "pretrain", "tune", "status", "watch", "restart", "release"])
    parser.add_argument("--minutes", type=float, default=45, help="how long to keep the GPUs from now")
    parser.add_argument("--no-follow", action="store_true", help="return at once instead of following progress")
    parser.add_argument("--name", default="", help="tune: a name for the model, which it learns to give when asked who it is")
    parser.add_argument("--dataset", default="", help="tune: a JSON file of extra training data to add")
    parser.add_argument("--json", action="store_true", help="status: print the whole snapshot")
    args = parser.parse_args()
    if args.command == "status":
        snapshot, lease = read_state()
        print(json.dumps(dict(snapshot=snapshot, lease=lease), indent=1) if args.json else describe(snapshot))
    elif args.command == "watch":
        follow()
    elif args.command == "warm":
        ask(args.minutes)
        print(f"asked for the GPUs for {args.minutes:.0f} minutes", flush=True)
        if not args.no_follow:
            print(describe(follow(("ready", "pretraining", "pretrained", "tuning", "tuned"))))
    elif args.command in ("pretrain", "tune"):
        options = dict(name=args.name) if args.name else {}
        if args.dataset:
            with open(args.dataset) as f:
                options["dataset"] = json.load(f)
        run_phase(args.command, args.minutes, not args.no_follow, options or None)
    elif args.command == "restart":
        ask(args.minutes, "restart")
        print("asked for a fresh trainer", flush=True)
        if not args.no_follow:
            time.sleep(3)
            print(describe(follow(("ready",))))
    elif args.command == "release":
        release()
        print("GPUs released; a running container stops in about a minute")


if __name__ == "__main__":
    main()
