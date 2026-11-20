# Modal app `nanogpt-live` (modal_run.py) and the live demo's trainer (live.py).
# Modal uploads the local working tree, so uncommitted edits are what get deployed or run.

modal := "uvx modal"
live := "uvx --with modal python live.py"

default:
    @just --list

# Deploy the app. Refuses while a live trainer holds GPUs, since a deploy makes it give them up.
deploy: check-idle
    @git diff --quiet HEAD || echo "warning: deploying uncommitted changes"
    {{modal}} deploy modal_run.py

# Deploy even if a live trainer is running; it releases its GPUs about a minute later.
deploy-force:
    {{modal}} deploy modal_run.py

# Fail unless the live trainer is offline or stopped.
check-idle:
    #!/usr/bin/env bash
    set -euo pipefail
    status=$({{live}} status --json | python3 -c 'import json,sys; print(json.load(sys.stdin)["snapshot"]["status"])')
    if [[ "$status" != offline && "$status" != stopped ]]; then
        echo "The live trainer is $status. Run 'just release' first, or 'just deploy-force'." >&2
        exit 1
    fi

# Serve an older run again from the inference containers.
publish run_id:
    {{modal}} run modal_run.py::publish_run --run-id {{run_id}}

# Delete expired runs from the Volume; pass dry_run=true to only list them.
prune dry_run="false":
    {{modal}} run modal_run.py::prune {{ if dry_run == "true" { "--dry-run" } else { "" } }}

# Live demo trainer: status, watch, warm [minutes], pretrain, tune, restart, release.
status:
    {{live}} status

watch:
    {{live}} watch

warm minutes="45":
    {{live}} warm --minutes {{minutes}}

pretrain:
    {{live}} pretrain

tune *args:
    {{live}} tune {{args}}

restart:
    {{live}} restart

release:
    {{live}} release

# Show this month's Modal spend.
billing:
    {{modal}} billing summary
