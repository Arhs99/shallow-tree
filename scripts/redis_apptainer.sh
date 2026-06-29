#!/usr/bin/env bash
# Run a Redis cache as an Apptainer instance, execute CMD with it up, then stop it.
# Requires an Apptainer image whose %startscript is:  exec redis-server "$@"
# Usage: REDIS_SIF=/path/to/redis_server.sif scripts/redis_apptainer.sh CMD...
#   e.g. REDIS_SIF=$PROJ/containers/redis_server.sif scripts/redis_apptainer.sh python input.py in.json
set -euo pipefail

SIF=${REDIS_SIF:?set REDIS_SIF to the redis_server.sif path}
PORT=${REDIS_PORT:-6379}

apptainer instance stop redis_server 2>/dev/null || true     # clear any stale instance
apptainer instance start "$SIF" redis_server \
  --bind 127.0.0.1 --port "$PORT" --save '' --appendonly no --maxmemory 1gb --maxmemory-policy allkeys-lru
trap 'apptainer instance stop redis_server' EXIT             # always stop Redis on exit (success, error, or interrupt)
until apptainer exec instance://redis_server redis-cli -p "$PORT" ping 2>/dev/null | grep -q PONG; do sleep 0.5; done

"$@"
