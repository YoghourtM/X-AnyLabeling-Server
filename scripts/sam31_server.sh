#!/usr/bin/env bash
# Build and manage the local SAM 3.1 server container.
#
#   SAM31_CHECKPOINT=/path/to/sam3.1_multiplex.pt scripts/sam31_server.sh start
#
# Commands: build | start | stop | restart | status | logs
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE="${SAM31_IMAGE:-xal-sam31-server}"
CONTAINER="${SAM31_CONTAINER:-xal-sam31}"
PORT="${SAM31_PORT:-8000}"
BASE_IMAGE="${BASE_IMAGE:-cgd-sam3:2.10.0-cu128-local}"

usage() {
    sed -n '2,6p' "$0" | sed 's/^# \{0,1\}//'
    exit 1
}

checkpoint_dir() {
    local ckpt="${SAM31_CHECKPOINT:-}"
    if [[ -z "$ckpt" ]]; then
        echo "Set SAM31_CHECKPOINT to the sam3.1_multiplex.pt file." >&2
        exit 2
    fi
    ckpt="$(readlink -f "$ckpt")"
    if [[ ! -f "$ckpt" || "$(basename "$ckpt")" != "sam3.1_multiplex.pt" ]]; then
        echo "Not a sam3.1_multiplex.pt file: $ckpt" >&2
        exit 2
    fi
    dirname "$ckpt"
}

health() {
    curl -fsS --max-time 3 "http://127.0.0.1:${PORT}/health" 2>/dev/null
}

case "${1:-}" in
    build)
        docker build --build-arg "BASE_IMAGE=${BASE_IMAGE}" \
            -t "$IMAGE" -f "$REPO/docker/sam31/Dockerfile" "$REPO"
        ;;
    start)
        ckpt_dir="$(checkpoint_dir)"
        if docker ps -a --format '{{.Names}}' | grep -qx "$CONTAINER"; then
            echo "Container $CONTAINER already exists; use restart or stop." >&2
            exit 1
        fi
        mkdir -p "$REPO/logs"
        docker run -d --name "$CONTAINER" --gpus all \
            --user "$(id -u):$(id -g)" \
            -p "127.0.0.1:${PORT}:8000" \
            -v "$REPO:/app" \
            -v "$ckpt_dir:/ckpt:ro" \
            "$IMAGE" >/dev/null
        echo "Started $CONTAINER; loading models (about 20 s)..."
        for _ in $(seq 1 90); do
            if health >/dev/null; then
                echo "Ready: http://127.0.0.1:${PORT}"
                exit 0
            fi
            if ! docker ps --format '{{.Names}}' | grep -qx "$CONTAINER"; then
                echo "Container exited:" >&2
                docker logs --tail 40 "$CONTAINER" >&2
                exit 1
            fi
            sleep 2
        done
        echo "Server not ready yet; check: $0 logs" >&2
        exit 1
        ;;
    stop)
        docker rm -f "$CONTAINER" >/dev/null 2>&1 && echo "Stopped $CONTAINER" \
            || echo "$CONTAINER is not running"
        ;;
    restart)
        "$0" stop
        "$0" start
        ;;
    status)
        docker ps -a --filter "name=^${CONTAINER}$" \
            --format '{{.Names}}  {{.Status}}  {{.Ports}}'
        health && echo || echo "health check failed on port ${PORT}"
        ;;
    logs)
        docker logs --tail "${2:-100}" -f "$CONTAINER"
        ;;
    *)
        usage
        ;;
esac
