#!/usr/bin/env bash
# Foley-Omni preview — one-shot deploy for Linux servers.
# Usage:
#   chmod +x deploy.sh
#   ./deploy.sh
# Optional:
#   ./deploy.sh --skip-download
#   ./deploy.sh --force-download
#   HF_REPO=CocoBro/Foley-Omni ./deploy.sh

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT_DIR}"

HF_REPO="${HF_REPO:-CocoBro/Foley-Omni}"
SKIP_DOWNLOAD=0
FORCE_DOWNLOAD=0
FOLLOW_LOGS=1

for arg in "$@"; do
  case "${arg}" in
    --skip-download) SKIP_DOWNLOAD=1 ;;
    --force-download) FORCE_DOWNLOAD=1 ;;
    --no-logs) FOLLOW_LOGS=0 ;;
    -h|--help)
      cat <<'EOF'
Foley-Omni one-shot deploy

  ./deploy.sh                 Download ckpts (if needed), build & start Docker
  ./deploy.sh --skip-download Skip Hugging Face download
  ./deploy.sh --force-download Re-download checkpoints
  ./deploy.sh --no-logs       Start containers then exit (do not tail logs)

Env:
  HF_REPO   Hugging Face repo id (default: CocoBro/Foley-Omni)
EOF
      exit 0
      ;;
    *)
      echo "Unknown argument: ${arg}" >&2
      exit 1
      ;;
  esac
done

log() { echo "[deploy] $*"; }
die() { echo "[deploy] ERROR: $*" >&2; exit 1; }

need_cmd() {
  command -v "$1" >/dev/null 2>&1 || die "Missing required command: $1"
}

compose() {
  if docker compose version >/dev/null 2>&1; then
    docker compose "$@"
  elif command -v docker-compose >/dev/null 2>&1; then
    docker-compose "$@"
  else
    die "Docker Compose not found (need 'docker compose' or 'docker-compose')"
  fi
}

log "Project root: ${ROOT_DIR}"

need_cmd docker
need_cmd python3

if ! docker info >/dev/null 2>&1; then
  die "Docker daemon is not reachable. Start Docker and retry."
fi

if ! docker info 2>/dev/null | grep -qiE 'Runtimes:.*nvidia|nvidia'; then
  log "WARNING: NVIDIA runtime not clearly detected. GPU inference may fail."
  log "         Install NVIDIA Container Toolkit and verify: docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi"
fi

# --- .env / tokens ---
mapfile -t TOKEN_LINES < <(python3 "${ROOT_DIR}/scripts/ensure_env_tokens.py")
[[ "${#TOKEN_LINES[@]}" -ge 3 ]] || die "Failed to prepare .env tokens"
TOKEN_STATUS="${TOKEN_LINES[0]}"
ACCESS_TOKEN="${TOKEN_LINES[1]}"
ADMIN_TOKEN="${TOKEN_LINES[2]}"

if [[ "${ACCESS_TOKEN}" == "${ADMIN_TOKEN}" ]]; then
  die "ACCESS_TOKEN and ADMIN_TOKEN must be different in .env"
fi

if [[ "${TOKEN_STATUS}" == "GENERATED" ]]; then
  log "Generated preview tokens (saved in .env). Share ACCESS_TOKEN with guests only:"
  echo "         ACCESS_TOKEN=${ACCESS_TOKEN}"
  echo "         ADMIN_TOKEN=${ADMIN_TOKEN}"
else
  log "Using existing tokens from .env"
fi

mkdir -p "${ROOT_DIR}/storage" "${ROOT_DIR}/ckpts"

# --- checkpoints ---
if [[ "${SKIP_DOWNLOAD}" -eq 0 ]]; then
  log "Ensuring huggingface_hub is available..."
  python3 -m pip install -q -U "huggingface_hub[cli]>=0.30.0,<1.0" || \
    die "Failed to install huggingface_hub (need pip / network)"

  DL_ARGS=("${HF_REPO}" "--root" "${ROOT_DIR}")
  if [[ "${FORCE_DOWNLOAD}" -eq 1 ]]; then
    DL_ARGS+=(--force)
  fi
  log "Downloading checkpoints from ${HF_REPO} (skipped automatically if already present)..."
  python3 "${ROOT_DIR}/scripts/download_release_ckpts.py" "${DL_ARGS[@]}"
else
  log "Skipping checkpoint download (--skip-download)"
fi

if [[ -f "${ROOT_DIR}/scripts/check_setup.py" ]]; then
  log "Verifying checkpoint layout..."
  python3 "${ROOT_DIR}/scripts/check_setup.py" \
    --ckpt-dir "${ROOT_DIR}/ckpts" \
    --model-checkpoint "${ROOT_DIR}/ckpts/Foley-Omni/v2st.pth" \
    || die "Checkpoint check failed. Re-run with --force-download or fix ./ckpts"
fi

# --- docker ---
log "Building and starting containers (this can take a long time on first build)..."
compose -f "${ROOT_DIR}/docker-compose.yml" --env-file "${ROOT_DIR}/.env" up --build -d

log "Containers are up."
log "UI:   http://localhost:8000/"
log "Health: http://localhost:8000/api/health"
log "Guest token (ACCESS_TOKEN): ${ACCESS_TOKEN}"
log "Useful: docker compose logs -f foley-omni"
log "Stop:   docker compose down"

if [[ "${FOLLOW_LOGS}" -eq 1 ]]; then
  log "Tailing logs (Ctrl+C stops logs only; containers keep running)..."
  compose -f "${ROOT_DIR}/docker-compose.yml" logs -f foley-omni
fi
