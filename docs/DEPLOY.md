# Foley-Omni Preview — Deploy Guide

Private GPU preview: guests open a simple webpage with an access token; jobs run **one at a time** on a single GPU.

**Public free Hugging Face Space (Gradio + ZeroGPU, Soundstage UI):** see **[SPACE_DEPLOY.md](SPACE_DEPLOY.md)**. Docker Spaces cannot use ZeroGPU — use that Gradio package for free A10G, or this Docker path with a dedicated GPU.

**Serving on a Linux PC over LAN / IP** (firewall, share URL + token, optional Nginx/VPN): see **[SERVE_ON_LINUX_PC.md](SERVE_ON_LINUX_PC.md)**.

## What you get

- Web UI at `http://<host>:8000/`
- REST API under `/api/*`
- Guest token (`ACCESS_TOKEN`) and admin token (`ADMIN_TOKEN`)
- Host-mounted checkpoints (`./ckpts`) — not baked into the image
- Persistent uploads/results in `./storage`

## Prerequisites

- Linux host with an NVIDIA GPU
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) (`docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi` must work)
- Docker Compose v2
- Enough disk for checkpoints (tens of GB) + storage for outputs
- Prefer **≥24GB VRAM**; if tighter, set `CPU_OFFLOAD=true` in `.env`

## One-shot deploy (recommended for DevOps)

After cloning, on a GPU server with Docker + NVIDIA Container Toolkit:

**Linux**

```bash
chmod +x deploy.sh deploy.bash
./deploy.sh
# or: ./deploy.bash
```

**Windows** (Docker Desktop + NVIDIA)

```bat
deploy.bat
```

These scripts will:

1. Create `.env` (and generate guest/admin tokens if still placeholders)
2. Download Hugging Face checkpoints into `./ckpts` (skipped if already present)
3. Verify checkpoint layout
4. `docker compose up --build -d` and print the UI URL + ACCESS_TOKEN

Useful flags:

```bash
./deploy.sh --skip-download    # ckpts already on disk
./deploy.sh --force-download   # re-fetch weights
./deploy.sh --no-logs          # do not tail container logs
```

Same flags work for `deploy.bat`.

---

## 1. Download checkpoints (manual)

From the repo root:

```bash
# Cross-platform
python scripts/download_release_ckpts.py CocoBro/Foley-Omni

# Or Linux shell helper
bash scripts/download_release_ckpts.sh CocoBro/Foley-Omni
python scripts/check_setup.py
```

Expected layout (abbreviated):

```text
ckpts/
├── Foley-Omni/v2st.pth
├── Wan2.2-TI2V-5B/...
└── mmaudio/ext_weights/...
```

## 2. Configure tokens

```bash
cp .env.example .env
```

Edit `.env`:

- `ACCESS_TOKEN` — share this with guests for the preview
- `ADMIN_TOKEN` — keep private; unlocks **Clear queue**
- Tokens **must** be different

## 3. Build and run

```bash
docker compose up --build -d
docker compose logs -f foley-omni
```

First start loads DiT + T5 + VAE into GPU memory (several minutes).  
`GET /api/health` is available early; `model_loaded` becomes `true` when inference is ready.

Open: `http://localhost:8000/`

### Docker deploy checklist

Before calling the stack production-ready, confirm:

1. `docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi` works  
2. `./ckpts/Foley-Omni/v2st.pth` (and related files) exist — `python scripts/check_setup.py`  
3. `.env` has distinct `ACCESS_TOKEN` / `ADMIN_TOKEN` (not placeholder values)  
4. Compose forces `SKIP_MODEL_LOAD=0` even if your local `.env` has `1` for UI smoke tests  
5. Port **8000** is free on the host; firewall opened if serving by IP  
6. Prefer **≥24GB VRAM**; set `CPU_OFFLOAD=true` in `.env` only if you must  

Verify after start:

```bash
docker compose ps
curl -s http://127.0.0.1:8000/api/health
# expect: "model_loaded": true  (may take several minutes)
docker compose logs -f foley-omni
```

## 4. Guest / admin usage

| Role | Token | Capabilities |
|------|--------|----------------|
| Guest | `ACCESS_TOKEN` | Submit jobs, cancel **own** queued jobs, download **own** results |
| Admin | `ADMIN_TOKEN` | Same + **Clear queue** (cancels queued only; running job continues) |

UI: paste token once (stored in browser `localStorage`).  
API: send header `X-API-Key: <token>`.

### Example API flow

```bash
# Who am I?
curl -s -H "X-API-Key: $ACCESS_TOKEN" http://localhost:8000/api/me

# Submit a job
curl -s -H "X-API-Key: $ACCESS_TOKEN" \
  -F "video=@clip.mp4" \
  -F "words=Hello there." \
  -F "audio_caption=Clear voice over soft room tone." \
  http://localhost:8000/api/jobs

# Poll
curl -s -H "X-API-Key: $ACCESS_TOKEN" http://localhost:8000/api/jobs/<job_id>

# Cancel if still queued
curl -s -X DELETE -H "X-API-Key: $ACCESS_TOKEN" \
  http://localhost:8000/api/jobs/<job_id>

# Download when done
curl -H "X-API-Key: $ACCESS_TOKEN" \
  -o out.mp4 http://localhost:8000/api/jobs/<job_id>/result

# Admin: empty the queue (not the running job)
curl -s -X POST -H "X-API-Key: $ADMIN_TOKEN" \
  http://localhost:8000/api/admin/queue/clear
```

## 5. Operational notes

- **One GPU owner**: Compose runs `uvicorn --workers 1`. Do not scale replicas against the same GPU.
- **Queue cap**: `MAX_QUEUE_SIZE` (default 10). Over capacity → HTTP `429`.
- **Video limit**: ≤30 seconds (configurable via `MAX_VIDEO_DURATION_SEC`). Clips over ~9.5s are generated in FOG-style chunks then stitched.
- **Cancel**: queued jobs only. Running CUDA work is not aborted in v1.
- **Restart**: queued/running jobs from a previous process are marked `failed`.
- **First visual extract**: OpenCLIP may download weights into the Hugging Face cache volume under `./storage/cache`.
- **Low VRAM**: set `CPU_OFFLOAD=true` in `.env` and recreate the container.

## 6. Local (non-Docker) smoke test

**Windows (UI only, no model):** double-click or run:

```bat
run_ui_local.bat
```

Uses Python **3.10–3.13** via the `py` launcher (creates `.venv-ui`).  
Python **3.14** is not supported for this smoke test (Pydantic wheels).

Then open `http://127.0.0.1:8000/` and paste the printed guest/admin token.  
Submitting a real job returns 503 while `SKIP_MODEL_LOAD=1`.

Manual / Linux equivalent:

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
source .venv/bin/activate
pip install -r requirements-web.txt omegaconf

export ACCESS_TOKEN=guest-demo
export ADMIN_TOKEN=admin-demo
export SKIP_MODEL_LOAD=1
uvicorn app.main:app --host 127.0.0.1 --port 8000 --workers 1
```

## 7. Useful commands

```bash
docker compose ps
docker compose logs -f foley-omni
docker compose restart foley-omni
docker compose down
```

## Security posture

This is a **shared-token preview**, not a public multi-tenant product:

- Put it behind a VPN, reverse proxy allowlist, or temporary tunnel you control
- Rotate `ACCESS_TOKEN` / `ADMIN_TOKEN` after each preview round if needed
- Do not expose port 8000 to the open internet without additional controls
