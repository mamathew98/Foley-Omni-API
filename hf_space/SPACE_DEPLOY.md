# Deploy Soundstage on Hugging Face (ZeroGPU)

Public Gradio Space with a guest-token gate and free **ZeroGPU (A10G)** inference.

**Do not use Docker SDK** for this path — ZeroGPU only works with Gradio Spaces.

## 0. Account rules (important)

Hugging Face changed hosting: **Gradio/Docker on `cpu-basic` needs PRO**.  
Free personal accounts in good standing (verified email, account ≥ ~30 days) can still host **up to 2 ZeroGPU Gradio Spaces** for free.

| Path | Works on free? |
|------|----------------|
| Create Gradio Space via CLI (`upload` auto-creates on cpu-basic) | **No** → `402 Payment Required` |
| Create Gradio Space in the **web UI** with hardware **ZeroGPU** | **Yes** (≤2 Spaces) |
| PRO / Team org | Gradio + Docker + more ZeroGPU slots |

Do **not** use `huggingface-cli upload …` to *create* the Space. Create it in the browser first (ZeroGPU), then push files.

## 1. Create the Space (web UI)

1. Open [huggingface.co/new-space](https://huggingface.co/new-space)
2. Owner: your **personal** user (ZeroGPU free tier is for personal accounts; orgs need Team/Enterprise)
3. Name: e.g. `Soundstage`
4. SDK: **Gradio**
5. Hardware: **ZeroGPU** (A10G / `zero-a10g`) — pick this at create time if shown
6. Visibility: **Public** (access is gated by `ACCESS_TOKEN` in the UI)
7. Create Space

## 2. Confirm hardware is ZeroGPU

1. Space → **Settings** → **Hardware**
2. Must be **ZeroGPU** / **Nvidia A10G · ZeroGPU** (`zero-a10g`), not CPU Basic
3. Save

Without ZeroGPU, Foley-Omni will not fit. If the UI only offers CPU Basic and asks for PRO, your account is not eligible for the free ZeroGPU host quota (too new / email unverified / already at 2 Spaces) — then you need [PRO](https://huggingface.co/pro) or reuse/replace an existing ZeroGPU Space (e.g. duplicate `Artimis/FOG` under a personal account that already has ZeroGPU).

## 3. Add secrets

Space → **Settings** → **Variables and secrets** → **New secret**:

| Secret | Purpose |
|--------|---------|
| `ACCESS_TOKEN` | Guest token shown on the gate screen |
| `ADMIN_TOKEN` | Admin token (optional clear-queue note; must differ from guest) |

Optional:

| Name | Purpose |
|------|---------|
| `HF_TOKEN` | Only if Hub downloads need auth |
| `SAMPLE_STEPS` | Default steps (Space default `8` = Fast preset) |
| `FOLEY_CPU_OFFLOAD` | `0`/`1` — ZeroGPU defaults to **off** for speed; set `1` if OOM |
| `FOLEY_GPU_SECONDS` | `@spaces.GPU` duration budget (default `180`) |
| `MAX_VIDEO_SECONDS` | Max clip length (default `30`; chunked ~9.5s) |

## 4. Push this package as the Space root

The Space repo root must contain `app.py`, `README.md` (with YAML), `requirements.txt`, etc.

### Option A — clone Space and copy files

```bash
git clone https://huggingface.co/spaces/<user>/Soundstage
cd Soundstage
# Copy contents of Foley-Omni-API/hf_space/ into this repo root
cp -r /path/to/Foley-Omni-API/hf_space/* .
git add -A
git commit -m "Initial Soundstage ZeroGPU Space"
git push
```

### Option B — git subtree (from Foley-Omni-API root)

```bash
git subtree push --prefix=hf_space https://huggingface.co/spaces/<user>/Soundstage main
```

(Requires HF git credentials / token with write access to the Space.)

### Option C — Python upload (recommended on free accounts)

`huggingface-cli upload` always calls `create_repo` first. On free accounts that API returns **402** even when the Space **already exists**. Bypass it:

```powershell
# From Foley-Omni-API root — Space must already exist (web UI + ZeroGPU)
py -3.11 -c "from pathlib import Path; from huggingface_hub import HfApi; print(HfApi().upload_folder(folder_path=str(Path('hf_space').resolve()), repo_id='mamathew/Soundstage', repo_type='space', commit_message='Update Soundstage Space'))"
```

Replace `mamathew/Soundstage` with your Space id.

## 5. First build

1. Open the Space page — build installs `requirements.txt` + `packages.txt` (ffmpeg, git)
2. Cold start downloads weights from `CocoBro/Foley-Omni` (can take a long time)
3. README sets `startup_duration_timeout: 1h` so the first boot can finish
4. When the app is up: `https://<user>-<space>.hf.space/`

Standalone Space (only `hf_space/` files): startup clones Foley-Omni into `vendor/` then downloads checkpoints.  
Monorepo push of the whole Foley-Omni-API tree also works — `setup_weights.py` detects parent `inference_v2st.py`.

## 6. Use it

1. Open the Space URL
2. Paste the guest `ACCESS_TOKEN` → **Enter studio**
3. Upload a ≤30s video, fill Words / Audio caption / Music, **Generate soundtrack** (longer clips use ~10s chunks)
4. Wait for ZeroGPU allocation + generation (first call is slowest)

## Troubleshooting

| Symptom | Check |
|---------|--------|
| Gate says secrets not set | Add `ACCESS_TOKEN` (and ideally `ADMIN_TOKEN`) under Secrets |
| OOM / CUDA errors | Hardware must be **ZeroGPU A10G**; set secret `FOLEY_CPU_OFFLOAD=1` if needed |
| Stuck on “Preparing model” | First weight download / engine load — watch Space logs |
| Build fails on torch | Stick to the pinned versions in `requirements.txt` |
| Foley name in UI | Guest UI is Soundstage-only; Hub model id may still appear in logs |

## Paid alternative

Exact FastAPI + static HTML from the main repo needs a **Docker** Space with a **dedicated** paid GPU (ZeroGPU is Gradio-only). See root [`DEPLOY.md`](../DEPLOY.md) for self-hosted Docker.
