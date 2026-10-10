---
name: HF Soundstage Space
overview: Package a public Gradio + ZeroGPU Hugging Face Space that reuses Foley-Omni inference (like Artimis/FOG) but presents the Soundstage-branded UI, guest-token gate, and free A10G via ZeroGPU—because Docker Spaces cannot use ZeroGPU.
todos:
  - id: hf-space-scaffold
    content: "Create hf_space/ package: README YAML, requirements, packages.txt, SPACE_DEPLOY.md"
    status: completed
  - id: hf-space-pipeline
    content: Implement ZeroGPU-safe pipeline (spaces.GPU, cpu_offload, weight download) wired to this repo's Foley-Omni inference
    status: completed
  - id: hf-space-ui
    content: Build Soundstage Gradio UI with token gate, custom CSS matching static webpage, responsive layout
    status: completed
  - id: hf-space-docs
    content: Document HF create Space + ZeroGPU + secrets + push steps; link from DEPLOY.md
    status: completed
isProject: false
---

# Hugging Face Space: Soundstage (ZeroGPU)

## What we learned from [artimis-fog.hf.space](https://artimis-fog.hf.space/)

Your existing Space is **[Artimis/FOG](https://huggingface.co/spaces/Artimis/FOG)**:

| Fact | Detail |
|------|--------|
| SDK | **Gradio** (not Docker) |
| Hardware | **`zero-a10g`** (ZeroGPU shared A10G) — free tier GPU, not CPU-only |
| Model | Foley-Omni (`CocoBro/Foley-Omni`) with `cpu_offload`, reduced `sample_steps` (~15) |
| GPU API | `import spaces` + `@spaces.GPU` around generation ([ZeroGPU docs](https://huggingface.co/docs/hub/spaces-zerogpu)) |

**Important constraint:** ZeroGPU works **only with Gradio Spaces**. A Docker/FastAPI Space cannot use free ZeroGPU; that path needs a **paid dedicated GPU**.

So for “public + free + actually generate audio,” the correct approach is:

**Gradio + ZeroGPU + Soundstage-styled UI** (not Docker FastAPI on free CPU).

```mermaid
flowchart LR
  user[Public_visitor]
  gate[Guest_token_gate]
  ui[Soundstage_Gradio_UI]
  zg[ZeroGPU_A10G]
  model[FoleyOmni_engine]

  user --> gate --> ui
  ui -->|"@spaces.GPU"| zg --> model
```

## Locked decisions

- **Visibility:** Public Space + guest token gate (same idea as `ACCESS_TOKEN` / `ADMIN_TOKEN`)
- **Compute:** Free **ZeroGPU (`zero-a10g`)** — same class as FOG (real inference possible)
- **Brand in UI:** **Soundstage** (do not show “Foley-Omni” to guests)
- **Not Docker** for this free Space (Docker = paid GPU or UI-only)

## What we will implement in this repo

Add a self-contained Space package under [`hf_space/`](hf_space/) that can be pushed as a new HF Space (recommended: e.g. `Artimis/Soundstage` or replace FOG later).

```text
hf_space/
  README.md              # HF YAML frontmatter (sdk: gradio, suggested ZeroGPU)
  app.py                 # Gradio Blocks: token gate + Soundstage layout
  theme.css              # Port Soundstage look (fonts, colors, layout)
  pipeline.py            # Adapt FOG ZeroGPU pipeline patterns for this repo
  setup_weights.py       # Download CocoBro/Foley-Omni ckpts on first start
  requirements.txt
  packages.txt           # ffmpeg etc.
  SPACE_DEPLOY.md        # Step-by-step: create Space, set ZeroGPU, secrets, push
```

### UI (match the webpage)

Rebuild the Soundstage experience in Gradio Blocks + custom CSS:

- Gate screen: brand **Soundstage**, “Hear the picture.”, access-token field, Enter studio
- Studio: video upload, Words / Audio caption / Music, Advanced (seed/steps), Generate
- Results: video player + status/progress
- Admin clear-queue only if token matches `ADMIN_TOKEN` (simple in-memory queue / Gradio `queue`)
- Responsive CSS (mobile-friendly), same visual system as [`static/styles.css`](static/styles.css) (Syne / Instrument Serif / teal-on-paper)

Token check: compare against Space **Secrets** `ACCESS_TOKEN` / `ADMIN_TOKEN` (set in HF Settings). No Foley name in title/markdown.

### Inference (make free plan work)

Port the proven ZeroGPU pattern from Artimis/FOG into `hf_space/pipeline.py`, wired to **this** codebase’s [`inference_v2st.py`](inference_v2st.py) / [`FoleyOmniEngine`](foley_omni/fusion_engine.py):

- `import spaces` before torch when available
- `@spaces.GPU(duration=...)` around the generate step
- Defaults tuned for A10G ZeroGPU: `cpu_offload=True`, `sample_steps≈15`, `qint8=False` on ZeroGPU, `sp_size=1`
- Single-flight lock (one job at a time)
- Download weights at startup via `huggingface_hub` into a local `ckpts/` (first cold start is long; set `startup_duration_timeout: 1h` like FOG)

### Docs / ops

[`hf_space/SPACE_DEPLOY.md`](hf_space/SPACE_DEPLOY.md) step-by-step:

1. Create public Gradio Space on HF  
2. Hardware → **ZeroGPU (A10G)**  
3. Secrets: `ACCESS_TOKEN`, `ADMIN_TOKEN`  
4. Push `hf_space/` contents as the Space repo root (or git subtree)  
5. Wait for first build + weight download  
6. Open `https://<user>-<space>.hf.space/`, paste guest token  

Also add a short pointer from [`DEPLOY.md`](DEPLOY.md).

## What we will NOT do in this free Space path

- Do not use Docker SDK (blocks ZeroGPU)
- Do not expect free **CPU-only** hardware to load Foley-Omni (impossible)
- Do not expose Foley-Omni branding in the guest UI (internal logs/setup may still download `CocoBro/Foley-Omni`)

## Optional later (paid)

If you later want the exact FastAPI + static HTML app on HF: Docker Space + dedicated A10G/A100 — separate from this free ZeroGPU package.
