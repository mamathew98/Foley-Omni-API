---
name: Longer Video Chunking
overview: Raise the video cap to 30s on both the HF Space and Docker/FastAPI app by generating ~9.5s chunks (FOG-style), stitching audio, then muxing—so quality stays good without one long expensive sampling pass.
todos:
  - id: chunk-helpers
    content: Implement FOG-style plan_chunks/cut_clip/stitch in hf_space/pipeline.py and app/chunking.py
    status: completed
  - id: space-pipeline
    content: Raise Space max to 30s; chunked generate + progress; UI + SPACE_DEPLOY copy
    status: completed
  - id: fastapi-chunk
    content: Raise MAX_VIDEO_DURATION_SEC to 30; InferenceService chunked run_job; .env.example/DEPLOY
    status: completed
  - id: static-ui
    content: Update static Soundstage copy to max 30 seconds
    status: completed
  - id: push-space
    content: Push updated hf_space to mamathew/Soundstage
    status: completed
isProject: false
---

# Longer videos via FOG-style chunk → stitch (30s)

## Approach

Foley-Omni quality degrades / VRAM blows up on long single-pass clips. Mirror [Artimis/FOG](https://huggingface.co/spaces/Artimis/FOG): cut muted video into **≤9.5s chunks**, run existing `generate_audio_for_video` per chunk, **stitch WAVs**, mux once.

```mermaid
flowchart LR
  upload[Upload_up_to_30s]
  mute[Mute_copy]
  plan[Plan_chunks_9.5s]
  gen[Generate_per_chunk]
  stitch[Stitch_WAV]
  mux[Mux_faststart_MP4]

  upload --> mute --> plan --> gen --> stitch --> mux
```

**Locked defaults**
- Max length: **30s** (`MAX_VIDEO_SECONDS` / `MAX_VIDEO_DURATION_SEC`)
- Chunk size: **9.5s** (FOG `MAX_CHUNK`)
- Scope: **Space + FastAPI/static UI**
- Presets on Space unchanged (Fast default still helps long jobs)

## Shared chunking module

Add [`app/chunking.py`](app/chunking.py) used by both paths (Space can import via monorepo, or thin-copy helpers into `hf_space/` for standalone Space clones):

- `plan_chunks(duration) -> list[(start, end)]` + freeze-pad for Synchformer min length
- `cut_clip(src, start, end, out, fps, freeze_extra)` via ffmpeg (libx264, muted)
- `stitch_audio(chunks, wav_map, full_duration, out_wav)` — FOG-style overlay/trim/light stretch
- Constants: `MAX_CHUNK=9.5`, `MIN_GEN_DURATION=1.0`

Standalone Space (`vendor/` clone) must still work: keep parallel helpers in [`hf_space/pipeline.py`](hf_space/pipeline.py) **or** vendor the same `chunking.py` next to Space files and import it. Prefer **duplicate minimal chunk helpers inside `hf_space/pipeline.py`** (self-contained Space) **and** implement the same logic in `app/chunking.py` for Docker—avoid a fragile cross-import when Space is pushed without `app/`.

## HF Space ([`hf_space/pipeline.py`](hf_space/pipeline.py), [`hf_space/app.py`](hf_space/app.py))

1. Raise `MAX_VIDEO_SECONDS` default **10 → 30**.
2. Rewrite `generate()`:
   - Mute full video
   - `plan_chunks` / `cut_clip`
   - For each chunk: existing `_generate_one` under `_GPU_LOCK` (each call keeps `@spaces.GPU`)
   - Stitch → mux with `-movflags +faststart`
   - Progress: `chunk i/N`
3. Bump `FOLEY_GPU_SECONDS` default if needed (**120–180 per chunk** is fine; wall time grows with chunk count).
4. UI copy: gate/studio “≤ 30 seconds”; note longer clips take longer (chunked).
5. Update [`hf_space/SPACE_DEPLOY.md`](hf_space/SPACE_DEPLOY.md) env table.
6. Push to `mamathew/Soundstage` after implement.

## Docker / FastAPI

1. [`app/config.py`](app/config.py): default `MAX_VIDEO_DURATION_SEC` **10 → 30**; `build_engine_config` `duration` stays the **cap**, not per-pass length.
2. [`app/inference_service.py`](app/inference_service.py): `run_job` uses chunk plan when `duration_sec > MAX_CHUNK`; else single-pass (current path). Progress callbacks: “Generating chunk 2/3…”.
3. [`app/video_utils.py`](app/video_utils.py) / callers: validate against 30s.
4. [`.env.example`](.env.example) / [`DEPLOY.md`](DEPLOY.md): document `MAX_VIDEO_DURATION_SEC=30`.
5. Static UI ([`static/index.html`](static/index.html), any JS duration check): “max 30 seconds”.

## Quality / time expectations (document in UI fineprint)

| Length | Chunks | Rough Fast-preset time (warm) |
|--------|--------|-------------------------------|
| ≤9.5s | 1 | ~same as today |
| ~20s | 2–3 | ~2–3× |
| 30s | 3–4 | ~3–4× |

ZeroGPU free daily quota still applies; 30s Fast jobs can burn several minutes of GPU quota.

## Out of scope

- Raising past 30s
- Changing Docker compose hardware
- Crossfading / music continuity across chunks beyond FOG stitch behavior
