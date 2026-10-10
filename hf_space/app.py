"""
Soundstage — public Gradio Space with guest-token gate + ZeroGPU inference.
"""

from __future__ import annotations

import os
import secrets
from pathlib import Path

import gradio as gr

# Import spaces before torch (ZeroGPU requirement) when available
try:
    import spaces  # noqa: F401
except ImportError:
    spaces = None  # type: ignore

from pipeline import (
    DEFAULT_PRESET,
    PRESETS,
    SAMPLE_STEPS_DEFAULT,
    ensure_faststart,
    generate,
    warmup,
)

HERE = Path(__file__).resolve().parent
THEME_CSS = (HERE / "theme.css").read_text(encoding="utf-8")

ACCESS_TOKEN = (os.getenv("ACCESS_TOKEN") or "").strip()
ADMIN_TOKEN = (os.getenv("ADMIN_TOKEN") or "").strip()
MAX_VIDEO_SECONDS = float(os.getenv("MAX_VIDEO_SECONDS", "10"))


def _video_path(video) -> str | None:
    if video is None:
        return None
    if isinstance(video, str):
        return video
    return getattr(video, "name", None) or str(video)


def _check_token(token: str) -> tuple[bool, bool]:
    """Return (ok, is_admin)."""
    t = (token or "").strip()
    if not t:
        return False, False
    if ACCESS_TOKEN and secrets.compare_digest(t, ACCESS_TOKEN):
        return True, False
    if ADMIN_TOKEN and secrets.compare_digest(t, ADMIN_TOKEN):
        return True, True
    if not ACCESS_TOKEN and not ADMIN_TOKEN:
        return False, False
    return False, False


def enter_studio(token: str):
    ok, is_admin = _check_token(token)
    if not ok:
        if not ACCESS_TOKEN and not ADMIN_TOKEN:
            msg = "Space secrets ACCESS_TOKEN / ADMIN_TOKEN are not set."
        else:
            msg = "Invalid access token."
        return (
            gr.update(visible=True),
            gr.update(visible=False),
            msg,
            False,
            False,
            gr.update(visible=False),
            "Private preview",
        )
    role = "Admin" if is_admin else "Guest"
    return (
        gr.update(visible=False),
        gr.update(visible=True),
        "",
        True,
        is_admin,
        gr.update(visible=is_admin),
        f"Private preview · {role}",
    )


def sign_out():
    return (
        gr.update(visible=True),
        gr.update(visible=False),
        "",
        False,
        False,
        gr.update(visible=False),
        "Private preview",
        None,  # result video
        None,  # download
        "",    # status
    )


def on_upload(video):
    """Remux with faststart so first Gradio play starts immediately."""
    path = _video_path(video)
    if not path:
        return None
    try:
        return ensure_faststart(path)
    except Exception:
        return path


def run_generate(
    authed: bool,
    video,
    words: str,
    sfx: str,
    music: str,
    seed: float,
    steps: float,
    progress=gr.Progress(track_tqdm=False),
):
    if not authed:
        raise gr.Error("Please enter a valid access token first.")
    if video is None:
        raise gr.Error("Please upload a video first.")

    path = _video_path(video)
    try:
        out = generate(
            video_path=path,
            words=words or "",
            sfx=sfx or "",
            music=music or "",
            seed=int(seed),
            sample_steps=int(steps),
            progress=progress,
        )
        # out already has faststart from mux; ensure again if needed
        web = ensure_faststart(out) if out else out
        return web, web, "Done — soundtrack ready. Play below or download the MP4."
    except ValueError as e:
        raise gr.Error(str(e)) from e
    except Exception as e:
        raise gr.Error(str(e)[-800:]) from e


def clear_queue_admin(authed: bool, is_admin: bool):
    if not authed or not is_admin:
        raise gr.Error("Admin token required.")
    return "Queue note: new jobs wait behind the current GPU job. Running work cannot be aborted on ZeroGPU."


if not os.environ.get("SPACES_ZERO_GPU"):
    try:
        warmup()
    except Exception:
        pass


with gr.Blocks(
    title="Soundstage Preview",
    css=THEME_CSS,
    theme=gr.themes.Base(
        primary_hue=gr.themes.Color(
            c50="#e8f5f3",
            c100="#c5e8e3",
            c200="#9fd5cf",
            c300="#6bb8b0",
            c400="#3d9189",
            c500="#0c6b66",
            c600="#084f4b",
            c700="#063d3a",
            c800="#042e2c",
            c900="#021f1e",
            c950="#011413",
        ),
        font=[gr.themes.GoogleFont("IBM Plex Sans"), "system-ui", "sans-serif"],
    ),
) as demo:
    is_authed = gr.State(False)
    is_admin_state = gr.State(False)

    with gr.Column(visible=True, elem_classes=["ss-panel"]) as gate_col:
        gr.HTML(
            """
            <p class="ss-brand">Soundstage</p>
            <h1 class="ss-headline">Hear the picture.</h1>
            <p class="ss-lede">
              Private preview — generate speech, sound effects, and music locked to your clip.
            </p>
            """
        )
        token_in = gr.Textbox(
            label="Access token",
            type="password",
            placeholder="Paste the token you were given",
            autofocus=True,
        )
        gate_err = gr.Markdown("", elem_classes=["ss-error"])
        enter_btn = gr.Button("Enter studio", variant="primary")
        gr.HTML(
            f'<p class="ss-fine">Token stays in this browser only. '
            f"Clips ≤ {MAX_VIDEO_SECONDS:.0f}s "
            f"(longer clips are generated in ~10s chunks).</p>"
        )

    with gr.Column(visible=False) as studio_col:
        with gr.Row(elem_classes=["ss-topbar"]):
            gr.HTML('<span class="ss-brand">Soundstage</span>')
            caption = gr.Markdown("Private preview")
            signout_btn = gr.Button("Sign out", variant="secondary", size="sm")

        with gr.Column(elem_classes=["ss-panel"]):
            gr.HTML(
                """
                <h2 class="ss-section-title">Compose</h2>
                <p class="ss-lede">Describe what should be heard. At least one field below is enough.</p>
                """
            )
            # No format= remux — we apply faststart ourselves (avoids Gradio re-encode lag)
            video_in = gr.Video(
                label="Video",
                sources=["upload"],
                height=280,
            )
            words = gr.Textbox(
                label="Words (speech)",
                lines=2,
                placeholder="What should be spoken…",
            )
            sfx = gr.Textbox(
                label="Audio caption (SFX · ambience · voice)",
                lines=3,
                placeholder="Footsteps on gravel, distant traffic, a calm female voice…",
            )
            music = gr.Textbox(
                label="Music (optional)",
                lines=2,
                placeholder="Soft piano, medium tempo, warm and understated…",
            )
            preset = gr.Radio(
                choices=list(PRESETS.keys()),
                value=DEFAULT_PRESET,
                label="Speed / quality",
                info="Fast ≈ fewer steps (quicker). Quality ≈ more steps (cleaner).",
            )
            with gr.Accordion("Advanced", open=False):
                with gr.Row():
                    seed = gr.Number(label="Seed", value=103, precision=0)
                    steps = gr.Slider(
                        label="Sample steps",
                        minimum=5,
                        maximum=50,
                        step=1,
                        value=SAMPLE_STEPS_DEFAULT,
                    )
            gen_btn = gr.Button("Generate soundtrack", variant="primary")
            gr.HTML(
                '<p class="ss-fine">Jobs run one at a time on the shared preview GPU. '
                "First generate after idle is slower (model load). "
                "Clips over ~10s run as multiple chunks — expect longer waits.</p>"
            )
            status = gr.Markdown("")
            result = gr.Video(label="Result", interactive=False)
            download = gr.File(
                label="Download result MP4",
                interactive=False,
                visible=True,
            )

        with gr.Column(visible=False) as admin_col:
            clear_btn = gr.Button("Clear queue note", variant="secondary")
            admin_msg = gr.Markdown("")

    gate_outputs = [
        gate_col,
        studio_col,
        gate_err,
        is_authed,
        is_admin_state,
        admin_col,
        caption,
    ]

    enter_btn.click(enter_studio, inputs=[token_in], outputs=gate_outputs)
    token_in.submit(enter_studio, inputs=[token_in], outputs=gate_outputs)

    # Only on upload — not .change (would re-trigger after we write the remuxed path)
    video_in.upload(on_upload, inputs=[video_in], outputs=[video_in])

    def apply_preset(name: str):
        return PRESETS.get(name, SAMPLE_STEPS_DEFAULT)

    preset.change(apply_preset, inputs=[preset], outputs=[steps])

    signout_btn.click(
        sign_out,
        outputs=[
            gate_col,
            studio_col,
            gate_err,
            is_authed,
            is_admin_state,
            admin_col,
            caption,
            result,
            download,
            status,
        ],
    )

    gen_btn.click(
        run_generate,
        inputs=[is_authed, video_in, words, sfx, music, seed, steps],
        outputs=[result, download, status],
    )

    clear_btn.click(
        clear_queue_admin,
        inputs=[is_authed, is_admin_state],
        outputs=[admin_msg],
    )


if __name__ == "__main__":
    demo.queue(max_size=8, default_concurrency_limit=1).launch()
