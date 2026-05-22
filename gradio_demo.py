import os
import shutil
import subprocess
import sys
import time
import uuid
from pathlib import Path

import gradio as gr
from PIL import Image


ROOT = Path(__file__).resolve().parent
OUTPUT_ROOT = ROOT / "demo_outputs"
INFER_SCRIPT = ROOT / "notebook" / "infer.py"
DEFAULT_EXAMPLE_DIR = Path("notebook/images/shutterstock_stylish_kidsroom_1640806567")
DEFAULT_IMAGE_PATH = ROOT / DEFAULT_EXAMPLE_DIR / "image.png"
DEFAULT_MASK_PATH = ROOT / DEFAULT_EXAMPLE_DIR / "14.png"

MODE_ARGS = {
    "full": [],
    "faster": ["--enable_acceleration"],
    "taylor": ["--enable_taylor"],
    "easy": ["--enable_easy"],
}


def _timestamp():
    return time.strftime("%Y%m%d_%H%M%S")


def _default_value(path):
    return str(path) if path.exists() else None


def _preview_file(path):
    if path and Path(path).exists():
        return path
    return None


def _format_elapsed(seconds):
    seconds = max(0.0, float(seconds))
    minutes = int(seconds // 60)
    remainder = seconds - minutes * 60
    if minutes:
        return f"{minutes}m {remainder:.1f}s"
    return f"{remainder:.1f}s"


def _status_html(state, mode, elapsed=None):
    elapsed_text = "--" if elapsed is None else _format_elapsed(elapsed)
    return f"""
    <div class="status-strip">
      <div>
        <span class="status-label">{state}</span>
        <span class="status-mode">{mode}</span>
      </div>
      <div class="elapsed-box">
        <span>Total Time</span>
        <strong>{elapsed_text}</strong>
        <em>Includes model loading and inference</em>
      </div>
    </div>
    """


def _new_session_dir():
    session_dir = OUTPUT_ROOT / f"{_timestamp()}_{uuid.uuid4().hex[:8]}"
    session_dir.mkdir(parents=True, exist_ok=False)
    return session_dir


def _save_image(src_path, dst_path):
    with Image.open(src_path) as image:
        image.convert("RGB").save(dst_path)


def _save_mask(src_path, dst_path):
    shutil.copyfile(src_path, dst_path)


def _latest_file(output_dir, suffix):
    files = sorted(output_dir.glob(f"*{suffix}"), key=lambda p: p.stat().st_mtime, reverse=True)
    return files[0] if files else None


def _build_command(
    image_path,
    output_dir,
    mode,
    seed,
    ss_faster_stride,
    ss_warmup,
    ss_order,
    ss_momentum_beta,
    slat_thresh,
    slat_warmup,
    slat_token_ratio,
    mesh_spectral_threshold_low,
    mesh_spectral_threshold_high,
):
    return [
        sys.executable,
        str(INFER_SCRIPT),
        "--image_path",
        str(image_path),
        "--mask_index",
        "0",
        "--output_dir",
        str(output_dir),
        "--seed",
        str(int(seed)),
        "--ss_faster_stride",
        str(int(ss_faster_stride)),
        "--ss_warmup",
        str(int(ss_warmup)),
        "--ss_order",
        str(int(ss_order)),
        "--ss_momentum_beta",
        str(ss_momentum_beta),
        "--slat_thresh",
        str(slat_thresh),
        "--slat_warmup",
        str(int(slat_warmup)),
        "--slat_token_ratio",
        str(slat_token_ratio),
        "--mesh_spectral_threshold_low",
        str(mesh_spectral_threshold_low),
        "--mesh_spectral_threshold_high",
        str(mesh_spectral_threshold_high),
        *MODE_ARGS[mode],
    ]


def generate(
    image_file,
    mask_file,
    mode,
    preview_format,
    seed,
    ss_faster_stride,
    ss_warmup,
    ss_order,
    ss_momentum_beta,
    slat_thresh,
    slat_warmup,
    slat_token_ratio,
    mesh_spectral_threshold_low,
    mesh_spectral_threshold_high,
):
    start_time = time.time()
    if image_file is None:
        yield None, [], _status_html("Waiting", mode, 0), "Please upload an image first.", None
        return
    if mask_file is None:
        yield None, [], _status_html("Waiting", mode, 0), "Please upload a mask first.", None
        return

    session_dir = _new_session_dir()
    input_dir = session_dir / "input"
    output_dir = session_dir / "output"
    input_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    image_path = input_dir / "image.png"
    mask_path = input_dir / "0.png"
    log_path = session_dir / "run.log"

    _save_image(image_file, image_path)
    _save_mask(mask_file, mask_path)

    cmd = _build_command(
        image_path=image_path,
        output_dir=output_dir,
        mode=mode,
        seed=seed,
        ss_faster_stride=ss_faster_stride,
        ss_warmup=ss_warmup,
        ss_order=ss_order,
        ss_momentum_beta=ss_momentum_beta,
        slat_thresh=slat_thresh,
        slat_warmup=slat_warmup,
        slat_token_ratio=slat_token_ratio,
        mesh_spectral_threshold_low=mesh_spectral_threshold_low,
        mesh_spectral_threshold_high=mesh_spectral_threshold_high,
    )

    env = os.environ.copy()
    env.setdefault("PYTHONPATH", str(ROOT))

    header = [
        f"Session: {session_dir}",
        f"Mode: {mode}",
        "Command:",
        " ".join(cmd),
        "",
    ]
    log_lines = header[:]
    yield None, [], _status_html("Running", mode, time.time() - start_time), "\n".join(log_lines), str(session_dir)

    with subprocess.Popen(
        cmd,
        cwd=str(ROOT),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    ) as proc:
        assert proc.stdout is not None
        with log_path.open("w", encoding="utf-8") as log_file:
            for line in proc.stdout:
                log_file.write(line)
                log_file.flush()
                log_lines.append(line.rstrip())
                yield None, [], _status_html("Running", mode, time.time() - start_time), "\n".join(log_lines[-120:]), str(session_dir)

        return_code = proc.wait()

    glb_path = _latest_file(output_dir, ".glb")
    ply_path = _latest_file(output_dir, ".ply")
    downloads = [str(p) for p in (glb_path, ply_path, log_path) if p is not None and Path(p).exists()]

    if return_code != 0:
        log_lines.append("")
        log_lines.append(f"Process failed with exit code {return_code}.")
        yield None, downloads, _status_html("Failed", mode, time.time() - start_time), "\n".join(log_lines[-160:]), str(session_dir)
        return

    preview_path = glb_path
    if preview_format == "ply" and ply_path is not None:
        preview_path = ply_path
    elif preview_format == "glb" and glb_path is not None:
        preview_path = glb_path

    if preview_path is None:
        log_lines.append("")
        log_lines.append("Generation finished, but no .glb or .ply file was found.")
        yield None, downloads, _status_html("No Output", mode, time.time() - start_time), "\n".join(log_lines[-160:]), str(session_dir)
        return

    log_lines.append("")
    log_lines.append("Done.")
    if glb_path is not None:
        log_lines.append(f"GLB: {glb_path}")
    if ply_path is not None:
        log_lines.append(f"PLY: {ply_path}")

    yield str(preview_path), downloads, _status_html("Completed", mode, time.time() - start_time), "\n".join(log_lines[-160:]), str(session_dir)


CSS = """
:root {
  --fast-bg: #f7f8fb;
  --fast-ink: #111827;
  --fast-muted: #64748b;
  --fast-line: #d9dee8;
  --fast-accent: #0f766e;
  --fast-accent-strong: #115e59;
}
.gradio-container {
  background: var(--fast-bg) !important;
  color: var(--fast-ink);
}
.fast-shell {
  max-width: 1380px;
  margin: 0 auto;
}
.fast-title {
  padding: 26px 0 20px;
  border-bottom: 1px solid var(--fast-line);
  margin-bottom: 18px;
  text-align: center;
}
.fast-title h1 {
  font-size: 68px;
  line-height: 1;
  margin: 0;
  letter-spacing: 0;
  font-weight: 900;
  background: linear-gradient(90deg, #0f766e 0%, #2563eb 38%, #9333ea 70%, #db2777 100%);
  -webkit-background-clip: text;
  background-clip: text;
  color: transparent;
}
.fast-title p {
  margin: 12px 0 0;
  color: #334155;
  font-size: 18px;
  font-weight: 650;
}
.status-strip {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 14px;
  padding: 14px 16px;
  border: 1px solid var(--fast-line);
  border-left: 5px solid var(--fast-accent);
  border-radius: 8px;
  background: #ffffff;
  margin-bottom: 12px;
}
.status-label {
  display: block;
  font-size: 22px;
  font-weight: 750;
  color: var(--fast-ink);
}
.status-mode {
  display: block;
  margin-top: 4px;
  text-transform: uppercase;
  letter-spacing: .08em;
  color: var(--fast-accent-strong);
  font-size: 12px;
  font-weight: 700;
}
.elapsed-box {
  min-width: 150px;
  text-align: right;
}
.elapsed-box span {
  display: block;
  color: var(--fast-muted);
  font-size: 12px;
  font-weight: 650;
  text-transform: uppercase;
  letter-spacing: .06em;
}
.elapsed-box strong {
  display: block;
  font-size: 30px;
  line-height: 1.05;
  color: var(--fast-accent-strong);
}
.elapsed-box em {
  display: block;
  margin-top: 5px;
  color: var(--fast-muted);
  font-size: 11px;
  font-style: normal;
  font-weight: 600;
}
.prominent-input label,
.prominent-input .label-wrap,
.prominent-input span[data-testid="block-info"] {
  font-size: 17px !important;
  font-weight: 800 !important;
  color: #0f172a !important;
}
.prominent-input input,
.prominent-input textarea {
  font-size: 15px !important;
}
.mode-panel label,
.mode-panel .label-wrap,
.mode-panel span[data-testid="block-info"] {
  font-size: 18px !important;
  font-weight: 850 !important;
  color: #0f172a !important;
}
.mode-panel .wrap label {
  font-size: 16px !important;
  font-weight: 750 !important;
}
.output-panel label,
.output-panel .label-wrap,
.output-panel span[data-testid="block-info"] {
  font-size: 18px !important;
  font-weight: 850 !important;
  color: #0f172a !important;
}
.upload-note {
  margin: -4px 0 10px;
  color: #475569;
  font-size: 13px;
  font-weight: 650;
}
.compact-log textarea {
  font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", monospace !important;
  font-size: 12px !important;
  line-height: 1.45 !important;
  color: #0f172a !important;
  background: #fbfcfe !important;
}
.panel-tight {
  border-radius: 8px !important;
}
.compact-download,
.compact-session {
  min-height: 96px !important;
}
.compact-download [data-testid="file"],
.compact-download .file-preview {
  min-height: 48px !important;
  max-height: 58px !important;
  overflow: auto !important;
}
.compact-session textarea {
  min-height: 58px !important;
  max-height: 58px !important;
  font-size: 12px !important;
  line-height: 1.35 !important;
}
button.primary {
  background: var(--fast-accent) !important;
  border-color: var(--fast-accent) !important;
}
"""


with gr.Blocks(title="Fast-SAM3D Demo", css=CSS) as demo:
    with gr.Column(elem_classes=["fast-shell"]):
        gr.HTML(
            """
            <div class="fast-title">
              <h1>Fast-SAM3D</h1>
              <p>Upload an image and mask, then generate a downloadable 3D asset</p>
            </div>
            """
        )

        with gr.Row():
            with gr.Column(scale=1, min_width=360):
                image_input = gr.Image(
                    label="🖼️ Upload Image",
                    type="filepath",
                    value=_default_value(DEFAULT_IMAGE_PATH),
                    elem_classes=["prominent-input"],
                )
                gr.HTML('<div class="upload-note">Original RGB image used by the generation pipeline.</div>')
                mask_input = gr.File(
                    label="🧩 Upload Mask PNG",
                    value=_default_value(DEFAULT_MASK_PATH),
                    file_types=[".png"],
                    type="filepath",
                    elem_classes=["prominent-input"],
                )
                gr.HTML('<div class="upload-note">Use the exact colorful mask PNG accepted by infer.py.</div>')
                mask_preview = gr.Image(
                    label="👁️ Mask Preview",
                    type="filepath",
                    value=_default_value(DEFAULT_MASK_PATH),
                    interactive=False,
                    height=180,
                    elem_classes=["prominent-input"],
                )
                mode_input = gr.Radio(
                    choices=["full", "faster", "taylor", "easy"],
                    value="faster",
                    label="🚀 Mode",
                    elem_classes=["mode-panel"],
                )
                preview_input = gr.Radio(
                    choices=["glb", "ply"],
                    value="glb",
                    label="👁️ Preview",
                    elem_classes=["mode-panel"],
                )
                seed_input = gr.Number(value=42, precision=0, label="🎲 Seed", elem_classes=["prominent-input"])

                with gr.Accordion("Faster Parameters", open=False):
                    ss_faster_stride_input = gr.Slider(1, 8, value=3, step=1, label="SS Faster Stride")
                    ss_warmup_input = gr.Slider(0, 8, value=2, step=1, label="SS Warmup")
                    ss_order_input = gr.Slider(1, 3, value=1, step=1, label="SS Order")
                    ss_momentum_beta_input = gr.Slider(0.0, 1.0, value=0.5, step=0.05, label="SS Momentum Beta")
                    slat_thresh_input = gr.Slider(0.0, 5.0, value=1.5, step=0.05, label="SLaT Threshold")
                    slat_warmup_input = gr.Slider(0, 8, value=3, step=1, label="SLaT Warmup")
                    slat_token_ratio_input = gr.Slider(0.0, 0.9, value=0.1, step=0.01, label="SLaT Token Ratio")

                with gr.Accordion("Mesh Parameters", open=False):
                    mesh_low_input = gr.Slider(0.0, 1.0, value=0.5, step=0.05, label="Spectral Threshold Low")
                    mesh_high_input = gr.Slider(0.0, 1.0, value=0.7, step=0.05, label="Spectral Threshold High")

                generate_button = gr.Button("Generate", variant="primary")

            with gr.Column(scale=2, min_width=680):
                status_output = gr.HTML(value=_status_html("Ready", "faster"))
                model_output = gr.Model3D(
                    label="🧊 3D Preview",
                    height=560,
                    elem_classes=["panel-tight", "output-panel"],
                )
                with gr.Row():
                    files_output = gr.Files(
                        label="📦 Downloads",
                        height=96,
                        elem_classes=["compact-download", "output-panel"],
                    )
                    session_output = gr.Textbox(
                        label="📁 Session",
                        lines=2,
                        max_lines=2,
                        interactive=False,
                        elem_classes=["compact-session", "output-panel"],
                    )
                log_output = gr.Textbox(
                    label="📝 Log",
                    lines=10,
                    interactive=False,
                    elem_classes=["compact-log", "output-panel"],
                )

    generate_button.click(
        fn=generate,
        inputs=[
            image_input,
            mask_input,
            mode_input,
            preview_input,
            seed_input,
            ss_faster_stride_input,
            ss_warmup_input,
            ss_order_input,
            ss_momentum_beta_input,
            slat_thresh_input,
            slat_warmup_input,
            slat_token_ratio_input,
            mesh_low_input,
            mesh_high_input,
        ],
        outputs=[model_output, files_output, status_output, log_output, session_output],
    )
    mask_input.change(fn=_preview_file, inputs=mask_input, outputs=mask_preview)


if __name__ == "__main__":
    demo.queue(default_concurrency_limit=1).launch(
        server_name="0.0.0.0",
        server_port=7861,
        share=True,
        show_error=True,
    )
