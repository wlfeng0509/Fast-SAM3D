import base64
import html
import io
import os
import re
import shutil
import subprocess
import sys
import time
import uuid
from pathlib import Path

os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")

import gradio as gr
from PIL import Image


ROOT = Path(__file__).resolve().parent
OUTPUT_ROOT = ROOT / "demo_outputs"
INFER_SCRIPT = ROOT / "notebook" / "infer.py"
SCENE_INFER_SCRIPT = ROOT / "notebook" / "infer_scene.py"
DEFAULT_SCENE_DIR = ROOT / "notebook" / "images" / "shutterstock_stylish_kidsroom_1640806567"
DEFAULT_IMAGE_PATH = DEFAULT_SCENE_DIR / "image.png"
DEFAULT_MASK_PATH = DEFAULT_SCENE_DIR / "14.png"
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
MAX_SCENE_OBJECT_PREVIEWS = 48


def _timestamp():
    return time.strftime("%Y%m%d_%H%M%S")


def _new_session_dir(prefix):
    session_dir = OUTPUT_ROOT / f"{prefix}_{_timestamp()}_{uuid.uuid4().hex[:8]}"
    session_dir.mkdir(parents=True, exist_ok=False)
    return session_dir


def _default_value(path):
    return str(path) if path.exists() else None


def _format_elapsed(seconds):
    seconds = max(0.0, float(seconds))
    minutes = int(seconds // 60)
    remainder = seconds - minutes * 60
    return f"{minutes}m {remainder:.1f}s" if minutes else f"{remainder:.1f}s"


def _status_html(state="Ready", elapsed=None, detail="", progress=None):
    elapsed_text = "--" if elapsed is None else _format_elapsed(elapsed)
    progress_html = ""
    if progress is not None:
        progress = max(0, min(100, int(progress)))
        progress_html = f"""
      <div class="mv-progress-bar"><div style="width:{progress}%"></div></div>
      <div class="mv-progress-meta"><span>{progress}%</span></div>
        """
    return f"""
    <div class="mv-status-card">
      <div class="mv-progress-head">
        <span>{html.escape(state)}</span>
        <b>{elapsed_text}</b>
      </div>
      {progress_html}
      <div class="mv-current"><div><b>Status</b><span>{html.escape(detail or state)}</span></div></div>
    </div>
    """


def _image_to_data_uri(path, max_side=900):
    with Image.open(path) as image:
        image = image.convert("RGB")
        image.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=90)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def _scene_mask_paths(scene_dir):
    scene_dir = Path(scene_dir)
    masks = []
    idx = 0
    while (scene_dir / f"{idx}.png").exists():
        masks.append(scene_dir / f"{idx}.png")
        idx += 1
    return masks


def _single_mask_choices(scene_dir):
    return [path.name for path in _scene_mask_paths(scene_dir)]


def _default_mask_choice(scene_dir=DEFAULT_SCENE_DIR):
    choices = _single_mask_choices(scene_dir)
    default_name = DEFAULT_MASK_PATH.name
    if default_name in choices:
        return default_name
    return choices[0] if choices else None


def _resolve_mask_choice(scene_dir, mask_choice):
    if not mask_choice:
        return None
    path = Path(mask_choice)
    if path.is_absolute():
        return str(path) if path.exists() else None
    path = Path(scene_dir) / str(mask_choice)
    return str(path) if path.exists() else None


def update_single_directory(scene_dir):
    scene_dir = Path(scene_dir)
    choices = _single_mask_choices(scene_dir)
    value = _default_mask_choice(scene_dir)
    image_path = scene_dir / "image.png"
    mask_path = _resolve_mask_choice(scene_dir, value)
    return (
        str(image_path) if image_path.exists() else None,
        gr.update(choices=choices, value=value),
        mask_path,
    )


def update_single_mask_preview(scene_dir, mask_choice):
    return _resolve_mask_choice(scene_dir, mask_choice)


def _scene_summary(scene_dir):
    scene_dir = Path(scene_dir)
    image_path = scene_dir / "image.png"
    masks = _scene_mask_paths(scene_dir)
    if not scene_dir.exists():
        return """
        <div class="mv-scene-summary">
          <div class="mv-summary-title">Scene path not found</div>
        </div>
        """
    badges = "".join(f"<span>{path.stem}</span>" for path in masks[:40])
    if len(masks) > 40:
        badges += f"<em>+{len(masks) - 40} more</em>"
    return f"""
    <div class="mv-scene-summary">
      <div class="mv-summary-title">{html.escape(scene_dir.name)}</div>
      <div class="mv-summary-row"><b>{'ready' if image_path.exists() else 'missing'}</b> image.png</div>
      <div class="mv-summary-row"><b>{len(masks)}</b> object masks</div>
      <div class="mv-badges">{badges or '<em>No masks found.</em>'}</div>
    </div>
    """


def _scene_preview_html(scene_dir):
    scene_dir = Path(scene_dir)
    image_path = scene_dir / "image.png"
    if not image_path.exists():
        return _scene_summary(scene_dir)
    return f"""
    <section class="mv-carousel-card">
      <div class="mv-carousel-head"><span>Input Scene</span><span>{html.escape(scene_dir.name)}</span></div>
      <div class="mv-carousel" style="height:300px;">
        <div class="mv-slide" style="display:flex;"><img src="{_image_to_data_uri(image_path)}" /></div>
      </div>
    </section>
    {_scene_summary(scene_dir)}
    """


def _preview_file(path):
    if path and Path(path).exists():
        return path
    return None


def _latest_file(output_dir, suffix):
    files = sorted(Path(output_dir).glob(f"*{suffix}"), key=lambda p: p.stat().st_mtime, reverse=True)
    return files[0] if files else None


def _scene_object_glbs(objects_dir):
    objects_dir = Path(objects_dir)
    if not objects_dir.exists():
        return []
    return sorted(objects_dir.glob("*.glb"), key=lambda p: p.name.lower())


def _empty_scene_object_updates():
    return [gr.update(value=None, visible=False) for _ in range(MAX_SCENE_OBJECT_PREVIEWS)]


def _scene_object_updates(glb_paths):
    updates = []
    for path in glb_paths[:MAX_SCENE_OBJECT_PREVIEWS]:
        updates.append(gr.update(value=str(path), visible=True))
    updates.extend(gr.update(value=None, visible=False) for _ in range(MAX_SCENE_OBJECT_PREVIEWS - len(updates)))
    return updates


def _voxel_occupancy_html(path=None):
    if not path or not Path(path).exists():
        return """
        <div class="mv-voxel-empty">
          <div>SS Voxel Occupancy</div>
          <span>Waiting for sparse-structure decode.</span>
        </div>
        """
    file_path = Path(path).resolve()
    try:
        iframe_doc = file_path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        return f"""
        <div class="mv-voxel-empty">
          <div>SS Voxel Occupancy</div>
          <span>Failed to load saved HTML: {html.escape(str(exc))}</span>
        </div>
        """
    srcdoc = html.escape(iframe_doc, quote=True)
    return f"""
    <section class="mv-voxel-card">
      <div class="mv-voxel-head">
        <span>SS Voxel Occupancy</span>
        <span>{html.escape(file_path.name)}</span>
      </div>
      <iframe srcdoc="{srcdoc}" loading="lazy"></iframe>
    </section>
    """


def _add_accel_flags(cmd, enable_ss, enable_slat, enable_mesh):
    if enable_ss:
        cmd.append("--enable_ss_faster")
    if enable_slat:
        cmd.append("--enable_slat_token")
    if enable_mesh:
        cmd.append("--enable_mesh_aggregation")
    return cmd


def _scene_progress_from_line(line, current_progress):
    text = line.strip()
    match = re.search(r"Processing object\s+(\d+)/(\d+)", text)
    if match:
        index = int(match.group(1))
        total = max(1, int(match.group(2)))
        progress = min(86, 8 + int(72 * index / total))
        return progress, f"Processing object {index}/{total}"
    if "Compositing scene" in text:
        return max(current_progress, 90), "Compositing scene"
    if "Rendering video" in text:
        return max(current_progress, 94), "Rendering scene video"
    if "GIF saved to" in text:
        return max(current_progress, 97), "GIF saved"
    if "MP4 saved to" in text:
        return 99, "MP4 saved"
    return current_progress, None


def _run_command(cmd, session_dir, start_time, state_label):
    log_path = session_dir / "run.log"
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT)
    env["TORCH_HOME"] = str(ROOT / "checkpoints" / "torch-cache")

    log_lines = [
        f"Session: {session_dir}",
        "Command:",
        " ".join(str(x) for x in cmd),
        "",
    ]
    yield None, [], _status_html("Running", time.time() - start_time, state_label), "\n".join(log_lines), str(session_dir)

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
                yield None, [], _status_html("Running", time.time() - start_time, state_label), "\n".join(log_lines[-120:]), str(session_dir)
        return_code = proc.wait()

    yield return_code, log_lines, log_path


def _save_image(src_path, dst_path):
    with Image.open(src_path) as image:
        image.convert("RGB").save(dst_path)


def generate_single(image_file, scene_dir, mask_choice, seed, enable_ss, enable_slat, enable_mesh):
    start_time = time.time()
    if image_file is None:
        yield None, [], _status_html("Waiting", 0, "Please provide an image."), _voxel_occupancy_html(), "", None
        return
    mask_file = _resolve_mask_choice(scene_dir, mask_choice)
    if mask_file is None:
        yield None, [], _status_html("Waiting", 0, "Please provide a mask."), _voxel_occupancy_html(), "", None
        return

    session_dir = _new_session_dir("single")
    input_dir = session_dir / "input"
    output_dir = session_dir / "output"
    input_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    image_path = input_dir / "image.png"
    mask_path = input_dir / "0.png"
    _save_image(image_file, image_path)
    shutil.copyfile(mask_file, mask_path)

    cmd = [
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
    ]
    _add_accel_flags(cmd, enable_ss, enable_slat, enable_mesh)

    runner = _run_command(cmd, session_dir, start_time, "Single-object generation")
    for update in runner:
        if isinstance(update[0], int):
            return_code, log_lines, log_path = update
            break
        yield update[0], update[1], update[2], _voxel_occupancy_html(), update[3], update[4]

    glb_path = _latest_file(output_dir, ".glb")
    ply_path = _latest_file(output_dir, ".ply")
    voxel_html = output_dir / "ss_voxel_occupancy.html"
    downloads = [str(p) for p in (glb_path, ply_path, voxel_html, log_path) if p is not None and Path(p).exists()]
    if return_code != 0:
        log_lines.append(f"Process failed with exit code {return_code}.")
        yield None, downloads, _status_html("Failed", time.time() - start_time, "Single-object generation failed"), _voxel_occupancy_html(voxel_html), "\n".join(log_lines[-160:]), str(session_dir)
        return
    preview = glb_path or ply_path
    yield str(preview) if preview else None, downloads, _status_html("Completed", time.time() - start_time, "Single object ready"), _voxel_occupancy_html(voxel_html), "\n".join(log_lines[-160:]), str(session_dir)


def generate_scene(scene_dir, seed, enable_ss, enable_slat, enable_mesh):
    start_time = time.time()
    scene_dir = Path(scene_dir)
    if not scene_dir.exists():
        yield None, None, [], _status_html("Waiting", 0, "Scene directory does not exist.", 0), "", None, *_empty_scene_object_updates()
        return
    if not (scene_dir / "image.png").exists():
        yield None, None, [], _status_html("Waiting", 0, "Scene directory must contain image.png.", 0), "", None, *_empty_scene_object_updates()
        return

    session_dir = _new_session_dir("scene")
    output_dir = session_dir / "output"
    objects_dir = session_dir / "objects"
    output_dir.mkdir(parents=True, exist_ok=True)
    objects_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        str(SCENE_INFER_SCRIPT),
        "--image_dir",
        str(scene_dir),
        "--output_dir",
        str(output_dir),
        "--objects_dir",
        str(objects_dir),
        "--seed",
        str(int(seed)),
    ]
    _add_accel_flags(cmd, enable_ss, enable_slat, enable_mesh)

    log_path = session_dir / "run.log"
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT)
    env["TORCH_HOME"] = str(ROOT / "checkpoints" / "torch-cache")
    log_lines = [
        f"Session: {session_dir}",
        "Command:",
        " ".join(str(x) for x in cmd),
        "",
    ]
    progress = 2
    detail = "Starting multi-object scene generation"
    yield None, None, [], _status_html("Running", time.time() - start_time, detail, progress), "\n".join(log_lines), str(session_dir), *_empty_scene_object_updates()

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
                progress, parsed_detail = _scene_progress_from_line(line, progress)
                if parsed_detail:
                    detail = parsed_detail
                object_glbs = _scene_object_glbs(objects_dir)
                yield None, None, [], _status_html("Running", time.time() - start_time, detail, progress), "\n".join(log_lines[-140:]), str(session_dir), *_scene_object_updates(object_glbs)
        return_code = proc.wait()

    ply_path = _latest_file(output_dir, ".ply")
    mp4_path = _latest_file(output_dir, ".mp4")
    gif_path = _latest_file(output_dir, ".gif")
    object_glbs = _scene_object_glbs(objects_dir)
    downloads = [str(p) for p in (ply_path, mp4_path, gif_path, log_path) if p is not None and Path(p).exists()]
    downloads.extend(str(p) for p in object_glbs if p.exists())
    if return_code != 0:
        log_lines.append(f"Process failed with exit code {return_code}.")
        yield None, None, downloads, _status_html("Failed", time.time() - start_time, "Scene generation failed", progress), "\n".join(log_lines[-180:]), str(session_dir), *_scene_object_updates(object_glbs)
        return

    video_preview = str(mp4_path or gif_path) if (mp4_path or gif_path) else None
    preview_detail = f"Scene video ready; {len(object_glbs)} object GLB file(s) saved"
    if video_preview is None:
        preview_detail = f"Merged PLY and {len(object_glbs)} object GLB file(s) are available; no video file was produced"
    yield None, video_preview, downloads, _status_html("Completed", time.time() - start_time, preview_detail, 100), "\n".join(log_lines[-180:]), str(session_dir), *_scene_object_updates(object_glbs)


CSS = """
:root {
  --mvsam-card: rgba(255, 255, 255, 0.94);
  --mvsam-ink: #101827;
  --mvsam-muted: #34425a;
  --mvsam-line: rgba(16, 24, 39, 0.14);
  --mvsam-green: #139b63;
  --mvsam-blue: #235dff;
  --mvsam-rose: #e84d7a;
}
body {
  background:
    radial-gradient(circle at 10% 10%, rgba(19, 155, 99, 0.23), transparent 26%),
    radial-gradient(circle at 90% 10%, rgba(35, 93, 255, 0.21), transparent 28%),
    linear-gradient(135deg, #effaf5 0%, #f7fbff 52%, #fff3f8 100%);
}
.gradio-container {
  max-width: 1200px !important;
  margin: 0 auto !important;
  padding: 22px 18px 36px !important;
  color: var(--mvsam-ink);
}
#app-title {
  max-width: 1040px;
  margin: 0 auto 18px;
  padding: 22px 28px;
  border: 1px solid var(--mvsam-line);
  border-radius: 22px;
  background:
    linear-gradient(135deg, rgba(255,255,255,0.98), rgba(255,255,255,0.76)),
    linear-gradient(135deg, rgba(19,155,99,0.16), rgba(35,93,255,0.12));
  box-shadow: 0 24px 80px rgba(35, 49, 73, 0.14);
  text-align: center;
}
#app-title h1 {
  margin: 0;
  font-size: 36px;
  line-height: 1.16;
  letter-spacing: 0;
  color: var(--mvsam-ink);
}
#app-title p {
  margin: 8px 0 0;
  color: var(--mvsam-muted);
  font-size: 15px;
  font-weight: 700;
}
.mvsam-panel {
  border: 1px solid var(--mvsam-line) !important;
  border-radius: 18px !important;
  background: var(--mvsam-card) !important;
  box-shadow: 0 18px 48px rgba(35, 49, 73, 0.12) !important;
  padding: 12px !important;
}
#run-button, #scene-run-button {
  border: 0 !important;
  min-height: 50px !important;
  border-radius: 14px !important;
  background: linear-gradient(135deg, var(--mvsam-green), var(--mvsam-blue) 68%, var(--mvsam-rose)) !important;
  color: white !important;
  font-size: 16px !important;
  font-weight: 800 !important;
}
.mv-status-card {
  margin-bottom: 14px;
  padding: 16px 18px;
  border-radius: 18px;
  background: linear-gradient(135deg, rgba(19,155,99,0.14), rgba(35,93,255,0.13), rgba(232,77,122,0.10));
  border: 1px solid rgba(16, 24, 39, 0.10);
}
.mv-progress-head {
  display: flex;
  justify-content: space-between;
  align-items: center;
  gap: 18px;
  font-size: 22px;
  font-weight: 850;
}
.mv-progress-head b {
  font-size: 34px;
  line-height: 1;
  color: var(--mvsam-ink);
}
.mv-progress-bar {
  height: 14px;
  margin-top: 16px;
  overflow: hidden;
  border-radius: 999px;
  background: rgba(16, 24, 39, 0.10);
}
.mv-progress-bar div {
  height: 100%;
  border-radius: 999px;
  background: linear-gradient(90deg, var(--mvsam-green), var(--mvsam-blue) 70%, var(--mvsam-rose));
  transition: width 0.25s ease;
}
.mv-progress-meta {
  display: flex;
  justify-content: flex-end;
  margin-top: 6px;
  color: var(--mvsam-muted);
  font-size: 17px;
  font-weight: 850;
}
.mv-current {
  margin-top: 12px;
  color: var(--mvsam-muted);
  font-size: 18px;
  font-weight: 700;
}
.mv-current b {
  margin-right: 8px;
  color: var(--mvsam-ink);
}
.mv-scene-summary {
  margin: 8px 0 12px;
  padding: 14px 16px;
  border-radius: 16px;
  background: rgba(255,255,255,0.72);
  border: 1px solid rgba(16, 24, 39, 0.10);
}
.mv-summary-title {
  font-size: 19px;
  font-weight: 850;
}
.mv-summary-row {
  margin-top: 7px;
  color: var(--mvsam-muted);
  font-size: 15px;
  font-weight: 650;
}
.mv-badges {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  margin-top: 12px;
}
.mv-badges span {
  padding: 6px 10px;
  border-radius: 999px;
  color: #0f5132 !important;
  background: rgba(19, 155, 99, 0.12);
  font-size: 13px !important;
  font-weight: 800;
}
.mv-carousel-card {
  border: 1px solid var(--mvsam-line);
  border-radius: 18px;
  background: rgba(255,255,255,0.90);
  overflow: hidden;
}
.mv-carousel-head {
  display: flex;
  justify-content: space-between;
  padding: 12px 14px;
  font-size: 15px;
  font-weight: 850;
  color: var(--mvsam-ink);
  border-bottom: 1px solid var(--mvsam-line);
}
.mv-carousel {
  position: relative;
  display: flex;
  align-items: center;
  justify-content: center;
  background: #f7fafc;
}
.mv-slide {
  width: 100%;
  height: 100%;
  align-items: center;
  justify-content: center;
}
.mv-slide img {
  max-width: 100%;
  max-height: 100%;
  object-fit: contain;
}
.mv-voxel-card {
  margin-top: 12px;
  overflow: hidden;
  border: 1px solid var(--mvsam-line);
  border-radius: 14px;
  background: rgba(255,255,255,0.92);
}
.mv-voxel-head {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 10px 12px;
  color: var(--mvsam-ink);
  font-size: 14px;
  font-weight: 850;
  border-bottom: 1px solid var(--mvsam-line);
}
.mv-voxel-head a {
  color: var(--mvsam-blue) !important;
  font-size: 13px;
  font-weight: 850;
  text-decoration: none;
}
.mv-voxel-card iframe {
  display: block;
  width: 100%;
  height: 260px;
  border: 0;
  background: #f7fafc;
}
.mv-voxel-empty {
  margin-top: 12px;
  padding: 18px;
  border: 1px dashed var(--mvsam-line);
  border-radius: 14px;
  color: var(--mvsam-muted);
  background: rgba(255,255,255,0.72);
}
.mv-voxel-empty div {
  color: var(--mvsam-ink);
  font-size: 14px;
  font-weight: 850;
}
.mv-voxel-empty span {
  display: block;
  margin-top: 4px;
  font-size: 13px;
  font-weight: 650;
}
.compact-log textarea {
  font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", monospace !important;
  font-size: 12px !important;
  line-height: 1.45 !important;
}
"""


with gr.Blocks(title="Fast-SAM3D", css=CSS) as demo:
    gr.HTML(
        """
        <div id="app-title">
          <h1>Fast-SAM3D Reconstruction</h1>
          <p>YAML-driven acceleration with optional SS, SLaT, and Mesh modules.</p>
        </div>
        """
    )

    with gr.Tabs():
        with gr.Tab("Single Object"):
            with gr.Row():
                with gr.Column(scale=1, min_width=360, elem_classes=["mvsam-panel"]):
                    single_dir = gr.Textbox(label="Image / Mask Directory", value=str(DEFAULT_SCENE_DIR), lines=2)
                    single_image = gr.Image(label="Image", type="filepath", value=_default_value(DEFAULT_IMAGE_PATH))
                    single_mask = gr.Dropdown(
                        label="Mask PNG",
                        choices=_single_mask_choices(DEFAULT_SCENE_DIR),
                        value=_default_mask_choice(DEFAULT_SCENE_DIR),
                        allow_custom_value=False,
                    )
                    single_mask_preview = gr.Image(label="Mask Preview", type="filepath", value=_default_value(DEFAULT_MASK_PATH), interactive=False, height=180)
                    single_seed = gr.Number(value=42, precision=0, label="Seed")
                    single_ss = gr.Checkbox(value=True, label="Enable SS acceleration")
                    single_slat = gr.Checkbox(value=True, label="Enable SLaT acceleration")
                    single_mesh = gr.Checkbox(value=True, label="Enable Mesh acceleration")
                    single_button = gr.Button("Generate Single Object", variant="primary", elem_id="run-button")

                with gr.Column(scale=2, min_width=620, elem_classes=["mvsam-panel"]):
                    single_status = gr.HTML(value=_status_html("Ready", None, "Single object"))
                    single_model = gr.Model3D(label="3D Preview", height=520)
                    single_voxel = gr.HTML(value=_voxel_occupancy_html())
                    with gr.Row():
                        single_files = gr.Files(label="Downloads", height=100)
                        single_session = gr.Textbox(label="Session", lines=2, interactive=False)
                    single_log = gr.Textbox(label="Log", lines=10, interactive=False, elem_classes=["compact-log"])

            single_button.click(
                fn=generate_single,
                inputs=[single_image, single_dir, single_mask, single_seed, single_ss, single_slat, single_mesh],
                outputs=[single_model, single_files, single_status, single_voxel, single_log, single_session],
            )
            single_dir.change(fn=update_single_directory, inputs=single_dir, outputs=[single_image, single_mask, single_mask_preview])
            single_mask.change(fn=update_single_mask_preview, inputs=[single_dir, single_mask], outputs=single_mask_preview)

        with gr.Tab("Multi-Object Scene"):
            with gr.Row():
                with gr.Column(scale=1, min_width=380, elem_classes=["mvsam-panel"]):
                    scene_dir = gr.Textbox(label="Scene Directory", value=str(DEFAULT_SCENE_DIR), lines=2)
                    scene_preview = gr.HTML(value=_scene_preview_html(DEFAULT_SCENE_DIR))
                    scene_masks = gr.Gallery(
                        label="Object Masks",
                        value=[str(p) for p in _scene_mask_paths(DEFAULT_SCENE_DIR)],
                        columns=4,
                        height=280,
                    )
                    scene_seed = gr.Number(value=42, precision=0, label="Seed")
                    scene_ss = gr.Checkbox(value=True, label="Enable SS acceleration")
                    scene_slat = gr.Checkbox(value=True, label="Enable SLaT acceleration")
                    scene_mesh = gr.Checkbox(value=True, label="Enable Mesh acceleration")
                    scene_button = gr.Button("Generate Full Scene", variant="primary", elem_id="scene-run-button")

                with gr.Column(scale=2, min_width=620, elem_classes=["mvsam-panel"]):
                    scene_status = gr.HTML(value=_status_html("Ready", None, "Multi-object scene"))
                    scene_video = gr.Video(label="Scene Turntable", height=520)
                    scene_model = gr.Model3D(label="Merged Scene PLY", height=240, visible=False)
                    with gr.Row():
                        scene_files = gr.Files(label="Downloads", height=100)
                        scene_session = gr.Textbox(label="Session", lines=2, interactive=False)
                    scene_log = gr.Textbox(label="Log", lines=10, interactive=False, elem_classes=["compact-log"])

            with gr.Row(elem_classes=["mv-object-grid"]):
                scene_object_models = []
                for index in range(MAX_SCENE_OBJECT_PREVIEWS):
                    scene_object_models.append(
                        gr.Model3D(
                            label=f"Object {index + 1}",
                            height=260,
                            visible=False,
                            elem_classes=["object-model-preview"],
                        )
                    )

            def update_scene_inputs(path):
                path = Path(path)
                return _scene_preview_html(path), [str(p) for p in _scene_mask_paths(path)]

            scene_dir.change(fn=update_scene_inputs, inputs=scene_dir, outputs=[scene_preview, scene_masks])
            scene_button.click(
                fn=generate_scene,
                inputs=[scene_dir, scene_seed, scene_ss, scene_slat, scene_mesh],
                outputs=[scene_model, scene_video, scene_files, scene_status, scene_log, scene_session, *scene_object_models],
            )


if __name__ == "__main__":
    demo.queue(default_concurrency_limit=1).launch(
        server_name="0.0.0.0",
        server_port=7861,
        share=True,
        show_error=True,
        allowed_paths=[str(OUTPUT_ROOT)],
    )
