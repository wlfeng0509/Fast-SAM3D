import os
import html
import queue
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Optional

import gradio as gr
import numpy as np
from PIL import Image


ROOT_DIR = Path("/data3/wmq/TRELLIS")
DEFAULT_IMAGE_PATH = ROOT_DIR / "assets/example_image/T.png"
DEFAULT_IMAGE_DIR = ROOT_DIR / "assets/multi1"
TMP_DIR = ROOT_DIR / "tmp/app_wmq"
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
INT32_MAX = np.iinfo(np.int32).max

TMP_DIR.mkdir(parents=True, exist_ok=True)


def session_dir(req: gr.Request) -> Path:
    sid = getattr(req, "session_hash", None) or "default"
    out_dir = TMP_DIR / sid
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def cleanup_session(req: gr.Request):
    sid = getattr(req, "session_hash", None)
    if sid:
        shutil.rmtree(TMP_DIR / sid, ignore_errors=True)


def image_paths_in_dir(image_dir: Path):
    if not image_dir.exists():
        return []
    return [
        path
        for path in sorted(image_dir.iterdir())
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]


def single_example_paths():
    return image_paths_in_dir(ROOT_DIR / "assets/example_image")


def multi_example_folders():
    assets_dir = ROOT_DIR / "assets"
    if not assets_dir.exists():
        return []

    folders = []
    for folder in sorted(path for path in assets_dir.rglob("*") if path.is_dir()):
        if not folder.is_dir():
            continue
        if "example_image" in folder.relative_to(assets_dir).parts:
            continue
        if len(image_paths_in_dir(folder)) >= 2:
            folders.append(folder)
    return folders


SINGLE_EXAMPLES = single_example_paths()
MULTI_EXAMPLE_DIRS = multi_example_folders()
MULTI_EXAMPLE_COVERS = [
    image_paths_in_dir(folder)[0]
    for folder in MULTI_EXAMPLE_DIRS
]


def select_index(evt: gr.SelectData) -> int:
    index = evt.index
    if isinstance(index, (list, tuple)):
        index = index[0]
    return int(index)


def load_single_example(evt: gr.SelectData):
    index = select_index(evt)
    path = SINGLE_EXAMPLES[index]
    return Image.open(path).convert("RGBA"), str(path)


def load_multi_example(evt: gr.SelectData):
    index = select_index(evt)
    folder = MULTI_EXAMPLE_DIRS[index]
    cover = MULTI_EXAMPLE_COVERS[index]
    return [Image.open(cover).convert("RGBA")], str(folder)


def reset_output_dir(out_dir: Path):
    shutil.rmtree(out_dir, ignore_errors=True)
    out_dir.mkdir(parents=True, exist_ok=True)


def save_pil_image(image: Image.Image, path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    if image.mode not in ("RGB", "RGBA"):
        image = image.convert("RGBA")
    image.save(path)
    return str(path)


def resolve_single_input(image: Optional[Image.Image], image_path: str, out_dir: Path) -> str:
    path_text = (image_path or "").strip()
    if path_text:
        path = Path(path_text).expanduser()
        if path.exists() and (image is None or path.resolve() != DEFAULT_IMAGE_PATH.resolve()):
            return str(path)

    if image is not None:
        return save_pil_image(image, out_dir / "input.png")

    if path_text:
        path = Path(path_text).expanduser()
        if path.exists():
            return str(path)
    raise gr.Error(f"Image path does not exist: {path_text}")


def normalize_gallery_item(item):
    if isinstance(item, tuple):
        return item[0]
    return item


def resolve_multi_input(multi_gallery, image_dir: str, out_dir: Path) -> str:
    if multi_gallery:
        input_dir = out_dir / "multi_inputs"
        input_dir.mkdir(parents=True, exist_ok=True)
        for index, item in enumerate(multi_gallery):
            image = normalize_gallery_item(item)
            if isinstance(image, Image.Image):
                save_pil_image(image, input_dir / f"{index:03d}.png")
            elif isinstance(image, str):
                src = Path(image).expanduser()
                if src.exists():
                    shutil.copy2(src, input_dir / f"{index:03d}{src.suffix.lower() or '.png'}")
        if any(p.suffix.lower() in IMAGE_EXTENSIONS for p in input_dir.iterdir()):
            return str(input_dir)
        raise gr.Error("No valid gallery images were provided.")

    path = Path((image_dir or "").strip()).expanduser()
    if not path.exists():
        raise gr.Error(f"Image directory does not exist: {path}")

    image_paths = sorted(
        p for p in path.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not image_paths:
        raise gr.Error(f"No images found in: {path}")
    return str(path)


def mode_args(accel_mode: str):
    if accel_mode == "Faster":
        return ["--enable_faster", "--enable_mesh"]
    if accel_mode == "Taylor":
        return ["--enable_taylor"]
    return []


def build_command(
    input_mode: str,
    input_path: str,
    out_dir: Path,
    accel_mode: str,
    multiimage_mode: str,
    seed: int,
    ss_guidance_strength: float,
    ss_sampling_steps: int,
    slat_guidance_strength: float,
    slat_sampling_steps: int,
    enable_voxel_visualization: bool,
):
    if input_mode == "Single Image":
        command = [
            sys.executable,
            str(ROOT_DIR / "example.py"),
            "--image_path",
            input_path,
        ]
    else:
        command = [
            sys.executable,
            str(ROOT_DIR / "example_multi_image.py"),
            "--image_dir",
            input_path,
            "--multiimage_mode",
            multiimage_mode,
        ]

    command.extend(
        [
            "--output_dir",
            str(out_dir),
            "--seed",
            str(int(seed)),
            "--ss_steps",
            str(int(ss_sampling_steps)),
            "--ss_cfg",
            str(float(ss_guidance_strength)),
            "--slat_steps",
            str(int(slat_sampling_steps)),
            "--slat_cfg",
            str(float(slat_guidance_strength)),
            "--skip_video",
        ]
    )
    command.extend(mode_args(accel_mode))
    if enable_voxel_visualization:
        command.append("--enable_voxel_visualization")
    return command


def run_entry_script_stream(command):
    env = os.environ.copy()
    env["SPCONV_ALGO"] = "native"
    env["TORCH_HOME"] = "/data3/wmq/Fast-sam3d-objects/checkpoints/torch-cache"
    process = subprocess.Popen(
        command,
        cwd=str(ROOT_DIR),
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=1,
    )
    logs = queue.Queue()

    def read_output():
        if process.stdout is None:
            return
        for line in process.stdout:
            logs.put(line.rstrip())

    threading.Thread(target=read_output, daemon=True).start()
    return process, logs


def drain_logs(log_queue, log_lines):
    while True:
        try:
            line = log_queue.get_nowait()
        except queue.Empty:
            break
        if line:
            log_lines.append(line)


def runtime_html(status: str, elapsed: float, input_mode: str, accel_mode: str, seed: int, args_text: str, log_lines=None):
    emoji = "⚡" if status == "running" else "✅" if status == "done" else "🌟" if status == "idle" else "❌"
    title = "Generating" if status == "running" else "Generation complete" if status == "done" else "Ready" if status == "idle" else "Generation failed"
    safe_args = html.escape(args_text)
    log_block = ""
    if log_lines:
        safe_logs = html.escape("\n".join(log_lines[-8:]))
        log_block = f"<pre class='runtime-log'>{safe_logs}</pre>"

    return (
        "<div class='runtime-card'>"
        f"<div class='runtime-label'>{emoji} {title}</div>"
        f"<div class='runtime-time'>{elapsed:.2f}s</div>"
        f"<div class='runtime-meta'>🎛️ {html.escape(input_mode)} / {html.escape(accel_mode)} · 🎲 Seed {seed}</div>"
        f"<div class='runtime-args'>Entry arguments: <code>{safe_args}</code></div>"
        f"{log_block}"
        "</div>"
    )


def run_generation(
    input_mode: str,
    image: Optional[Image.Image],
    image_path: str,
    multi_gallery,
    image_dir: str,
    accel_mode: str,
    multiimage_mode: str,
    seed: int,
    randomize_seed: bool,
    ss_guidance_strength: float,
    ss_sampling_steps: int,
    slat_guidance_strength: float,
    slat_sampling_steps: int,
    enable_voxel_visualization: bool,
    req: gr.Request,
):
    out_dir = session_dir(req)
    reset_output_dir(out_dir)

    actual_seed = int(np.random.randint(0, INT32_MAX)) if randomize_seed else int(seed)
    if input_mode == "Single Image":
        input_path = resolve_single_input(image, image_path, out_dir)
    else:
        input_path = resolve_multi_input(multi_gallery, image_dir, out_dir)

    command = build_command(
        input_mode=input_mode,
        input_path=input_path,
        out_dir=out_dir,
        accel_mode=accel_mode,
        multiimage_mode=multiimage_mode,
        seed=actual_seed,
        ss_guidance_strength=ss_guidance_strength,
        ss_sampling_steps=ss_sampling_steps,
        slat_guidance_strength=slat_guidance_strength,
        slat_sampling_steps=slat_sampling_steps,
        enable_voxel_visualization=enable_voxel_visualization,
    )

    args_text = " ".join(command[2:])
    start = time.perf_counter()
    process, log_queue = run_entry_script_stream(command)
    log_lines = []

    yield (
        None,
        gr.DownloadButton(value=None, interactive=False),
        gr.DownloadButton(value=None, interactive=False),
        runtime_html("running", 0.0, input_mode, accel_mode, actual_seed, args_text),
    )

    while process.poll() is None:
        elapsed = time.perf_counter() - start
        drain_logs(log_queue, log_lines)
        yield (
            None,
            gr.DownloadButton(value=None, interactive=False),
            gr.DownloadButton(value=None, interactive=False),
            runtime_html("running", elapsed, input_mode, accel_mode, actual_seed, args_text, log_lines),
        )
        time.sleep(0.5)

    drain_logs(log_queue, log_lines)
    elapsed = time.perf_counter() - start
    log_tail = "\n".join(log_lines[-12:])

    if process.returncode != 0:
        yield (
            None,
            gr.DownloadButton(value=None, interactive=False),
            gr.DownloadButton(value=None, interactive=False),
            runtime_html("failed", elapsed, input_mode, accel_mode, actual_seed, args_text, log_lines),
        )
        raise gr.Error(f"Generation failed.\n\n{log_tail}")

    glb_path = out_dir / "sample.glb"
    ply_path = out_dir / "sample.ply"
    if not glb_path.exists() or not ply_path.exists():
        yield (
            None,
            gr.DownloadButton(value=None, interactive=False),
            gr.DownloadButton(value=None, interactive=False),
            runtime_html("failed", elapsed, input_mode, accel_mode, actual_seed, args_text, log_lines),
        )
        raise gr.Error(f"Generation finished but output files are incomplete.\n\n{log_tail}")

    yield (
        str(glb_path),
        gr.DownloadButton(value=str(glb_path), interactive=True),
        gr.DownloadButton(value=str(ply_path), interactive=True),
        runtime_html("done", elapsed, input_mode, accel_mode, actual_seed, args_text, log_lines),
    )


CSS = """
.fast-title {
    font-size: 48px;
    font-weight: 800;
    line-height: 1.05;
    margin: 4px 0 2px;
    background: linear-gradient(90deg, #00c2ff 0%, #7c4dff 45%, #ff4ecd 100%);
    -webkit-background-clip: text;
    background-clip: text;
    color: transparent;
}
.fast-subtitle {
    color: #666;
    font-size: 16px;
    margin-bottom: 14px;
}
.runtime-card {
    border: 1px solid rgba(124, 77, 255, 0.25);
    border-radius: 8px;
    padding: 18px 20px;
    background:
        linear-gradient(135deg, rgba(0, 194, 255, 0.12), rgba(255, 78, 205, 0.10)),
        rgba(255, 255, 255, 0.78);
    box-shadow: 0 14px 42px rgba(66, 56, 126, 0.12);
}
.runtime-label {
    font-size: 18px;
    font-weight: 800;
    color: #4b3cff;
}
.runtime-time {
    margin: 4px 0 6px;
    font-size: 46px;
    line-height: 1;
    font-weight: 900;
    background: linear-gradient(90deg, #00b4d8, #7c4dff, #ff4ecd);
    -webkit-background-clip: text;
    background-clip: text;
    color: transparent;
}
.runtime-meta {
    font-size: 16px;
    font-weight: 700;
    color: #333;
}
.runtime-args {
    margin-top: 8px;
    font-size: 13px;
    color: #555;
    word-break: break-all;
}
.runtime-log {
    margin-top: 12px;
    max-height: 145px;
    overflow: auto;
    font-size: 12px;
    line-height: 1.45;
    border-radius: 6px;
    padding: 10px;
    background: rgba(20, 20, 32, 0.88);
    color: #e8e7ff;
}
.gradio-container label,
.gradio-container label span,
.gradio-container .wrap label {
    font-size: 16px !important;
    font-weight: 700 !important;
}
.gradio-container input,
.gradio-container textarea,
.gradio-container button,
.gradio-container .tabitem,
.gradio-container .form label {
    font-size: 15px !important;
}
.examples-heading {
    margin-top: 18px;
    font-size: 20px;
    font-weight: 850;
    background: linear-gradient(90deg, #00b4d8, #7c4dff, #ff4ecd);
    -webkit-background-clip: text;
    background-clip: text;
    color: transparent;
}
.example-gallery img {
    aspect-ratio: 1 / 1;
    object-fit: cover !important;
}
.example-gallery .thumbnail-item {
    aspect-ratio: 1 / 1;
}
"""


with gr.Blocks(delete_cache=(600, 600), title="Fast Trellis") as demo:
    gr.HTML(
        """
        <div class="fast-title">🚀 Fast Trellis</div>
        <div class="fast-subtitle">✨ Fast-SAM3D implementation on TRELLIS</div>
        """
    )

    with gr.Row():
        with gr.Column(scale=5):
            input_mode = gr.Radio(
                ["Single Image", "Multi Image"],
                label="🖼️ Input Mode",
                value="Single Image",
            )

            with gr.Tabs():
                with gr.Tab("Single Image"):
                    image_input = gr.Image(
                        label="📷 Image",
                        value=str(DEFAULT_IMAGE_PATH),
                        image_mode="RGBA",
                        type="pil",
                        height=300,
                    )
                    image_path = gr.Textbox(
                        label="📁 Image Path",
                        value=str(DEFAULT_IMAGE_PATH),
                    )

                with gr.Tab("Multi Image"):
                    multi_gallery = gr.Gallery(
                        label="🖼️ Images",
                        type="pil",
                        columns=3,
                        height=300,
                    )
                    image_dir = gr.Textbox(
                        label="📂 Image Directory",
                        value=str(DEFAULT_IMAGE_DIR),
                    )

            with gr.Accordion("⚙️ Generation Settings", open=True):
                accel_mode = gr.Radio(
                    ["Original", "Taylor", "Faster"],
                    label="⚡ Mode",
                    value="Faster",
                )
                multiimage_mode = gr.Radio(
                    ["stochastic", "multidiffusion"],
                    label="🧩 Multi-image Algorithm",
                    value="stochastic",
                )
                enable_voxel_visualization = gr.Checkbox(label="🌈 Voxel Visualization HTML", value=False)
                with gr.Row():
                    seed = gr.Slider(0, INT32_MAX, label="🎲 Seed", value=1, step=1)
                    randomize_seed = gr.Checkbox(label="🎰 Randomize Seed", value=False)
                gr.Markdown("### 🧱 Stage 1: Sparse Structure Generation")
                with gr.Row():
                    ss_guidance_strength = gr.Slider(0.0, 10.0, label="Guidance Strength", value=7.5, step=0.1)
                    ss_sampling_steps = gr.Slider(1, 50, label="Sampling Steps", value=12, step=1)
                gr.Markdown("### 🧬 Stage 2: Structured Latent Generation")
                with gr.Row():
                    slat_guidance_strength = gr.Slider(0.0, 10.0, label="Guidance Strength", value=3.0, step=0.1)
                    slat_sampling_steps = gr.Slider(1, 50, label="Sampling Steps", value=12, step=1)

            generate_btn = gr.Button("🚀 Generate", variant="primary")

        with gr.Column(scale=6):
            runtime = gr.HTML(runtime_html("idle", 0.0, "Single Image", "Faster", 1, "Waiting to start"))
            model_output = gr.Model3D(label="✨ GLB Preview", height=460)
            with gr.Row():
                download_glb = gr.DownloadButton(label="📦 Download GLB", interactive=False)
                download_ply = gr.DownloadButton(label="☁️ Download Gaussian PLY", interactive=False)

    gr.HTML("<div class='examples-heading'>🪄 Default Examples</div>")
    with gr.Row() as single_image_examples:
        single_example_gallery = gr.Gallery(
            value=[str(path) for path in SINGLE_EXAMPLES],
            label="📷 Single-Image Examples",
            columns=8,
            rows=2,
            height=640,
            object_fit="cover",
            allow_preview=False,
            buttons=[],
            type="filepath",
            elem_classes=["example-gallery"],
        )
    with gr.Row(visible=False) as multi_folder_examples_row:
        multi_example_gallery = gr.Gallery(
            value=[str(path) for path in MULTI_EXAMPLE_COVERS],
            label="🖼️ Multi-Image Examples",
            columns=8,
            rows=1,
            height=360,
            object_fit="cover",
            allow_preview=False,
            buttons=[],
            type="filepath",
            elem_classes=["example-gallery"],
        )

    demo.unload(cleanup_session)
    generate_btn.click(
        run_generation,
        inputs=[
            input_mode,
            image_input,
            image_path,
            multi_gallery,
            image_dir,
            accel_mode,
            multiimage_mode,
            seed,
            randomize_seed,
            ss_guidance_strength,
            ss_sampling_steps,
            slat_guidance_strength,
            slat_sampling_steps,
            enable_voxel_visualization,
        ],
        outputs=[model_output, download_glb, download_ply, runtime],
    )

    input_mode.change(
        lambda mode: (
            gr.update(visible=mode == "Single Image"),
            gr.update(visible=mode == "Multi Image"),
        ),
        inputs=[input_mode],
        outputs=[single_image_examples, multi_folder_examples_row],
    )

    single_example_gallery.select(
        load_single_example,
        outputs=[image_input, image_path],
    )
    multi_example_gallery.select(
        load_multi_example,
        outputs=[multi_gallery, image_dir],
    )


if __name__ == "__main__":
    demo.queue(default_concurrency_limit=1).launch(
        server_name="0.0.0.0",
        server_port=7861,
        css=CSS,
    )
