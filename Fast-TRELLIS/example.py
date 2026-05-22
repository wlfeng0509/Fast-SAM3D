import os
import sys
import json
import time
from typing import Optional
from pathlib import Path
from PIL import Image
import imageio
import numpy as np
import trimesh
from trellis.pipelines import TrellisImageTo3DPipeline
from trellis.utils import render_utils, postprocessing_utils
import argparse

# Environment setup
os.environ['SPCONV_ALGO'] = 'native'
os.environ['TORCH_HOME'] = '/data3/wmq/Fast-sam3d-objects/checkpoints/torch-cache'

sys.path.append("/data3/wmq/TRELLIS/trellis")
sys.path.append("/data3/wmq/TRELLIS/trellis/pipelines")

# ---------- JSON update helper ----------
def update_pipeline_sampler_names(
    input_path: str,
    output_path: str,
    sparse_sampler_name: Optional[str] = "FlowEulerGuidanceIntervalSampler",
    slat_sampler_name: Optional[str] = "FlowEulerGuidanceIntervalSampler"
):
    """
    Update the sparse_structure_sampler and slat_sampler names, then save the pipeline.
    """
    with open(input_path, 'r', encoding='utf-8') as f:
        pipeline = json.load(f)

    if 'args' in pipeline:
        if 'sparse_structure_sampler' in pipeline['args'] and sparse_sampler_name:
            pipeline['args']['sparse_structure_sampler']['name'] = sparse_sampler_name
        if 'slat_sampler' in pipeline['args'] and slat_sampler_name:
            pipeline['args']['slat_sampler']['name'] = slat_sampler_name

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(pipeline, f, indent=4)
    print(f"✅ Saved modified pipeline to {output_path}")


def export_untextured_mesh(mesh, output_path):
    vertices = mesh.vertices.detach().cpu().numpy()
    faces = mesh.faces.detach().cpu().numpy()
    vertices = vertices @ np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
    mesh_asset = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    mesh_asset.export(str(output_path))

# ---------- Command-line arguments ----------
parser = argparse.ArgumentParser(description="Modify pipeline sampler names")
parser.add_argument('--image_path', default="assets/example_image/T.png", help='Input image path')
parser.add_argument('--enable', action='append', choices=['faster', 'taylor', 'mesh'], default=[], help='Enable an inference option. Can be repeated, e.g. --enable faster --enable mesh')
parser.add_argument('--enable_faster', action='store_true', help='Enable faster sampler names')
parser.add_argument('--enable_taylor', action='store_true', help='Enable taylor sampler names')
parser.add_argument('--enable_mesh', action='store_true', help='Enable mesh')
parser.add_argument('--enable_voxel_visualization', action='store_true', help='Write voxel visualization HTML files')
parser.add_argument('--output_dir', default='.', help='Directory to save generated outputs')
parser.add_argument('--seed', type=int, default=1, help='Random seed')
parser.add_argument('--ss_steps', type=int, default=12, help='Sparse structure sampling steps')
parser.add_argument('--ss_cfg', type=float, default=7.5, help='Sparse structure CFG strength')
parser.add_argument('--slat_steps', type=int, default=12, help='SLAT sampling steps')
parser.add_argument('--slat_cfg', type=float, default=3.0, help='SLAT CFG strength')
parser.add_argument('--skip_video', action='store_true', help='Skip preview video export')
args = parser.parse_args()
args.enable_faster = args.enable_faster or 'faster' in args.enable
args.enable_taylor = args.enable_taylor or 'taylor' in args.enable
args.enable_mesh = args.enable_mesh or 'mesh' in args.enable
output_dir = Path(args.output_dir)
output_dir.mkdir(parents=True, exist_ok=True)

# Pipeline path
pipeline_dir = "/data3/wmq/TRELLIS/checkpoints/TRELLIS-image-large"
input_json = os.path.join(pipeline_dir, "pipeline_raw.json")
output_json = os.path.join(pipeline_dir, "pipeline.json")

# ---------- Update pipeline according to arguments ----------
if args.enable_faster:
    update_pipeline_sampler_names(
        input_path=input_json,
        output_path=output_json,
        sparse_sampler_name="FlowEulerGuidanceIntervalSampler_taylor",
        slat_sampler_name="FlowEulerGuidanceIntervalSampler_faster"
    )
elif args.enable_taylor:
    update_pipeline_sampler_names(
        input_path=input_json,
        output_path=output_json,
        sparse_sampler_name="FlowEulerGuidanceIntervalSampler_taylor",
        slat_sampler_name="FlowEulerGuidanceIntervalSampler_taylor"
    )
else:
    update_pipeline_sampler_names(
        input_path = input_json,
        output_path = output_json,
    )

# ---------- Load pipeline ----------
pipeline = TrellisImageTo3DPipeline.from_pretrained(pipeline_dir)
pipeline.cuda()

# ---------- Load image and run pipeline ----------
image = Image.open(args.image_path)
pipeline.enable_faster = args.enable_faster
pipeline.enable_voxel_visualization = args.enable_voxel_visualization
if args.enable_mesh:
    pipeline.enable_mesh = True
else:
    pipeline.enable_mesh = False

infer_start = time.time()
outputs = pipeline.run(
    image,
    seed=args.seed,
    sparse_structure_sampler_params={
        "steps": args.ss_steps,
        "cfg_strength": args.ss_cfg,
    },
    slat_sampler_params={
        "steps": args.slat_steps,
        "cfg_strength": args.slat_cfg,
    },
)
infer_time = time.time() - infer_start
print(f"✅ [Inference] run completed, inference time only: {infer_time:.3f}s")


# ---------- Save videos ----------
if not args.skip_video:
    print("🍔 Processing preview videos")
    video = render_utils.render_video(outputs['gaussian'][0])['color']
    imageio.mimsave(str(output_dir / "sample_gs.mp4"), video, fps=30)
    video = render_utils.render_video(outputs['radiance_field'][0])['color']
    imageio.mimsave(str(output_dir / "sample_rf.mp4"), video, fps=30)
    video = render_utils.render_video(outputs['mesh'][0])['normal']
    imageio.mimsave(str(output_dir / "sample_mesh.mp4"), video, fps=30)

# ---------- GLB post-processing ----------
print("👍 GLB post-processing")
glb = postprocessing_utils.to_glb(
    outputs['gaussian'][0],
    outputs['mesh'][0],
    simplify=0.95,
    texture_size=1024
)
glb.export(str(output_dir / "sample.glb"))

# ---------- Save untextured mesh ----------
export_untextured_mesh(outputs['mesh'][0], output_dir / "sample_mesh.obj")

# ---------- Save PLY ----------
outputs['gaussian'][0].save_ply(str(output_dir / "sample.ply"))
