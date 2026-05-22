import os
import argparse
import json
import time
from pathlib import Path
from typing import Optional
# os.environ['ATTN_BACKEND'] = 'xformers'   # Can be 'flash-attn' or 'xformers', default is 'flash-attn'
os.environ['SPCONV_ALGO'] = 'native'        # Can be 'native' or 'auto', default is 'auto'.
                                            # 'auto' is faster but will do benchmarking at the beginning.
                                            # Recommended to set to 'native' if run only once.

import numpy as np
import imageio
import trimesh
from PIL import Image
from trellis.pipelines import TrellisImageTo3DPipeline
from trellis.utils import postprocessing_utils, render_utils

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


def update_pipeline_sampler_names(
    input_path: str,
    output_path: str,
    sparse_sampler_name: Optional[str] = "FlowEulerGuidanceIntervalSampler",
    slat_sampler_name: Optional[str] = "FlowEulerGuidanceIntervalSampler",
):
    """
    Update the sparse_structure_sampler and slat_sampler names, then save the pipeline.
    """
    with open(input_path, "r", encoding="utf-8") as f:
        pipeline = json.load(f)

    if "args" in pipeline:
        if "sparse_structure_sampler" in pipeline["args"] and sparse_sampler_name:
            pipeline["args"]["sparse_structure_sampler"]["name"] = sparse_sampler_name
        if "slat_sampler" in pipeline["args"] and slat_sampler_name:
            pipeline["args"]["slat_sampler"]["name"] = slat_sampler_name

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(pipeline, f, indent=4)
    print(f"✅ Saved modified pipeline to {output_path}")


def export_untextured_mesh(mesh, output_path):
    vertices = mesh.vertices.detach().cpu().numpy()
    faces = mesh.faces.detach().cpu().numpy()
    vertices = vertices @ np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
    mesh_asset = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    mesh_asset.export(str(output_path))


parser = argparse.ArgumentParser(description="TRELLIS multi-image inference")
parser.add_argument(
    "--image_dir",
    default="/data3/wmq/TRELLIS/assets/example_multi_image",
    help="Directory containing input images",
)
parser.add_argument("--enable", action="append", choices=["faster", "taylor", "mesh"], default=[], help="Enable an inference option. Can be repeated, e.g. --enable faster --enable mesh")
parser.add_argument("--enable_faster", action="store_true", help="Enable faster sampler names")
parser.add_argument("--enable_taylor", action="store_true", help="Enable taylor sampler names")
parser.add_argument("--enable_mesh", action="store_true", help="Enable mesh-aware sparse aggregation")
parser.add_argument("--enable_voxel_visualization", action="store_true", help="Write voxel visualization HTML files")
parser.add_argument("--multiimage_mode", choices=["stochastic", "multidiffusion"], default="stochastic")
parser.add_argument("--output_dir", default=".", help="Directory to save generated outputs")
parser.add_argument("--seed", type=int, default=1, help="Random seed")
parser.add_argument("--ss_steps", type=int, default=25, help="Sparse structure sampling steps")
parser.add_argument("--ss_cfg", type=float, default=7.5, help="Sparse structure CFG strength")
parser.add_argument("--slat_steps", type=int, default=25, help="SLAT sampling steps")
parser.add_argument("--slat_cfg", type=float, default=3.0, help="SLAT CFG strength")
parser.add_argument("--skip_video", action="store_true", help="Skip preview video export")
args = parser.parse_args()
args.enable_faster = args.enable_faster or "faster" in args.enable
args.enable_taylor = args.enable_taylor or "taylor" in args.enable
args.enable_mesh = args.enable_mesh or "mesh" in args.enable
output_dir = Path(args.output_dir)
output_dir.mkdir(parents=True, exist_ok=True)

pipeline_dir = "/data3/wmq/TRELLIS/checkpoints/TRELLIS-image-large"
input_json = os.path.join(pipeline_dir, "pipeline_raw.json")
output_json = os.path.join(pipeline_dir, "pipeline.json")

if args.enable_faster:
    update_pipeline_sampler_names(
        input_path=input_json,
        output_path=output_json,
        sparse_sampler_name="FlowEulerGuidanceIntervalSampler_taylor",
        slat_sampler_name="FlowEulerGuidanceIntervalSampler_faster",
    )
elif args.enable_taylor:
    update_pipeline_sampler_names(
        input_path=input_json,
        output_path=output_json,
        sparse_sampler_name="FlowEulerGuidanceIntervalSampler_taylor",
        slat_sampler_name="FlowEulerGuidanceIntervalSampler_taylor",
    )
else:
    update_pipeline_sampler_names(
        input_path=input_json,
        output_path=output_json,
    )

# Load a pipeline from a model folder or a Hugging Face model hub.
pipeline = TrellisImageTo3DPipeline.from_pretrained(pipeline_dir)
pipeline.cuda()
pipeline.enable_faster = args.enable_faster
pipeline.enable_mesh = args.enable_mesh
pipeline.enable_voxel_visualization = args.enable_voxel_visualization

# Load an image
image_dir = Path(args.image_dir)
image_paths = sorted(
    p for p in image_dir.iterdir()
    if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
)
if not image_paths:
    raise ValueError(f"No images found in {image_dir}")

images = [Image.open(path) for path in image_paths]

# Run the pipeline
infer_start = time.time()
outputs = pipeline.run_multi_image(
    images,
    seed=args.seed,
    # Optional parameters
    sparse_structure_sampler_params={
        "steps": args.ss_steps,
        "cfg_strength": args.ss_cfg,
    },
    slat_sampler_params={
        "steps": args.slat_steps,
        "cfg_strength": args.slat_cfg,
    },
    mode=args.multiimage_mode,
)
infer_time = time.time() - infer_start
print(f"✅ [Inference] run_multi_image completed, inference time only: {infer_time:.3f}s")
# outputs is a dictionary containing generated 3D assets in different formats:
# - outputs['gaussian']: a list of 3D Gaussians
# - outputs['radiance_field']: a list of radiance fields
# - outputs['mesh']: a list of meshes

if not args.skip_video:
    video_gs = render_utils.render_video(outputs['gaussian'][0])['color']
    video_mesh = render_utils.render_video(outputs['mesh'][0])['normal']
    video = [np.concatenate([frame_gs, frame_mesh], axis=1) for frame_gs, frame_mesh in zip(video_gs, video_mesh)]
    imageio.mimsave(str(output_dir / "sample_multi.mp4"), video, fps=30)

print("👍 GLB post-processing")
glb = postprocessing_utils.to_glb(
    outputs['gaussian'][0],
    outputs['mesh'][0],
    simplify=0.95,
    texture_size=1024,
)
glb.export(str(output_dir / "sample.glb"))
export_untextured_mesh(outputs['mesh'][0], output_dir / "sample_mesh.obj")
outputs['gaussian'][0].save_ply(str(output_dir / "sample.ply"))
