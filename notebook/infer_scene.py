import sys
import os
import time
import shutil
import torch
import numpy as np
import argparse
import imageio
from omegaconf import OmegaConf
from inference import Inference, ready_gaussian_for_video_rendering, load_image, load_masks, load_hfers, make_scene, render_video
sys.path.append("notebook")
os.environ['TORCH_HOME'] = 'checkpoints/torch-cache'

def save_visual_ply(gs_model, path):
    from plyfile import PlyData, PlyElement
    folder_path = os.path.dirname(path)
    if folder_path and not os.path.exists(folder_path):
        os.makedirs(folder_path, exist_ok=True)

    xyz = gs_model._xyz.detach().cpu().numpy()
    f_dc = gs_model._features_dc.detach().contiguous().cpu().numpy()
    SH_C0 = 0.28209479177387814
    rgb = 0.5 + (SH_C0 * f_dc)
    
    rgb = np.clip(rgb, 0, 1) * 255
    rgb = rgb.astype(np.uint8)
    rgb = rgb.squeeze(1)

    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
             ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    
    elements = np.empty(xyz.shape[0], dtype=dtype)
    elements['x'] = xyz[:, 0]
    elements['y'] = xyz[:, 1]
    elements['z'] = xyz[:, 2]
    elements['red'] = rgb[:, 0]
    elements['green'] = rgb[:, 1]
    elements['blue'] = rgb[:, 2]

    el = PlyElement.describe(elements, 'vertex')
    PlyData([el]).write(path)
    print(f"Saved colored PLY to {path}")

def main():
    parser = argparse.ArgumentParser(description="3D Scene Inference Script")

    parser.add_argument("--tag", type=str, default="hf", help="model Tag")
    parser.add_argument("--image_dir", type=str, required=True, help="image path")
    parser.add_argument("--output_dir", type=str, default="./Generate/Scene", help="output dir")
    parser.add_argument("--seed", type=int, default=42, help="seed")
    
    # --- SSG  ---
    parser.add_argument("--ss_cache_stride", type=int, default=3)
    parser.add_argument("--ss_warmup", type=int, default=2)
    parser.add_argument("--ss_order", type=int, default=1)
    parser.add_argument("--ss_momentum_beta", type=float, default=0.5)
    
    # --- SLaT ---
    parser.add_argument("--slat_thresh", type=float, default=0.5)
    parser.add_argument("--slat_warmup", type=int, default=2)
    parser.add_argument("--slat_carving_ratio", type=float, default=0.15)
    
    # --- Mesh ---
    parser.add_argument("--mesh_spectral_threshold_low", type=float, default=0.5)
    parser.add_argument("--mesh_spectral_threshold_high", type=float, default=0.7)
    
    parser.add_argument("--enable_ss_cache", action="store_true")
    parser.add_argument("--enable_slat_carving", action="store_true")
    parser.add_argument("--enable_mesh_aggregation", action="store_true")
    parser.add_argument("--enable_acceleration", action="store_true")
    
    args, unknown = parser.parse_known_args()


    def get_enable_params(args):
        args_dict = vars(args)
        enable_params = {k: v for k, v in args_dict.items() if k.startswith("enable_")}
        
        if enable_params.get('enable_acceleration', False):
            enable_params['enable_ss_cache'] = True
            enable_params['enable_slat_carving'] = True
            enable_params['enable_mesh_aggregation'] = True
        
        for k, v in enable_params.items():
            setattr(args, k, v)
        
        return enable_params

    enable_params = get_enable_params(args)
    print(f"✅: SS:{enable_params['enable_ss_cache']}, SLaT:{enable_params['enable_slat_carving']}, Mesh:{enable_params['enable_mesh_aggregation']}")

    config_path = f"checkpoints/{args.tag}/pipeline.yaml"


    config = OmegaConf.load(config_path) 
    config.workspace_dir = os.path.dirname(config_path)
    
    if enable_params['enable_ss_cache']:
        config['ss_generator_config_path'] = "ss_generator_faster.yaml" 
    if enable_params['enable_slat_carving']:
        config['slat_generator_config_path'] = "slat_generator_faster.yaml" 

    inference = Inference(config, compile=False, args=args)

    image_name = os.path.basename(args.image_dir)
    image_path = os.path.join(args.image_dir,"image.png")
    print(f"📂 Loading data from: {args.image_dir}")
    image = load_image(image_path)
    masks = load_masks(args.image_dir, extension=".png")
    hfers = load_hfers(args.image_dir, extension=".png")

    print(f"🚀 Begin Inference, total {len(masks)} views...")

    outputs = []
    s_time = time.time()
    
    for i in range(len(masks)):
        if hasattr(inference, 'get_hfer'):
            inference.get_hfer(hfers[i])
        if hasattr(inference, 'get_params'):
            inference.get_params(args)
            
        print(f"  -> Processing object {i+1}/{len(masks)}...")
        
        output = inference(
            image, 
            masks[i], 
            seed=args.seed
        )
        outputs.append(output)

    e_time = time.time()
    print(f"⏱️ Total Inference Time: {e_time - s_time:.2f}s")

    print("🧩 Compositing scene...")
    scene_gs = make_scene(*outputs)
    scene_gs = ready_gaussian_for_video_rendering(scene_gs)

    os.makedirs(args.output_dir, exist_ok=True)
    
    ply_path = os.path.join(args.output_dir, f"{image_name}_scene.ply")
    save_visual_ply(scene_gs, ply_path)

    print("🎥 Rendering video...")
    video_frames = render_video(
        scene_gs,
        r=2.5, 
        fov=60,
        resolution=1024,
    )["color"]

    # Save GIF
    gif_path = os.path.join(args.output_dir, f"{image_name}.gif")
    imageio.mimsave(
        gif_path,
        video_frames,
        format="GIF",
        duration=1000 / 30, 
        loop=0,
    )
    print(f"GIF saved to: {gif_path}")

   

if __name__ == "__main__":
    main()