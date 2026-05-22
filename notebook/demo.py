import sys
import torch
import numpy as np
import argparse
from inference import Inference, load_image, load_single_mask
from fft.fft2d import calculate_hfer_robust
import os
import time

sys.path.append("notebook")
os.environ['TORCH_HOME'] = '/data3/wmq/Fast-sam3d-objects/checkpoints/torch-cache'

def clear_directory(directory_path):
    try:
        if not os.path.exists(directory_path):
            return False
        for filename in os.listdir(directory_path):
            file_path = os.path.join(directory_path, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)  
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)  
            except Exception as e:
                print(f"Error {file_path} : {e}")
        return True
        
    except Exception as e:
        return False

def inspect_dict(output_dict):
  
    for key, value in output_dict.items():
        if isinstance(value, torch.Tensor):
            info = str(list(value.shape))
            type_name = "torch.Tensor"
        elif isinstance(value, np.ndarray):
            info = str(value.shape)
            type_name = "np.ndarray"
        elif isinstance(value, list):
            info = f"len={len(value)}"
            type_name = "List"
        else:
            info = str(value)[:30] + "..." if len(str(value)) > 30 else str(value)
            type_name = type(value).__name__          

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


def main():
    parser = argparse.ArgumentParser(description="3D GS Inference Script")
    
    parser.add_argument("--tag", type=str, default="hf", help="model Tag")
    parser.add_argument("--image_path", type=str, required=True, help="image path")
    parser.add_argument("--mask_index", type=int, default=14, help="mask index")
    parser.add_argument("--output_dir", type=str, default="./Generate", help="ply and glb")
    
    parser.add_argument("--seed", type=int, default=42, help="seed")
    #---SSG
    parser.add_argument("--ss_faster_stride", type=int, default=3)
    parser.add_argument("--ss_warmup", type=int, default=2)
    parser.add_argument("--ss_order", type=float, default=1)
    parser.add_argument("--ss_momentum_beta", type=float, default=0.5)
    #---SLaT
    parser.add_argument("--slat_thresh", type=float, default=0.5)
    parser.add_argument("--slat_warmup", type=int, default=2)
    parser.add_argument("--slat_token_ratio", type=float, default=0.15)
    
    args = parser.parse_args()
    # 1. load model
    config_path = f"checkpoints/{args.tag}/pipeline.yaml"
    inference = Inference(config_path, compile=False)

    # 2. load image and mask
    image = load_image(args.image_path)
    
    # If load_single_mask needs a directory and index, split them from the path.
    # Keep the original logic while passing values through args.
    folder_path = os.path.dirname(args.image_path)
    mask = load_single_mask(folder_path, index=args.index)
    
    # 3. Calculate HFER and set it on the model.
    hfer = calculate_hfer_robust(args.mask_path)
    inference.get_HFER(hfer)

    # 4. Run inference.
    print(f"Start inference: {args.image_path}")
    s_time = time.time()
    
    # Pass additional inference hyperparameters.
    output = inference(
        image, 
        mask, 
        seed=args.seed,
        cache_stride=args.cache_stride,
        momentum_beta=args.momentum_beta,
        carving_ratio=args.carving_ratio
    )
    
    print(f"Inference finished in {time.time() - s_time:.2f}s")

    # 5. Save results.
    os.makedirs(args.output_dir, exist_ok=True)
    
    ply_path = os.path.join(args.output_dir, f"splat-easy-{args.index}.ply")
    save_visual_ply(output["gs"], ply_path)
    
    glb_path = os.path.join(args.output_dir, f"splat-easy-{args.index}.glb")
    output["glb"].export(glb_path)
    
    print(f"Files saved to:\n - {ply_path}\n - {glb_path}")

if __name__ == "__main__":
    main()