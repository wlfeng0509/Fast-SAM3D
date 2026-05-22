import os
import argparse
import torch
import torch.nn as nn
from PIL import Image
import open_clip
from os.path import expanduser
from urllib.request import urlretrieve
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from queue import Queue


def get_aesthetic_model(clip_model="vit_l_14"):
    """load the aethetic model"""
    home = expanduser("~")
    cache_folder = home + "/.cache/emb_reader"
    path_to_model = cache_folder + "/sa_0_4_"+clip_model+"_linear.pth"
    if not os.path.exists(path_to_model):
        os.makedirs(cache_folder, exist_ok=True)
        url_model = (
            "https://github.com/LAION-AI/aesthetic-predictor/blob/main/sa_0_4_"+clip_model+"_linear.pth?raw=true"
        )
        urlretrieve(url_model, path_to_model)
    if clip_model == "vit_l_14":
        m = nn.Linear(768, 1)
    elif clip_model == "vit_b_32":
        m = nn.Linear(512, 1)
    else:
        raise ValueError()
    s = torch.load(path_to_model)
    m.load_state_dict(s)
    m.eval()
    return m


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--clip_model", type=str, default="vit_l_14")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--world_size", type=int, default=1)
    opt = parser.parse_args()
    
    amodel = get_aesthetic_model(clip_model="vit_l_14")
    amodel.eval()
    model, _, preprocess = open_clip.create_model_and_transforms('ViT-L-14', pretrained='openai')
    model = model.cuda()
    amodel = amodel.cuda()
    
    metadata = pd.read_csv(os.path.join(opt.output_dir, 'metadata.csv'))
    metadata = metadata[metadata['snapshotted'] == 1]
    sha256s = metadata['sha256'].values
    
    # filter out objects that are already calculated
    if os.path.exists(os.path.join(opt.output_dir, 'aesthetic_scores.csv')):
        with open(os.path.join(opt.output_dir, 'aesthetic_scores.csv'), 'r') as f:
            old_metadata = pd.read_csv(f)
        sha256s = list(set(sha256s) - set(old_metadata['sha256'].values))

    sha256s = sorted(sha256s)
    sha256s = sha256s[len(sha256s) * opt.rank // opt.world_size: len(sha256s) * (opt.rank + 1) // opt.world_size]
    
    rows = []
        
    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        finished = Queue(maxsize=128)
        
        def load_image(sha256):
            try:
                files = os.listdir(os.path.join(opt.output_dir, 'snapshots', sha256))
                files = [f for f in files if f.endswith('.png')]
                processed = []
                for file in files:
                    image = Image.open(os.path.join(opt.output_dir, 'snapshots', sha256, file))
                    processed.append(preprocess(image))
                processed = torch.stack(processed, dim=0)
            except Exception as e:
                print(e)
                processed = None
            finished.put((sha256, processed))
                
        executor.map(load_image, sha256s)
        for _ in tqdm(range(len(sha256s)), desc='Calculating aesthetic scores'):
            sha256, processed = finished.get()
            if processed is not None:
                with torch.no_grad():
                    image_features = model.encode_image(processed.cuda())
                    image_features /= image_features.norm(dim=-1, keepdim=True)
                    aesthetic_score = amodel(image_features).cpu()
                    rows.append(pd.DataFrame({
                        'sha256': [sha256],
                        'mean': [aesthetic_score.mean().item()],
                        'std': [aesthetic_score.std().item()],
                        'min': [aesthetic_score.min().item()],
                        'max': [aesthetic_score.max().item()],
                        'median': [aesthetic_score.median().item()]
                    }))
                    
    with open(os.path.join(opt.output_dir, f'aesthetic_scores_{opt.rank}.csv'), 'w') as f:
        pd.concat(rows).to_csv(f, index=False)
