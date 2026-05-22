from typing import *
from contextlib import contextmanager
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision import transforms
from PIL import Image
import rembg
from .base import Pipeline
from . import samplers
from .sparse_sampling import (
    calculate_adaptive_factor as _calculate_adaptive_factor,
    downsample_with_feature_fusion as _downsample_with_feature_fusion,
)
from ..modules import sparse as sp

import sys
sys.path.append("/data3/wmq/TRELLIS/trellis")
sys.path.append("/data3/wmq/TRELLIS/trellis/pipelines")

from fft.fft2d import calculate_hfer_robust
from fft.fft3d import get_coords_value, process_and_visualize


class TrellisImageTo3DPipeline(Pipeline):
    """
    Pipeline for inferring Trellis image-to-3D models.

    Args:
        models (dict[str, nn.Module]): The models to use in the pipeline.
        sparse_structure_sampler (samplers.Sampler): The sampler for the sparse structure.
        slat_sampler (samplers.Sampler): The sampler for the structured latent.
        slat_normalization (dict): The normalization parameters for the structured latent.
        image_cond_model (str): The name of the image conditioning model.
    """
    def __init__(
        self,
        models: dict[str, nn.Module] = None,
        sparse_structure_sampler: samplers.Sampler = None,
        slat_sampler: samplers.Sampler = None,
        slat_normalization: dict = None,
        image_cond_model: str = None,
    ):
        if models is None:
            return
        super().__init__(models)
        self.sparse_structure_sampler = sparse_structure_sampler
        self.slat_sampler = slat_sampler
        self.sparse_structure_sampler_params = {}
        self.slat_sampler_params = {}
        self.slat_normalization = slat_normalization
        self.rembg_session = None
        self._init_image_cond_model(image_cond_model)
        
        self.hfer_2d = 0
        self.enable_faster = False
        self.enable_mesh = False
        self.enable_voxel_visualization = False

    # Load model.
    @staticmethod
    def from_pretrained(path: str) -> "TrellisImageTo3DPipeline":
        """
        Load a pretrained model.

        Args:
            path (str): The path to the model. Can be either local path or a Hugging Face repository.
        """
        pipeline = super(TrellisImageTo3DPipeline, TrellisImageTo3DPipeline).from_pretrained(path)
        new_pipeline = TrellisImageTo3DPipeline()
        new_pipeline.__dict__ = pipeline.__dict__
        args = pipeline._pretrained_args

        # Resolve samplers.
        new_pipeline.sparse_structure_sampler = getattr(samplers, args['sparse_structure_sampler']['name'])(**args['sparse_structure_sampler']['args'])
        new_pipeline.sparse_structure_sampler_params = args['sparse_structure_sampler']['params']

        new_pipeline.slat_sampler = getattr(samplers, args['slat_sampler']['name'])(**args['slat_sampler']['args'])
        new_pipeline.slat_sampler_params = args['slat_sampler']['params']
        new_pipeline.slat_normalization = args['slat_normalization']

        new_pipeline._init_image_cond_model(args['image_cond_model'])

        return new_pipeline
    
    def _init_image_cond_model(self, name: str):
        """
        Initialize the image conditioning model.
        """
        dinov2_model = torch.hub.load(
            '/data3/wmq/TRELLIS/checkpoints/hub/facebookresearch_dinov2_main',
            name,
            pretrained=True,
            source='local'
        )
        dinov2_model.eval()
        self.models['image_cond_model'] = dinov2_model
        transform = transforms.Compose([
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.image_cond_model_transform = transform

    # Image segmentation.
    def preprocess_image(self, input: Image.Image) -> Image.Image:
        """
        Preprocess the input image.
        """
        # if has alpha channel, use it directly; otherwise, remove background
        has_alpha = False
        if input.mode == 'RGBA':
            alpha = np.array(input)[:, :, 3]
            if not np.all(alpha == 255):
                has_alpha = True
        if has_alpha:
            output = input
        else:
            input = input.convert('RGB')
            max_size = max(input.size)
            scale = min(1, 1024 / max_size)
            if scale < 1:
                input = input.resize((int(input.width * scale), int(input.height * scale)), Image.Resampling.LANCZOS)
            if getattr(self, 'rembg_session', None) is None:
                self.rembg_session = rembg.new_session('u2net')
            output = rembg.remove(input, session=self.rembg_session)
        output_np = np.array(output)
        alpha = output_np[:, :, 3]
        bbox = np.argwhere(alpha > 0.8 * 255)
        bbox = np.min(bbox[:, 1]), np.min(bbox[:, 0]), np.max(bbox[:, 1]), np.max(bbox[:, 0])
        center = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
        size = max(bbox[2] - bbox[0], bbox[3] - bbox[1])
        size = int(size * 1.2)
        bbox = center[0] - size // 2, center[1] - size // 2, center[0] + size // 2, center[1] + size // 2
        output = output.crop(bbox)  # type: ignore
        output = output.resize((518, 518), Image.Resampling.LANCZOS)
        output = np.array(output).astype(np.float32) / 255
        output = output[:, :, :3] * output[:, :, 3:4]
        output = Image.fromarray((output * 255).astype(np.uint8))
        return output


    # Visualize the image segmentation result.
    def save_unet_segmentation(self,img: Image.Image, output_dir: str):
        """
        Use U2Net (rembg) to segment the image and save the foreground mask and overlay.
        Args:
            image_path (str): Input image path.
            output_dir (str): Directory for saved outputs.
        """
        import numpy as np
        import rembg
        import os
        # Create the output directory.
        os.makedirs(output_dir, exist_ok=True)
        # Read image.
        # Initialize the rembg session.
        session = rembg.new_session('u2net')
        # Get the foreground RGBA image.
        fg = rembg.remove(img, session=session)
        fg_np = np.array(fg)

        # alpha mask
        alpha = fg_np[:, :, 3]  # 0-255
        # Save the foreground overlay with a black background.
        foreground = Image.fromarray(fg_np[:, :, :3])
        foreground.putalpha(Image.fromarray(alpha))
        overlay_path = os.path.join(output_dir, 'foreground.png')
        foreground.save(overlay_path)
        # 
        hfer = calculate_hfer_robust(overlay_path)
        print("hfer",hfer)
        self.hfer_2d = hfer
        return hfer

    
        
    @torch.no_grad()
    def encode_image(self, image: Union[torch.Tensor, list[Image.Image]]) -> torch.Tensor:
        """
        Encode the image.

        Args:
            image (Union[torch.Tensor, list[Image.Image]]): The image to encode

        Returns:
            torch.Tensor: The encoded features.
        """
        if isinstance(image, torch.Tensor):
            assert image.ndim == 4, "Image tensor should be batched (B, C, H, W)"
        elif isinstance(image, list):
            assert all(isinstance(i, Image.Image) for i in image), "Image list should be list of PIL images"
            image = [i.resize((518, 518), Image.LANCZOS) for i in image]
            image = [np.array(i.convert('RGB')).astype(np.float32) / 255 for i in image]
            image = [torch.from_numpy(i).permute(2, 0, 1).float() for i in image]
            image = torch.stack(image).to(self.device)
        else:
            raise ValueError(f"Unsupported type of image: {type(image)}")
        
        image = self.image_cond_model_transform(image).to(self.device)
        features = self.models['image_cond_model'](image, is_training=True)['x_prenorm']
        patchtokens = F.layer_norm(features, features.shape[-1:])
        return patchtokens
        
    def get_cond(self, image: Union[torch.Tensor, list[Image.Image]]) -> dict:
        """
        Get the conditioning information for the model.

        Args:
            image (Union[torch.Tensor, list[Image.Image]]): The image prompts.

        Returns:
            dict: The conditioning information
        """
        cond = self.encode_image(image)
        neg_cond = torch.zeros_like(cond)
        return {
            'cond': cond,
            'neg_cond': neg_cond,
        }

    # Sparse-structure sampling.
    def sample_sparse_structure(
        self,
        cond: dict,
        num_samples: int = 1,
        sampler_params: dict = {},
    ) -> torch.Tensor:
        """
        Sample sparse structures with the given conditioning.
        
        Args:
            cond (dict): The conditioning information.
            num_samples (int): The number of samples to generate.
            sampler_params (dict): Additional parameters for the sampler.
        """
        # Sample occupancy latent
        flow_model = self.models['sparse_structure_flow_model']

        reso = flow_model.resolution
        noise = torch.randn(num_samples, flow_model.in_channels, reso, reso, reso).to(self.device)
        sampler_params = {**self.sparse_structure_sampler_params, **sampler_params}
        
        z_s = self.sparse_structure_sampler.sample(
            flow_model,
            noise,
            **cond,
            **sampler_params,
            verbose=True
        ).samples
        
        # Decode occupancy latent
        decoder = self.models['sparse_structure_decoder']
        
        occupancy = decoder(z_s)
        coords = torch.argwhere(occupancy > 0)[:, [0, 2, 3, 4]].int()
        # Mesh-aware aggregation can be enabled here.
        coords_value = get_coords_value(occupancy)
        coords_scores ,hfer_3d = process_and_visualize(
            coords_value,
            output_dir="./visualization",
            filter_radius=8,
            draw_spatial=self.enable_voxel_visualization,
            draw_freq=self.enable_voxel_visualization,
        )

        print(coords_scores.shape)
        print(hfer_3d)
        # import pdb;pdb.set_trace()

        sample_type = "raw"
        if self.enable_mesh:
            sample_type = "double"
            print("Enable_mesh_aggregation!")

        if sample_type == "raw":
            factor = 1
            coords_sample, coords_scores, downsample_factor = _downsample_with_feature_fusion(
            coords, 
            coords_scores, 
            max_coords=42000, 
            downsample_factor = factor,
            fusion_mode='max') 

        if sample_type == "double": 
            # factor,score = self.calculate_adaptive_factor(self.hfer_2d, hfer_3d,
            #                                             high_thresh = self.mesh_params['mesh_spectral_threshold_high'],
            #                                             low_thresh =self.mesh_params['mesh_spectral_threshold_low'] )
            factor,score = _calculate_adaptive_factor(self.hfer_2d, hfer_3d,
                                                        high_thresh = 0.55,
                                                        low_thresh = 0.3 )
            
            print()
            coords_sample, coords_scores, downsample_factor = _downsample_with_feature_fusion(
            coords, 
            coords_scores, 
            max_coords=42000, 
            downsample_factor = factor,
            fusion_mode='max') 

        print(f"factor: {factor}")
        print(f"downsample_factor:  {downsample_factor}")
        print(f"coords:  {coords.shape}")
        print(f"coords_sample:  {coords_sample.shape}")
        # import pdb;pdb.set_trace()
        return coords_sample,coords_scores

    # Structured latent decoding.
    def decode_slat(
        self,
        slat: sp.SparseTensor,
        formats: List[str] = ['mesh', 'gaussian', 'radiance_field'],
    ) -> dict:
        """
        Decode the structured latent.

        Args:
            slat (sp.SparseTensor): The structured latent.
            formats (List[str]): The formats to decode the structured latent to.

        Returns:
            dict: The decoded structured latent.
        """
        ret = {}
        if 'mesh' in formats:
            ret['mesh'] = self.models['slat_decoder_mesh'](slat)
        if 'gaussian' in formats:
            ret['gaussian'] = self.models['slat_decoder_gs'](slat)
        if 'radiance_field' in formats:
            ret['radiance_field'] = self.models['slat_decoder_rf'](slat)
        return ret
    

    def sample_slat(
        self,
        cond: dict,
        coords: torch.Tensor,
        coords_scores: Optional[torch.Tensor] = None,
        sampler_params: dict = {},
    ) -> sp.SparseTensor:
        """
        Sample structured latent with the given conditioning.
        
        Args:
            cond (dict): The conditioning information.
            coords (torch.Tensor): The coordinates of the sparse structure.
            sampler_params (dict): Additional parameters for the sampler.
        """
        # Sample structured latent
        flow_model = self.models['slat_flow_model']
        
        noise = sp.SparseTensor(
            feats=torch.randn(coords.shape[0], flow_model.in_channels).to(self.device),
            coords=coords,
        )

        # noise_feats:torch.Size([15201, 8]),noise_coords:torch.Size([15201, 4])
        # print(f"noise_feats:{noise.feats.shape},noise_coords:{noise.coords.shape}")

        # import pdb;pdb.set_trace()
        sampler_params = {**self.slat_sampler_params, **sampler_params}
        if coords_scores is not None and self.enable_faster and hasattr(self.slat_sampler, "set_coords_scores"):
            print("Injecting coords_scores")
            self.slat_sampler.set_coords_scores(coords_scores)

        slat = self.slat_sampler.sample(
            flow_model,
            noise,
            **cond,
            **sampler_params,
            verbose=True
        ).samples

        std = torch.tensor(self.slat_normalization['std'])[None].to(slat.device)
        mean = torch.tensor(self.slat_normalization['mean'])[None].to(slat.device)
        slat = slat * std + mean
        
        return slat
    
    # End-to-end single-image inference.
    @torch.no_grad()
    def run(
        self,
        image: Image.Image,
        num_samples: int = 1,
        seed: int = 42,
        sparse_structure_sampler_params: dict = {},
        slat_sampler_params: dict = {},
        formats: List[str] = ['mesh', 'gaussian', 'radiance_field'],
        preprocess_image: bool = True,
    ) -> dict:
        """
        Run the pipeline with time and type logging for each stage.

        Args:
            image (Image.Image): The image prompt.
            num_samples (int): The number of samples to generate.
            seed (int): The random seed.
            sparse_structure_sampler_params (dict): Additional parameters for the sparse structure sampler.
            slat_sampler_params (dict): Additional parameters for the structured latent sampler.
            formats (List[str]): The formats to decode the structured latent to.
            preprocess_image (bool): Whether to preprocess the image.
        """
        import time
        from typing import List
        t_s = time.time()
        print("⭐ Starting inference")
        if preprocess_image:
            image = self.preprocess_image(image)
            self.save_unet_segmentation(image, "segmentation")
            # import pdb;pdb.set_trace()
        
        cond = self.get_cond([image])
        torch.manual_seed(seed)

        # -------------------------------
        # Step 1: Sparse Structure Sampling
        # -------------------------------
        t0 = time.time()
        coords,coords_scores = self.sample_sparse_structure(cond, num_samples, sparse_structure_sampler_params)
        t1 = time.time()
        print(f"✅ [Step 1] sample_sparse_structure completed, time: {t1-t0:.3f}s, output type: {type(coords)}")
        print(f"✅ coords: {coords.shape}") # torch.Size([14948, 4])
        print("\n")


        # -------------------------------
        # Step 2: Slat Sampling
        # -------------------------------
        t0 = time.time()
        slat = self.sample_slat(cond, coords,coords_scores, slat_sampler_params)
        t1 = time.time()
        print(f"✅ [Step 2] sample_slat completed, time: {t1-t0:.3f}s, output type: {type(slat)}")
        slat_coords, slat_feats = slat.coords, slat.feats
        print(f"✅ slat_coords:{slat_coords.shape}, slat_feats:{slat_feats.shape}")
        print("\n")

        # -------------------------------
        # Step 3: Decode Slat
        # -------------------------------
        t0 = time.time()
        decoded = self.decode_slat(slat, formats)
        t1 = time.time()
        print(f"✅ [Step 3] decode_slat completed, time: {t1-t0:.3f}s, output type: {type(decoded)}")
        print(f"✅ decoded:{decoded.keys()}") # ['mesh', 'gaussian', 'radiance_field'])
        print(f"mesh: {type(decoded['mesh'])},{len(decoded['mesh'])}")
        print(f"gaussian: {type(decoded['gaussian'])},{len(decoded['gaussian'])}")
        print(f"radiance_field: {type(decoded['radiance_field'])},{len(decoded['radiance_field'])}")
        print("\n")

        t_e = time.time()
        print(f"⭐ Inference finished, time: {t_e-t_s:.3f}s")
        # import pdb; pdb.set_trace()
        return  decoded

    @contextmanager
    def inject_sampler_multi_image(
        self,
        sampler_name: str,
        num_images: int,
        num_steps: int,
        mode: Literal['stochastic', 'multidiffusion'] = 'stochastic',
    ):
        """
        Inject a sampler with multiple images as condition.
        
        Args:
            sampler_name (str): The name of the sampler to inject.
            num_images (int): The number of images to condition on.
            num_steps (int): The number of steps to run the sampler for.
        """
        sampler = getattr(self, sampler_name)
        setattr(sampler, f'_old_inference_model', sampler._inference_model)

        if mode == 'stochastic':
            if num_images > num_steps:
                print(f"\033[93mWarning: number of conditioning images is greater than number of steps for {sampler_name}. "
                    "This may lead to performance degradation.\033[0m")

            cond_indices = (np.arange(num_steps) % num_images).tolist()
            def _new_inference_model(self, model, x_t, t, cond, **kwargs):
                cond_idx = cond_indices.pop(0)
                cond_i = cond[cond_idx:cond_idx+1]
                return self._old_inference_model(model, x_t, t, cond=cond_i, **kwargs)
        
        elif mode =='multidiffusion':
            from .samplers import FlowEulerSampler
            def _new_inference_model(self, model, x_t, t, cond, neg_cond, cfg_strength, cfg_interval, **kwargs):
                if cfg_interval[0] <= t <= cfg_interval[1]:
                    preds = []
                    for i in range(len(cond)):
                        preds.append(FlowEulerSampler._inference_model(self, model, x_t, t, cond[i:i+1], **kwargs))
                    pred = sum(preds) / len(preds)
                    neg_pred = FlowEulerSampler._inference_model(self, model, x_t, t, neg_cond, **kwargs)
                    return (1 + cfg_strength) * pred - cfg_strength * neg_pred
                else:
                    preds = []
                    for i in range(len(cond)):
                        preds.append(FlowEulerSampler._inference_model(self, model, x_t, t, cond[i:i+1], **kwargs))
                    pred = sum(preds) / len(preds)
                    return pred
            
        else:
            raise ValueError(f"Unsupported mode: {mode}")
            
        sampler._inference_model = _new_inference_model.__get__(sampler, type(sampler))

        yield

        sampler._inference_model = sampler._old_inference_model
        delattr(sampler, f'_old_inference_model')

    @torch.no_grad()
    def run_multi_image(
        self,
        images: List[Image.Image],
        num_samples: int = 1,
        seed: int = 42,
        sparse_structure_sampler_params: dict = {},
        slat_sampler_params: dict = {},
        formats: List[str] = ['mesh', 'gaussian', 'radiance_field'],
        preprocess_image: bool = True,
        mode: Literal['stochastic', 'multidiffusion'] = 'stochastic',
    ) -> dict:
        """
        Run the pipeline with multiple images as condition

        Args:
            images (List[Image.Image]): The multi-view images of the assets
            num_samples (int): The number of samples to generate.
            sparse_structure_sampler_params (dict): Additional parameters for the sparse structure sampler.
            slat_sampler_params (dict): Additional parameters for the structured latent sampler.
            preprocess_image (bool): Whether to preprocess the image.
        """
        if preprocess_image:
            images = [self.preprocess_image(image) for image in images]
        if self.enable_mesh and images:
            self.save_unet_segmentation(images[0], "segmentation")
        cond = self.get_cond(images)
        cond['neg_cond'] = cond['neg_cond'][:1]
        torch.manual_seed(seed)

        ss_steps = {**self.sparse_structure_sampler_params, **sparse_structure_sampler_params}.get('steps')

        with self.inject_sampler_multi_image('sparse_structure_sampler', len(images), ss_steps, mode=mode):
            coords, coords_scores = self.sample_sparse_structure(cond, num_samples, sparse_structure_sampler_params)

        slat_steps = {**self.slat_sampler_params, **slat_sampler_params}.get('steps')

        with self.inject_sampler_multi_image('slat_sampler', len(images), slat_steps, mode=mode):
            slat = self.sample_slat(cond, coords, coords_scores, slat_sampler_params)
            
        return self.decode_slat(slat, formats)
    

    # Calculate the downsampling ratio.
    def calculate_adaptive_factor(self,hfer_2d, hfer_3d, high_thresh = 0.7,low_thresh = 0.5):

        factor, combined_score = _calculate_adaptive_factor(
            hfer_2d,
            hfer_3d,
            high_thresh=high_thresh,
            low_thresh=low_thresh,
        )
        print(f"⭐⭐ 2d:{hfer_2d}, 3d:{hfer_3d}, combined score: {combined_score}")
        return factor, combined_score

    # Token fusion.
    def downsample_with_feature_fusion(
        self,
        coord_batch, 
        coords_scores, 
        max_coords=42000, 
        downsample_factor=2,
        fusion_mode='mean',
    ):
    
        return _downsample_with_feature_fusion(
            coord_batch,
            coords_scores,
            max_coords=max_coords,
            downsample_factor=downsample_factor,
            fusion_mode=fusion_mode,
        )
