import torch
import torch.nn.functional as F

class AdvancedStabilityTracker:
    def __init__(self, num_tokens=4096, channels=8, device='cpu'):
        self.num_tokens = num_tokens
        self.device = device
        self.cached_streak_counter = None
        self.prev_pred_v = None
        self.FIXED_THRESHOLD = 5

        self.ACCELERATION_WEIGHT = 0.7
        self.coords_scores = None


        self.motion_score = None
        self.spatial_score = None

    def reset(self, num_tokens = 4096, latent_channels=8 , device='cpu'):
        self.device = device
        self.num_tokens = num_tokens
        self.cached_streak_counter = torch.zeros(self.num_tokens, device=self.device, dtype=torch.long)
        self.prev_pred_v = None

    def set_hyperparameters(self, args):
        pass

    @torch.no_grad()
    def update_and_select(self, pred_v: torch.Tensor, num_to_skip: int, t: float, **kwargs):
        pred_v = pred_v.float()

        if num_to_skip <= 0:
            if self.cached_streak_counter is not None:
                 self.cached_streak_counter.zero_()
            if self.prev_pred_v is None:
                self.prev_pred_v = torch.zeros_like(pred_v)
            self.prev_pred_v.copy_(pred_v)

            fast_update_indices = torch.arange(self.num_tokens, device=self.device)
            return torch.tensor([], device=self.device, dtype=torch.long), fast_update_indices

        l2_scores = torch.norm(pred_v, p=2, dim=2).squeeze(0).view(-1)
        if self.prev_pred_v is None:
            acceleration_scores = torch.zeros_like(l2_scores)
        else:
             if self.prev_pred_v.shape != pred_v.shape:
                  print(f"Warning: prev_pred_v shape {self.prev_pred_v.shape} mismatch with pred_v shape {pred_v.shape}. Resetting prev_pred_v.")
                  self.prev_pred_v = torch.zeros_like(pred_v)
                  acceleration_scores = torch.zeros_like(l2_scores)
             else:
                acceleration_scores = torch.norm(pred_v - self.prev_pred_v, p=2, dim=2).squeeze(0).view(-1)

        eps = 1e-6
        l2_range = l2_scores.max() - l2_scores.min() + eps
        accel_range = acceleration_scores.max() - acceleration_scores.min() + eps
        
        l2_norm_scores = (l2_scores - l2_scores.min()) / l2_range
        accel_norm_scores = (acceleration_scores - acceleration_scores.min()) / accel_range
        L2_NORM_WEIGHT = 1.0 - self.ACCELERATION_WEIGHT
        combined_scores = (self.ACCELERATION_WEIGHT * accel_norm_scores) + (L2_NORM_WEIGHT * l2_norm_scores)

        num_to_pick = min(num_to_skip, self.num_tokens)
        if num_to_pick <= 0:
             preliminary_cached_indices = torch.tensor([], device=self.device, dtype=torch.long)
        else:
             if combined_scores.numel() == 0:
                  print("Warning: combined_scores is empty. Cannot select tokens.")
                  preliminary_cached_indices = torch.tensor([], device=self.device, dtype=torch.long)
             else:
                _, preliminary_cached_indices = torch.topk(combined_scores, k=num_to_pick, largest=False)

        stale_indices = torch.tensor([], device=self.device, dtype=torch.long)
        final_cached_indices = torch.tensor([], device=self.device, dtype=torch.long)

        if preliminary_cached_indices.numel() > 0:
            current_streaks = self.cached_streak_counter[preliminary_cached_indices]
            is_stale_mask = current_streaks >= self.FIXED_THRESHOLD - 1

            stale_indices = preliminary_cached_indices[is_stale_mask]
            final_cached_indices = preliminary_cached_indices[~is_stale_mask]

        update_mask = torch.ones(self.num_tokens, dtype=torch.bool, device=self.device)
        if final_cached_indices.numel() > 0:
            update_mask[final_cached_indices] = False
        fast_update_indices = torch.where(update_mask)[0]

        self.cached_streak_counter[fast_update_indices] = 0
        if final_cached_indices.numel() > 0:
            self.cached_streak_counter[final_cached_indices] += 1

        if self.prev_pred_v is None or self.prev_pred_v.shape != pred_v.shape:
            self.prev_pred_v = torch.zeros_like(pred_v)
        self.prev_pred_v.copy_(pred_v)
            
        return final_cached_indices, fast_update_indices
 
    @torch.no_grad()
    def update_and_select_combined(self, pred_v: torch.Tensor, num_to_skip: int, t: float, coords_scores: torch.Tensor, spatial_weight=0.5, **kwargs):
        pred_v = pred_v.float()
        
        if self.prev_pred_v is None or self.prev_pred_v.shape != pred_v.shape:
            
             self.prev_pred_v = torch.zeros_like(pred_v)
             is_first_frame = True
        else:
             is_first_frame = False

        if num_to_skip <= 0:
            if self.cached_streak_counter is not None:
                 self.cached_streak_counter.zero_()
            self.prev_pred_v.copy_(pred_v)
            return torch.tensor([], device=self.device, dtype=torch.long), torch.arange(self.num_tokens, device=self.device)

      
        current_coords = coords_scores[:, 1:] 
        l2_scores = torch.norm(pred_v, p=2, dim=-1).view(-1) #
        
        if is_first_frame:
            acceleration_scores = torch.zeros_like(l2_scores)
        else:
            acceleration_scores = torch.norm(pred_v - self.prev_pred_v, p=2, dim=-1).view(-1)

        eps = 1e-6
        l2_norm = (l2_scores - l2_scores.min()) / (l2_scores.max() - l2_scores.min() + eps)
 
        accel_norm = (acceleration_scores - acceleration_scores.min()) / (acceleration_scores.max() - acceleration_scores.min() + eps)

        acc_w = getattr(self, 'ACCELERATION_WEIGHT', 0.7)
        motion_score = (acc_w * accel_norm) + ((1.0 - acc_w) * l2_norm)
        
        self.motion_score = torch.cat([motion_score.unsqueeze(1), current_coords], dim=1)
  
       
        spatial_raw = coords_scores[:, 0].view(-1)
        spatial_score = (spatial_raw - spatial_raw.min()) / (spatial_raw.max() - spatial_raw.min() + eps)
        self.spatial_score = torch.cat([spatial_score.unsqueeze(1), current_coords], dim=1)

        final_scores = (spatial_weight * spatial_score) + ((1.0 - spatial_weight) * motion_score)
        self.final_scores = torch.cat([final_scores.unsqueeze(1), current_coords], dim=1)

        
        num_to_pick = min(num_to_skip, self.num_tokens)
        
        if num_to_pick > 0:
            _, preliminary_cached_indices = torch.topk(final_scores, k=num_to_pick, largest=False)
        else:
            preliminary_cached_indices = torch.tensor([], device=self.device, dtype=torch.long)
        
        stale_indices = torch.tensor([], device=self.device, dtype=torch.long)
        final_cached_indices = torch.tensor([], device=self.device, dtype=torch.long)

        if preliminary_cached_indices.numel() > 0:
            current_streaks = self.cached_streak_counter[preliminary_cached_indices]
            is_stale_mask = current_streaks >= self.FIXED_THRESHOLD - 1
            
            stale_indices = preliminary_cached_indices[is_stale_mask]
            final_cached_indices = preliminary_cached_indices[~is_stale_mask]
        
        update_mask = torch.ones(self.num_tokens, dtype=torch.bool, device=self.device)
        if final_cached_indices.numel() > 0:
            update_mask[final_cached_indices] = False 
            
        fast_update_indices = torch.where(update_mask)[0]
    
        self.cached_streak_counter[fast_update_indices] = 0
        if final_cached_indices.numel() > 0:
            self.cached_streak_counter[final_cached_indices] += 1
            
        self.prev_pred_v.copy_(pred_v)

        return final_cached_indices, fast_update_indices
