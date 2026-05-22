import torch


def calculate_adaptive_factor(hfer_2d, hfer_3d, high_thresh=0.7, low_thresh=0.5):
    """Choose a sparse-structure aggregation factor from 2D/3D frequency scores."""
    combined_score = (hfer_2d * 0.9) + (hfer_3d * 0.1)
    if combined_score >= high_thresh:
        factor = 1.25
    elif combined_score > low_thresh:
        factor = 1.50
    else:
        factor = 1.00
    return factor, combined_score


def downsample_with_feature_fusion(
    coord_batch,
    coords_scores,
    max_coords=42000,
    downsample_factor=2,
    fusion_mode="mean",
):
    """Merge nearby sparse-structure coordinates while carrying a fused spatial score."""
    assert coord_batch.shape[0] == coords_scores.shape[0]

    min_coords = 2048
    current_factor = downsample_factor
    if coord_batch.shape[0] >= max_coords:
        current_factor = 2
    if coord_batch.shape[0] < min_coords:
        current_factor = 1

    device = coord_batch.device
    coords = coord_batch[:, 1:].float()
    batch_indices = coord_batch[:, 0:1]

    coords_min = coords.min(dim=0)[0]
    coords_max = coords.max(dim=0)[0]
    original_size = coords_max - coords_min + 1
    safe_extent = torch.clamp(coords_max - coords_min, min=1)

    target_size = original_size / current_factor
    target_min = coords_min + ((original_size - target_size) / 2)
    target_max = target_min + target_size - 1

    coords_normalized = (coords - coords_min) / safe_extent
    coords_rescaled = coords_normalized * (target_size - 1) + target_min
    coords_rescaled = torch.round(coords_rescaled).int()
    coords_rescaled = torch.clamp(coords_rescaled, target_min.int(), target_max.int())

    combined_keys = torch.cat([batch_indices, coords_rescaled], dim=1)
    unique_keys, inverse_indices = torch.unique(combined_keys, dim=0, return_inverse=True)
    inverse_indices = inverse_indices.to(device)

    raw_scores = coords_scores[:, 0].float().to(device)
    fused_scores = torch.zeros(unique_keys.shape[0], device=device, dtype=raw_scores.dtype)

    if fusion_mode == "mean":
        fused_scores.index_add_(0, inverse_indices, raw_scores)
        counts = torch.zeros_like(fused_scores)
        counts.index_add_(0, inverse_indices, torch.ones_like(raw_scores))
        fused_scores = fused_scores / (counts + 1e-6)
    elif fusion_mode == "max":
        fused_scores.fill_(-1e9)
        fused_scores.scatter_reduce_(0, inverse_indices, raw_scores, reduce="amax", include_self=False)
    else:
        raise ValueError(f"Unsupported fusion_mode: {fusion_mode}")

    new_coord_batch = unique_keys.int()
    new_coords_scores = torch.cat(
        [fused_scores.unsqueeze(1), new_coord_batch[:, 1:].float()],
        dim=1,
    )

    if new_coord_batch.shape[0] > max_coords:
        perm = torch.randperm(new_coord_batch.shape[0], device=device)[: int(max_coords)]
        new_coord_batch = new_coord_batch[perm]
        new_coords_scores = new_coords_scores[perm]

    return new_coord_batch, new_coords_scores, current_factor
