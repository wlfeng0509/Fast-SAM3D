CUDA_VISIBLE_DEVICES=7 python /data/wmq/Fast-SAM3D/notebook/infer.py \
    --image_path /data/wmq/Fast-SAM3D/notebook/images/shutterstock_stylish_kidsroom_1640806567/image.png \
    --mask_index 14 \
    --output_dir /data/wmq/Fast-SAM3D/Look \
    --ss_cache_stride 3 \
    --ss_warmup 2 \
    --ss_order 1 \
    --ss_momentum_beta 0.5 \
    --slat_thresh 1.5 \
    --slat_warmup 3 \
    --slat_carving_ratio 0.1 \
    --mesh_spectral_threshold_low 0.5 \
    --mesh_spectral_threshold_high 0.7 \
    --enable_acceleration

