#!/usr/bin/env bash
set -e

IMAGE_DIR="${1:-/data3/wmq/TRELLIS/assets/multi1}"
if [ "$#" -gt 0 ]; then
    shift
fi

export CUDA_VISIBLE_DEVICES=0
python /data3/wmq/TRELLIS/example_multi_image.py \
    --image_dir "$IMAGE_DIR" \
    "$@"


# export CUDA_VISIBLE_DEVICES=0
# python /data3/wmq/TRELLIS/example_multi_image.py \
#     --image_dir "$IMAGE_DIR" \
#     --enable_faster \
#     --enable_mesh \
#     "$@"

