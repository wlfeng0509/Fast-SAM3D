
#!/usr/bin/env bash
set -e

IMAGE_PATH="${1:-/data3/wmq/TRELLIS/assets/example_image/T.png}"
if [ "$#" -gt 0 ]; then
    shift
fi

# export CUDA_VISIBLE_DEVICES=0
# python /data3/wmq/TRELLIS/example.py \
#     --image_path "$IMAGE_PATH" \
#     --enable_faster \
#     --enable_mesh \
#     "$@"

export CUDA_VISIBLE_DEVICES=0
python /data3/wmq/TRELLIS/example.py \
    --image_path "$IMAGE_PATH" \
    "$@"

