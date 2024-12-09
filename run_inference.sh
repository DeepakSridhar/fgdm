#!/bin/bash
export PYTHONPATH=$PWD

# Set default values
DEFAULT_PROMPT="a bride and a groom against the backdrop of a mountain"
DEFAULT_NAME="outputs/txt2img-samples/seg/"

# Check if arguments are provided, else use defaults
PROMPT=${1:-$DEFAULT_PROMPT}
NAME=${2:-$DEFAULT_NAME}

# Display the arguments being used
echo "Using prompt: $PROMPT"
echo "Output directory: $NAME"

# Run the inference script
python scripts/txt2img_fgdm_inference.py \
    --config models/config.yaml \
    --prompt "$PROMPT" \
    --ddim_eta 0.0 \
    --n_samples 5 \
    --n_iter 1 \
    --scale 7.5 \
    --ddim_steps 50 \
    --ckpt models/fgdm_seg.pth \
    --H 256 \
    --W 256 \
    --outdir "$NAME" \
    --C 4 \
    --use_controlnet
