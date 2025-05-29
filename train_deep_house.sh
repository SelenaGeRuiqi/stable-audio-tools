#!/bin/bash
# Deep House Fine-tuning Script

echo "🎵 Starting Deep House Fine-tuning..."
echo "Dataset: 43 tracks, 20 seconds each"
echo "Target: Improve Deep House generation quality"

# Setup Weights & Biases (optional)
# wandb login

# Start training
python train.py \
    --dataset-config configs/deep_house_dataset_config.json \
    --model-config configs/deep_house_model_config.json \
    --name deep_house_finetuning \
    --pretrained-ckpt-path stabilityai/stable-audio-open-1.0 \
    --save-dir checkpoints/deep_house \
    --batch-size 1 \
    --accum-batches 4 \
    --num-gpus 1

echo "✅ Fine-tuning complete!"
echo "Check checkpoints/deep_house/ for saved models"
