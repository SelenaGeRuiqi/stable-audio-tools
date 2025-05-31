## Download the pretrained model
huggingface-cli download stabilityai/stable-audio-open-1.0 --local-dir ./stable-audio-open-1.0/

## With SingleA6000 62G RAM: 2 hours for 1000 steps

# Option 1: Train the whole model
python train.py \
  --dataset-config classical_piano_dataset/dataset_config.json \
  --model-config ./stable-audio-open-1.0/model_config.json \
  --pretrained-ckpt-path ./stable-audio-open-1.0/model.ckpt \
  --name piano_1000_clips \
  --batch-size 4 \
  --accum-batches 4 \
  --precision 16-mixed \
  --checkpoint-every 500 \
  --save-dir ./checkpoints

# Option 2: Freeze part of the model (T5 encoder and part of the unet)
FREEZE_T5=true FREEZE_UNET=true UNET_FREEZE_RATIO=0.5 python train_freeze.py \
  --dataset-config classical_piano_dataset/dataset_config.json \
  --model-config ./stable-audio-open-1.0/model_config.json \
  --pretrained-ckpt-path ./stable-audio-open-1.0/model.ckpt \
  --name piano_1000_clips \
  --batch-size 4 \
  --accum-batches 4 \
  --precision 16-mixed \
  --checkpoint-every 500 \
  --save-dir ./checkpoints


python ./unwrap_model.py \
  --model-config ./stable-audio-open-1.0/model_config.json \
  --ckpt-path ./checkpoints/piano_1000_clips/jf94czsu/checkpoints/epoch=15-step=1000.ckpt \
  --name sao_piano_1000clips

python run_gradio.py \
  --model-config ./stable-audio-open-1.0/model_config.json \
  --ckpt-path ./sao_piano_1000clips.ckpt

python evaluation_piano.py