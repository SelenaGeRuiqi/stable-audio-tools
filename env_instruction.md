# 1. Clone the repository
git clone https://github.com/SelenaGeRuiqi/stable-audio-tools.git

# 2. Install MiniConda (remember to set the installation path under /workspace)
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && chmod +x Miniconda3-latest-Linux-x86_64.sh && ./Miniconda3-latest-Linux-x86_64.sh 

source /workspace/miniconda3/etc/profile.d/conda.sh

conda --version

# 3. Create eivronment
conda create -n stable_audio python=3.10 -y
conda activate stable_audio
which python


# 3. Check the cuda version and install the corresponding torch
nvidia-smi
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia

# 4. install project related stuff
cd stable-audio-tools
conda install -c conda-forge librosa wandb -y
pip install soundfile
pip install -e .
pip install einops
pip install gradio
pip install beautifulsoup4
pip install stable-audio-tools torch torchaudio librosa essentia-tensorflow laion_clap scipy matplotlib pandas
conda install -c conda-forge ffmpeg -y

## Optional: install flash-attn for faster inference
pip install flash-attn --no-build-isolation

# Install HuggingFace CLI
pip install huggingface_hub

# For remote servers, use token-based auth (more reliable)
# Get your token from: https://huggingface.co/settings/tokens
huggingface-cli login

# Verify authentication
huggingface-cli whoami

# For memory optimization
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

# Clear GPU memory
python -c "import torch; torch.cuda.empty_cache(); print('✅ GPU cleared')"

# 5. Unzip the dataset
apt-get update && apt-get install -y unzip
unzip deep_house_processed_20s.zip

# 6. Finetune with 20s clips
# 6.1. Download the pretrained model
huggingface-cli download stabilityai/stable-audio-open-1.0 --local-dir ./stable-audio-open-1.0/

# 6.2. Finetune

## With SingleA6000 62G RAM: 2 hours for 1000 steps

# Train the whole model
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

# Freeze part of the model (T5 encoder and part of the unet)
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

# 7. After each run cleanup

# Clear GPU memory
python -c "import torch; torch.cuda.empty_cache(); print('✅ GPU cleared')"

# Check disk space usage in key directories
du -sh ./checkpoints/
du -sh ~/.cache/torch/
df -h .


## Conda env setup
export PATH="/workspace/miniconda3/bin:$PATH"
conda --version
echo 'export PATH="/workspace/miniconda3/bin:$PATH"' >> ~/.bashrc
conda init
source ~/.bashrc
conda activate stable_audio

cd stable-audio-tools

## To scp something from remote server to local Downloads
### Folder
scp -r -i ~/.ssh/id_ed25519 -P 43549 root@38.147.83.14:/workspace/stable-audio-tools/evaluation_results ~/Downloads/
### File
scp -i ~/.ssh/id_ed25519 -P 43549 root@38.147.83.14:/workspace/stable-audio-tools/evaluation_results/results/comparison_table.txt ~/Downloads/

