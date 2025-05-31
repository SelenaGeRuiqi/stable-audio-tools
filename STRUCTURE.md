### Dataset Collection and Preprocessing

|-- piano_collector.py ## See [piano_collector](piano_collector.py)
|-- classical_piano_dataset
|   |-- processed_10s # Store .wav files
|   |-- classical_piano_prompts.json
|   |-- custom_metadata.py
|   |-- dataset_summary.json
|   `-- dataset_config.json

### Training

|-- stable-audio-open-1.0
|   |-- model.ckpt  ## Baseline checkpoint
|   |-- model_config.json  # Use this for training and evaluation, see [model config](stable-audio-open-1.0/model_config.json)
|   `-- ...
|-- stable_audio_tools
|   |-- models
|   |   |-- diffusion.py
|   |   |-- autoencoders.py
|   |   |-- pretransforms.py
|   |   |-- convnext.py
|   |   |-- pretrained.py
|   |   |-- conditioners.py
|   |   `-- ...
|   `-- ...
|-- train.py  ## Train the whole model, see [train instruction](train_instruction.md)
|-- train_freeze.py  ## Train the model with frozen T5 encoder and part of the unet, see [train_freeze](train_freeze.py)
`-- wandb
|-- checkpoints
|   `-- piano_1000_clips

### Music Generation(Samples will be shown at the end of the presentation)

|-- unwrap_model.py  ## Unwrap the finetuned model to get the checkpoint
|-- sao_piano_1000clips.ckpt  ## After unwrap
|-- run_gradio.py  ## Run the gradio api to test the finetuned model and generate samples

### Evaluation

|-- evaluation_piano.py ## See [evaluation_piano](evaluation_piano.py)
|-- evaluation_results
|   |-- baseline_samples
|   |-- finetuned_samples
|   `-- results 
`-- ...