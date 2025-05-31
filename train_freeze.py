import torch
import json
import os
import pytorch_lightning as pl
import argparse

from typing import Dict, Optional, Union
from prefigure.prefigure import get_all_args, push_wandb_config
from stable_audio_tools.data.dataset import create_dataloader_from_config, fast_scandir
from stable_audio_tools.models import create_model_from_config
from stable_audio_tools.models.utils import copy_state_dict, load_ckpt_state_dict, remove_weight_norm_from_model
from stable_audio_tools.training import create_training_wrapper_from_config, create_demo_callback_from_config

class ExceptionCallback(pl.Callback):
    def on_exception(self, trainer, module, err):
        print(f'{type(err).__name__}: {err}')

class ModelConfigEmbedderCallback(pl.Callback):
    def __init__(self, model_config):
        self.model_config = model_config

    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        checkpoint["model_config"] = self.model_config

def freeze_t5_encoder(model):
    """Freeze T5 encoder parameters"""
    frozen_params = 0
    total_params = 0
    
    # Look for T5 encoder in different possible locations
    t5_components = []
    
    # Check common attribute names for T5 encoder
    if hasattr(model, 'text_encoder'):
        t5_components.append(model.text_encoder)
    if hasattr(model, 't5_encoder'):
        t5_components.append(model.t5_encoder)
    if hasattr(model, 'encoder'):
        t5_components.append(model.encoder)
    if hasattr(model, 'conditioner') and hasattr(model.conditioner, 'text_encoder'):
        t5_components.append(model.conditioner.text_encoder)
    if hasattr(model, 'conditioner') and hasattr(model.conditioner, 't5'):
        t5_components.append(model.conditioner.t5)
    
    # Try to find T5 components by searching through all modules
    for name, module in model.named_modules():
        if 't5' in name.lower() and 'encoder' in name.lower():
            t5_components.append(module)
    
    # Freeze found T5 components
    for component in t5_components:
        for param in component.parameters():
            param.requires_grad = False
            frozen_params += param.numel()
            total_params += param.numel()
    
    print(f"Frozen T5 encoder parameters: {frozen_params:,}")
    return frozen_params

def freeze_unet_layers(model, freeze_ratio=0.5):
    """Freeze front to mid part of UNet layers"""
    frozen_params = 0
    total_params = 0
    
    # Look for UNet in different possible locations
    unet_component = None
    
    if hasattr(model, 'unet'):
        unet_component = model.unet
    elif hasattr(model, 'diffusion_model'):
        unet_component = model.diffusion_model
    elif hasattr(model, 'model'):
        unet_component = model.model
    elif hasattr(model, 'backbone'):
        unet_component = model.backbone
    
    # Try to find UNet by searching through modules
    if unet_component is None:
        for name, module in model.named_modules():
            if 'unet' in name.lower() or 'diffusion' in name.lower():
                unet_component = module
                break
    
    if unet_component is None:
        print("Warning: Could not find UNet component to freeze")
        return 0
    
    # Get all UNet layers
    unet_layers = []
    for name, module in unet_component.named_modules():
        # Look for common UNet layer patterns
        if any(layer_type in name.lower() for layer_type in ['downsample', 'upsample', 'resblock', 'attn', 'conv']):
            unet_layers.append((name, module))
    
    # If no specific layers found, use all parameters
    if not unet_layers:
        unet_layers = [(name, module) for name, module in unet_component.named_modules() if len(list(module.parameters())) > 0]
    
    # Calculate how many layers to freeze
    num_layers_to_freeze = int(len(unet_layers) * freeze_ratio)
    
    print(f"Found {len(unet_layers)} UNet layers, freezing first {num_layers_to_freeze} layers ({freeze_ratio*100:.1f}%)")
    
    # Freeze the front portion of layers
    for name, module in unet_layers[:num_layers_to_freeze]:
        for param in module.parameters():
            if param.requires_grad:  # Only count if it was trainable
                param.requires_grad = False
                frozen_params += param.numel()
        total_params += sum(p.numel() for p in module.parameters())
    
    print(f"Frozen UNet parameters: {frozen_params:,} out of {total_params:,}")
    return frozen_params

def add_freezing_args(parser):
    """Add freezing-related arguments to the parser"""
    freeze_group = parser.add_argument_group('Freezing Options')
    freeze_group.add_argument('--freeze-t5', action='store_true', 
                             help='Freeze T5 encoder parameters')
    freeze_group.add_argument('--freeze-unet', action='store_true',
                             help='Freeze part of UNet layers (front to mid part)')
    freeze_group.add_argument('--unet-freeze-ratio', type=float, default=0.5,
                             help='Ratio of UNet layers to freeze from the front (default: 0.5)')
    return parser

def main():
    torch.multiprocessing.set_sharing_strategy('file_system')
    
    # Get args using prefigure, but also add our custom freezing args
    args = get_all_args()
    
    # Add freezing arguments to the existing args
    # Check for environment variables as fallback
    freeze_t5 = getattr(args, 'freeze_t5', False) or os.environ.get('FREEZE_T5', '').lower() == 'true'
    freeze_unet = getattr(args, 'freeze_unet', False) or os.environ.get('FREEZE_UNET', '').lower() == 'true'
    unet_freeze_ratio = getattr(args, 'unet_freeze_ratio', 0.5)
    try:
        unet_freeze_ratio = float(os.environ.get('UNET_FREEZE_RATIO', str(unet_freeze_ratio)))
    except ValueError:
        unet_freeze_ratio = 0.5
    
    seed = args.seed

    # Set a different seed for each process if using SLURM
    if os.environ.get("SLURM_PROCID") is not None:
        seed += int(os.environ.get("SLURM_PROCID"))

    pl.seed_everything(seed, workers=True)

    #Get JSON config from args.model_config
    with open(args.model_config) as f:
        model_config = json.load(f)

    with open(args.dataset_config) as f:
        dataset_config = json.load(f)

    train_dl = create_dataloader_from_config(
        dataset_config,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        sample_rate=model_config["sample_rate"],
        sample_size=model_config["sample_size"],
        audio_channels=model_config.get("audio_channels", 2),
    )

    val_dl = None
    val_dataset_config = None

    if args.val_dataset_config:
        with open(args.val_dataset_config) as f:
            val_dataset_config = json.load(f)

        val_dl = create_dataloader_from_config(
            val_dataset_config,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            sample_rate=model_config["sample_rate"],
            sample_size=model_config["sample_size"],
            audio_channels=model_config.get("audio_channels", 2),
            shuffle=False
        )

    model = create_model_from_config(model_config)

    if args.pretrained_ckpt_path:
        copy_state_dict(model, load_ckpt_state_dict(args.pretrained_ckpt_path))

    if args.remove_pretransform_weight_norm == "pre_load":
        remove_weight_norm_from_model(model.pretransform)

    if args.pretransform_ckpt_path:
        model.pretransform.load_state_dict(load_ckpt_state_dict(args.pretransform_ckpt_path))

    # Remove weight_norm from the pretransform if specified
    if args.remove_pretransform_weight_norm == "post_load":
        remove_weight_norm_from_model(model.pretransform)

    # Apply freezing after model loading
    total_frozen_params = 0
    if freeze_t5:
        print("Freezing T5 encoder...")
        total_frozen_params += freeze_t5_encoder(model)
    
    if freeze_unet:
        print(f"Freezing UNet layers (ratio: {unet_freeze_ratio})...")
        total_frozen_params += freeze_unet_layers(model, unet_freeze_ratio)
    
    if total_frozen_params > 0:
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"\nParameter Summary:")
        print(f"Total parameters: {total_params:,}")
        print(f"Frozen parameters: {total_frozen_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        print(f"Frozen ratio: {(total_frozen_params/total_params)*100:.1f}%")

    training_wrapper = create_training_wrapper_from_config(model_config, model)

    exc_callback = ExceptionCallback()

    if args.logger == 'wandb':
        logger = pl.loggers.WandbLogger(project=args.name)
        logger.watch(training_wrapper)
    
        if args.save_dir and isinstance(logger.experiment.id, str):
            checkpoint_dir = os.path.join(args.save_dir, logger.experiment.project, logger.experiment.id, "checkpoints") 
        else:
            checkpoint_dir = None
    elif args.logger == 'comet':
        logger = pl.loggers.CometLogger(project_name=args.name)
        if args.save_dir and isinstance(logger.version, str):
            checkpoint_dir = os.path.join(args.save_dir, logger.name, logger.version, "checkpoints") 
        else:
            checkpoint_dir = args.save_dir if args.save_dir else None
    else:
        logger = None
        checkpoint_dir = args.save_dir if args.save_dir else None
        
    ckpt_callback = pl.callbacks.ModelCheckpoint(every_n_train_steps=args.checkpoint_every, dirpath=checkpoint_dir, save_top_k=-1)
    save_model_config_callback = ModelConfigEmbedderCallback(model_config)

    if args.val_dataset_config:
        demo_callback = create_demo_callback_from_config(model_config, demo_dl=val_dl)
    else:
        demo_callback = create_demo_callback_from_config(model_config, demo_dl=train_dl)

    #Combine args and config dicts
    args_dict = vars(args)
    args_dict.update({"model_config": model_config})
    args_dict.update({"dataset_config": dataset_config})
    args_dict.update({"val_dataset_config": val_dataset_config})
    # Add freezing info to logged parameters
    args_dict.update({
        "freeze_t5": freeze_t5,
        "freeze_unet": freeze_unet,
        "unet_freeze_ratio": unet_freeze_ratio
    })

    if args.logger == 'wandb':
        push_wandb_config(logger, args_dict)
    elif args.logger == 'comet':
        logger.log_hyperparams(args_dict)

    #Set multi-GPU strategy if specified
    if args.strategy:
        if args.strategy == "deepspeed":
            from pytorch_lightning.strategies import DeepSpeedStrategy
            strategy = DeepSpeedStrategy(stage=2,
                                        contiguous_gradients=True,
                                        overlap_comm=True,
                                        reduce_scatter=True,
                                        reduce_bucket_size=5e8,
                                        allgather_bucket_size=5e8,
                                        load_full_weights=True)
        else:
            strategy = args.strategy
    else:
        strategy = 'ddp_find_unused_parameters_true' if args.num_gpus > 1 else "auto"

    val_args = {}
    
    if args.val_every > 0:
        val_args.update({
            "check_val_every_n_epoch": None,
            "val_check_interval": args.val_every,
        })

    trainer = pl.Trainer(
        devices="auto",
        accelerator="gpu",
        num_nodes = args.num_nodes,
        strategy=strategy,
        precision=args.precision,
        accumulate_grad_batches=args.accum_batches, 
        callbacks=[ckpt_callback, demo_callback, exc_callback, save_model_config_callback],
        logger=logger,
        log_every_n_steps=1,
        max_epochs=20,
        default_root_dir=args.save_dir,
        gradient_clip_val=args.gradient_clip_val,
        reload_dataloaders_every_n_epochs = 0,
        num_sanity_val_steps=0, # If you need to debug validation, change this line
        **val_args      
    )

    trainer.fit(training_wrapper, train_dl, val_dl, ckpt_path=args.ckpt_path if args.ckpt_path else None)

if __name__ == '__main__':
    main()