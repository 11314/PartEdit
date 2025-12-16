#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import sys
import yaml
import json
import torch
import logging
import argparse
import gc
import random
import numpy as np
from typing import Tuple, List
from torch.utils.data import DataLoader
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
from typing import Union, Optional

# Import the OVAM library
from ovam.stable_diffusion.daam_module import StableDiffusionDAAM, StableDiffusionXLDAAM
from ovam.stable_diffusion.locator import SlimeAttentionLocator

from diffusers import (
    StableDiffusionImg2ImgPipeline,
    StableDiffusionPipeline,
    StableDiffusionXLImg2ImgPipeline,
    StableDiffusionXLPipeline,
)
from src.dataset_hf import HFImageMaskDataset

from .main_logic import (
    process_text,
    save_opt_embedding,
    initial_forwardpass,
    train_embedding,
    generate_images,
    prepare_idx,
    prepare_masks,
    get_init_embedding,
    load_model,
    load_embd
)

def move_to_gpu(model, device, offload_text: bool = False, upcast: bool = False):
    model = model.to(device)
    if offload_text:
        model.text_encoder = model.text_encoder.cpu()
        model.text_encoder_2 = model.text_encoder.cpu()
    if upcast:
        model = model.to(torch.float32)
    return model

def get_logger(
    level=logging.INFO, display_to_terminal: bool = True, file_name: str = None
):
    logger = logging.getLogger(__name__)
    logger.setLevel(level)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    if file_name is not None and file_name != "":
        file_handler = logging.FileHandler(file_name)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    if display_to_terminal:
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

    return logger

def run_training(
    pipeI2I: Union[StableDiffusionImg2ImgPipeline, StableDiffusionXLImg2ImgPipeline],
    example_imgs: torch.Tensor,
    example_masks: torch.Tensor,
    example_imgs_test: torch.Tensor,
    example_masks_test: torch.Tensor,
    device: Union[str, torch.device],
    # Hyperparameters below
    cur_idx: list[int],
    init_seed: int = 0,
    strength: float = 0.25,
    guidance_scale: float = 7.5,
    TEXT="Quadruped Head",
    TEXT_INIT="Head",
    init_embed_format="N{}_0.pt",  # "N{}_{runidx}.pt",
    # decoded_full_idx: int = 3,  # Change this based on TEXT
    gamma=0.7,
    step_size: int = 80,  # StepLR step
    initial_lr: float = 3000,
    n_epochs: int = 1000,
    loss_type="bce",  # bce, blog, l2, nll, cross
    save_dir="tests",
    prompt="",
    dtype=torch.float32,
    start_strategy: str = "random",  # random, token, token_random
    dataset_name: str = "not provided",  # added for logging purposes
    N_train: int = 100,  # added for logging purposes
    N_test: int = 50,  # added for logging purposes
    optimizer: str = "adam",
    subset_steps: int = 0,
    subset_size: int = 0,
    skip_saving:bool = True,
    subset: Optional[str] = None,
    expand_size: int = 512,
    logger = None
):
    """
    This function runs the OVAM training process, saves the trained embeddings,
    generates images, and saves the losses.

    Parameters:
    pipeI2I: The image-to-image pipeline.
    example_imgs: The example images for training.
    example_masks: The example masks for training.
    example_imgs_test: The example images for testing.
    example_masks_test: The example masks for testing.
    cur_idx: The current indices.
    device: The device to run the training on.
    init_seed: The initial seed for random number generation.
    strength: The strength parameter for the initial forward pass.
    guidance_scale: The guidance scale parameter for the initial forward pass.
    TEXT: The text prompt for the process_text function.
    init_embed_format: The format for saving the trained embeddings.
    gamma: The gamma parameter for the train_embedding function.
    step_size: The step size parameter for the train_embedding function.
    initial_lr: The initial learning rate for the train_embedding function.
    n_epochs: The number of epochs for the train_embedding function.
    loss_type: The loss type for the train_embedding function.
    save_dir: The directory to save the losses and embeddings.
    prompt: The prompt for the process_text function.
    dtype: The data type for the train_embedding function.
    Following is for logging purposes
    dataset_name: The name of the dataset.
    N_train: The number of training examples.
    N_test: The number of testing examples.

    """
    assert save_dir != "", f"Please provide a non empty path, got {save_dir=}"
    # create save_dir location
    os.makedirs(save_dir, exist_ok=True)
    _N = len(cur_idx)
    _log_name = init_embed_format.format(_N).split(".")[0]
    embed_format = f"{save_dir}/{init_embed_format}"  # should leave {}
    # write a json file with all the hyperparams used to make this training
    embed_format_half = f"{save_dir}/half{n_epochs//2}_{init_embed_format}"  # should leave {}
    _embedding, decoded_str = (
        process_text(
            pipeI2I,
            TEXT
        )
    )
    # We get index of TEXT_INIT
    decoded_full_idx = [i for i, x in enumerate(decoded_str) if x == TEXT_INIT.lower()]
    if len(decoded_full_idx) == 0:
        msg = f"Could not find {TEXT_INIT=} in {decoded_str=}, using {decoded_str[-2]}"
        if logger:
            logger.warn(msg)
        else:
            print(f"WARNING: {msg}")
        decoded_full_idx = -2
    else:
        decoded_full_idx = decoded_full_idx[0]
    if not os.path.exists(f"{save_dir}/hyperparameters.json"):
        with open(f"{save_dir}/hyperparameters.json", "w") as f:
            json.dump(
                {
                    "init_seed": init_seed,
                    "strength": strength,
                    "guidance_scale": guidance_scale,
                    "TEXT": TEXT,
                    "TEXT_INIT": TEXT_INIT,
                    "init_embed_format": init_embed_format,
                    "decoded_full_idx": decoded_full_idx,
                    "gamma": gamma,
                    "step_size": step_size,
                    "initial_lr": initial_lr,
                    "n_epochs": n_epochs,
                    "loss_type": loss_type,
                    "save_dir": save_dir,
                    "use_SDXL": "XL" in pipeI2I.__class__.__name__,
                    "use_fp16": str(dtype),
                    "dataset_name": dataset_name,
                    "N_train": N_train,
                    "N_test": N_test,
                    "optimizer": optimizer,
                    "subset_steps": subset_steps,
                    "subset_size": subset_size,
                    "subset": subset,
                },
                f,
            )

    with open(f"{save_dir}/hp_{_log_name}.json", "w") as f:
        _cur_idx = cur_idx.tolist() if isinstance(cur_idx, torch.Tensor) else cur_idx
        json.dump(
            {
                "cur_idx": _cur_idx,
            },
            f,
        )
    if isinstance(subset, str):
        subset = [int(a) for a in subset.split(",")]
        subset = None if len(subset) == 0 else subset
    use_SDXL = "XL" in pipeI2I.__class__.__name__
    # expand_sizes = (1024, 1024) if use_SDXL else (512, 512)
    hooker_kwargs = {
        "daam_module_class": StableDiffusionXLDAAM if use_SDXL else StableDiffusionDAAM,
        "block_hooker_kwargs": {
            "subset":subset,
        },
        "locator_hooker_class": SlimeAttentionLocator, #NOTE(ALEX): We use this as optimization
    }

    
     # _layers = [
    #     'up_blocks.1.attentions.0.transformer_blocks.0.attn2',
    #     'up_blocks.1.attentions.1.transformer_blocks.0.attn2',
    #     'up_blocks.1.attentions.2.transformer_blocks.0.attn2',
    #     'up_blocks.2.attentions.0.transformer_blocks.0.attn2',
    #     'up_blocks.2.attentions.1.transformer_blocks.0.attn2',
    #     'up_blocks.3.attentions.0.transformer_blocks.0.attn1',
    #     'up_blocks.3.attentions.1.transformer_blocks.0.attn1',
    #     'up_blocks.3.attentions.2.transformer_blocks.0.attn1'
    # ] # SD2.1

    _layers = [
        'up_blocks.0.attentions.0.transformer_blocks.0.attn2',
        'up_blocks.0.attentions.0.transformer_blocks.1.attn2',
        'up_blocks.0.attentions.0.transformer_blocks.2.attn2',
        'up_blocks.0.attentions.0.transformer_blocks.3.attn2',
        'up_blocks.0.attentions.0.transformer_blocks.4.attn2',
        'up_blocks.0.attentions.0.transformer_blocks.5.attn2',
        'up_blocks.0.attentions.0.transformer_blocks.6.attn2',
        'up_blocks.0.attentions.0.transformer_blocks.7.attn2',
    ] # SDXL

    if "locator_kwargs" not in hooker_kwargs or hooker_kwargs["locator_kwargs"] is None:
        # add "layers" inside
        hooker_kwargs["locator_kwargs"] = {"layers": _layers}
    else:
        # check that we have "layers"
        if "layers" not in hooker_kwargs['locator_kwargs']:
            hooker_kwargs["locator_kwargs"].update({"layers": _layers})
    # update verbose
    hooker_kwargs['locator_kwargs'].update({'verbose': True})

    _selected_masks, selected_masks = prepare_masks(
        example_masks, cur_idx
    )
    example_imgs_in, cur_idx = prepare_idx(cur_idx, example_imgs, device)

    _options = "|".join(["{}:{}".format(k, v) for k, v in enumerate(decoded_str)])
    assert (
        len(decoded_str) - 1 >= decoded_full_idx
    ), f"{decoded_full_idx=} should corespond to options [{_options}]"

    init_embedding, _clone_init = get_init_embedding(
        start_strategy, pipeI2I, _embedding, -1, device
    )
    
    print(f"Using {start_strategy=} initialization")
    with torch.autocast(device_type="cuda", enabled=True):
        # Initial forward pass
        evaluator, img_pil = initial_forwardpass(
            pipeI2I,
            example_imgs_in.to(device),
            cur_idx,
            strength,
            guidance_scale,
            seed=init_seed,
            prompt=prompt,
            expand_size=(expand_size, expand_size),
            hooker_kwargs=hooker_kwargs,
            no_grad_context=True,  # Note(Alex): Pretty sure it will just speedup
            verbose=True
        )
    # Move the model before training
    pipeI2I.to("cpu")
    gc.collect()
    torch.cuda.empty_cache()

    # we need to cast back to full precision the evaluator
    if dtype == torch.float32:
        msg = "Upcasting EVALUATOR"
        if logger:
            logger.info(msg)
        print(msg)
        evaluator = move_to_gpu(
            evaluator, device, offload_text=True, upcast=True
        )

    # Train embedding
    trained_new, opt_embedding, _half, my_callback = train_embedding(
        init_embedding=init_embedding,
        double_target=selected_masks,
        evaluator=evaluator.to(device),
        device=device,
        n_epochs=n_epochs,
        _initial_lr=initial_lr,
        _step_size=step_size,
        _gamma=gamma,
        loss_type=loss_type,
        cast=True if dtype == torch.float16 else False,
        optimizer=optimizer,
    )

    trained_new, _, _ = save_opt_embedding(
        opt_embedding=opt_embedding,
        trained_new=trained_new,
        direct_embd_name=_N,
        file_save=embed_format,
    )

    _, _, _ = save_opt_embedding(
        opt_embedding=_half,
        trained_new=True,
        direct_embd_name=_N,
        file_save=embed_format_half,
    )
    
    # Plot metrics
    my_callback.plot_metrics(loss_type).save(f"{save_dir}/metrics_{_log_name}.png")
    # evaluator = move_to_gpu(evaluator, torch.device('cpu'),
    #                         offload_text=False, upcast=False)
    # Actually we don't need it anymore
    del evaluator
    gc.collect()
    torch.cuda.empty_cache()
    pipeI2I.to(device)  # Move back to device
    # Generate images and save losses for training data
    opt_embd, _v = load_embd(_N, embed_format=embed_format)

    # Change verbos for inference
    hooker_kwargs['locator_kwargs'].update({'verbose': False})

    rimg, _ = generate_images(
        opt_embedding=opt_embd,
        init_embd=_clone_init,
        word_embd= _embedding,
        init_seed=init_seed,
        strength=strength,
        guidance_scale=guidance_scale,
        img_indices=cur_idx,  # Essentially train
        decoded_full_idx=decoded_full_idx,  # Head
        example_imgs=example_imgs,
        example_masks=example_masks,
        pipeI2I=pipeI2I,
        prompt=prompt,
        return_preds=True,
        device=device,
        use_SDXL=use_SDXL,
        TEXT=TEXT,
        TEXT_INIT=TEXT_INIT,
        _v= _v,
        hooker_kwargs = hooker_kwargs
    )
    rimg.save(f"{save_dir}/vis_train_{_log_name}.png")

    gc.collect()
    torch.cuda.empty_cache()

    # Generate images and save losses for testing data
    rimg_test, _ = generate_images(
        opt_embedding=opt_embd,
        init_embd=_clone_init,
        word_embd= _embedding,
        init_seed=init_seed,
        strength=strength,
        guidance_scale=guidance_scale,
        img_indices=list(range(len(example_imgs_test))),  # Essentially whole test
        decoded_full_idx=decoded_full_idx,  # Head
        example_imgs=example_imgs_test,
        example_masks=example_masks_test,
        pipeI2I=pipeI2I,
        prompt=prompt,
        return_preds=True,
        device=device,
        use_SDXL=use_SDXL,
        TEXT=TEXT,
        TEXT_INIT=TEXT_INIT,
        _v= _v,
        hooker_kwargs=hooker_kwargs
    )
    rimg_test.save(f"{save_dir}/vis_val_{_log_name}.png")

    return



# ============ Reproducibility helpers ============

def set_global_seed(seed: int):
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # (Optional) Make cudnn deterministic for bitwise reproducibility.
    # NOTE: This can slow things down. Set via flag if you want it optional.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


# ============ IO batching → full-split concat (preserve old semantics) ============

def stack_full_split(loader: DataLoader) -> Tuple[torch.Tensor, torch.Tensor]:
    imgs, masks = [], []
    for b_imgs, b_masks in loader:
        imgs.append(b_imgs)
        masks.append(b_masks)
    return torch.cat(imgs, dim=0), torch.cat(masks, dim=0)


# ============ TQDM visibility: patch old MyCallback to be loud on stdout ============

def patch_tqdm_callback():
    """
    The original MyCallback in main_logic creates a tqdm that can be too quiet.
    We replace it at runtime with a subclass that uses stdout + dynamic width + label.
    """
    try:
        import main_logic as ml
        from tqdm import tqdm
        import sys as _sys

        if not hasattr(ml, "MyCallback"):
            return  # older variant? nothing to do

        class _PatchedCallback(ml.MyCallback):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                try:
                    # Close old bar if it exists, then recreate visibly
                    if hasattr(self, "progress_bar") and self.progress_bar is not None:
                        try:
                            self.progress_bar.close()
                        except Exception:
                            pass
                    total_epochs = getattr(self, "total_epochs", None)
                    # Some variants store total in 'progress_bar.total'; fall back if needed
                    if total_epochs is None and hasattr(self, "progress_bar") and self.progress_bar is not None:
                        total_epochs = getattr(self.progress_bar, "total", None)
                    total_epochs = total_epochs or 0
                    self.progress_bar = tqdm(
                        total=total_epochs,
                        desc="OVAM train",
                        position=0,
                        dynamic_ncols=True,
                        leave=True,
                        file=_sys.stdout,
                    )
                except Exception:
                    # don't break training if tqdm patch fails
                    pass

            def __call__(self, epoch, embedding, mask, loss):
                # Preserve original behavior, then add a clearer description + postfix
                out = super().__call__(epoch, embedding, mask, loss)
                try:
                    self.progress_bar.set_description(f"OVAM train | epoch {epoch+1}")
                    # Some variants store last loss in self.loss or self.l2; support both
                    last = None
                    if hasattr(self, "loss") and len(self.loss) > 0:
                        last = float(self.loss[-1])
                        first = float(self.loss[0])
                    elif hasattr(self, "l2") and len(self.l2) > 0:
                        last = float(self.l2[-1])
                        first = float(self.l2[0])
                    if last is not None:
                        self.progress_bar.set_postfix({
                            "l": f"{last:.3g}",
                            "~l": f"{last - first:.3g}"
                        })
                except Exception:
                    pass
                return out

        ml.MyCallback = _PatchedCallback
    except Exception:
        # Keep training even if we cannot patch (safe fallback)
        pass


# ============ CLI ============

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser("Legacy unified (old stack) – HF dataset")

    # data
    p.add_argument("--hf_id", type=str, default="Aleksandar/partedit_parts")
    p.add_argument("--train_split", type=str, default="train")
    p.add_argument("--val_split", type=str, default="val")
    p.add_argument("--class_name", type=str, default=None, 
                   help="Filter dataset by class_name (e.g., 'biped_foot', 'quadruped_head')")
    p.add_argument("--resize", type=int, default=1024)
    p.add_argument("--batch_size_io", type=int, default=64)
    p.add_argument("--num_workers", type=int, default=4)

    # text tokens
    p.add_argument("--TEXT", type=str, required=False, default=None)
    p.add_argument("--TEXT_INIT", type=str, default="head")

    # train hparams (keep names from old runs)
    p.add_argument("--n_epochs", type=int, default=2000)
    p.add_argument("--initial_lr", type=float, default=30.0)
    p.add_argument("--strength", type=float, default=0.25)
    p.add_argument("--guidance_scale", type=float, default=7.5)
    p.add_argument("--gamma", type=float, default=0.7)
    p.add_argument("--step_size", type=int, default=80)
    p.add_argument("--loss_type", type=str, default="l2", choices=["l2", "bce", "bcelog", "cross", "nll"])
    p.add_argument("--optimizer", type=str, default="adam", choices=["adam", "sgd"])

    # infra
    p.add_argument("--save_dir", type=str, required=False, default=None)
    p.add_argument("--use_SDXL", action="store_true")
    p.add_argument("--use_fp16", action="store_true")
    p.add_argument("--log_file", type=str, default="run.log")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--deterministic", action="store_true", help="Enable CuDNN deterministic mode")
    p.add_argument("--config", type=str, default=None)
    return p


def main():
    p = build_parser()

    # Phase 1: partial parse for YAML
    args, _ = p.parse_known_args()
    if args.config is not None:
        with open(args.config, "r") as f:
            cfg = yaml.safe_load(f) or {}
        # Set defaults from YAML; CLI overrides YAML; YAML overrides parser defaults
        for k, v in cfg.items():
            if f"--{k}" not in sys.argv:
                p.set_defaults(**{k: v})

    # Phase 2: final parse
    args = p.parse_args()

    # Validate required arguments
    if args.TEXT is None:
        p.error("--TEXT is required (either via CLI or config file)")
    if args.save_dir is None:
        p.error("--save_dir is required (either via CLI or config file)")

    # Reproducibility
    set_global_seed(args.seed)
    if args.deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    # Prepare logging (old dual logger)
    os.makedirs(args.save_dir, exist_ok=True)
    
    logger = get_logger(
        level=logging.INFO,
        display_to_terminal=True,
        file_name=os.path.join(args.save_dir, args.log_file) if args.log_file else None
    )

    # Make tqdm visible (safe monkey patch)
    patch_tqdm_callback()

    # Device/dtype
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if args.use_fp16 else torch.float32

    # Transforms consistent with SDXL default
    tfm = T.Compose([
        T.Resize((args.resize, args.resize), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor()
    ])

    # HF datasets
    ds_tr = HFImageMaskDataset(args.hf_id, args.train_split, class_name=args.class_name, transform=tfm)
    ds_va = HFImageMaskDataset(args.hf_id, args.val_split, class_name=args.class_name, transform=tfm)

    # IO batching only (concat afterwards) to keep full-dataset gradients
    dl_tr = DataLoader(ds_tr, batch_size=args.batch_size_io, shuffle=False, num_workers=args.num_workers)
    dl_va = DataLoader(ds_va, batch_size=args.batch_size_io, shuffle=False, num_workers=args.num_workers)
    example_imgs,  example_masks   = stack_full_split(dl_tr)
    example_imgs_v, example_masks_v= stack_full_split(dl_va)

    # SDXL pipeline (your original loader)
    logger.info(f"Loading {'SDXL' if args.use_SDXL else 'SD'} model...")
    pipe = load_model(
        device=device, use_sdxl=args.use_SDXL, torch_dtype=dtype, img2img=True, disable_progress_bar=True
    )
    logger.info("Model loaded successfully")
    
    # Log timestep information
    num_inference_steps = 50  # default
    timesteps, actual_steps = pipe.get_timesteps(num_inference_steps, args.strength, device)
    logger.info(f"Diffusion timesteps: using {num_inference_steps} steps (strength={args.strength})")
    logger.info(f"Timestep range: {timesteps[0].item()} -> {timesteps[-1].item()}")
    # logger.info(f"Timesteps: {timesteps}")

    # Single-run config (no N_list / Nc)
    cur_idx = list(range(len(example_imgs)))

    # Log hyperparameters (like the old code did)
    hparams = dict(
        TEXT=args.TEXT, TEXT_INIT=args.TEXT_INIT, strength=args.strength, guidance_scale=args.guidance_scale,
        gamma=args.gamma, step_size=args.step_size, initial_lr=args.initial_lr, n_epochs=args.n_epochs,
        loss_type=args.loss_type, dtype=str(dtype), start_strategy="random",
        dataset_name="hf", N_train=len(example_imgs), N_test=len(example_imgs_v),
        optimizer=args.optimizer, expand_size=args.resize, init_seed=args.seed
    )
    with open(os.path.join(args.save_dir, "hyperparameters.json"), "w") as f:
        json.dump(hparams, f, indent=2)
    logger.info("Hyperparameters:\n" + json.dumps(hparams, indent=2))

    # Call the original trainer (DAAM stack intact)
    result = run_training(
        pipeI2I=pipe,
        example_imgs=example_imgs,
        example_masks=example_masks,
        example_imgs_test=example_imgs_v,
        example_masks_test=example_masks_v,
        device=device,
        cur_idx=cur_idx,
        init_seed=args.seed,
        strength=args.strength,
        guidance_scale=args.guidance_scale,
        TEXT=args.TEXT,
        TEXT_INIT=args.TEXT_INIT,
        init_embed_format="N{}_fold0.pt",     # keep naming layout
        gamma=args.gamma,
        step_size=args.step_size,
        initial_lr=args.initial_lr,
        n_epochs=args.n_epochs,
        loss_type=args.loss_type,
        save_dir=args.save_dir,
        dtype=dtype,
        start_strategy="random",
        dataset_name="hf",
        N_train=len(example_imgs),
        N_test=len(example_imgs_v),
        optimizer=args.optimizer,
        skip_saving=False,
        expand_size=args.resize,
        logger = logger
    )

    # Save minimal metrics (trainer already writes its own artifacts)
    with open(os.path.join(args.save_dir, "metrics_unified.json"), "w") as f:
        json.dump(result if isinstance(result, dict) else {"result": str(result)}, f, indent=2)

    logger.info("Training finished.")


if __name__ == "__main__":
    main()
