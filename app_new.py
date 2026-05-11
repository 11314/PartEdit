#!/usr/bin/env python

import os
import random
from typing import Optional, Tuple, Union, List

import numpy as np
import PIL.Image
import gradio as gr
import torch
import spaces  # 👈 ZeroGPU support

from model import PartEditSDXLModel, PART_TOKENS, PART_NAME_MAP
from datasets import load_dataset

import base64
from io import BytesIO
import tempfile
import uuid

MAX_SEED = np.iinfo(np.int32).max
CACHE_EXAMPLES = os.environ.get("CACHE_EXAMPLES") == "1"
AVAILABLE_TOKENS = list(PART_TOKENS.keys())
AVAILABLE_NAME_MAP = list(PART_NAME_MAP.keys())

# Download examples directly from the huggingface PartEdit-Bench
# Login using e.g. `huggingface-cli login` or `hf login` if needed.
# bench = load_dataset("Aleksandar/PartEdit-Bench", revision="v1.1", split="synth")
bench = load_dataset(
    "parquet",
    data_files={
        "synth": "/hxp/zy/PartEdit/PartEdit-Bench/data/synth-00000-of-00001.parquet",
        "real": "/hxp/zy/PartEdit/PartEdit-Bench/data/real-00000-of-00001.parquet",
    },
    split="synth"
)



def get_example(idx, bench):
    # [prompt_original, subject, part(token_cls), edit, "", 50, 7.5, seed, 50]
    example = bench[idx]
    return [
        example["prompt_original"],
        example["subject"],
        example["part"],   # 现数据集没有 token_cls 字段，换成part字段
        example["edit"],
        "",
        50,
        7.5,
        example["seed"],
        50,
    ]





def run(
    model,
    image:str,
    prompt: str,
    subject: str,
    part: str,
    edit: str,
    negative_prompt: str,
    num_inference_steps: int = 50,
    guidance_scale: float = 7.5,
    seed: int = 0,
    t_e: int = 50,
    n_cross_replace: float = 0.4,
    # progress=gr.Progress(track_tqdm=True),
) -> Tuple[List, Optional[PIL.Image.Image]]:
    if seed == -1:
        seed = random.randint(0, MAX_SEED)
    n_cross_replace = float(n_cross_replace) # to make sure 0 and 1 are float
    print("The parameters of the program are ",prompt, subject, part, edit, negative_prompt, num_inference_steps,guidance_scale, seed, t_e, n_cross_replace)
    out = model.edit(
        image = image,
        prompt=prompt,
        subject=subject,
        part=part,
        edit=edit,
        negative_prompt=negative_prompt,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        seed=seed,
        t_e=t_e,
        n_cross_replace=n_cross_replace
    )

    # Accept either (image, mask) or just image from model.edit
    if isinstance(out, tuple) and len(out) == 2:
        edited, mask_img = out
    else:
        edited, mask_img = out, None

    # download_path = _save_image_for_download(edited)
    return edited, mask_img


import argparse
from PIL import Image

def run_cli():
    parser = argparse.ArgumentParser(description="PartEdit CLI")

    parser.add_argument("--input_path", type = str, default = "", help = "folder to load image")
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--subject", type=str, required=True)
    parser.add_argument("--part", type=str, required=True, choices=AVAILABLE_NAME_MAP)
    parser.add_argument("--edit", type=str, required=True)
    parser.add_argument("--negative_prompt", type=str, default="")
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--t_e", type=int, default=50)
    parser.add_argument("--n_cross_replace", type=float, default=0.4)
    parser.add_argument("--output", type=str, default="output.png")
    # parser.add_argument("--id", type=int, default=0)
    args = parser.parse_args()

    # examples = [get_example(idx, bench) for idx in (use_examples if use_examples is not None else range(len(bench)))]
    # first_ex = examples[args.id] if len(examples) else ["", "", AVAILABLE_TOKENS[0], "", "", 50, 7.5, 0, 50]

    print("Loading model...")
    model = PartEditSDXLModel()

    print("Running edit...")
    # prompt = first_ex[0]
    # subject = first_ex[1]
    # part = first_ex[2]
    # edit = first_ex[3]
    # negative_prompt = first_ex[4]
    # num_inference_steps = int(first_ex[5])
    # guidance_scale = float(first_ex[6])
    # seed = int(first_ex[7])
    # t_e = int(first_ex[8])
    image_path = args.input_path
    prompt = args.prompt
    subject = args.subject
    part = args.part
    edit = args.edit
    negative_prompt = args.negative_prompt
    num_inference_steps = args.num_inference_steps
    guidance_scale = args.guidance_scale
    seed = args.seed
    t_e = args.t_e
    n_cross_replace = args.n_cross_replace
    print(prompt,subject,part,edit,negative_prompt)
    image = Image.open(args.input_path).resize((512, 512)).convert('RGB')
    edited, mask = run(
        model,
        image,
        prompt=prompt,
        subject=subject,
        part=part,
        edit=edit,
        negative_prompt=negative_prompt,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        seed=seed,
        t_e=t_e,
        n_cross_replace=n_cross_replace
    )

    # 保存图像
    img = edited[0] if isinstance(edited, list) else edited

    if isinstance(img, np.ndarray):
        img = Image.fromarray(img)

    count = len(os.listdir(args.output))
    outpath = os.path.join(args.output, f"{count}")
    os.makedirs(outpath, exist_ok=False)
    outpath_img = os.path.join(outpath, "result.jpg")
    img.save(outpath_img)
    print(f"Image saved to {outpath_img}")

    # 保存 mask（可选）
    if mask is not None:
        mask_path = args.output.replace(".png", "_mask.png")
        if isinstance(mask, np.ndarray):
            mask = Image.fromarray(mask)
        
        outpath_mask = os.path.join(outpath, "result_mask.jpg")
        mask.save(outpath_mask)
        print(f"Mask saved to {outpath_mask}")


if __name__ == "__main__":
    run_cli()
