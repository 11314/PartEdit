import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


# from ovam.optimize import optimize_embedding
# We put it here cos I changed the original implementation to fit costumization
# Import the OVAM library
from ovam import StableDiffusionHooker
from ovam.stable_diffusion.locator import SlimeAttentionLocator
from ovam.utils import set_seed, get_device
from ovam.utils.dcrf import densecrf
from ovam.stable_diffusion.daam_module import StableDiffusionDAAM, StableDiffusionXLDAAM
from diffusers import (
    StableDiffusionImg2ImgPipeline,
    StableDiffusionPipeline,
    StableDiffusionXLImg2ImgPipeline,
    StableDiffusionXLPipeline,
    AutoencoderKL
)
import io
from glob import glob
from torchvision.transforms import ToPILImage, ToTensor
from tqdm.auto import tqdm
from typing import Union, Tuple
import gc

try:
    import ipdb
except ModuleNotFoundError as m:
    import pdb as ipdb
from IPython.display import display
from torchvision.utils import make_grid

from typing import TYPE_CHECKING, Callable, Optional, Union

import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F
import os
import torchmetrics
import logging

if TYPE_CHECKING:
    from ovam.base.daam_module import DAAMModule
from ovam.utils.text_encoding import full_encode_sdxl, encode_text

OG_INIT_CHOICES = ["random", "token", "token_random", "average"]
INIT_CHOICES = OG_INIT_CHOICES + [f"{x}_norm" for x in OG_INIT_CHOICES]



def min_max(attn: torch.Tensor, _eps=1e-8) -> torch.Tensor:
    _min, _max = attn.min(), attn.max()
    return (attn - _min) / (_max - _min + _eps)


def display_count_images(batch, nrow=8):
    print(f"{len(batch)=}")
    display(ToPILImage()(make_grid([a["image"] for a in batch], nrow=nrow)))
    display(
        ToPILImage()(
            make_grid(
                [
                    (
                        a["part_instances"]
                        .get_fields()["gt_masks"][:-1]
                        .sum(dim=0)
                        .bool()[None]
                        * 255
                    ).to(torch.uint8)
                    for a in batch
                ],
                nrow=8,
            )
        )
    )


# 略微改变了ovam的实现。优化导入optimize_embedding
def optimize_embedding(
    daam_module: "DAAMModule",
    embedding: "torch.Tensor",
    target: "torch.Tensor",
    device: Optional[str] = None,
    callback: Optional[Callable] = None,
    initial_lr: float = 300,
    epochs: int = 1000,
    step_size: int = 80,
    gamma: float = 0.7,
    apply_min_max: Union[bool, int] = 3720,
    squeezed_target: bool = False,
    loss_type: str = "cross",
    autocast_enabled: bool = False,
    optimizer: str = "adam",
) -> "torch.Tensor":
    """Basic optimization function for the embedding.

    Arguments
    ---------
    daam_module : DAAMModule
        The DAAM module used to evaluate the embedding.
    embedding : torch.Tensor
        The embedding to optimize.
    target : torch.Tensor
        The target to optimize the embedding.
    device : str, optional
        The device to use for the optimization, by default uses
        the device of the embedding.
    callback : Callable, optional
        A callback function to call at each epoch, by default None.
        Is called with the following arguments:
            - epoch: int
            - embedding: torch.Tensor
            - mask: torch.Tensor
            - loss: torch.Tensor
    initial_lr : float, optional
        The initial learning rate, by default 3.
    epochs : int, optional
        The number of epochs, by default 100.
    step_size : int, optional
        The step size for the scheduler, by default 80.
    gamma : float, optional
        The gamma for the scheduler, by default 0.7.

    Returns
    -------
    torch.Tensor
        The optimized embedding.

    Notes
    -----
    To obtain the losses during optimization use the callback function.

    """
    # 略微改变了ovam的实现。优化导入optimize_embedding
    # 保证 embedding / target / model 在同一设备
    device = embedding.device if device is None else device

    # 将嵌入克隆为可训练张量/冻结模型，只让 embedding 可学习
    x = embedding.detach().clone().requires_grad_(True)
    x.retain_grad() # 保留梯度
    x.to(device)
    daam_module.to(device)
    # 将target移动到设备上
    target.to(device)

    # 定义优化器、调度程序和损失函数

    if optimizer == "adam":
        optimizer = optim.AdamW([x], lr=initial_lr) # 只传x，彻底冻结模型参数
    else:
        optimizer = optim.SGD([x], lr=initial_lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)  # 学习率调度器，防止 embedding 后期震荡，让 attention mask 稳定收敛
    if loss_type == "bce":  # 根据传入的loss类型，选取
        loss_fn = nn.BCELoss(reduction="mean")
    elif loss_type == "l2":
        loss_fn = nn.MSELoss(reduction="mean")
    elif loss_type == "bcelog":
        loss_fn = nn.BCEWithLogitsLoss(reduction="mean")
    elif loss_type == "nll":
        loss_fn = nn.NLLLoss(reduction="mean")
        target = torch.argmax(target, dim=1)
        # squeezed_target = True
        # loss_fn = nn.NLLLoss2d(reduction="mean")
    elif loss_type == "cross":
        loss_fn = nn.CrossEntropyLoss(reduction="mean")
    else:
        raise ValueError(f"Loss type {loss_type} not supported.")
    _x_half = None
    
    # 只有在没有提供回调时才使用tqdm（回调有自己的进度条）
    iterator = range(epochs) if callback is not None else tqdm(range(epochs), desc="Optimizing embedding", position=0, dynamic_ncols=True, leave=True, file=sys.stdout)
    
    for i in iterator:
        optimizer.zero_grad()   # optimizer.zero_grad()
        with torch.autocast(device_type="cuda", enabled=autocast_enabled):
            # import ipdb; ipdb.set_trace()
            mask = daam_module.forward(x)   # 用 embedding 生成 attention mask
            # 应用最小最大归一化
            if loss_type != "bcelog":
                if isinstance(apply_min_max, float):
                    mask = mask / apply_min_max
                elif apply_min_max:  # 对于线性情况
                    minimun, maximun = mask.min(), mask.max()
                    mask = (mask - minimun) / (maximun - minimun + 1e-8)    # 归一化attention
                else:
                    mask = mask / mask.sum(dim=1, keepdim=True)
            else:
                target = target * mask.max().item()

            if squeezed_target:
                mask = mask.squeeze()
                if mask.ndim == 2:
                    mask = mask.unsqueeze(0)
            if loss_type in ["nll", "cross"]:
                mask = torch.log(mask + 1e-8)   # 让 token attention 在 GT 区域概率最大
            try:
               # print(f'{mask.shape=} {target.shape=}, {mask.dtype=} {target.dtype=}, {mask.max()=}, {mask.min()=}, {target.max()=}, {target.min()=}')
                loss = loss_fn(mask, target)    # 计算 loss
            except Exception as e:
                ipdb.set_trace()
                raise e

        if callback is not None:
            callback(i, x, mask, loss)

        loss.backward() # 反向传播
        optimizer.step()    # 更新embedding
        scheduler.step()
        if epochs //2 == i:
            _x_half = x.clone().detach()    # _x_half的作用是对比早起embedding、中期embedding和后期embedding
    return x.detach().cpu(), _x_half


def fig_to_image(fig, add_tight=True):
    """Convert a Matplotlib figure to a PIL Image and return it"""
    buf = io.BytesIO()
    _extra_kwargs = {"bbox_inches": "tight", "pad_inches": 0.0} if add_tight else {}
    fig.savefig(buf, format="png", **_extra_kwargs)
    buf.seek(0)
    return Image.open(buf)


def stack_images(images):
    widths, heights = zip(*(i.size for i in images))

    total_height = sum(heights)
    max_width = max(widths)

    new_im = Image.new("RGB", (max_width, total_height))

    y_offset = 0
    for im in images:
        new_im.paste(im, (0, y_offset))
        y_offset += im.height

    return new_im

def stack_horizontally(images):
    widths, heights = zip(*(i.size for i in images))

    total_width = sum(widths)
    max_height = max(heights)

    new_im = Image.new("RGB", (total_width, max_height))

    x_offset = 0
    for im in images:
        new_im.paste(im, (x_offset, 0))
        x_offset += im.width

    return new_im

def encode_decode(ovam_evaluator, text, add_special: bool = True) -> list[str]:
    text_encoded = ovam_evaluator.tokenizer.encode(text, add_special_tokens=add_special)    # 编码文本为 token IDs（模型真实使用的）
    decoded_str = [ovam_evaluator.tokenizer.decode(k) for k in text_encoded]    # 把 token ID 解码回字符串（用于对齐）
    # replace <|startoftext|> and <|endoftext|> with <SoT> and <EoT>/特殊 token 的人类可读替换（非训练必需，但很重要）
    decoded_str = [
        k.replace("<|startoftext|>", "<SoT>").replace("<|endoftext|>", "<EoT>")
        for k in decoded_str
    ]
    return decoded_str


def visualize_one(
    og_img,
    gen_img,
    vis_prompt,
    attention_maps,
    ovam_evaluator,
    save_fig_file: str = None,
    skip_special: bool = False,
    norm: bool = True,
    set_titles: bool = True,
    save_individual: bool = False
):
    if isinstance(vis_prompt, str):
        # This is normal prompt we need to encode,
        decoded_str = encode_decode(ovam_evaluator, vis_prompt)
    elif isinstance(vis_prompt, torch.Tensor):
        # This is embedding, we will say its V1, V2 .... len(vis_prompt)
        decoded_str = [f"V{k}" for k in range(len(vis_prompt))]
    assert len(decoded_str) == len(
        attention_maps
    ), f"{len(decoded_str)=} != {len(attention_maps)=}"
    no_og = True if og_img is None else False
    # offset = 2 - int(no_og) if not skip_special else 0
    _begging_offset = 2 - int(no_og)
    offset = 2 if skip_special else 0
    fig, axes = plt.subplots(
        1,
        len(decoded_str) - offset + _begging_offset,
        figsize=(20, 5),
        gridspec_kw={
            "wspace": 0.0,
            "hspace": 0.0,
            "left": 0,
            "right": 1,
            "top": 0.975 if set_titles else 1,
            "bottom": 0,
        },
    )
    for ax in axes:
        ax.axis("off")
    if no_og:
        axes[0].imshow(gen_img)
        axes[0].set_title("Generated image")
    else:
        axes[0].imshow(og_img.astype(np.float32))
        axes[0].set_title("Original image")
        axes[1].imshow(gen_img)
        axes[1].set_title("Generated image")
    j = _begging_offset
    for i in range(len(decoded_str)):
        if skip_special and (i == 0 or i >= len(decoded_str) - 1):
            continue
        attn = attention_maps[i]
        if norm:
            _min, _max = attn.min(), attn.max()
            attn = (attn - _min) / (_max - _min)
        attn = attn.astype(np.float32)
        axes[j].imshow(gen_img)
        axes[j].imshow(attn, alpha=attn, cmap="jet")
        axes[j].set_title(f"Attn `{str(decoded_str[i])}`")
        
        # Save individual attention maps if requested
        if save_individual and save_fig_file:
            # Create individual figure for this attention map
            ind_fig, ind_ax = plt.subplots(figsize=(10, 10))
            ind_ax.axis("off")
            ind_ax.imshow(gen_img)
            ind_ax.imshow(attn, alpha=attn, cmap="jet")
            
            # Generate filename for individual map
            base_dir = os.path.dirname(save_fig_file)
            base_name = os.path.splitext(os.path.basename(save_fig_file))[0]
            word_filename = f"{base_dir}/{base_name}_{decoded_str[i].replace(' ', '_')}.png"
            
            # Save and close individual figure
            ind_fig.savefig(word_filename, bbox_inches='tight', pad_inches=0)
            plt.close(ind_fig)
            
        j += 1
    # plt.tight_layout()
    img = fig_to_image(fig, add_tight=True)
    plt.close(fig)
    if save_fig_file:
        img.save(save_fig_file)
    return img


def maybe_bitmask(tensor):
    if isinstance(tensor, torch.Tensor):
        return tensor
    else:  # BitMask
        return tensor.tensor

def sum_zero_or_zero(tensor, cur_idx_of_gt_class):
    if cur_idx_of_gt_class is None or cur_idx_of_gt_class.shape[0] == 0:
        return torch.zeros(tensor.shape[-2:]).bool()
    else:
        return tensor[cur_idx_of_gt_class].sum(dim=0).bool()

def add_bg_to_mask(masks:torch.Tensor) -> torch.Tensor:
    # We add in first channel the background, which is the inverse of the masks 
    assert masks.ndim == 4, "Expected masks to have 4 dimensions"
    bg = (1 - masks.sum(dim=1, keepdim=True)).to(device=masks.device, dtype=masks.dtype)
    return torch.cat([bg, masks], dim=1)

def get_correct_masks_and_images(
    idx_or_subidx: int,
    batch_composer,
    n: int = -1,
    return_type="pt",
    key_access: str = "part_instances",
    field_access: str = "gt_classes",
    final_mask_access: str = "gt_masks",
    return_dtype=torch.float32,
):
    if return_type == "pt":
        # we return stacked images and stacked selected_idx_or_subidx of part_instances
        images = torch.stack([a["image"] for a in batch_composer], dim=0)
        all_masks = []
        for idx in idx_or_subidx:
            idx_of_gt_classes = [
                torch.where(a[key_access].get_fields()[field_access] == idx)[0]
                for a in batch_composer
            ]

            # now we just want those gt_masks

            masks = [
                sum_zero_or_zero(
                    maybe_bitmask(a[key_access].get_fields()[final_mask_access]),
                    idx_of_gt_classes[cur_idx]
                )
                for cur_idx, a in enumerate(batch_composer)
                #if len(idx_of_gt_classes[cur_idx]) > 0
            ]  # bool
            masks = torch.stack(
                masks,
                dim=0,
            )
            all_masks.append(masks)

        masks = torch.stack(all_masks, dim=1) # B N H W, for one idx its B 1 N W
        #if masks.ndim == 3:
        #    masks = masks.unsqueeze(1)

        images = images.to(return_dtype) / 255.0
        masks = masks.to(return_dtype)  # bool => float 0..1

        # extra check that per each masks batch, we have at least one non zero N
        # we select indexes of those that have at least one non zero mask
        valid_idx = masks.sum(dim=[1, 2, 3]) > 0
        print(f"Valid idx {valid_idx.sum()} out of {valid_idx.shape[0]}")
        images = images[valid_idx]
        masks = masks[valid_idx]
        assert images.shape[0] == masks.shape[0], f"{images.shape=} {masks.shape=}"
        assert images.ndim == 4, f"{images.shape=}"
        assert masks.ndim == 4, f"{masks.shape=}"

    elif return_type == "PIL":
        raise NotImplementedError("Not implemented yet")
    elif return_type == "np":
        raise NotImplementedError("Not implemented yet")
    else:
        raise ValueError("return_type must be either 'pt' or 'np' or PIL")

    if n > 0:
        return images[:n], masks[:n]
    return images, masks


def visualize_prepared(imgs):
    assert len(imgs.shape) == 4, f"Provide in batch format"

    return ToPILImage()(make_grid(imgs.to(torch.float32)))

# 把数据集里的 GT mask，转换成一个“可用于 token–attention 监督的二分类目标”
def prepare_masks(
    example_masks: torch.Tensor, cur_idx: list[int]
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Select masks from the given example masks using the list current index and prepare a negative.

    Parameters:
    example_masks (torch.Tensor): Tensor of example masks.
    cur_idx (list[int]): Current index.

    Returns:
    tuple: A tuple containing the following:
        - _selected_masks (torch.Tensor): The raw selected masks.
        - selected_masks (torch.Tensor): The processed selected masks.
    """
    cur_idx = [cur_idx] if isinstance(cur_idx, int) else cur_idx    # 统一索引格式（保证 batch 维度）
    _selected_masks = example_masks[cur_idx, ...]   # 从完整数据集中取出当前训练样本的 mask
    if _selected_masks.ndim == 3:   # 处理“单样本维度坍塌”的特殊情况
        assert isinstance(
            cur_idx, int
        ), f"Dimensionality is 3, but {cur_idx=} is not int ({type(cur_idx)})"
        _selected_masks = _selected_masks[None]
    # 为选出的mask制作底片
    selected_masks = torch.cat([1 - _selected_masks, _selected_masks], dim=1)
    assert selected_masks.ndim == 4, f"{selected_masks.shape=} should be B C H W, C=2"  # 维度断言（确保训练安全）
    return _selected_masks, selected_masks

# 把文本 TEXT 转换成“可训练 token embedding 的原始起点”
def process_text(
    ovam_evaluator: Union["StableDiffusionDAAM", "StableDiffusionXLPipeline"],
    TEXT: str,
):
    """
    Process the given text using the OVAM evaluator, and return the embeddings and masks.

    Parameters:
    ovam_evaluator (object): OVAM evaluator object used to encode the text.
    TEXT (str): The text to be processed.

    Returns:
    tuple: A tuple containing the following:
        - _embedding (torch.Tensor): The raw embedding of the text.
        - decoded_str (str): The decoded string after encoding and decoding the text.
    """
    # 强制 Text Encoder 使用 FP32（数值稳定性）为 token-level optimization 特别做的稳定性处理
    if ovam_evaluator.text_encoder.dtype == torch.float16:
        ovam_evaluator.text_encoder = ovam_evaluator.text_encoder.to(torch.float32)
        if hasattr(ovam_evaluator, "text_encoder_2"):
            ovam_evaluator.text_encoder_2 = ovam_evaluator.text_encoder_2.to(
                torch.float32
            )
    if "XL" in ovam_evaluator.__class__.__name__:            
        # 删除 OVAM 获取文本的依赖
        _embedding = full_encode_sdxl(
            ovam_evaluator,
            text=TEXT,
            context_sentence=None,
            remove_special_tokens=False,
            padding=False,
        )[
            :-1
        ]  # 跳过被池化的 token，因为没有空间对应，无法对齐 mask
    else:
        _embedding = encode_text(
            ovam_evaluator.tokenizer,
            ovam_evaluator.text_encoder,
            text=TEXT,
            context_sentence=None,
            remove_special_tokens=False,
            padding=False,
        )
        print(f"Embedding shape {_embedding.shape}")
        _embedding = _embedding[:-1] # 从OVAM_TRAIN的ipynb复制
    # 编码，解码文本（token 对齐用）
    decoded_str = encode_decode(ovam_evaluator, TEXT)
    return _embedding, decoded_str



def extract_opt_masks(embd, ovam_evaluator, device):
    ovam_evaluator.to(device)
    with torch.no_grad():
        mask = ovam_evaluator(embd.to(device))
        if isinstance(mask, torch.Tensor):
            mask = mask.squeeze().cpu()
    return mask    


def plot_attention_maps(
    ovam_evaluator,
    _embedding, 
    embedding,
    opt_embedding,
    image,
    TEXT:str = "",
    TEXT_INIT:str = "",
    device="cpu",
    _selected_masks=None,
    set_titles: bool = True,
    decoded_full_idx: int = 3,
    _opt_text=None,
    _first_text=None,
    return_preds: bool = False,
    plot_binary: bool = True,
    use_headmap:bool = False,
):
    non_optimized_map = extract_opt_masks(_embedding, ovam_evaluator, device)[[0, decoded_full_idx]] 
    non_optimized_avg = extract_opt_masks(embedding, ovam_evaluator, device)[[0, 1]]
    optimized_map = extract_opt_masks(opt_embedding, ovam_evaluator, device)[[0, 1]]
    # Plot result using
    fig, axs = plt.subplots(
        1,
        4 + int(plot_binary) + int(_selected_masks != None),
        figsize=(14, 4),
        # constrained_layout=constrained,
        gridspec_kw={
            "wspace": 0.0,
            "hspace": 0.0,
            "left": 0,
            "right": 1,
            "top": 0.975 if set_titles else 1,
            "bottom": 0,
        },
    )

    axs[0].imshow(image)
    for ax in axs:
        ax.axis("off")

    if set_titles:
        axs[0].set_title("Synthetized image" if _first_text is None else _first_text)

    non_opt_map = min_max(non_optimized_map[-1]).numpy()
    non_opt_avg = min_max(non_optimized_avg[-1]).numpy()
    opt_map = min_max(optimized_map[-1]).numpy()
    axs[1].imshow(non_opt_map)
    axs[2].imshow(non_opt_avg)
    axs[3].imshow(opt_map)
    if set_titles:
        axs[1].set_title(f"Attn word {TEXT_INIT}")
        axs[2].set_title(f"Init token")
        axs[3].set_title(f"Opt {TEXT if _opt_text is None else _opt_text}")
    if _selected_masks is not None:
        axs[-2].imshow(ToPILImage()(_selected_masks[0]))
        if set_titles:
            axs[-2].set_title("Target mask")
    # import ipdb; ipdb.set_trace()
    if plot_binary:
        axs[-1].imshow((opt_map > 0.5).astype(np.float32))
        if set_titles:
            axs[-1].set_title("Binary pred")

    image = fig_to_image(fig, add_tight=True)
    plt.close(fig)
    if return_preds:
        return image, torch.stack(
            # [optimized_map]
            [non_optimized_map, optimized_map, non_optimized_avg]
        )  # will be 3 2 H W
    return image


def combine_pil_vertically(pil1, pil2):
    # combine them to a new image that is stacked pil1 then pil2
    new_im = Image.new("RGB", (pil1.width, pil1.height + pil2.height))
    new_im.paste(pil1, (0, 0))
    new_im.paste(pil2, (0, pil1.height))
    return new_im


# 它负责把“已经训练完成的 token embedding”以一个可复现实验的方式保存到磁盘，并返回保存状态。
def save_opt_embedding(
    opt_embedding,
    trained_new,
    file_save="ovam_tokens/test_{}.pt",
    direct_embd_name: int = None,
):
    if direct_embd_name is not None:
        embd_name = direct_embd_name    # 如果没有指定embedding名字，就使用指定编号生成文件名
        f_l = file_save.format(embd_name)
    else:
        potential_files = glob(file_save.format("*"))   # 否则：自动推断 embedding 编号
        offset = 0 if trained_new else -1   # 根据训练状态计算偏移量
        embd_name = len(potential_files) + offset   # 生成embedding名字
        f_l = file_save.format(embd_name)
    if trained_new: # 如果是新训练，真正保存
        if os.path.exists(f_l): # 如果文件已存在，记录日志
            # if "debug" not in f_l:
            #     raise ValueError(f"File {f_l} already exists")
            # else:
            logging.info(f"File {f_l} already exists - overriding")
        torch.save(opt_embedding, f_l)  # 保存embedding
        trained_new = False # 修改训练状态，防止多次调用
    return trained_new, f_l, embd_name

import sys
class MyCallback:
    def __init__(self, initial_embedding, device, total_epochs):
        self.embedding = initial_embedding.to(device)
        self.l2 = []
        # self.cosine = []
        self.loss = []
        self.progress_bar = tqdm(total=total_epochs, desc="OVAM Train", position=0,
                                 dynamic_ncols=True,
                                 leave=True,
                                 file=sys.stdout)

    def __call__(self, epoch, embedding, mask, loss):
        # e_clone = embedding.clone().detach()
        # with torch.no_grad():
        # we want only L2 of second token
        # self.l2.append(torch.norm(e_clone[1, ...] - self.embedding[1, ...], p=2).cpu().numpy())
        # self.cosine.append(
        #     F.cosine_similarity(e_clone, self.embedding).cpu().numpy()
        # )
        # print(f'{loss=}')
        self.loss.append(loss.item())
        self.progress_bar.update(1)
        # write in tqdm list version of l2, cosine and loss value
        self.progress_bar.set_postfix(
            {
                "l": f"{self.loss[-1] :.3g}",
                "~l": f"{self.loss[-1] - self.loss[0]:.3g}",
                # "c0": f"{self.cosine[-1][0]:.3f}",
                # "c1": f"{self.cosine[-1][1]:.3f}",
            }
        )

    def close(self):
        self.progress_bar.close()

    def plot_metrics(self, loss_type):
        fig, ax = plt.subplots(1, 1, figsize=(15, 5))
        ax.plot(self.loss)
        ax.set_title(f"Loss mask ({loss_type})")
        # ax[0].plot(self.loss)
        # ax[0].set_title(f"Loss mask ({loss_type})")
        # ax[1].plot(self.l2)
        # ax[1].set_title("L2 from original embed")
        # ax[1].plot(self.cosine)
        # ax[1].set_title("Cosine original embed")
        # ax[1].legend(["<SoT>", "V1"])  # Add legend
        plt.tight_layout()
        img = fig_to_image(fig)
        plt.close(fig)
        return img

# 把“你要训练的样本索引”转换成“真正送入扩散模型的 img2img 输入张量”，并确保索引与 token–mask–attention 在整个训练中保持一
def prepare_idx(cur_idx, tensor_imgs, device) -> tuple[torch.Tensor, list[int]]:
    cur_idx = [cur_idx] if isinstance(cur_idx, int) else cur_idx    # 统一索引为 list（训练一致性）
    # 严格断言索引合法（防止 silent bug）
    assert isinstance(cur_idx, list), f"Got {cur_idx=} should be list[int]"
    assert len(cur_idx) >= 1, f"Got empty list {cur_idx=}"
    assert isinstance(
        cur_idx[0], int
    ), f"list elements should be integers got {cur_idx=}"
    # 从数据集中取出对应图像（img2img 的关键）
    if tensor_imgs is not None and len(tensor_imgs) > 0:
        # 这就是Img2Img
        selected_tensor = tensor_imgs[cur_idx, ...].to(device)
    else:
        selected_tensor = None

    return selected_tensor, cur_idx

# 在“不训练任何参数”的前提下，用真实图像跑一次扩散模型，并把 UNet 的 cross-attention 全部 hook 起来，
# 生成一个“可用于后续 token embedding 训练的评估器（evaluator）”。
def initial_forwardpass(
    pipeI2I,
    example_imgs: torch.Tensor,
    cur_idx: list[int],
    strength: float,
    guidance_scale: float,
    seed: int = 0,
    prompt: str = "",
    extract_self_attentions: bool = False,
    expand_size: tuple[int, int] = (512, 512),
    hooker_kwargs: dict = {},
    ovam_callable_kwargs: dict = {},
    no_grad_context: bool = False,
    return_hooker: bool = False,
    verbose: bool = False
):
    """
    Performs an initial forward pass through a given pipeline.

    Parameters:
    pipeI2I (object): The pipeline through which the images will be passed.
    example_imgs (torch.Tensor): A tensor of example images.
    cur_idx (list): The current index or indices of the images to be processed.
    strength (float): Parameter for the pipeline.
    guidance_scale (float): Parameter for the pipeline.
    seed (int, optional): A seed for random number generation. Defaults to 0.
    prompt (str, optional): A string prompt. Defaults to "".
    extract_self_attentions (bool, optional): Whether to extract self-attentions. Defaults to False.
    expand_size (tuple, optional): The size to which the images will be expanded. Defaults to (512, 512).
    Returns:
    callable: A callable from hooker.get_ovam_callable with expand_size set to (512, 512).
    list: A list of images from the output of the pipeline.


    Example:
    cur_idx = [0, 1, 2, 3]
    strength, guidance_scale = 0.25, 7.5
    ovam_evaluator, img_pil_list = initial_forwardpass(
        pipeI2I,
        example_imgs,
        cur_idx,
        strength,
        guidance_scale,
        seed=0,
        prompt=""
    )
    """
    hooker_kwargs = {} if hooker_kwargs is None else hooker_kwargs

    # 判断当前是不是 img2img 条件分支（训练前提检查）
    _cond = "Img2Img" in pipeI2I.__class__.__name__ or hasattr(
        pipeI2I, "text_encoder_2"
    )
    if prompt == "": # 强制校验训练模式一致性，prompt为空-使用image条件。prompt非空-用txt2img
        assert example_imgs is not None and isinstance(
            example_imgs, torch.Tensor
        ), f"{type(example_imgs)=}"
        assert _cond, f"wrong pipe passed expected Img2Img got {type(pipeI2I)=}"
    else:
        assert not _cond, f"{type(pipeI2I)=} should be normal SDP"
    selected_init_tensor = example_imgs
    img_pil = []
    # print(f"INFO: Initial forward pass with {len(cur_idx)} images")
    with torch.set_grad_enabled(not no_grad_context):   # 进入“是否计算梯度”的控制区,目的不是训练，而是 “搭建训练环境”
        with StableDiffusionHooker( # hook Unet的Attention层，记录token到spatial attention，为后续训练提供attention extraction接口
            pipeI2I, extract_self_attentions=extract_self_attentions, **hooker_kwargs
        ) as hooker:
            # 对每一张训练图像跑一次扩散 forward
            for img_idx, img_seed in enumerate(cur_idx):
                set_seed(1 + seed) if _cond else set_seed(seed + img_seed)  # 控制随机性（保证可复现）
                _kwargs = ( # img2img 条件扩散 forward（关键）/将image->latent，latent+prompt ->Unet,在Unet内部产生Cross-Attention，使用hook 把attention抓下来。
                    {"image": selected_init_tensor[img_idx][None], "strength": strength}
                    if _cond
                    else {}
                )
                out = pipeI2I(prompt=prompt, **_kwargs, guidance_scale=guidance_scale)
                img_pil.append(out.images[0])   # 收集可视化图像（辅助，不影响训练）
    return (    # 返回“训练用 evaluator”（极其重要）
        hooker.get_ovam_callable(expand_size=expand_size, **ovam_callable_kwargs) if not return_hooker else hooker,
        img_pil,
    )


def get_init_embedding(
    start_strategy: str, pipeI2I, _embedding, decoded_full_idx, device
) -> Tuple[torch.Tensor, torch.Tensor]:
    # 解析初始化策略（是否归一化）
    do_norm = start_strategy.endswith("_norm")
    start_strategy = start_strategy.replace("_norm", "")
    # 不是从零开始学token，是从原有的语义上微调
    if start_strategy == "token":
        init_embedding = torch.cat(
            [_embedding[:1], _embedding[decoded_full_idx][None]], dim=0
        )
    elif start_strategy == "token_random": # 如果是从完全随机语义开始的话，Attention是完全无结构的，mask supervision很难收敛。所以直接禁用这个策略
        init_embedding = pipeI2I.text_encoder.get_input_embeddings().weight.data.clone()
        raise NotImplementedError("This is not implemented yet")
    elif start_strategy == "average":   # 被训练的token初始化为prompt中所有token的平均语义。
        init_embedding = torch.cat(
            [_embedding[:1], _embedding[1:].mean(dim=0, keepdim=True)], dim=0
        )
    else:  # random。在训练中意味着token没有任何语义先验，Attention初始是噪声。
        init_embedding = torch.randn_like(_embedding[:2]).to(device)
    if do_norm: # 可选归一化（对 attention 稳定性很重要）
        init_embedding = nn.functional.normalize(init_embedding, p=2, dim=-1)
    _clone_init = init_embedding.clone().detach().to(device)    # 保存一份“不可训练的原始副本”
    return init_embedding, _clone_init


def train_embedding(
    init_embedding: torch.Tensor,
    device: Union[str, torch.device],
    double_target: torch.Tensor,
    evaluator: "StableDiffusionDAAM",
    n_epochs: int = 1000,
    _initial_lr: int = 3000,
    _step_size: int = 80,
    _gamma: float = 0.7,
    loss_type="nll",
    apply_min_max: Union[bool, float] = True,
    cast=False,
    disable_logging: bool = False,
    optimizer: str = "adam",
) -> tuple[bool, torch.Tensor, MyCallback]:
    """
    trained_new, opt_embedding, my_callback = train_embedding(
        ...
    )
    Trains an embedding using the provided parameters and returns the training status, optimized embedding, and callback.

    Args:
        init_embedding (torch.Tensor): Initial embedding tensor to be optimized.
        device (Union[str, torch.device]): Device to perform computations on. Can be a string (e.g., 'cpu', 'cuda') or a torch.device object.
        double_target (torch.Tensor): Target tensor for the optimization.
        ovam_evaluator (StableDiffusionDAAM): Evaluator object to compute the loss for optimization.
        n_epochs (int, optional): Number of epochs for the optimization. Defaults to 1000.
        _initial_lr (int, optional): Initial learning rate for the optimization. Defaults to 3000.
        _step_size (int, optional): Step size for the learning rate scheduler. Defaults to 80.
        _gamma (float, optional): Multiplicative factor of learning rate decay. Defaults to 0.7.
        loss_type (str, optional): Type of loss to use for optimization. Defaults to "nll".
        apply_min_max (Union[bool, float]): Whether to apply min-max normalization to the mask. 3740 is their default.
        If a float is provided, it is used as the denominator for normalization. Defaults to True.
        cast (bool, optional): If True, enables autocasting for mixed precision. Currently disabled and should be False.
        disable_logging (bool, optional): If True, disables logging. Defaults to False.
    Raises:
        e: Any exception that occurs during the optimization process.

    Returns:
        tuple[bool, torch.Tensor, MyCallback]: A tuple containing a boolean indicating whether the training was successful, the optimized embedding tensor, and the callback object used during training.
    """
    # assert cast == False, f"Disabled cast as it never works"
    trained_new = True  # 用于在异常时返回失败状态
    gc.collect()    # 清理 Python 和 CUDA 显存。防止显存碎片影响结果
    torch.cuda.empty_cache()
    set_seed(0)  # 增加了检查多图像训练时的再现性/固定 seed → 多图 / 多次实验可复现
    double_target = double_target.to(device)    # 这是监督信号（GT mask + negative mask）
    try:
        my_callback = ( # 记录每一个epoch中的当前embedding、attention mask和loss
            MyCallback(init_embedding, device, n_epochs)
            if not disable_logging
            else None
        )
        opt_embedding, _half = optimize_embedding(
            evaluator.to(device),
            embedding=init_embedding,
            target=double_target,
            device=device,
            callback=my_callback,
            initial_lr=_initial_lr,
            apply_min_max=apply_min_max,
            epochs=n_epochs,
            step_size=_step_size,
            gamma=_gamma,
            loss_type=loss_type,
            autocast_enabled=cast,
            optimizer=optimizer,
        )
    except Exception as e:  # 异常处理，如果训练失败-标记失败、抛出异常。
        trained_new = False
        raise e
    finally:
        my_callback.close() # 训练完毕，关闭Callback
    return trained_new, opt_embedding, _half, my_callback


def load_model(
    device: torch.device,
    use_sdxl: bool = True,
    torch_dtype=torch.float32,
    img2img: bool = True,
    disable_progress_bar: bool = True,
    use_sd15: bool = False,
    extra_kwargs:dict = {}
) -> Union[
    StableDiffusionXLPipeline,
    StableDiffusionXLImg2ImgPipeline,
    StableDiffusionPipeline,
    StableDiffusionImg2ImgPipeline,
]:
    str_non_sdxl = "runwayml/stable-diffusion-v1-5" if use_sd15 else "stabilityai/stable-diffusion-2-1-base"
    model_id = (
        "stabilityai/stable-diffusion-xl-base-1.0"
        if use_sdxl
        else str_non_sdxl
    )
    
    if use_sdxl:
        _cls = (
            StableDiffusionXLPipeline
            if not img2img
            else StableDiffusionXLImg2ImgPipeline
        )
    else:
        _cls = (
            StableDiffusionPipeline if not img2img else StableDiffusionImg2ImgPipeline
        )
    vae_id = model_id if not use_sdxl else "madebyollin/sdxl-vae-fp16-fix"
    _extra = {**extra_kwargs} if use_sdxl else {"subfolder": "vae", **extra_kwargs}
    
    # Suppress verbose model loading output
    vae = AutoencoderKL.from_pretrained(vae_id, torch_dtype=torch_dtype, **_extra)
    pipe = _cls.from_pretrained(
        model_id, vae=vae, torch_dtype=torch_dtype, safety_checker=None
    ).to(device)
    
    if disable_progress_bar:
        pipe.set_progress_bar_config(disable=True)
    return pipe

# 把“已经训练好的 token embedding”，重新加载出来，用于生成图像、验证训练效果、评估泛化性
def load_embd(embd_name, embed_format):
    _v = "optimized"    # 默认版本标识
    if isinstance(embd_name, (str, int)):   # 判断输入类型：名字还是 tensor？
        embd_loc = embed_format.format(embd_name)  # 如果是已生成的文件，生成 embedding 文件路径,用于下面进行确认
        if not os.path.exists(embd_loc):    # 确保训练产物存在
            raise ValueError(f"File {embd_loc} does not exist")
        _v = embd_loc.split("/")[-1]    # 用文件名作为版本标识
        opt_embedding = torch.load(embd_loc)    # 加载 embedding
    else:
        opt_embedding = embd_name
        assert torch.is_tensor(opt_embedding), f"{type(opt_embedding)=}"
    return opt_embedding, _v

# 在训练完成后，用“刚学到的 token embedding”，再跑一遍扩散过程，验证：embedding 是否真的学会了“目标区域语义对齐”
def generate_images(
    opt_embedding: torch.Tensor,
    init_embd: torch.Tensor,
    word_embd: torch.Tensor, # 77, D
    device,
    TEXT: str,
    TEXT_INIT:str,
    init_seed: int,
    strength: float,
    guidance_scale: float,
    img_indices: list[int],
    decoded_full_idx: int,
    example_imgs: torch.Tensor,
    example_masks: torch.Tensor,
    pipeI2I: Union[StableDiffusionImg2ImgPipeline, StableDiffusionXLImg2ImgPipeline],
    prompt: str,
    #embed_format: Optional[str] = None,
    #init_embd_int_strategy: str = "random",  # strategy to use when init_embd is None
    _v: str = "optimized",
    return_preds: bool = False,
    use_SDXL: bool = False,
    ovam_callable_kwargs: dict = {},
    hooker_kwargs = None
) -> tuple[Image.Image, list[torch.Tensor]]:
    # 我们只需要example_mask用于可视化，占用CPU
    if example_masks is not None:   # 清理 GPU / 数据迁移到 CPU
        example_masks.cpu()
    example_imgs = example_imgs.cpu()
    gc.collect()
    torch.cuda.empty_cache()
    img_indices = [img_indices] if isinstance(img_indices, int) else img_indices    # 统一索引格式

    assert isinstance(
        prompt, str
    ), f"Even though we can pass list or str, we use same prompt {prompt=}"
    
    assert hooker_kwargs is not None, "Provide hooker_kwargs in generate_images()"  # 校验推理环境，如果没有attention hook，这个函数就无法验证token embedding是否控制了空间区域。

    img_size = 1024 if use_SDXL else 512    # 根据模型版本确定分辨率
    imgs = []
    preds = []
    with torch.no_grad():   # 进入 no_grad 推理模式
        for _iidx, img_idx in enumerate(tqdm(img_indices, desc='Inference...')):    # 遍历训练 / 测试样本
            _selected_masks, selected_masks = prepare_masks(example_masks, img_idx) # 为当前样本准备 mask 与 image
            example_imgs_in, cur_idx = prepare_idx(img_idx, example_imgs, device)
            if selected_masks is not None:  # 防止文本到图像的管道
                _selected_masks = (selected_masks[:, 1, ...],)
            with torch.autocast("cuda", enabled=True):
                ovam_evaluator, img_pil_list = initial_forwardpass( # 调用 initial_forwardpass,用训练好的embedding所在的pipeline，再跑一次img2img Diffusion
                    pipeI2I,
                    example_imgs_in,
                    cur_idx,
                    strength,
                    guidance_scale,
                    init_seed,
                    prompt,
                    expand_size=(img_size, img_size),
                    hooker_kwargs = hooker_kwargs,
                    # hooker_kwargs={
                    #     "daam_module_class": (
                    #         StableDiffusionXLDAAM if use_SDXL else StableDiffusionDAAM
                    #     ),
                    #     "locator_hooker_class": SlimeAttentionLocator, #NOTE(ALEX): Centrilized as a parameter now
                    # },
                    ovam_callable_kwargs=ovam_callable_kwargs,
                )
                image = img_pil_list[0] # 这是被优化 embedding 影响后的生成结果。
                # 我们只想绘制主体mask
                pil_img = plot_attention_maps(  # 关键一步：attention / mask 可视化
                    ovam_evaluator,
                    word_embd,
                    init_embd,  # 未训练的token
                    opt_embedding,
                    image,
                    TEXT,
                    TEXT_INIT,
                    device,
                    decoded_full_idx=decoded_full_idx,
                    _selected_masks=_selected_masks,
                    set_titles=_iidx == 0,  # 只在第一张图片上面添加文字
                    _opt_text=_v,  # 哪个文件用于opt
                    _first_text="Inverted Image",
                    return_preds=return_preds,
                )

               #  optimized_map = extract_opt_masks(opt_embedding, ovam_evaluator, device)[[0, 1]]

            if isinstance(pil_img, tuple):  # 可选：返回定量指标,用于训练后定量评估
                pil_img, loss_values = (
                    pil_img  # pil_img, torch.stack([non_optimized_map, optimized_map, non_optimized_avg])
                )
                preds.append(loss_values)
                # display(pil_img)
            imgs.append(pil_img)
        if len(imgs) == 0:
            raise ValueError("No images were generated")
        elif len(imgs) == 1:
            combined_img = imgs[0]
        else:
            combined_img = combine_pil_vertically(  # 多图合并展示
                imgs[0],
                ToPILImage()(
                    make_grid([ToTensor()(a) for a in imgs[1:]], nrow=1, padding=0)
                ),
            )
        # 将图像返回到CPU
        example_imgs = example_imgs.cpu()
        if return_preds:  # in theory should unify, but left for now
            return combined_img, torch.stack(preds)
        return combined_img


# 这是为了事后计算损失
@torch.no_grad()
def calculate_loss(
    loss_type: str, masks, targets, apply_min_max: bool = True, eps=1e-8
):
    assert isinstance(masks, torch.Tensor), f"Expect tensor got {type(masks)=}"
    assert isinstance(targets, torch.Tensor), f"Expect tensor got {type(targets)=}"
    assert (
        masks.ndim == targets.ndim
    ), f"Expect same shape got {masks.shape=} {targets.shape=}"
    # both are in range 0..1, will be converted to appropriate argmax for some losses
    # assert masks.min() >= 0.0 and masks.max() <= 1.0, f"{masks.max()=}, {masks.min()=}"
    assert (
        targets.min() >= 0.0 and targets.max() <= 1.0
    ), f"{targets.max()=}, {targets.min()=}"

    if loss_type == "bce":
        loss_fn = nn.BCELoss(reduction="mean")
    elif loss_type == "l2":
        loss_fn = nn.MSELoss(reduction="mean")
    elif loss_type == "bcelog":
        loss_fn = nn.BCEWithLogitsLoss(reduction="mean")
    elif loss_type == "nll":
        loss_fn = nn.NLLLoss(reduction="mean")
        targets = torch.argmax(targets, dim=1)
    elif loss_type == "cross":
        loss_fn = nn.CrossEntropyLoss(reduction="mean")
    elif loss_type == "IoU":
        loss_fn = torchmetrics.JaccardIndex(
            task="binary", threshold=0.5, ignore_index=None, num_classes=2
        )
        # B C H W, where C=0 is essentially 1 - target
        # will get B 1 H W
        targets = targets[:, 1:2, ...].int()  # gives as same as argmax
    else:
        raise ValueError(f"Loss type {loss_type} not supported.")

    if loss_type != "bcelog":  # Same as in optimization
        if isinstance(apply_min_max, float):
            masks = masks / apply_min_max
        elif apply_min_max:  # For the lineal case
            minimun, maximun = masks.min(), masks.max()
            masks = (masks - minimun) / (maximun - minimun)
        else:
            masks = masks / masks.sum(dim=1, keepdim=True)
    else:
        targets = targets * masks.max().item()

    if loss_type in [
        "nll",
        "cross",
    ]:  # Note(Alex): don't remember should we do this for cross
        # but for nll works...
        masks = torch.log(masks + eps)

    return loss_fn(masks, targets)
