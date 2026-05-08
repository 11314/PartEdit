import gc

import PIL.Image
import torch

from stable_diffusion_xl_partedit import PartEditPipeline, DotDictExtra, Binarization, PaddingStrategy, EmptyControl
from diffusers import AutoencoderKL
from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
from transformers import CLIPImageProcessor

from huggingface_hub import hf_hub_download

available_pts = [
    "pt/torso_custom.pt", # 这只是人体躯干
    "pt/chair_custom.pt", # 这只是椅子的座位
    "pt/carhood_custom.pt", 
    "pt/partimage_biped_head.pt", # 这是猴子
    "pt/partimage_carbody.pt", # 这是除了轮子以外的所有东西
    "pt/partimage_human_hair.pt", 
    "pt/partimage_human_head.pt", # 这是人的头的地方
    "pt/partimage_human_torso.pt", # 用custom on代替这个
    "pt/partimage_quadruped_head.pt", # 这是一种四条腿的动物
]

# def download_part(index):
#     return hf_hub_download(
#         repo_id="Aleksandar/PartEdit-extra",
#         repo_type="dataset",
#         filename=available_pts[index]
#     )

import os

LOCAL_PARTEDIT_EXTRA = "/hxp/zy/PartEdit/Aleksandar/PartEdit-extra"

def download_part(index):
    pt_rel_path = available_pts[index]      # e.g. "pt/partimage_human_head.pt"
    pt_abs_path = os.path.join(LOCAL_PARTEDIT_EXTRA, pt_rel_path)

    if not os.path.exists(pt_abs_path):
        raise FileNotFoundError(
            f"[PartEdit] 本地 pt 文件不存在: {pt_abs_path}"
        )

    return pt_abs_path

PART_NAME_MAP = {
    "head": "human_head",
    "hair": "human_hair",
    "torso": "human_torso_custom",          # 或 human_torso_custom
    "chair": "chair_custom",
    "carhood": "carhood_custom",
    "carbody": "carbody",
    "biped_head": "biped_head",
    "quadruped_head": "quadruped_head",
}

PART_TOKENS = {
    "human_head": download_part(6),
    "human_hair": download_part(5),
    "human_torso_custom": download_part(0), # custom one
    "chair_custom": download_part(1),
    "carhood_custom": download_part(2),
    "carbody": download_part(4),
    "biped_head": download_part(8),
    "quadruped_head": download_part(3),
    "human_torso": download_part(7), # based on partimage
}


class PartEditSDXLModel:
    MAX_NUM_INFERENCE_STEPS = 50

    def __init__(self):
        if torch.cuda.is_available():   # 检测当前运行环境是否有可用的 CUDA GPU
            print(">>> init PartEditSDXLModel")
            self.device = torch.device(f"cuda:{torch.cuda.current_device()}" if torch.cuda.is_available() else "cpu")   # 返回当前 PyTorch 选中的 GPU 编号
            self.sd_pipe, self.partedit_pipe = PartEditPipeline.default_pipeline(self.device)   # 这里调用了stable_diffusion_xl_partedit文件中定义的函数,并返回两个pipeline
        else:
            self.pipe = None    # 如果没有GPU，不加载模型

    def generate(
        self,
        prompt: str,
        negative_prompt: str = "",
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        seed: int = 0,
        eta: float = 0,
    ) -> PIL.Image.Image:

        if not torch.cuda.is_available():   # 检查CUDA
            raise RuntimeError("This demo does not work on CPU!")

        out = self.sd_pipe( # 调用_init_返回的pipeline，应该是个标准推理流程
            prompt=prompt,
            # negative_prompt=negative_prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            eta=eta,
            generator=torch.Generator().manual_seed(seed),
        ).images[0] # 取第一张图

        gc.collect()    # 触发 Python 垃圾回收
        torch.cuda.empty_cache()    # 释放 PyTorch 不再使用的显存缓存
        return out

    def edit(
        self,
        prompt: str,
        subject: str,
        part: str,
        edit: str,
        negative_prompt: str = "",
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        seed: int = 0,
        eta: int = 0,
        t_e: int = 50,
        n_cross_replace: float = 0.4
    ) -> PIL.Image.Image:
        print(">>> In PartEditSDXLModel edit")
        # 性能检查
        if not torch.cuda.is_available():
            raise RuntimeError("This demo does not work on CPU!")

        # if part in PART_TOKENS: # 部件（part）检查与加载 token embedding
        #     token_path = PART_TOKENS[part]
        # else:
        #     raise ValueError(f"Part `{part}` is not supported!")
        print(f"part is {part}")
        if part not in PART_NAME_MAP:
            raise ValueError(f"Part `{part}` is not supported!")

        token_key = PART_NAME_MAP[part]

        if token_key not in PART_TOKENS:
            raise ValueError(f"Token `{token_key}` is not available!")

        token_path = PART_TOKENS[token_key]

        if subject not in prompt:   # 检查 subject 是否存在于 prompt 中
            raise ValueError(f"The subject `{subject}` does not exist in the original prompt!")

        prompts = [ # 构造“原始 / 编辑”prompt 对（非常关键）
            prompt,
            prompt.replace(subject, edit),
        ]

        # PartEdit 参数
        cross_attention_kwargs = {
            "edit_type": "replace",
            "n_self_replace": 0.0,
            "n_cross_replace": {"default_": 1.0, edit: n_cross_replace},
            # "local_blend_words": ["hair", "face"],
        }
        extra_params = DotDictExtra()   # 配置额外的 PartEdit 参数（时间与阈值）
        extra_params.update({"omega": 1.5, "edit_steps": t_e})

        out = self.partedit_pipe(   # 真正执行 PartEdit（最重的一步）
            prompt=prompts,
            # negative_prompt=negative_prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            eta=eta,
            generator=torch.Generator().manual_seed(seed),
            cross_attention_kwargs=cross_attention_kwargs,
            extra_kwargs=extra_params,
            embedding_opt=token_path,
        ).images[:2][::-1]  # 取前两张（原始 + 编辑），再反转

        mask = self.partedit_pipe.visualize_map_across_time()   # 生成并取出编辑 mask，这里调用的是另外一个代码文件中的函数。
        gc.collect()
        torch.cuda.empty_cache()
        return out, mask
