# Based on stable_diffusion_reference.py
# Based on https://github.com/RoyiRa/prompt-to-prompt-with-sdxl
from __future__ import annotations

import abc
import typing
from collections.abc import Iterable
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import einops
import numpy as np
import torch
import torch.nn.functional as F
from diffusers import AutoencoderKL, StableDiffusionXLPipeline, UNet2DConditionModel
from diffusers import __version__ as diffusers_version
from diffusers.models.lora import adjust_lora_scale_text_encoder
from diffusers.models.attention_processor import AttnProcessor2_0
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import (
    rescale_noise_cfg,
)
from diffusers.pipelines.stable_diffusion.safety_checker import (
    StableDiffusionSafetyChecker,
)

from diffusers.pipelines.stable_diffusion_xl.pipeline_output import (
    StableDiffusionXLPipelineOutput,
)
from diffusers.utils import (
    USE_PEFT_BACKEND,
    logging,
    scale_lora_layers,
    unscale_lora_layers,
)
from diffusers.utils.import_utils import is_invisible_watermark_available
from packaging import version
from PIL import Image
from safetensors.torch import load_file
from torchvision.transforms import ToPILImage, ToTensor
from torchvision.utils import make_grid
from transformers import CLIPImageProcessor

if is_invisible_watermark_available():
    from diffusers.pipelines.stable_diffusion_xl.watermark import (
        StableDiffusionXLWatermarker,
    )


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

try: # 尝试导入高级编辑管线和调度器
    from diffusers import LEditsPPPipelineStableDiffusionXL, EulerDiscreteScheduler, DDIMScheduler, DPMSolverMultistepScheduler
except ImportError as e: # 如果由于diffusers版本较低导入失败，
    logger.error("DPMSolverMultistepScheduler or LEditsPPPipelineStableDiffusionXL not found. Verified on >= 0.29.1")
    from diffusers import DDIMScheduler, EulerDiscreteScheduler # 降级导入保证最基本的生成和编辑能力

if typing.TYPE_CHECKING: # 这些导入仅用于静态类型检查
    from diffusers import MultiPipelineCallbacks, PipelineCallback
    from transformers import (
        CLIPTextModel,
        CLIPTextModelWithProjection,
        CLIPTokenizer,
        CLIPVisionModelWithProjection,
    )
    from diffusers.models.attention import Attention
    from diffusers.schedulers import KarrasDiffusionSchedulers


# Original implementation from
# Updated to reflect
class PartEditPipeline(StableDiffusionXLPipeline):
    r"""
    PartEditPipeline for text-to-image generation Pusing Stable Diffusion XL with SD1.5 NSFW checker.

    This model inherits from [`StableDiffusionXLPipeline`]. Check the superclass documentation for the generic methods the
    library implements for all the pipelines (such as downloading or saving, running on a particular device, etc.)

    The pipeline also inherits the following loading methods:
        - [`~loaders.TextualInversionLoaderMixin.load_textual_inversion`] for loading textual inversion embeddings
        - [`~loaders.FromSingleFileMixin.from_single_file`] for loading `.ckpt` files
        - [`~loaders.StableDiffusionXLLoraLoaderMixin.load_lora_weights`] for loading LoRA weights
        - [`~loaders.StableDiffusionXLLoraLoaderMixin.save_lora_weights`] for saving LoRA weights
        - [`~loaders.IPAdapterMixin.load_ip_adapter`] for loading IP Adapters

    Args:
        vae ([`AutoencoderKL`]):
            Variational Auto-Encoder (VAE) Model to encode and decode images to and from latent representations.
        text_encoder ([`CLIPTextModel`]):
            Frozen text-encoder. Stable Diffusion XL uses the text portion of
            [CLIP](https://huggingface.co/docs/transformers/model_doc/clip#transformers.CLIPTextModel), specifically
            the [clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14) variant.
        text_encoder_2 ([` CLIPTextModelWithProjection`]):
            Second frozen text-encoder. Stable Diffusion XL uses the text and pool portion of
            [CLIP](https://huggingface.co/docs/transformers/model_doc/clip#transformers.CLIPTextModelWithProjection),
            specifically the
            [laion/CLIP-ViT-bigG-14-laion2B-39B-b160k](https://huggingface.co/laion/CLIP-ViT-bigG-14-laion2B-39B-b160k)
            variant.
        tokenizer (`CLIPTokenizer`):
            Tokenizer of class
            [CLIPTokenizer](https://huggingface.co/docs/transformers/v4.21.0/en/model_doc/clip#transformers.CLIPTokenizer).
        tokenizer_2 (`CLIPTokenizer`):
            Second Tokenizer of class
            [CLIPTokenizer](https://huggingface.co/docs/transformers/v4.21.0/en/model_doc/clip#transformers.CLIPTokenizer).
        unet ([`UNet2DConditionModel`]): Conditional U-Net architecture to denoise the encoded image latents.
        scheduler ([`SchedulerMixin`]):
            A scheduler to be used in combination with `unet` to denoise the encoded image latents. Can be one of
            [`DDIMScheduler`], [`LMSDiscreteScheduler`], or [`PNDMScheduler`].
        force_zeros_for_empty_prompt (`bool`, *optional*, defaults to `"True"`):
            Whether the negative prompt embeddings shall be forced to always be set to 0. Also see the config of
            `stabilityai/stable-diffusion-xl-base-1-0`.
        add_watermarker (`bool`, *optional*):
            Whether to use the [invisible_watermark library](https://github.com/ShieldMnt/invisible-watermark/) to
            watermark output images. If not defined, it will default to True if the package is installed, otherwise no
            watermarker will be used.
    """

    _optional_components = ["feature_extractor", "add_watermarker, safety_checker"]

    # 从stable_diffusion_reference.py添加了safey_check来实例化SD1.5中的NSFW检查器
    def __init__(
        self,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        text_encoder_2: CLIPTextModelWithProjection,
        tokenizer: CLIPTokenizer,
        tokenizer_2: CLIPTokenizer,
        unet: UNet2DConditionModel,
        scheduler: KarrasDiffusionSchedulers,
        image_encoder: CLIPVisionModelWithProjection = None,
        feature_extractor: CLIPImageProcessor = None,
        force_zeros_for_empty_prompt: bool = True,
        add_watermarker: Optional[bool] = None,
        safety_checker: Optional[StableDiffusionSafetyChecker] = None,
    ):
        if safety_checker is not None:  # 判断用户 是否传入 NSFW 检测器，如果没传就跳过。
            assert isinstance(safety_checker, StableDiffusionSafetyChecker), f"Expected safety_checker to be of type StableDiffusionSafetyChecker, got {type(safety_checker)}"
            assert feature_extractor is not None, "Feature Extractor must be present to use the NSFW checker"
        super().__init__(   # 调用父类（SDXL）的构造函数
            vae=vae,
            text_encoder=text_encoder,
            text_encoder_2=text_encoder_2,
            tokenizer=tokenizer,
            tokenizer_2=tokenizer_2,
            unet=unet,
            scheduler=scheduler,
            image_encoder=image_encoder,
            feature_extractor=feature_extractor,
            force_zeros_for_empty_prompt=force_zeros_for_empty_prompt,
            add_watermarker=add_watermarker,
        )   # 到此你拥有一个“完整、可用的 SDXL pipeline”
        self.register_modules(  # 将 safety_checker 注册到 pipeline 内部，生成之后，pipeline 会自动调用它做 NSFW 检测
            safety_checker=safety_checker,
        )
        # self.warn_once_callback = True

    @staticmethod   # 这是一个静态方法，不依赖类实例（self），可以直接通过类名调用。
    def default_pipeline(device, precision=torch.float16, scheduler_type: str = "euler", load_safety: bool = False) -> Tuple[StableDiffusionXLPipeline, PartEditPipeline]:
        if scheduler_type.strip().lower() in ["ddim", "editfriendly"]:  # 对输入字符串做：去空格，转小写。判断是否是 DDIM / Edit-friendly 模式
            scheduler = DDIMScheduler.from_pretrained("/hxp/zy/pretrain_models/stable-diffusion-xl-base-1.0", subfolder="scheduler", torch_dtype=precision)  # 从模型中加载DDIM
        elif scheduler_type.strip().lower() in "leditspp":

            scheduler = DPMSolverMultistepScheduler.from_pretrained(    # 构造 SDE-DPM-Solver++，这是 LEdits++ 中使用的调度器
                "/hxp/zy/pretrain_models/stable-diffusion-xl-base-1.0", subfolder="scheduler", algorithm_type="sde-dpmsolver++", solver_order=2
            )  # LEdits
        else:   # 默认情况下，使用 Euler 离散调度器，速度快，稳定。作为默认推理调度器
            scheduler = EulerDiscreteScheduler.from_pretrained("/hxp/zy/pretrain_models/stable-diffusion-xl-base-1.0", subfolder="scheduler", torch_dtype=precision)

        # 加载VAE
        vae = AutoencoderKL.from_pretrained(
            "madebyollin/sdxl-vae-fp16-fix",
            torch_dtype=precision,
            use_safetensors=True,
            resume_download=None,
        )
        # 构造“原始 SDXL pipeline”
        default_pipe = StableDiffusionXLPipeline.from_pretrained(
            "/hxp/zy/pretrain_models/stable-diffusion-xl-base-1.0",
            device=device,
            vae=vae,
            resume_download=None,
            scheduler=DDIMScheduler.from_pretrained("/hxp/zy/pretrain_models/stable-diffusion-xl-base-1.0", subfolder="scheduler", torch_dtype=precision),
            torch_dtype=precision,
        )

        # 是否加载 NSFW 安全模块（可选）
        safety_checker = (
            StableDiffusionSafetyChecker.from_pretrained(
                "/hxp/zy/pretrain_models/stable-diffusion-v1-5",  # runwayml/stable-diffusion-v1-5",
                device_map=device,
                torch_dtype=precision,
                subfolder="safety_checker",
            )
            if load_safety
            else None
        )
        # 在NSFW开启时，给 safety checker 提供图像特征
        feature_extractor = (
            CLIPImageProcessor.from_pretrained(
                "/hxp/zy/pretrain_models/stable-diffusion-v1-5",  # "runwayml/stable-diffusion-v1-5",
                subfolder="feature_extractor",
                device_map=device,
            )
            if load_safety
            else None
        )
        # 构造 PartEditPipeline（关键）
        pipeline: PartEditPipeline = PartEditPipeline(
            vae=vae,
            tokenizer=default_pipe.tokenizer,
            tokenizer_2=default_pipe.tokenizer_2,
            text_encoder=default_pipe.text_encoder,
            text_encoder_2=default_pipe.text_encoder_2,
            unet=default_pipe.unet,
            scheduler=scheduler,
            image_encoder=default_pipe.image_encoder,
            safety_checker=safety_checker,
            feature_extractor=feature_extractor,
        )
        return default_pipe.to(device), pipeline.to(device)

    # 输入合法 + PartEdit 编辑条件齐备。
    def check_inputs(
        self,
        prompt,
        prompt_2,
        height,
        width,
        callback_steps,
        negative_prompt=None,
        negative_prompt_2=None,
        prompt_embeds=None,
        negative_prompt_embeds=None,
        pooled_prompt_embeds=None,
        negative_pooled_prompt_embeds=None,
        ip_adapter_image=None,
        ip_adapter_image_embeds=None,
        callback_on_step_end_tensor_inputs=None,
        # PartEdit 部分
        embedding_opt: Optional[torch.FloatTensor] = None,  # 用于 token embedding 优化 / 编辑
    ):
        # 检查diffusers版本
        extra_params = (
            {
                "ip_adapter_image": ip_adapter_image,
                "ip_adapter_image_embeds": ip_adapter_image_embeds,
            }
            if version.parse(diffusers_version) >= version.parse("0.27.0")
            else {}
        )

        # 使用super来检查来自父类的输入
        super(PartEditPipeline, self).check_inputs(
            prompt,
            prompt_2,
            height,
            width,
            callback_steps,
            negative_prompt,
            negative_prompt_2,
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
            callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
            **extra_params,
        )
        # PartEdit 专属：检查 embedding_opt（编辑核心）
        if embedding_opt is not None:
            assert embedding_opt.ndim == 2, f"Embedding should be of shape (2, features), got {embedding_opt.shape}"    # 强制要求 embedding 是二维张量，对应 classifier-free guidance 的双通道 embedding
            assert embedding_opt.shape[-1] == 2048, f"SDXL Embedding should have 2048 features, got {embedding_opt.shape[1]}"   # SDXL 的 text embedding 维度是 2048
            assert embedding_opt.dtype in [ # 防止用 int / bf16 / 乱七八糟的 dtype
                torch.float32,
                torch.float16,
            ], f"Embedding should be of type fp32/fp16, got {embedding_opt.dtype}"

        assert hasattr(self, "controller"), "Controller should be present"  # 确保 PartEdit Controller 已正确注入
        assert hasattr(self.controller, "extra_kwargs"), "Controller should have extra_kwargs"

        # 从 controller 中取出：阈值 / 掩码策略。th_strategy 决定：attention → mask 的方式
        extra_kwargs: DotDictExtra = self.controller.extra_kwargs
        strategy: Binarization = extra_kwargs.th_strategy

        assert isinstance(strategy, Binarization), f"Expected strategy to be of type Binarization, got {type(strategy)}"    # 防止传入非法字符串 / 数字
        assert hasattr(extra_kwargs, "pad_strategy"), "Controller should have pad_strategy" # 控制mask是否膨胀，是否做边缘padding
        assert isinstance(extra_kwargs.pad_strategy, PaddingStrategy), f"Expected pad_strategy to be of type PaddingStrategy, got {type(self.controller.extra_kwargs.pad_strategy)}"

        if strategy in [Binarization.PROVIDED_MASK]:    # 当策略是“用户提供掩码”时的强校验。外部 mask 编辑模式
            assert hasattr(extra_kwargs, "mask_edit"), "Mask should be present in extra_kwargs"

    def _aggregate_and_get_attention_maps_per_token(self, with_softmax, select: int = 0, res: int = 32):
        attention_maps = self.controller.aggregate_attention(
            res=res,
            from_where=("up", "down", "mid"),
            batch_size=self.controller.batch_size,
            is_cross=True,
            select=select,
        )   # 把 UNet 各层的 cross-attention 全部收集 → 对齐 → 聚合
        attention_maps_list = self._get_attention_maps_list(attention_maps=attention_maps, with_softmax=with_softmax)   # 拆分成“每个 token 一张图”，见下面函数
        return attention_maps_list

    @staticmethod
    def _get_attention_maps_list(attention_maps: torch.Tensor, with_softmax) -> List[torch.Tensor]:
        attention_maps *= 100   # 放大图

        if with_softmax:    # 可选 softmax（token 竞争机制）
            attention_maps = torch.nn.functional.softmax(attention_maps, dim=-1)

        attention_maps_list = [attention_maps[:, :, i] for i in range(attention_maps.shape[2])] # 这是 PartEdit 能“知道哪个词控制哪块区域”的根源
        return attention_maps_list

    @torch.inference_mode()  # 如果出现问题，请返回@torch.no_grad（）
    def __call__(
        self,
        prompt: Union[str, List[str]],
        prompt_2: Optional[Union[str, List[str]]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        denoising_end: Optional[float] = None,
        guidance_scale: float = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        negative_prompt_2: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: Optional[int] = 1,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        guidance_rescale: float = 0.0,
        original_size: Optional[Tuple[int, int]] = None,
        crops_coords_top_left: Tuple[int, int] = (0, 0),
        target_size: Optional[Tuple[int, int]] = None,
        negative_original_size: Optional[Tuple[int, int]] = None,
        negative_crops_coords_top_left: Tuple[int, int] = (0, 0),
        negative_target_size: Optional[Tuple[int, int]] = None,
        attn_res=None,
        callback_on_step_end: Optional[Union[Callable[[int, int, Dict], None], PipelineCallback, MultiPipelineCallbacks]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        # PartEdit
        embedding_opt: Optional[Union[torch.FloatTensor, str]] = None,
        extra_kwargs: Optional[Union[dict, DotDictExtra]] = None,  # 所有参数，检查DotDictExtra
        uncond_embeds: Optional[torch.FloatTensor] = None,  # 无条件嵌入从空文本反转
        latents_list=None,
        zs=None,
    ):
        r"""
        Function invoked when calling the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`):
                The prompt or prompts to guide the image generation.
            height (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
                The height in pixels of the generated image.
            width (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
                The width in pixels of the generated image.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            guidance_scale (`float`, *optional*, defaults to 7.5):
                Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
                `guidance_scale` is defined as `w` of equation 2. of [Imagen
                Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >
                1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
                usually at the expense of lower image quality.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. Ignored when not using guidance (i.e., ignored
                if `guidance_scale` is less than `1`).
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            eta (`float`, *optional*, defaults to 0.0):
                Corresponds to parameter eta (η) in the DDIM paper: https://arxiv.org/abs/2010.02502. Only applies to
                [`schedulers.DDIMScheduler`], will be ignored for others.
            generator (`torch.Generator`, *optional*):
                One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html)
                to make generation deterministic.
            latents (`torch.FloatTensor`, *optional*):
                Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor will ge generated by sampling using the supplied random `generator`.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generate image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] instead of a
                plain tuple.
            callback (`Callable`, *optional*):
                A function that will be called every `callback_steps` steps during inference. The function will be
                called with the following arguments: `callback(step: int, timestep: int, latents: torch.FloatTensor)`.
            callback_steps (`int`, *optional*, defaults to 1):
                The frequency at which the `callback` function will be called. If not specified, the callback will be
                called at every step.
            cross_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the [`AttentionProcessor`] as defined in
                [`self.processor`](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).

                The keyword arguments to configure the edit are:
                - edit_type (`str`). The edit type to apply. Can be either of `replace`, `refine`, `reweight`.
                - n_cross_replace (`int`): Number of diffusion steps in which cross attention should be replaced
                - n_self_replace (`int`): Number of diffusion steps in which self attention should be replaced
                - local_blend_words(`List[str]`, *optional*, default to `None`): Determines which area should be
                  changed. If None, then the whole image can be changed.
                - equalizer_words(`List[str]`, *optional*, default to `None`): Required for edit type `reweight`.
                  Determines which words should be enhanced.
                - equalizer_strengths (`List[float]`, *optional*, default to `None`) Required for edit type `reweight`.
                  Determines which how much the words in `equalizer_words` should be enhanced.

            guidance_rescale (`float`, *optional*, defaults to 0.0):
                Guidance rescale factor from [Common Diffusion Noise Schedules and Sample Steps are
                Flawed](https://arxiv.org/pdf/2305.08891.pdf). Guidance rescale factor should fix overexposure when
                using zero terminal SNR.
        PartEdit Parameters:
            embedding_opt (`Union[torch.FloatTensor, str]`, *optional*): The embedding to be inserted in the prompt. The embedding
                will be inserted as third batch dimension.
            extra_kwargs (`dict`, *optional*): A dictionary with extra parameters to be passed to the pipeline.
                - Check `pipe.part_edit_available_params()` for the available parameters.
        Returns:
            [`~pipelines.stable_diffusion_xl.StableDiffusionXLPipelineOutput`] or `tuple`:
            [`~pipelines.stable_diffusion_xl.StableDiffusionXLPipelineOutput`] if `return_dict` is True, otherwise a
            `tuple`. When returning a tuple, the first element is a list with the generated images.
        """

        print(">>> In PartEditPipeline")
        # 0. Default height and width to unet/默认高度和宽度为unet的值
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor

        original_size = original_size or (height, width)
        target_size = target_size or (height, width)

        # PartEdit setup
        extra_kwargs = DotDictExtra() if extra_kwargs is None else DotDictExtra(extra_kwargs)   # 把所有编辑参数（阈值、策略、mask 等）统一封装
        prompt = prompt + [prompt[0]] if prompt[0] != prompt[-1] else prompt  # 构造 “原 prompt + 编辑 prompt”
        extra_kwargs.batch_indx = len(prompt) - 1 if extra_kwargs.batch_indx == -1 else extra_kwargs.batch_indx # 设置哪个 batch 是“编辑目标”，默认使用最后一个
        add_extra_step = extra_kwargs.add_extra_step

        if attn_res is None:
            attn_res = int(np.ceil(width / 32)), int(np.ceil(height / 32))  # Attention 分辨率设置
        self.attn_res = attn_res
        # _prompts = prompt if embedding_opt is None else prompt + [prompt[-1]]
        if hasattr(self, "controller"): # 如果之前用过 controller，清空其内部状态
            self.controller.reset()

        # 创建 Attention Controller（核心）
        self.controller = create_controller(
            prompt,
            cross_attention_kwargs,
            num_inference_steps,
            tokenizer=self.tokenizer,
            device=self.device,
            attn_res=self.attn_res,
            extra_kwargs=extra_kwargs,
        )
        assert self.controller is not None  # 运行期安全检查
        assert issubclass(type(self.controller), AttentionControl)  # 判断controller的类型是AttentionControl的子类
        self.register_attention_control(    
            self.controller,
        )  # 把 controller 挂到 UNet 的 cross-attention 上

        # 1. 校验 prompt / embedding / mask / strategy
        self.check_inputs(
            prompt,
            prompt_2,
            height,
            width,
            callback_steps,
            negative_prompt,
            negative_prompt_2,
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        )

        # 确定 batch 数量，用于后面latent/embedding对齐
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]
        # batch_size = batch_size + 1 if embedding_opt is not None else batch_size

        device = self._execution_device # 获取当前 pipeline 使用的设备（cuda）
        # 此处guidance_scale的定义类似于式(2)中的指导权重w。
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # 对应于不做分类器自由引导.
        do_classifier_free_guidance = guidance_scale > 1.0  # batch / CFG 判断

        # 3. Encode input prompt 编码图片
        # 文本到embedding，文本编码（编辑开始介入）
        text_encoder_lora_scale = cross_attention_kwargs.get("scale", None) if cross_attention_kwargs is not None else None # 若使用 LoRA，读取其 scale
        (
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        ) = self.encode_prompt(
            prompt=prompt,
            prompt_2=prompt_2,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            do_classifier_free_guidance=do_classifier_free_guidance,
            negative_prompt=negative_prompt,
            negative_prompt_2=negative_prompt_2,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
            lora_scale=text_encoder_lora_scale,
        )

        # 4. 生成扩散反向过程的时间步序列
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # 5. 准备潜在变量
        num_channels_latents = self.unet.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )
        latents[1] = latents[0] # 强制编辑 prompt 和原 prompt 从同一个噪声起点开始

        # 6. 准备额外的步骤。TODO:理想情况下，逻辑应该从管道中移出
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 7. /准备添加的时间id和嵌入
        add_text_embeds = pooled_prompt_embeds
        add_time_ids = self._get_add_time_ids(
            original_size,
            crops_coords_top_left,
            target_size,
            dtype=prompt_embeds.dtype,
            text_encoder_projection_dim=self.text_encoder_2.config.projection_dim,  # if none should be changed to enc1
        )
        if negative_original_size is not None and negative_target_size is not None:
            negative_add_time_ids = self._get_add_time_ids(
                negative_original_size,
                negative_crops_coords_top_left,
                negative_target_size,
                dtype=prompt_embeds.dtype,
            )
        else:
            negative_add_time_ids = add_time_ids

        # PartEdit:embedding 修改，会对特定 token embedding 做：替换,padding，对齐
        prompt_embeds = self.process_embeddings(embedding_opt, prompt_embeds, self.controller.pad_strategy)
        self.prompt_embeds = prompt_embeds  # 保存供 controller 使用

        if do_classifier_free_guidance: # 构造 [uncond, cond] embedding
            _og_prompt_embeds = prompt_embeds.clone()
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
            add_text_embeds = torch.cat([negative_pooled_prompt_embeds, add_text_embeds], dim=0)
            add_time_ids = torch.cat([negative_add_time_ids, add_time_ids], dim=0)

        prompt_embeds = prompt_embeds.to(device)
        add_text_embeds = add_text_embeds.to(device)
        add_time_ids = add_time_ids.to(device).repeat(batch_size * num_images_per_prompt, 1)

        # 8. Denoising loop/去噪循环
        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)

        # 7.1 Apply denoising_end
        if denoising_end is not None and isinstance(denoising_end, float) and denoising_end > 0 and denoising_end < 1:
            discrete_timestep_cutoff = int(round(self.scheduler.config.num_train_timesteps - (denoising_end * self.scheduler.config.num_train_timesteps)))
            num_inference_steps = len(list(filter(lambda ts: ts >= discrete_timestep_cutoff, timesteps)))
            timesteps = timesteps[:num_inference_steps]
        # PartEdit
        if hasattr(self, "debug_list"):  # 如果它是禁用的，并且有一个列表
            del self.debug_list
        if extra_kwargs.debug_vis:
            self.debug_list = []
        if add_extra_step:
            num_inference_steps += 1
            timesteps = torch.cat([timesteps[[0]], timesteps], dim=-1)
            _latents = latents.clone()

        self._num_timesteps = len(timesteps)  # Same as in SDXL
        added_cond_kwargs = {"text_embeds": add_text_embeds, "time_ids": add_time_ids}
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                # if i in range(50):
                #     latents[0] = latents_list[i]
                # 如果我们在做分类器自由引导，就扩展潜在向量
                latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents

                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)    # 按 scheduler 要求缩放 latent

                # 注（Alex）： Null text inversion用法
                if uncond_embeds is not None:
                    # if callback_on_step_end is not None and self.warn_once_callback:
                    #     self.warn_once_callback = False
                    #     logger.warning("Callback on step end is not supported with Null text inversion - Know what you are doing!")
                    _indx_to_use = i if i < len(uncond_embeds) else len(uncond_embeds) - 1  # use last if we have extra steps
                    # _og_prompt_embeds
                    curr = uncond_embeds[_indx_to_use].to(dtype=prompt_embeds.dtype).to(device).repeat(_og_prompt_embeds.shape[0], 1, 1)
                    prompt_embeds = torch.cat([curr, _og_prompt_embeds], dim=0)  # 目前不更改池提示嵌入
                    # if prompt_embeds.shape != (2, 77, 2048):
                    #     print(f"Prompt Embeds should be of shape (2, 77, 2048), got {prompt_embeds.shape}")

                # UNet 前向预测噪声
                noise_pred = self.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=prompt_embeds,
                    added_cond_kwargs=added_cond_kwargs,
                ).sample

                if add_extra_step:  # PartEdit
                    latents = _latents.clone()
                    add_extra_step = False
                    progress_bar.update()
                    self.scheduler._init_step_index(t)
                    continue  # we just wanted the unet, not to do the step

                # perform guidance
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    # gs = torch.tensor([guidance_scale] * len(noise_pred_uncond),
                    #                   device=noise_pred.device, dtype= noise_pred.dtype).view(-1, 1, 1, 1)
                    # gs[0] = 7.5
                    # our_gs = torch.FloatTensor([1.0, guidance_scale, 1.0]).view(-1, 1, 1, 1).to(latents.device, dtype=latents.dtype)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond) # CFG 合成最终噪声预测

                if do_classifier_free_guidance and guidance_rescale > 0.0:
                    # Based on 3.4. in https://arxiv.org/pdf/2305.08891.pdf
                    noise_pred = rescale_noise_cfg(noise_pred, noise_pred_text, guidance_rescale=guidance_rescale)

                # 执行一步扩散反向更新 # synth
                latents = self.scheduler.step(
                    noise_pred, t, latents, **extra_step_kwargs
                )
                # inv
                # latents = self.scheduler.step(noise_pred, t, latents, variance_noise=zs[i], **extra_step_kwargs)

                if extra_kwargs.debug_vis:  # PartEdit
                    # Could be removed, with .prev_sample above
                    self.debug_list.append(latents.pred_original_sample.cpu())

                latents = latents.prev_sample  # 这里需要，因为将上面计入日志

                # 让 controller：更新 attention，控制 mask 生效时间
                latents = self.controller.step_callback(latents)    # 查看latents参数内容

                # 注释（Alex）：从SDXL复制
                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                    negative_prompt_embeds = callback_outputs.pop("negative_prompt_embeds", negative_prompt_embeds)
                    add_text_embeds = callback_outputs.pop("add_text_embeds", add_text_embeds)
                    negative_pooled_prompt_embeds = callback_outputs.pop("negative_pooled_prompt_embeds", negative_pooled_prompt_embeds)
                    add_time_ids = callback_outputs.pop("add_time_ids", add_time_ids)
                    negative_add_time_ids = callback_outputs.pop("negative_add_time_ids", negative_add_time_ids)
                if embedding_opt is not None:  # PartEdit
                    us_dx = 0
                    if i == 0 and us_dx != 0:
                        print(f'Using lantents[{us_dx}] instead of latents[0]')
                    latents[-1:] = latents[us_dx]  # 总是把困难的过程联系起来
                # 如果embedding_opt不为None且callback_on_step_end不为None，则 \
                # callback_on_step_end.reversed_latents is not None:
                #     latents[-1:] = callback_on_step_end.reversed_latents[i]

                # 如果提供的话，调用回调
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        step_idx = i // getattr(self.scheduler, "order", 1)
                        callback(step_idx, t, latents)

        # 8. 后处理
        if output_type == "latent":
            image = latents
        else:
            self.final_map = self.controller.visualize_final_map(False) # 生成最终 attention / mask 可视化
            # 增加支持更低的VRAM gpu
            self.controller.offload_stores(torch.device("cpu"))
            image = self.latent2image(latents, device, output_type, force_upcast=False) # VAE 解码 latent → image

        # 卸载所有模型
        self.maybe_free_model_hooks()

        if not return_dict:
            return image

        self.grid = self.visualize_maps()
        # 在以下情况下禁用编辑
        self.unregister_attention_control() # 移除 attention hook，防止影响下一次推理

        # 没有添加NSFW输出，因为它不是xlpipelineoutput的一部分
        return StableDiffusionXLPipelineOutput(images=image)

    # 将扩散模型的潜在空间输出（latent）转换为可视化的图像（通过 VAE 解码），并执行后处理（如安全检测、水印和图像标准化）。
    @torch.no_grad()
    def latent2image(
        self: PartEditPipeline,
        latents: torch.Tensor,
        device: torch.device,
        output_type: str = "pil",  # ['latent', 'pt', 'np', 'pil']
        force_upcast: bool = False,
    ) -> Union[torch.Tensor, np.ndarray, Image.Image]:
        # 确保VAE处于float32模式，因为它在float16模式下溢出
        needs_upcasting = self.vae.dtype == torch.float16 and self.vae.config.force_upcast or force_upcast
        latents = latents.to(device)
        if needs_upcasting:
            self.upcast_vae()
        latents = latents.to(next(iter(self.vae.post_quant_conv.parameters())).dtype)   # 将 latents 数据转换为 VAE 解码层参数的类型，确保解码过程的数据类型匹配。

        image = self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False)[0] # 通过 VAE 解码器将 latents 转换为图像。
        # 如果需要的话，回到fp16
        if needs_upcasting and not force_upcast:
            self.vae.to(dtype=torch.float16)
        image, has_nsfw_concept = self.run_safety_checker(image, device, latents.dtype) # 对生成的图像进行 NSFW 检查（如成人内容）。

        if has_nsfw_concept is None:    # 根据 NSFW 检测结果决定是否对图像进行标准化。do_denormalize 表示哪些图像需要恢复为正常的像素范围。
            do_denormalize = [True] * image.shape[0]
        else:
            do_denormalize = [not has_nsfw for has_nsfw in has_nsfw_concept]
            if not all(do_denormalize):
                logger.warn(
                    "NSFW detected in the following images: %s",
                    ", ".join([f"image {i + 1}" for i, has_nsfw in enumerate(has_nsfw_concept) if has_nsfw]),
                )
        if self.watermark is not None:  # 如果启用了水印功能，将水印应用到图像上。
            image = self.watermark.apply_watermark(image)
        image = self.image_processor.postprocess(image, output_type=output_type, do_denormalize=do_denormalize)
        if output_type in ["pt", "latent"]:
            image = image.cpu()
            latents = latents.cpu()
        return image

    def run_safety_checker(self, image: Union[np.ndarray, torch.Tensor], device: torch.device, dtype: type):
        if self.safety_checker is None:
            has_nsfw_concept = None
        else:
            if torch.is_tensor(image):  # 如果输入图像是 torch.Tensor 类型，则先经过 postprocess 转换为 PIL 图像格式。
                feature_extractor_input = self.image_processor.postprocess(image, output_type="pil")
            else:   # 如果输入是 numpy.ndarray，则直接通过 numpy_to_pil 转换为 PIL 图像。
                feature_extractor_input = self.image_processor.numpy_to_pil(image)
            safety_checker_input = self.feature_extractor(feature_extractor_input, return_tensors="pt").to(device)  # 使用 feature_extractor 提取图像特征，将其转换为 PyTorch tensor 格式，方便传入模型进行处理。
            image, has_nsfw_concept = self.safety_checker(images=image, clip_input=safety_checker_input.pixel_values.to(dtype))
        return image, has_nsfw_concept

    # 在 UNet 中注册并配置注意力（attention）控制器，允许 PartEdit 对扩散模型的 cross-attention 层进行控制。
    def register_attention_control(self, controller):
        attn_procs = {}
        cross_att_count = 0
        self.attn_names = {}  # Name => Idx
        for name in self.unet.attn_processors:  # 这里开始循环遍历 UNet 中的所有 attention 层。
            (None if name.endswith("attn1.processor") else self.unet.config.cross_attention_dim)    # 跳过指定的层
            if name.startswith("mid_block"):    # 判断当前层是否属于 UNet 的中间部分（mid_block）。 
                self.unet.config.block_out_channels[-1]
                place_in_unet = "mid"
            elif name.startswith("up_blocks"):  # 判断当前层是否属于 UNet 的上采样部分（up_blocks）。
                block_id = int(name[len("up_blocks.")]) # 如果是，那就提取块的编号，
                list(reversed(self.unet.config.block_out_channels))[block_id]
                place_in_unet = "up"
            elif name.startswith("down_blocks"):    # 同理
                block_id = int(name[len("down_blocks.")])
                self.unet.config.block_out_channels[block_id]
                place_in_unet = "down"
            else:
                continue
            attn_procs[name] = PartEditCrossAttnProcessor(controller=controller, place_in_unet=place_in_unet)   # 创建 PartEditCrossAttnProcessor，下面有具体定义。
            # print(f'{cross_att_count}=>{name}')
            cross_att_count += 1    # 每添加一个 cross-attention 层，cross_att_count 就加 1，统计需要控制的层数。

        self.unet.set_attn_processor(attn_procs)    # 将 attn_procs 传给 self.unet。
        controller.num_att_layers = cross_att_count # 更新 controller 对象中的 num_att_layers，标记有多少个 attention 层需要控制。

    def unregister_attention_control(self):
        # if pytorch >= 2.0
        self.unet.set_attn_processor(AttnProcessor2_0())    # 将 UNet 的 attention 处理器恢复为标准的 UNet cross-attention 处理器。
        if hasattr(self, "controller") and self.controller is not None: # 检查 self 是否具有 controller 属性，且 controller 不为 None。
            if hasattr(self.controller, "last_otsu"):   # 如果 controller 具有 last_otsu 属性，就将 最后一个 OTSU 阈值保存到 self.last_otsu_value 中。
                self.last_otsu_value = self.controller.last_otsu[-1]
            del self.controller # 删除 controller 对象，释放内存。
            # self.controller.allow_edit_control = False

    def available_params(self) -> str:

        pipeline_params = """
        Pipeline Parameters: 
            embedding_opt (`Union[torch.FloatTensor, str]`, *optional*): The embedding to be inserted in the prompt. The embedding
                will be inserted as third batch dimension.
            extra_kwargs (`dict`, *optional*): A dictionary with extra parameters to be passed to the pipeline. 
                - Check `pipe.part_edit_available_params()` for the available parameters.
        """
        # 将 pipeline_params 字符串与 self.part_edit_available_params() 返回的字符串连接起来，形成 完整的参数文档。
        return pipeline_params + "\n" + self.part_edit_available_params()

    def process_embeddings(
            self,
            embedding_opt: Optional[Union[torch.FloatTensor, str]],
            prompt_embeds: torch.FloatTensor,
            padd_strategy: PaddingStrategy,
    ) -> torch.Tensor:
        return process_embeddings(embedding_opt, prompt_embeds, padd_strategy)

    def part_edit_available_params(self) -> str:
        return DotDictExtra().explain() # DotDictExtra 是一个对象，它包含了一些额外的参数（例如，PartEdit 专用的参数）。它的 explain() 方法返回该对象的参数说明。

    # def run_sa

    def visualize_maps(self, make_grid_kwargs: dict = None):
        """Wrapper function to select correct storage location"""
        if not hasattr(self, "controller") or self.controller is None:
            return self.grid if hasattr(self, "grid") else None

        return self.controller.visualize_maps_agg(
            self.controller.use_agg_store,
            make_grid_kwargs=make_grid_kwargs,
        )

    def visualize_map_across_time(self):
        """Wrapper function to visualize the same as above, but as one mask"""
        if hasattr(self, "final_map") and self.final_map is not None:   # 首先检查 self 是否具有 final_map 属性，且它不为 None。
            return self.final_map   # self.final_map 存储了之前计算得到的 attention map 或掩码
        return self.controller.visualize_final_map(self.controller.use_agg_store)   # 如果 final_map 不存在，那么通过控制器来生成最终的 attention map 或掩码。

def process_embeddings(
        embedding_opt: Optional[Union[torch.Tensor, str]],
        prompt_embeds: torch.Tensor,
        padd_strategy: PaddingStrategy,
    ) -> torch.Tensor:
        if embedding_opt is None:   # 如果没有传入 embedding_opt，则直接返回 prompt_embeds。
            return prompt_embeds
        assert isinstance(padd_strategy, PaddingStrategy), f"padd_strategy must be of type PaddingStrategy, got {type(padd_strategy)}"  # 确保传入的 padd_strategy 是 PaddingStrategy 类型

        if isinstance(embedding_opt, str):  # 如果 embedding_opt 是一个文件路径（字符串），根据文件类型（safetensors 或 PyTorch）加载文件。
            embedding_opt = load_file(embedding_opt)["embedding"] if "safetensors" in embedding_opt else torch.load(embedding_opt)
        elif isinstance(embedding_opt, list):   # 如果 embedding_opt 是一个文件路径列表，它会循环加载每个文件中的 embedding，并将它们拼接在一起。
            e = [load_file(i)["embedding"] if "safetensors" in i else torch.load(i) for i in embedding_opt]
            embedding_opt = torch.cat(e, dim=0)
            print(f'Embedding Opt shape: {embedding_opt.shape=}')
        embedding_opt = embedding_opt.to(device=prompt_embeds.device, dtype=prompt_embeds.dtype)    # 将 embedding_opt 移动到 prompt_embeds 所在的设备，并确保数据类型匹配。
        if embedding_opt.ndim == 2: # 如果 embedding_opt 是二维张量（通常是 [batch_size, features]），将其变为三维张量（[1, batch_size, features]），以便能够与 prompt_embeds 正常合并。
            embedding_opt = embedding_opt[None]
        num_embeds = embedding_opt.shape[1] # BG + Num of classes
        prompt_embeds[-1:, :num_embeds, :] = embedding_opt[:, :num_embeds, :]   # 将 embedding_opt 的前 num_embeds 维度（即 embedding 的一部分）插入到 prompt_embeds 的最后一行。

        if PaddingStrategy.context == padd_strategy:    # 如果当前的 padding 策略是 context，则直接返回 prompt_embeds，无需做任何处理
            return prompt_embeds
        if not (hasattr(padd_strategy, "norm") and hasattr(padd_strategy, "scale")):
            raise ValueError(f"PaddingStrategy with {padd_strategy} not recognized")
        _norm, _scale = padd_strategy.norm, padd_strategy.scale # 检查 padd_strategy 是否具有 norm 和 scale 属性，这两个属性决定了是否对 embedding_opt 进行 归一化 和 缩放。

        # 根据不同的 padding 策略进行具体填充
        if padd_strategy == PaddingStrategy.BG:
            prompt_embeds[-1:, num_embeds:, :] = embedding_opt[:, :1, :]
        elif padd_strategy == PaddingStrategy.EOS:
            prompt_embeds[-1:, num_embeds:, :] = prompt_embeds[-1:, -1:, :]
        elif padd_strategy == PaddingStrategy.ZERO:
            prompt_embeds[-1:, num_embeds:, :] = 0.0
        elif padd_strategy == PaddingStrategy.SOT_E:
            prompt_embeds[-1:, num_embeds:, :] = prompt_embeds[-1:, :1, :]
        else:
            raise ValueError(f"{padd_strategy} not recognized")
        # 不推荐
        if _norm:
            prompt_embeds[-1:, :, :] = F.normalize(prompt_embeds[-1:, :, :], p=2, dim=-1)
        if _scale:
            _eps = 1e-8
            _min, _max = prompt_embeds[:1].min(), prompt_embeds[:1].max()
            if _norm:
                prompt_embeds = (prompt_embeds - _min) / (_max - _min + _eps)
            else:
                _new_min, _new_max = (
                    prompt_embeds[-1:, num_embeds:, :].min(),
                    prompt_embeds[-1:, num_embeds:, :].max(),
                )
                prompt_embeds[-1:, num_embeds:, :] = (prompt_embeds[-1:, num_embeds:, :] - _new_min) / (_new_max - _new_min + _eps)
                prompt_embeds[-1:, num_embeds:, :] = prompt_embeds[-1:, num_embeds:, :] * (_max - _min + _eps) + _min
        return prompt_embeds

# 取决于训练时使用的层
LAYERS_TO_USE = [
    24,
    25,
    26,
    27,
    28,
    29,
    30,
    31,
    32,
    33,
    34,
    35,
    36,
    37,
    38,
    39,
    40,
    41,
    42,
    43,
    44,
    45,
    46,
    47,
    48,
    49,
    50,
    51,
    52,
    53,
    54,
    55,
    56,
    57,
    58,
    59,
    0,
    1,
    2,
    3,
]  # noqa: E501


# 它定义了 PartEdit 中“如何把注意力图（attention map）变成可用掩码”的策略集合。,阈值策略
class Binarization(Enum):
    """Controls the binarization of attn maps
    in case of use_otsu lower_binarize and upper_binarizer are multilpiers of otsu threshold

    args:
        strategy: str: name of the strategy
        enabled: bool: if binarization is enabled
        lower_binarize: float: lower threshold for binarization
        upper_binarize: float: upper threshold for binarization
        use_otsu: bool: if otsu is used for binarization
    """

    P2P = "p2p", False, 0.5, 0.5, False  # Baseline
    PROVIDED_MASK = "mask", True, 0.5, 0.5, False
    BINARY_0_5 = "binary_0.5", True, 0.5, 0.5, False
    BINARY_OTSU = "binary_otsu", True, 1.0, 1.0, True
    PARTEDIT = "partedit", True, 0.5, 1.5, True
    DISABLED = "disabled", False, 0.5, 0.5, False

    # 让每一个枚举值，不只是一个字符串，而是一个“带属性的对象”
    def __new__(
        cls,
        strategy: str,
        enabled: bool,
        lower_binarize: float,
        upper_binarize: float,
        use_otsu: bool,
    ) -> "Binarization":
        obj = object.__new__(cls)
        obj._value_ = strategy
        obj.enabled = enabled   # 这里是给创建的对象绑定额外属性
        obj.lower_binarize = lower_binarize
        obj.upper_binarize = upper_binarize
        obj.use_otsu = use_otsu
        assert isinstance(obj.enabled, bool), "enabled should be of type bool"  # 类型检查，防止策略定义错误
        assert isinstance(obj.lower_binarize, float), "lower_binarize should be of type float"
        assert isinstance(obj.upper_binarize, float), "upper_binarize should be of type float"
        assert isinstance(obj.use_otsu, bool), "use_otsu should be of type bool"
        return obj

    # 重写等号
    def __eq__(self, other: Optional[Union[Binarization, str]] = None) -> bool:
        if not other:
            return False
        if isinstance(other, Binarization):
            return self.value.lower() == other.value.lower()
        if isinstance(other, str):
            return self.value.lower() == other.lower()

    @staticmethod
    def available_strategies() -> List[str]:    # 返回所有可用策略名称
        return [strategy.name for strategy in Binarization]

    def __str__(self) -> str:   # 打印使用
        return f"Binarization: {self.name} (Enabled: {self.enabled} Lower: {self.lower_binarize} Upper: {self.upper_binarize} Otsu: {self.use_otsu})"

    @staticmethod   # 把用户输入的字符串参数，转换成真正的 Binarization 策略对象
    def from_string(
        strategy: str,
        enabled: Optional[bool] = None,
        lower_binarize: Optional[bool] = None,
        upper_binarize: Optional[float] = None,
        use_otsu: Optional[bool] = None,
    ) -> Binarization:
        strategy = strategy.strip().lower() # 修改输入
        for _strategy in Binarization:  # 遍历所有策略
            if _strategy.name.lower() == strategy:  # 匹配
                if enabled is not None: # 允许修改策略参数
                    _strategy.enabled = enabled
                if lower_binarize is not None:
                    _strategy.lower_binarize = lower_binarize
                if upper_binarize is not None:
                    _strategy.upper_binarize = upper_binarize
                if use_otsu is not None:
                    _strategy.use_otsu = use_otsu
                return _strategy
        raise ValueError(f"binarization_strategy={strategy} not recognized")

# 当 PartEdit 往 prompt embeddings 中插入额外 token embedding 后，其余 token 位应该用什么方式填充，以及是否对填充结果做归一化 / 缩放。
class PaddingStrategy(Enum):
    # 默认
    BG = "BG", False, False
    # 其他添加只是为了实验
    context = "context", False, False
    EOS = "EoS", False, False
    ZERO = "zero", False, False
    SOT_E = "SoT_E", False, False

    def __new__(cls, strategy: str, norm: bool, scale: bool) -> "PaddingStrategy":
        obj = object.__new__(cls)
        obj._value_ = strategy
        obj.norm = norm
        obj.scale = scale
        return obj

    # 按value比较
    def __eq__(self, other: Optional[Union[PaddingStrategy, str]] = None) -> bool:
        if not other:
            return False
        if isinstance(other, PaddingStrategy):
            return self.value.lower() == other.value.lower()
        if isinstance(other, str):
            return self.value.lower() == other.lower()

    @staticmethod
    def available_strategies() -> List[str]:
        return [strategy.name for strategy in PaddingStrategy]

    def __str__(self) -> str:
        return f"PaddStrategy: {self.name} Norm: {self.norm} Scale: {self.scale}"

    @staticmethod
    def from_string(strategy_str, norm: Optional[bool] = False, scale: Optional[bool] = False) -> "PaddingStrategy":
        for strategy in PaddingStrategy:
            if strategy.name.lower() == strategy_str.lower():
                if norm is not None:
                    strategy.norm = norm
                if scale is not None:
                    strategy.scale = scale
                return strategy
        raise ValueError(f"padd_strategy={strategy} not recognized")

# 是一个“带默认值 + 自动解析 + 自动规范化”的配置容器，用来承载并规范 PartEdit 所有 extra_kwargs 参数。
class DotDictExtra(dict):
    """
    dot.notation access to dictionary attributes
    Holds default values for the extra_kwargs
    """
    # 点访问
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    _layers_to_use = LAYERS_TO_USE  # 训练参数，不直接暴露给用户，类级“隐藏配置参数”
    _enable_non_agg_storing = False  # 是否保存未聚合 attention，但非常占显存！~35GB无卸载14GB带卸载
    _cpu_offload = False  # 是否把 attention store 放到 CPU，降低VRAM，但大幅减速，隐藏
    _default = {    # 默认参数表
        "th_strategy": Binarization.PARTEDIT,
        "pad_strategy": PaddingStrategy.BG,
        "omega": 1.5,  # 值应该在0.25到2.0之间  
        "use_agg_store": False,
        "edit_mask": None,
        "edit_steps": 50, # 在这个时间步结束
        "start_editing_at": 0,  # 推荐，但是会在想要改变的时候暴露
        "use_layer_subset_idx": None,  # 以防我们想要使用特定的层, NOTE: 顺序不与Unet层对齐
        "add_extra_step": False,
        "batch_indx": -1,  # 最后一个
        "blend_layers": None,
        "force_cross_attn": False,  # 强迫交叉注意力到图
        # 优化部分
        "VRAM_low": True,  # 默认情况下保持开启状态，除非会导致错误
        "grounding": None,
    }
    _default_explanations = {   # 参数解释说明表（用于文档）
        "th_strategy": "Binarization strategy for attention maps",
        "pad_strategy": "Padding strategy for the added tokens",
        "omega": "Omega value for the PartEdit",
        "use_agg_store": "If the attention maps should be aggregated",
        "add_extra_step": "If extra 0 step should be added to the diffusion process",
        "edit_mask": "Mask for the edit when using ProvidedMask strategy",
        "edit_steps": "Number of edit steps",
        "start_editing_at": "Step at which the edit should start",
        "use_layer_subset_idx": "Sublayers to use, recommended 0-8 if really needed to use some",
        "VRAM_low": "Recommended to not change",
        "force_cross_attn": "Force cross attention to use OPT token maps",
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)   # 调用 dict 初始化
        for key, value in self._default.items():    # 注入默认参数
            if key not in self:
                self[key] = value

        print(">>> init DotDictExtra")
        # 对二值化，填充策略进行了额外的更改
        if isinstance(self["th_strategy"], str):    # 把字符串策略转成枚举
            self["th_strategy"] = Binarization.from_string(self["th_strategy"])
        if isinstance(self["pad_strategy"], str):
            self["pad_strategy"] = PaddingStrategy.from_string(self["pad_strategy"])
        self["edit_steps"] = self["edit_steps"] + self["add_extra_step"]    # 时间步对齐

        if self.edit_mask is not None :
            if isinstance(self.edit_mask, str):
                # 从PIL or torch/safetensors中加载  
                if self.edit_mask.endswith(".safetensors"):
                    self.edit_mask = load_file(self.edit_mask)["edit_mask"]
                elif self.edit_mask.endswith(".pt"):
                    self.edit_mask = torch.load(self.edit_mask)["edit_mask"]
                else:
                    self.edit_mask = Image.open(self.edit_mask)
            if isinstance(self.edit_mask, Image.Image):
                self.edit_mask = ToTensor()(self.edit_mask.convert("L"))
            elif isinstance(self.edit_mask, np.ndarray):
                self.edit_mask = torch.from_numpy(self.edit_mask).unsqueeze(0)
            if self.edit_mask.ndim == 2:
                self.edit_mask = self.edit_mask[None, None, ...]
            elif self.edit_mask.ndim == 3:
                self.edit_mask = self.edit_mask[None, ...]
            
            if self.edit_mask.max() > 1.0:# 归一化
                self.edit_mask = self.edit_mask / self.edit_mask.max()
        if self.grounding is not None: # same as above, but slightly different function
            if isinstance(self.grounding, Image.Image):
                self.grounding = ToTensor()(self.grounding.convert("L"))
            elif isinstance(self.grounding, np.ndarray):
                self.grounding = torch.from_numpy(self.grounding).unsqueeze(0)
            if self.grounding.ndim == 2:
                self.grounding = self.grounding[None, None, ...]
            elif self.grounding.ndim == 3:
                self.grounding = self.grounding[None, ...]
            if self.grounding.max() > 1.0:  
                self.grounding = self.grounding / self.grounding.max()

        assert isinstance(self.th_strategy, Binarization), "th_strategy should be of type Binarization"
        assert isinstance(self.pad_strategy, PaddingStrategy), "pad_strategy should be of type PaddingStrategy"

    def th_from_str(self, strategy: str):
        return Binarization.from_string(strategy)

    @staticmethod   # 返回参数说明字符串
    def explain() -> str:
        """Returns a string with all the explanations of the parameters"""
        return "\n".join(
            [
                f"{key}: {DotDictExtra._default_explanations[key]}"
                for key in DotDictExtra._default
                if DotDictExtra._default_explanations.get(key, "Recommended to not change") != "Recommended to not change"
            ]
        )

# 把 attention 从“token/flatten 形式”变成“空间特征图”，做插值，再还原回原来的 attention 形状。
def pack_interpolate_unpack(att, size, interpolation_mode, unwrap_last_dim=True, rewrap=False):
    has_last_dim = att.shape[-1] in [77, 1] # 77是CLIPtoken数，判断最后一维是不是“token 维”
    _last_dim = att.shape[-1]   # 记录 token 维大小
    if unwrap_last_dim: # 是否展开为二维空间，
        if has_last_dim:    # 有token维
            sq = int(att.shape[-2] ** 0.5)  # attention 是 flatten 的 HW。反推空间尺寸
            att = att.reshape(att.shape[0], sq, sq, -1).permute(0, 3, 1, 2)  # B x HW x D => B x D x H x W
        else:   # 没有token维，也同样反推。
            sq = int(att.shape[-1] ** 0.5)
            att = att.reshape(*att.shape[:-1], sq, sq)  # B x H x W
    att = att.unsqueeze(-3)  # 添加通道尺寸
    if att.shape[-2:] != size:  # 判断是否需要resize
        att, ps = einops.pack(att, "* c h w")   # 打包成任意 batch 维
        att = F.interpolate(    # 对 所有 token / channel 的 attention map，同时resize到目标空间。
            att,
            size=size,
            mode=interpolation_mode,
        )
        att = torch.stack(einops.unpack(att, ps, "* c h w"))    # 恢复原batch维
    if rewrap:  # 是否重新 wrap 回 token 形式
        if has_last_dim:
            att = att.reshape(att.shape[0], -1, att.shape[-1] * att.shape[-1], _last_dim)
        else:
            att = att.reshape(att.shape[0], -1, att.shape[-1] * att.shape[-1])
    # returns
    # rewrap True:
    # B x heads x D
    # B x heads X D x N
    # rewrap FALSE:
    # B x heads x H x W
    # B x N x heads X H x W x  if has_last_dim
    return att

# otsu阈值，这里定义otsu阈值
# 根据图像（或 attention map）的直方图，自适应计算一个能最大化前景/背景类间方差的全局阈值。
@torch.no_grad()
def threshold_otsu(image: torch.Tensor = None, nbins=256, hist=None):
    """Return threshold value based on Otsu's method using PyTorch.
    This is a reimplementation from scikit-image
    https://github.com/scikit-image/scikit-image/blob/b76ff13478a5123e4d8b422586aaa54c791f2604/skimage/filters/thresholding.py#L336

    Args:
    image: torch.Tensor
        Grayscale input image.
    nbins: int
        Number of bins used to calculate histogram.
    hist: torch.Tensor or tuple
        Histogram of the input image. If None, it will be calculated using the input image.
    Returns
    -------
    threshold : float
        Upper threshold value. All pixels with an intensity higher than
        this value are assumed to be foreground.
    """
    if image is not None and image.dim() > 2 and image.shape[-1] in (3, 4): # 检查是否是RGB类型的图片
        raise ValueError(f"threshold_otsu is expected to work correctly only for " f"grayscale images; image shape {image.shape} looks like " f"that of an RGB image.")
    # 在设备上将bbin转换为张量，将 nbins 放到同一设备
    nbins = torch.tensor(nbins, device=image.device)

    # 检查图像是否常量图像；如果不是，则返回该值
    if image is not None:
        first_pixel = image.view(-1)[0]
        if torch.all(image == first_pixel):
            return first_pixel.item()
    # counts：每个 bin 中的像素数量。bin_centers：每个 bin 的中心值
    counts, bin_centers = _validate_image_histogram(image, hist, nbins)

    # 所有可能阈值的类概率
    weight1 = torch.cumsum(counts, dim=0)
    weight2 = torch.cumsum(counts.flip(dims=[0]), dim=0).flip(dims=[0])
    # 所有可能阈值的类均值
    mean1 = torch.cumsum(counts * bin_centers, dim=0) / weight1
    mean2 = (torch.cumsum((counts * bin_centers).flip(dims=[0]), dim=0).flip(dims=[0])) / weight2

    # Clip ends to align class 1 and class 2 variables:
    # The last value of ``weight1``/``mean1`` should pair with zero values in
    # ``weight2``/``mean2``, which do not exist.
    variance12 = weight1[:-1] * weight2[1:] * (mean1[:-1] - mean2[1:]) ** 2 # 计算类间方差

    idx = torch.argmax(variance12)  # 选择最大类间方差的阈值
    threshold = bin_centers[idx]    # 将对应的灰度值就是

    return threshold.item()

# 校验并构造灰度直方图表示，确保 Otsu 阈值计算阶段始终拿到合法、规范的 counts 和 bin_centers。
def _validate_image_histogram(image: torch.Tensor, hist, nbins):
    """Helper function to validate and compute histogram if necessary."""
    if hist is not None:    # 判断是否传入了hist
        if isinstance(hist, tuple) and len(hist) == 2:  # 如果hist 是 (counts, bin_centers) 形式
            counts, bin_centers = hist
            if not (isinstance(counts, torch.Tensor) and isinstance(bin_centers, torch.Tensor)):
                counts = torch.tensor(counts)
                bin_centers = torch.tensor(bin_centers)
        else:
            counts = torch.tensor(hist)
            bin_centers = torch.linspace(0, 1, len(counts))
    else:# 如果没有传入hist，从image计算
        if image is None:
            raise ValueError("Either image or hist must be provided.")
        image = image.to(torch.float32)
        counts, bin_edges = histogram(image, nbins)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    return counts, bin_centers

# 用于在GPU上统计张量的直方图，返回每个bin的计数和bin边界，作为 Otsu 阈值等算法的输入。
def histogram(xs: torch.Tensor, bins):
    # Like torch.histogram, but works with cuda
    # https://github.com/pytorch/pytorch/issues/69519#issuecomment-1183866843
    min, max = xs.min(), xs.max()
    counts = torch.histc(xs, bins, min=min, max=max).to(xs.device)  # 使用 torch.histc 统计直方图
    boundaries = torch.linspace(min, max, bins + 1, device=xs.device)
    return counts, boundaries

# 把 UNet 各层、各位置、各 step 的 attention map 汇总成统一空间分辨率的 attention 张量
# Modification of the original from
# https://github.com/google/prompt-to-prompt/blob/9c472e44aa1b607da59fea94820f7be9480ec545/prompt-to-prompt_stable.ipynb
def aggregate_attention(
    attention_store: AttentionStore,
    res: int,
    batch_size: int,
    from_where: List[str],
    is_cross: bool,
    upsample_everything: int = None,
    return_all_layers: bool = False,
    use_same_layers_as_train: bool = False,
    train_layers: Optional[list[int]] = None,
    use_layer_subset_idx: list[int] = None,
    use_step_store: bool = False,
):
    out = []    # 初始化输出列表
    attention_maps = attention_store.get_average_attention(use_step_store)  # 从 AttentionStore 中取出 attention
    num_pixels = res**2 # 计算目标像素数
    for location in from_where: # 遍历Unet的位置（up,middle,down)
        for item in attention_maps[f"{location}_{'cross' if is_cross else 'self'}"]:    # 遍历该位置下的所有 attention 层

            if upsample_everything or (use_same_layers_as_train and is_cross):  # 是否需要插值（空间对齐）
                item = pack_interpolate_unpack(item, (res, res), "bilinear", rewrap=True)
            if item.shape[-2] == num_pixels:    # 只保留空间大小正确的 attention
                cross_maps = item.reshape(batch_size, -1, res, res, item.shape[-1])[None]   # 重排Attention维度
                out.append(cross_maps)
    _dim = 0    # 设定聚合维度
    if is_cross and use_same_layers_as_train and train_layers is not None:  # 训练层对齐（高级用法）
        out = [out[i] for i in train_layers]
        if use_layer_subset_idx is not None:  # 再次筛选 layer，只用特定深度的 attention
            out = [out[i] for i in use_layer_subset_idx]

    out = torch.cat(out, dim=_dim)  # 拼接所有 attention
    if return_all_layers:   # 是否返回所有层
        return out
    else:
        out = out.sum(_dim) / out.shape[_dim]   # 聚合所有层
    return out


def min_max_norm(a, _min=None, _max=None, eps=1e-6):
    _max = a.max() if _max is None else _max
    _min = a.min() if _min is None else _min
    return (a - _min) / (_max - _min + eps)


# Copied from https://github.com/RoyiRa/prompt-to-prompt-with-sdxl/blob/e579861f06962b697b37f3c6dd4813c2acdd55bd/processors.py#L209
# 在 latent 空间里，用 cross-attention 生成的空间 mask，只在“与指定词语相关的区域”应用编辑，其余区域保持原样。
class LocalBlend:
    def __call__(self, x_t, attention_store):   # 在 每个 diffusion step 中，用 attention 生成 mask 并应用到 latent
        print(">>> In LocalBlend")
        # 请注意，此代码在潜在层上工作！
        k = 1
        # maps = attention_store["down_cross"][2:4] + attention_store["up_cross"][:3]
        # 这些是数字，因为我们想要取256x256的图层，我认为这可以改变得更聪明一些...
        # 比如，把所有的注意力放在第二个身上。Attn_res [0] * self。Attn_res[1]在上下交叉。
        # NOTE(Alex): 这将需要激活注意力图的保存（更改DotDictExtra _enable_non_agg_storage)
        # NOTE(Alex): 另一种选择是像在其他示例中一样使用聚合掩码
        # 在下、中、上采样过程中，收集所有层的cross-attention，然后只保留满足空间分辨率
        maps = [m for m in attention_store["down_cross"] + attention_store["mid_cross"] + attention_store["up_cross"] if m.shape[1] == self.attn_res[0] * self.attn_res[1]]
        maps = [    # 重塑
            item.reshape(
                self.alpha_layers.shape[0],
                -1,
                1,
                self.attn_res[0],
                self.attn_res[1],
                self.max_num_words,
            )
            for item in maps
        ]
        maps = torch.cat(maps, dim=1)   # 拼接不同层的 attention
        maps = (maps * self.alpha_layers).sum(-1).mean(1)   # 只保留“被编辑词语”的 attention
        # 因为alpha_layers除了我们编辑的部分外都是0，所以product将除我们修改的部分外的所有内容归零。
        # 然后，将原始值和我们编辑的值相加。
        # 然后，我们取dim=1的平均值，这是层数.
        mask = F.max_pool2d(maps, (k * 2 + 1, k * 2 + 1), (1, 1), padding=(k, k))   # 局部膨胀（平滑 mask）
        mask = F.interpolate(mask, size=(x_t.shape[2:]))    # 插值到 latent 分辨率
        mask = mask / mask.max(2, keepdims=True)[0].max(3, keepdims=True)[0]    # 归一化
        mask = mask.gt(self.threshold)  # 二值化

        mask = mask[:1] + mask[1:]  # source + target mask 合并
        mask = mask.to(torch.float16)
        if mask.shape[0] < x_t.shape[0]:  # PartEdit 的 batch 对齐补丁
            # 再次连接最后一个掩码
            mask = torch.cat([mask, mask[-1:]], dim=0)

        # ## NOTE(Alex): 这是与掩码的局部混合
        # assert isinstance(attention_store, AttentionStore), "AttentionStore expected"
        # cur_res = x_t.shape[-1]

        # if attention_store.th_strategy == Binarization.PROVIDED_MASK:
        #     mask = attention_store.edit_mask.to(x_t.device)
        #     # resize to res
        #     mask = F.interpolate(
        #         mask, (cur_res, cur_res), mode="bilinear"
        #     ) # ).reshape(1, -1, 1)
        # else:
        #     mask =  attention_store.get_maps_agg(
        #         res=cur_res,
        #         device=x_t.device,
        #         use_agg_store=attention_store.use_agg_store,  # Agg is across time, Step is last step without time agg
        #         keepshape=True
        #     )  # provide in cross_attention_kwargs in pipeline
        # x_t[1:] = mask * x_t[1:] + (1 - mask) * x_t[0]
        # ## END NOTE(Alex): this is local blending with the mask

        x_t = x_t[:1] + mask * (x_t - x_t[:1])
        # 代码对原始图像和每个生成的图像之间的图像差异应用掩码，有效地只保留所需的区域
        return x_t

    # NOTE(Alex): 复制到LocalBlend
    def __init__(
        self,
        prompts: List[str],
        words: List[List[str]],
        tokenizer,
        device,
        threshold=0.3,
        attn_res=None,
    ):
        print(">>> init LocalBlend")
        self.max_num_words = 77 # token固定长度=77
        self.attn_res = attn_res

        alpha_layers = torch.zeros(len(prompts), 1, 1, 1, 1, self.max_num_words)
        for i, (prompt, words_) in enumerate(zip(prompts, words)):  # 标记需要编辑的 token
            if isinstance(words_, str):
                words_ = [words_]
            for word in words_:
                ind = get_word_inds(prompt, word, tokenizer)
                alpha_layers[i, :, :, :, :, ind] = 1
        self.alpha_layers = alpha_layers.to(device)  # 一个单热向量，其中1是我们修改的单词（源和目标）
        self.threshold = threshold


# Copied from https://github.com/RoyiRa/prompt-to-prompt-with-sdxl/blob/e579861f06962b697b37f3c6dd4813c2acdd55bd/processors.py#L129
# 定义了一个注意力控制抽象基类，对 attention map 进行干预
class AttentionControl(abc.ABC):
    def step_callback(self, x_t):   # 扩散后的回调
        return x_t

    def between_steps(self):
        return

    @property
    def num_uncond_att_layers(self):    # 前多少层 attention 是 无条件分支
        return 0

    @abc.abstractmethod
    def forward(self, attn, is_cross: bool, place_in_unet: str, store: bool = True):    # 控制注意力的地方
        raise NotImplementedError

    def __call__(self, attn, is_cross: bool, place_in_unet: str, store: bool = True):
        # print(">>> In AttentionControl")
        if self.cur_att_layer >= self.num_uncond_att_layers:    # 是否当前层 >= uncond层数
            h = attn.shape[0]   # 获取batch size
            attn[h // 2 :] = self.forward(attn[h // 2 :], is_cross, place_in_unet, store)   # 只对 conditional 部分做编辑
        self.cur_att_layer += 1 # 当前层数+1
        if self.cur_att_layer == self.num_att_layers + self.num_uncond_att_layers:  # 是否当前 step 的所有 attention 层已经走完
            self.cur_att_layer = 0
            self.cur_step += 1
            self.between_steps()
        return attn

    def reset(self):    # 初始化
        self.cur_step = 0
        self.cur_att_layer = 0
        self.allow_edit_control = True

    def __init__(self, attn_res=None, extra_kwargs: DotDictExtra = None):   # 初始化参数
        # PartEdit
        print(">>> init AttentionControl")
        self.extra_kwargs = extra_kwargs
        self.index_inside_batch = extra_kwargs.get("index_inside_batch", 1) # 默认值是我们之前设置的1!
        if not isinstance(self.index_inside_batch, list):
            self.index_inside_batch = [self.index_inside_batch]
        self.layers_to_use = extra_kwargs.get("_layers_to_use", LAYERS_TO_USE)  # Training parameter, not exposed directly
        # Params
        self.th_strategy: Binarization = extra_kwargs.get("th_strategy", Binarization.P2P)
        self.pad_strategy: PaddingStrategy = extra_kwargs.get("pad_strategy", PaddingStrategy.BG)
        self.omega: float = extra_kwargs.get("omega", 1.0)
        self.use_agg_store: bool = extra_kwargs.get("use_agg_store", False)
        self.edit_mask: Optional[torch.Tensor] = extra_kwargs.get("edit_mask", None)  # edit_mask_t
        self.edit_steps: int = extra_kwargs.get("edit_steps", 50) # NOTE(Alex): This is the end step, IMPORTANT
        self.blend_layers: Optional[List] = None
        self.start_editing_at: int = extra_kwargs.get("start_editing_at", 0)
        self.use_layer_subset_idx: Optional[list[int]] = extra_kwargs.get("use_layer_subset_idx", None)
        self.batch_indx: int = extra_kwargs.get("batch_indx", 0)
        self.VRAM_low: bool = extra_kwargs.get("VRAM_low", False)
        self.allow_edit_control = True
        # Old
        self.cur_step: int = 0
        self.num_att_layers: int = -1
        self.cur_att_layer: int = 0
        self.attn_res: int = attn_res

    def get_maps_agg(self, resized_res, device):    # 用于返回聚合 attention map，默认未实现
        return None

    def _editing_allowed(self): # 当前时间步是否允许编辑
        return self.allow_edit_control  # TODO(Alex): Maybe make this only param, instead of unregister attn control?


# Copied from https://github.com/RoyiRa/prompt-to-prompt-with-sdxl/blob/e579861f06962b697b37f3c6dd4813c2acdd55bd/processors.py#L166
class EmptyControl(AttentionControl):   # 对上面的AttentionControl进行简单实现，提供一个不改变 attention 的基类实现
    def forward(self, attn, is_cross: bool, place_in_unet: str, store:bool = True):
        return attn


# Modified from https://github.com/RoyiRa/prompt-to-prompt-with-sdxl/blob/e579861f06962b697b37f3c6dd4813c2acdd55bd/processors.py#L171
# 保存Attention，生成空间mask
class AttentionStore(AttentionControl):
    @staticmethod
    def get_empty_store():  # 初始化容器，存储上中下采样过程中的自注意力和交叉注意力，优化的注意力，和背景注意力
        return {
            "down_cross": [],
            "mid_cross": [],
            "up_cross": [],
            "down_self": [],
            "mid_self": [],
            "up_self": [],
            "opt_cross": [],
            "opt_bg_cross": [],
        }

    def maybe_offload(self, attn_device, attn_dtype):   # 显存优化
        if self.extra_kwargs.get("_cpu_offload", False):    # 如果开启，把 attention 从 GPU → CPU，并转成 float32（稳定）
            attn_device, attn_dtype = torch.device("cpu"), torch.float32
        return attn_device, attn_dtype

    def forward(self, attn, is_cross: bool, place_in_unet: str, store: bool = True):
        key = f"{place_in_unet}_{'cross' if is_cross else 'self'}"  # 创建key
        _device, _dtype = self.maybe_offload(attn.device, attn.dtype)   # 存储设备
        if store and self.batch_indx is not None and is_cross:  # 提取并存储attention
            # We always store for our method
            _dim = attn.shape[0] // self.num_prompt # 计算每个prompt对应的chunk，↓选取第batch_indx个样本，取所有空间位置，选指定token，聚合，转设备
            _val = attn[_dim * self.batch_indx : _dim * (self.batch_indx + 1), ..., self.index_inside_batch].sum(0, keepdim=True).to(_device, _dtype)
            if _val.shape[-1] != 1: # 归一化+token聚合
                # min_max each -1 seperately
                _max = _val.max()
                for i in range(_val.shape[-1]): # 对每个token单独归一化
                    _val[..., i] = min_max_norm(_val[..., i], _max=_max)
                _val = _val.sum(-1, keepdim=True)   # 跨token求和
            self.step_store["opt_cross"].append(_val)   # 存入opt_cross
        if self.extra_kwargs.get("_enable_non_agg_storing", False) and store:   # 可选存原始attention
            _attn = attn.clone().detach().to(_device, _dtype, non_blocking=True)
            if attn.shape[1] <= 32**2:  # avoid memory overhead
                self.step_store[key].append(_attn)
        return attn

    def offload_stores(self, device):   # 显存释放
        """Created for low VRAM usage, where we want to do this before Decoder"""
        for key in self.step_store:
            self.step_store[key] = [a.to(device) for a in self.step_store[key]]
        for key in self.attention_store:
            self.attention_store[key] = [a.to(device) for a in self.attention_store[key]]
        torch.cuda.empty_cache()

    @torch.no_grad()
    def calculate_mask_t_res(self, use_step_store: bool = False):   # 将注意力转换成mask
        mask_t_res = aggregate_attention(   # 输入opt_cross，多层注意力，输出[H,W]
            self,
            res=1024,
            from_where=["opt"],
            batch_size=1,
            is_cross=True,
            upsample_everything=False,
            return_all_layers=False, # Removed sum in this function
            use_same_layers_as_train=True,
            train_layers=self.layers_to_use,
            use_step_store=use_step_store,
            use_layer_subset_idx=self.use_layer_subset_idx,
        )[..., 0]   # 取单通道

        strategy: Binarization = self.th_strategy   # 获取阈值

        mask_t_res = min_max_norm(mask_t_res)   # 归一化

        upper_threshold = strategy.upper_binarize
        lower_threshold = strategy.lower_binarize
        use_otsu = strategy.use_otsu
        tt = threshold_otsu(mask_t_res)  # Otsu阈值
        if not hasattr(self, "last_otsu") or self.last_otsu == []:  # 判断是否保存历史阈值
            self.last_otsu = [tt]
        else:
            self.last_otsu.append(tt)
        if use_otsu:    # 根据策略调整阈值
            upper_threshold, lower_threshold = (
                tt * upper_threshold,
                tt * lower_threshold,
            )

        if strategy == Binarization.PARTEDIT:   # 特殊策略
            upper_threshold = self.omega * tt  # Assuming we are not chaning upper in PartEdit

        if strategy in [Binarization.P2P, Binarization.PROVIDED_MASK]:  # 不二值化情况
            return mask_t_res

        mask_t_res[mask_t_res < lower_threshold] = 0    # 掩码二值化
        mask_t_res[mask_t_res >= upper_threshold] = 1.0

        return mask_t_res

    def has_maps(self) -> bool:
        return len(self.mask_storage_step) > 0 or len(self.mask_storage_agg) > 0

    def _store_agg_map(self) -> None:   # 存mask
        if self.use_agg_store:
            self.mask_storage_agg[self.cur_step] = self.calculate_mask_t_res().cpu()    # 跨步累计
        else:
            self.mask_storage_step[self.cur_step] = self.calculate_mask_t_res(True).cpu()   # 每步单独

    def between_steps(self):
        no_items = len(self.attention_store) == 0   # 是否是第一次运行
        if no_items:
            self.attention_store = self.step_store  # 初始化
        else:
            for key in self.attention_store:# 跨step累加
                for i in range(len(self.attention_store[key])):
                    self.attention_store[key][i] += self.step_store[key][i]

        self._store_agg_map()   # 存mask
        if not no_items:
            # only in this case, otherwise we are just assigning it
            for key in self.step_store:
                # Clear the list while maintaining the dictionary structure
                del self.step_store[key][:] # 清空step_store

        self.step_store = self.get_empty_store()    # 重建空store

    def get_maps_agg(self, res, device, use_agg_store: bool = None, keepshape: bool = False):   # 获取mask
        if use_agg_store is None:   # 确定使用的哪种store
            use_agg_store = self.use_agg_store
        _store = self.mask_storage_agg if use_agg_store else self.mask_storage_step # 确定mask存储来源
        last_idx = sorted(_store.keys())[-1]    # 取最后一个step
        mask_t_res = _store[last_idx].to(device)  # Should be 1 1 H W   # 取出对应的mask
        mask_t_res = F.interpolate(mask_t_res, (res, res), mode="bilinear") # 上采样到分辨率
        if not keepshape:   # 是否reshape
            mask_t_res = mask_t_res.reshape(1, -1, 1)
        return mask_t_res

    def visualize_maps_agg(self, use_agg_store: bool, make_grid_kwargs: dict = None):   # 三个都是将mask转成图片
        _store = self.mask_storage_agg if use_agg_store else self.mask_storage_step
        if make_grid_kwargs is None:
            make_grid_kwargs = {"nrow": 10}
        return ToPILImage()(make_grid(torch.cat(list(_store.values())), **make_grid_kwargs))

    def visualize_one_map(self, use_agg_store: bool, idx: int):
        _store = self.mask_storage_agg if use_agg_store else self.mask_storage_step
        return ToPILImage()(_store[idx])

    def visualize_final_map(self, use_agg_store: bool):
        """This method returns the agg non-binarized attn map of the whole process

        Args:
            use_agg_store (bool): If True, it will return the agg store, otherwise the step store

        Returns:
            [PIL.Image]: The non-binarized attention map
        """
        _store = self.mask_storage_agg if use_agg_store else self.mask_storage_step
        return ToPILImage()(torch.cat(list(_store.values())).mean(0))

    # 对存储的attention按step做平均
    def get_average_attention(self, step: bool = False):
        _store = self.attention_store if not step else self.step_store
        average_attention = {key: [item / self.cur_step for item in _store[key]] for key in _store}
        return average_attention

    def reset(self):    # 清空运行状态
        super(AttentionStore, self).reset()
        for key in self.step_store:
            del self.step_store[key][:]
        for key in self.attention_store:
            del self.attention_store[key][:]
        self.step_store = self.get_empty_store()
        self.attention_store = {}
        self.last_otsu = []

    def __init__(   # 初始化整个数据结构
        self,
        num_prompt: int,
        attn_res=None,
        extra_kwargs: DotDictExtra = None,
    ):
        super(AttentionStore, self).__init__(attn_res, extra_kwargs)

        print(">>> init AttentionStore")
        self.num_prompt = num_prompt
        self.mask_storage_step = {}
        self.mask_storage_agg = {}
        if self.batch_indx is not None:
            assert num_prompt > 0, "num_prompt must be greater than 0 if batch_indx is not None"
        self.step_store = self.get_empty_store()
        self.attention_store = {}
        self.last_otsu = []


# Copied from https://github.com/RoyiRa/prompt-to-prompt-with-sdxl/blob/e579861f06962b697b37f3c6dd4813c2acdd55bd/processors.py#L246
class AttentionControlEdit(AttentionStore, abc.ABC):
    def step_callback(self, x_t):   # 回调使用，这里的x_t是传入的latents
        if self.local_blend is not None:    # 是否进行局部融合
            # x_t = self.local_blend(x_t, self.attention_store) # TODO: Check if there is more memory efficient way
            x_t = self.local_blend(x_t, self)   # 查看，这两个参数
        return x_t

    def replace_self_attention(self, attn_base, att_replace):   # 自注意力替换策略
        if att_replace.shape[2] <= self.attn_res[0] ** 2:   # 判断是否是低分辨率层
            return attn_base.unsqueeze(0).expand(att_replace.shape[0], *attn_base.shape)
        else:
            return att_replace

    @abc.abstractmethod # 抽象函数是什么？
    def replace_cross_attention(self, attn_base, att_replace):
        raise NotImplementedError

    def forward(self, attn, is_cross: bool, place_in_unet: str, store: bool = True):
        super(AttentionControlEdit, self).forward(attn, is_cross, place_in_unet, store) # 先执行父类
        if is_cross or (self.num_self_replace[0] <= self.cur_step < self.num_self_replace[1]):  # 判断是否需要编辑
            h = attn.shape[0] // (self.batch_size)  # 计算每个prompt的块大小
            try:
                attn = attn.reshape(self.batch_size, h, *attn.shape[1:])    # 重塑attn的维度
            except RuntimeError as e:   # 如果重置失败就打印
                logger.error(f"Batch size: {self.batch_size}, h: {h}, attn.shape: {attn.shape}")
                raise e

            attn_base, attn_replace = attn[0], attn[1:] # 分离base和replace
            if is_cross:    # 交叉注意力分支
                alpha_words = self.cross_replace_alpha[self.cur_step].to(attn_base.device)  # 获取alpha
                attn_replace_new = self.replace_cross_attention(attn_base, attn_replace) * alpha_words + (1 - alpha_words) * attn_replace   # 替换
                

                attn[1:] = attn_replace_new
                if self.has_maps() and self.extra_kwargs.get("force_cross_attn", False):  # 强制mask控制
                    mask_t_res = self.get_maps_agg( # 获取mask
                        res=int(attn_base.shape[1] ** 0.5),
                        device=attn_base.device,
                        use_agg_store=self.use_agg_store,  # Agg is across time, Step is last step without time agg
                        keepshape=False,
                    ).repeat(h, 1, 1)   # 扩展到head数
                    zero_index = torch.argmax(torch.eq(self.cross_replace_alpha[0], 0).to(mask_t_res.dtype)).item() # 找到被替换的token
                    # zero_index = torch.eq(self.cross_replace_alpha[0].flatten(), 0)
                    mean_curr = attn[1:2, ..., zero_index].mean()   # 当前attention均值
                    ratio_to_mean = mean_curr / mask_t_res[..., 0].mean()   # mask均值
                    # print(f'{ratio_to_mean=}')
                    extra_mask = torch.where(mask_t_res[..., 0] > self.last_otsu[-1], ratio_to_mean * 2, 0.5)   # 构建额外mask

                    attn[1:2, ..., zero_index : zero_index + 1] += mask_t_res[None] * extra_mask[None, ..., None]  # 加到attention上
                    # attn[1:2, ..., zero_index] = (mask_t_res[..., 0][None] > self.last_otsu[-1] * 1.5).to(mask_t_res.dtype) * mean_curr
            else:   # 自注意分支
                attn[1:] = self.replace_self_attention(attn_base, attn_replace)
            attn = attn.reshape(self.batch_size * h, *attn.shape[2:])   # reshape到原始形状
        return attn

    def __init__(
        self,
        prompts: list[str],
        num_steps: int,
        cross_replace_steps: Union[float, Tuple[float, float], Dict[str, Tuple[float, float]]],
        self_replace_steps: Union[float, Tuple[float, float]],
        local_blend: Optional[LocalBlend],
        tokenizer,
        device: torch.device,
        attn_res=None,
        extra_kwargs: DotDictExtra = None,
    ):
        super(AttentionControlEdit, self).__init__(
            attn_res=attn_res,
            num_prompt=len(prompts),
            extra_kwargs=extra_kwargs,
        )
        # 在这里添加分词器和设备
        print(">>> init AttentionControlEdit")
        self.tokenizer = tokenizer
        self.device = device

        self.batch_size = len(prompts)
        self.cross_replace_alpha = get_time_words_attention_alpha(prompts, num_steps, cross_replace_steps, self.tokenizer).to(self.device)
        if isinstance(self_replace_steps, float):
            self_replace_steps = 0, self_replace_steps
        self.num_self_replace = int(num_steps * self_replace_steps[0]), int(num_steps * self_replace_steps[1])
        self.local_blend = local_blend


# Copied from https://github.com/RoyiRa/prompt-to-prompt-with-sdxl/blob/e579861f06962b697b37f3c6dd4813c2acdd55bd/processors.py#L307
# 具体实现的编辑控制器，用“token映射（mapper）”来替换 cross-attention
class AttentionReplace(AttentionControlEdit):
    def replace_cross_attention(self, attn_base, att_replace):  # 对替换具体实现
        return torch.einsum("hpw,bwn->bhpn", attn_base, self.mapper.to(attn_base.device))

    def __init__(
        self,
        prompts,
        num_steps: int,
        cross_replace_steps: float,
        self_replace_steps: float,
        local_blend: Optional[LocalBlend] = None,   # localblend可以是Localblend实例，也可以是None
        tokenizer=None,
        device=None,
        attn_res=None, 
        extra_kwargs: DotDictExtra = None,
    ):
        super(AttentionReplace, self).__init__(
            prompts,
            num_steps,
            cross_replace_steps,
            self_replace_steps,
            local_blend,
            tokenizer,
            device,
            attn_res,
            extra_kwargs,
        )
        print(">>> init AttentionReplace")
        self.mapper = get_replacement_mapper(prompts, self.tokenizer).to(self.device)


# Copied from https://github.com/RoyiRa/prompt-to-prompt-with-sdxl/blob/e579861f06962b697b37f3c6dd4813c2acdd55bd/processors.py#L328
class AttentionRefine(AttentionControlEdit):
    def replace_cross_attention(self, attn_base, att_replace):  # 父类抽象函数的具体实现
        attn_base_replace = attn_base[:, :, self.mapper].permute(2, 0, 1, 3)    # 使用mapper重排token，按照维度顺序
        attn_replace = attn_base_replace * self.alphas + att_replace * (1 - self.alphas)    # 融合
        return attn_replace

    def __init__(
        self,
        prompts,
        num_steps: int,
        cross_replace_steps: float,
        self_replace_steps: float,
        local_blend: Optional[LocalBlend] = None,
        tokenizer=None,
        device=None,
        attn_res=None,
        extra_kwargs: DotDictExtra = None,
    ):
        super(AttentionRefine, self).__init__(
            prompts,
            num_steps,
            cross_replace_steps,
            self_replace_steps,
            local_blend,
            tokenizer,
            device,
            attn_res,
            extra_kwargs,
        )
        print(">>> init AttentionRefine")
        self.mapper, alphas = get_refinement_mapper(prompts, self.tokenizer)
        self.mapper, alphas = self.mapper.to(self.device), alphas.to(self.device)
        self.alphas = alphas.reshape(alphas.shape[0], 1, 1, alphas.shape[1])


# Copied from https://github.com/RoyiRa/prompt-to-prompt-with-sdxl/blob/e579861f06962b697b37f3c6dd4813c2acdd55bd/processors.py#L353
# 对token的attention加权，
class AttentionReweight(AttentionControlEdit):
    def replace_cross_attention(self, attn_base: torch.Tensor, att_replace: torch.Tensor):
        if self.prev_controller is not None:    # 是否调用签一个controller，链式控制
            attn_base = self.prev_controller.replace_cross_attention(attn_base, att_replace)
        attn_replace = attn_base[None, :, :, :] * self.equalizer[:, None, None, :]  # 加权
        return attn_replace

    def __init__(
        self,
        prompts: list[str],
        num_steps: int,
        cross_replace_steps: float,
        self_replace_steps: float,
        equalizer,
        local_blend: Optional[LocalBlend] = None,
        controller: Optional[AttentionControlEdit] = None,
        tokenizer=None,
        device=None,
        attn_res=None,
        extra_kwargs: DotDictExtra = None,
    ):
        super(AttentionReweight, self).__init__(
            prompts,
            num_steps,
            cross_replace_steps,
            self_replace_steps,
            local_blend,
            tokenizer,
            device,
            attn_res,
            extra_kwargs,
        )
        print(">>> init AttentionReweight")
        self.equalizer = equalizer.to(self.device)  # 设置每个prompt的每个token权重 
        self.prev_controller = controller   # 保存前一个controller


# 连接 UNet attention 与 controller
class PartEditCrossAttnProcessor:
    # Modified from https://github.com/RoyiRa/prompt-to-prompt-with-sdxl/blob/e579861f06962b697b37f3c6dd4813c2acdd55bd/processors.py#L11
    def __init__(   # 初始化
        self,
        controller: AttentionStore,
        place_in_unet,
        store_this_layer: bool = True,
    ):
        super().__init__()
        # print(">>> init PartEditCrossAttnProcessor")
        self.controller = controller
        assert issubclass(type(controller), AttentionControl), f"{controller} isn't subclass of AttentionControl"   # 检查controller来源
        self.place_in_unet = place_in_unet  # 保存层位置
        self.store_this_layer = store_this_layer    # 是否存储

    def has_maps(self) -> bool: # 检查是否有mask，(跨step生成，聚合mask，外部提供mask)
        return len(self.controller.mask_storage_step) > 0 or len(self.controller.mask_storage_agg) > 0 or self.controller.edit_mask is not None

    def condition_for_editing(self) -> bool:    # 是否允许编辑
        # If we have a given mask
        # If we are using PartEdit
        return self.controller.th_strategy.enabled

    def __call__(   # UNet attention layer的forward hook
        self,
        attn: Attention,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
    ):
        # print(">>> In PartEditCrossAttnProcessor")
        batch_size, sequence_length, _ = hidden_states.shape    # shape处理
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)   # 标准处理mask

        query = attn.to_q(hidden_states)    # 计算Q

        is_cross = encoder_hidden_states is not None    # 是否是交叉注意力
        encoder_hidden_states = encoder_hidden_states if encoder_hidden_states is not None else hidden_states
        key = attn.to_k(encoder_hidden_states)  # K和V
        value = attn.to_v(encoder_hidden_states)

        # initial_condition = hasattr(self, "controller") and hasattr(self.controller, "batch_indx") and batch_size > self.controller.batch_size

        if hasattr(self, "controller") and self.controller._editing_allowed() and self.controller.batch_indx > 0:   # batch对齐
            # Set the negative/positive of the batch index to the zero image
            batch_indx = self.controller.batch_indx
            _bs = self.controller.batch_size    # prompt数量
            query[[batch_indx, batch_indx + _bs]] = query[[0, _bs]] # 把编辑样本的Q替换为原始样本的Q
            # value[[batch_indx, batch_indx+_bs]] = value[[0, _bs]]

        query = attn.head_to_batch_dim(query)   # 将这是三个reshape到multi-head格式
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        attention_probs = attn.get_attention_scores(query, key, attention_mask) # 计算attention

        self.controller(attention_probs, is_cross, self.place_in_unet, self.store_this_layer)   # 调用controller

        hidden_states = torch.bmm(attention_probs, value)   # attention * v
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)   # 线性投影
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        res = int(np.sqrt(hidden_states.shape[1]))  # 计算空间尺寸

        should_edit = ( # 是否执行Partedit
            hasattr(self, "controller")
            and self.controller._editing_allowed()  # allow_edit_control
            and self.has_maps() 
            and self.condition_for_editing()
            and self.controller.cur_step > self.controller.start_editing_at
            and self.controller.cur_step < self.controller.edit_steps
        )
        # print("if inter Partedit?")
        if should_edit: # 进入核心
            # print("inter Partedit ")
            if self.controller.th_strategy == Binarization.PROVIDED_MASK:   # 如果用户提供mask
                mask_t_res = self.controller.edit_mask.to(hidden_states.device)
                # resize to reshape
                mask_t_res = F.interpolate(mask_t_res, (res, res), mode="bilinear").reshape(1, -1, 1)
            else:   # attention生成mask
                mask_t_res = self.controller.get_maps_agg(
                    res=res,
                    device=hidden_states.device,
                    use_agg_store=self.controller.use_agg_store,  # Agg is across time, Step is last step without time agg
                )  # 在cross_attention_kwargs管道中提供
                # Note: Additional blending with grounding
                _extra_grounding = self.controller.extra_kwargs.get("grounding", None)
                if _extra_grounding is not None:
                    mask_t_res = mask_t_res * F.interpolate(_extra_grounding, (res, res), mode="bilinear").reshape(1, -1, 1).to(hidden_states.device)

            # 核心融合
            b1_u = 0
            b1_c = self.controller.batch_size
            b2_u = 1
            b2_c = self.controller.batch_size + 1
            hidden_states[b2_u] = (1 - mask_t_res) * hidden_states[b1_u] + mask_t_res * hidden_states[b2_u]
            hidden_states[b2_c] = (1 - mask_t_res) * hidden_states[b1_c] + mask_t_res * hidden_states[b2_c]
            # mask区域使用编辑结果，非mask区域用原始结果

        return hidden_states


# Adapted from https://github.com/RoyiRa/prompt-to-prompt-with-sdxl/blob/e579861f06962b697b37f3c6dd4813c2acdd55bd/processors.py#L48
def create_controller(
    prompts: List[str],
    cross_attention_kwargs: Dict,
    num_inference_steps: int,
    tokenizer,
    device: torch.device,
    attn_res: Tuple[int, int],
    extra_kwargs: dict,
) -> AttentionControl:
    edit_type = cross_attention_kwargs.get("edit_type", "replace")  # 从字典里取编辑类型
    local_blend_words = cross_attention_kwargs.get("local_blend_words") # 是否局部编辑
    equalizer_words = cross_attention_kwargs.get("equalizer_words") # 
    equalizer_strengths = cross_attention_kwargs.get("equalizer_strengths")
    n_cross_replace = cross_attention_kwargs.get("n_cross_replace", 0.4)    # 注意力替换比例
    n_self_replace = cross_attention_kwargs.get("n_self_replace", 0.4)
    print("local_blend_words is ",local_blend_words)
    print("cross_attention_kwargs is ", cross_attention_kwargs)
    print ("Whatever use LB?")

    # 纯替换,用的是这个？
    if edit_type == "replace" and local_blend_words is None:
        print("no")
        return AttentionReplace(    # 返回一个替换的实例
            prompts,
            num_inference_steps,
            n_cross_replace,
            n_self_replace,
            tokenizer=tokenizer,
            device=device,
            attn_res=attn_res,
            extra_kwargs=extra_kwargs,
        )

    # 局部替换，使用的是这个分支
    if edit_type == "replace" and local_blend_words is not None:
        print("yes")
        lb = LocalBlend(
            prompts,
            local_blend_words,
            tokenizer=tokenizer,
            device=device,
            attn_res=attn_res,
        )
        return AttentionReplace(
            prompts,
            num_inference_steps,
            n_cross_replace,
            n_self_replace,
            lb,
            tokenizer=tokenizer,
            device=device,
            attn_res=attn_res,
            extra_kwargs=extra_kwargs,
        )

    # 仅细化
    if edit_type == "refine" and local_blend_words is None:
        return AttentionRefine(
            prompts,
            num_inference_steps,
            n_cross_replace,
            n_self_replace,
            tokenizer=tokenizer,
            device=device,
            attn_res=attn_res,
            extra_kwargs=extra_kwargs,
        )

    # 局部+渐进细化
    if edit_type == "refine" and local_blend_words is not None:
        lb = LocalBlend(
            prompts,
            local_blend_words,
            tokenizer=tokenizer,
            device=device,
            attn_res=attn_res,
        )
        return AttentionRefine(
            prompts,
            num_inference_steps,
            n_cross_replace,
            n_self_replace,
            lb,
            tokenizer=tokenizer,
            device=device,
            attn_res=attn_res,
            extra_kwargs=extra_kwargs,
        )

    # 权重调整
    if edit_type == "reweight" and local_blend_words is None:
        assert equalizer_words is not None and equalizer_strengths is not None, "To use reweight edit, please specify equalizer_words and equalizer_strengths."
        assert len(equalizer_words) == len(equalizer_strengths), "equalizer_words and equalizer_strengths must be of same length."
        equalizer = get_equalizer(prompts[1], equalizer_words, equalizer_strengths, tokenizer=tokenizer)
        return AttentionReweight(
            prompts,
            num_inference_steps,
            n_cross_replace,
            n_self_replace,
            tokenizer=tokenizer,
            device=device,
            equalizer=equalizer,
            attn_res=attn_res,
            extra_kwargs=extra_kwargs,
        )

    # 调整权重并限制区域
    if edit_type == "reweight" and local_blend_words:
        assert equalizer_words is not None and equalizer_strengths is not None, "To use reweight edit, please specify equalizer_words and equalizer_strengths."
        assert len(equalizer_words) == len(equalizer_strengths), "equalizer_words and equalizer_strengths must be of same length."
        equalizer = get_equalizer(prompts[1], equalizer_words, equalizer_strengths, tokenizer=tokenizer)
        lb = LocalBlend(
            prompts,
            local_blend_words,
            tokenizer=tokenizer,
            device=device,
            attn_res=attn_res,
        )
        return AttentionReweight(
            prompts,
            num_inference_steps,
            n_cross_replace,
            n_self_replace,
            tokenizer=tokenizer,
            device=device,
            equalizer=equalizer,
            attn_res=attn_res,
            local_blend=lb,
            extra_kwargs=extra_kwargs,
        )

    raise ValueError(f"Edit type {edit_type} not recognized. Use one of: replace, refine, reweight.")


# Copied from https://github.com/RoyiRa/prompt-to-prompt-with-sdxl/blob/e579861f06962b697b37f3c6dd4813c2acdd55bd/processors.py#L380-L596
### util functions for all Edits

# 在“时间维度（扩散步）+ 单词维度”上，控制某些词的 attention 何时生效
def update_alpha_time_word(
    alpha,
    bounds: Union[float, Tuple[float, float]],
    prompt_ind: int,
    word_inds: Optional[torch.Tensor] = None,
):
    if isinstance(bounds, float):   # 判断 bounds 是否是 float
        bounds = 0, bounds
    start, end = int(bounds[0] * alpha.shape[0]), int(bounds[1] * alpha.shape[0])   # 计算时间步范围
    if word_inds is None:   # 如果没有指定词，那就使用全部词
        word_inds = torch.arange(alpha.shape[2])
    alpha[:start, prompt_ind, word_inds] = 0
    alpha[start:end, prompt_ind, word_inds] = 1 # 前部分关闭，中间打开，后部分关闭
    alpha[end:, prompt_ind, word_inds] = 0
    return alpha

# 为所有 prompt + 所有词，构建一个完整的“时间-词 attention 控制表（alpha）
def get_time_words_attention_alpha(
    prompts,
    num_steps,
    cross_replace_steps: Union[float, Dict[str, Tuple[float, float]]],
    tokenizer,
    max_num_words=77,
):
    if not isinstance(cross_replace_steps, dict):   # 如果不是 dict → 转成 dict
        cross_replace_steps = {"default_": cross_replace_steps}
    if "default_" not in cross_replace_steps:   # 如果没有default，补一个
        cross_replace_steps["default_"] = (0.0, 1.0)
    alpha_time_words = torch.zeros(num_steps + 1, len(prompts) - 1, max_num_words)  # 初始化alpha_tensor
    for i in range(len(prompts) - 1):   # 第一轮：给所有词应用 default 规则
        alpha_time_words = update_alpha_time_word(alpha_time_words, cross_replace_steps["default_"], i)
    for key, item in cross_replace_steps.items():   # 第二轮：对特定词进行覆盖（override）
        if key != "default_":
            inds = [get_word_inds(prompts[i], key, tokenizer) for i in range(1, len(prompts))]
            for i, ind in enumerate(inds):
                if len(ind) > 0:
                    alpha_time_words = update_alpha_time_word(alpha_time_words, item, i, ind)
    alpha_time_words = alpha_time_words.reshape(num_steps + 1, len(prompts) - 1, 1, 1, max_num_words)   # reshape
    return alpha_time_words


### util函数用于LocalBlend和replacentedit
def get_word_inds(text: str, word_place: int, tokenizer):
    split_text = text.split(" ")    # 按空格切分文本
    if isinstance(word_place, str): # 如果输入的是“词字符串”
        word_place = [i for i, word in enumerate(split_text) if word_place == word] # 找到这个词在句子中的位置
    elif isinstance(word_place, int):   # 否则，统一成 list 格式
        word_place = [word_place]
    out = []    # 初始化输出
    if len(word_place) > 0: # 如果确实找到了词
        words_encode = [tokenizer.decode([item]).strip("#") for item in tokenizer.encode(text)][1:-1]   # 将一个词拆分成多个 token
        cur_len, ptr = 0, 0 # 初始化两个指针

        for i in range(len(words_encode)):  # 遍历token
            cur_len += len(words_encode[i]) # 累加token长度
            if ptr in word_place:   # 如果当前词是目标词
                out.append(i + 1)   # 把 token index 加入结果
            if cur_len >= len(split_text[ptr]):
                ptr += 1
                cur_len = 0
    return np.array(out)


### util函数用于replacementit，构造一个 token-level 映射矩阵（mapper），用来把「原 prompt 的 cross-attention」映射到「新 prompt」
def get_replacement_mapper_(x: str, y: str, tokenizer, max_len=77):
    words_x = x.split(" ")  # 按空格切词
    words_y = y.split(" ")
    print(f"words_x is {words_x},length is {len(words_x)}")
    print(f"words_y is {words_y},length is {len(words_y)}")
    if len(words_x) != len(words_y):    # 检查长度是否一样
        raise ValueError(
            f"attention replacement edit can only be applied on prompts with the same length" f" but prompt A has {len(words_x)} words and prompt B has {len(words_y)} words."
        )
    inds_replace = [i for i in range(len(words_y)) if words_y[i] != words_x[i]] # 找到哪些位置的词变了
    inds_source = [get_word_inds(x, i, tokenizer) for i in inds_replace]    # 找到它在 token 级别的位置
    inds_target = [get_word_inds(y, i, tokenizer) for i in inds_replace]    # 同样
    mapper = np.zeros((max_len, max_len))   # 初始化 mapper 矩阵
    i = j = 0   # 初始化指针
    cur_inds = 0
    while i < max_len and j < max_len:  # 遍历整个 token 空间
        if cur_inds < len(inds_source) and inds_source[cur_inds][0] == i:   # 判断当前是否遇到替换词
            inds_source_, inds_target_ = inds_source[cur_inds], inds_target[cur_inds]   # 取出对应的token
            if len(inds_source_) == len(inds_target_):  # token相同
                mapper[inds_source_, inds_target_] = 1
            else:
                ratio = 1 / len(inds_target_)
                for i_t in inds_target_:
                    mapper[inds_source_, i_t] = ratio
            cur_inds += 1   # 移动指针
            i += len(inds_source_)
            j += len(inds_target_)
        elif cur_inds < len(inds_source):   # 如果是没到替换词，那就一一对齐
            mapper[i, j] = 1
            i += 1
            j += 1
        else:
            mapper[j, j] = 1
            i += 1
            j += 1

    # return torch.from_numpy(mapper).float()
    return torch.from_numpy(mapper).to(torch.float16)

# 是对上面函数的批量封装
def get_replacement_mapper(prompts, tokenizer, max_len=77):
    x_seq = prompts[0]
    mappers = []
    for i in range(1, len(prompts)):    # 遍历所有prompt
        mapper = get_replacement_mapper_(x_seq, prompts[i], tokenizer, max_len)
        mappers.append(mapper)
    return torch.stack(mappers)


### util函数用于 ReweightEdit，Prompt-to-Prompt 中 reweight（权重调节）模式的核心函数。
def get_equalizer(
    text: str,
    word_select: Union[int, Tuple[int, ...]],
    values: Union[List[float], Tuple[float, ...]],
    tokenizer,
):
    if isinstance(word_select, (int, str)): # 统一 word_select 类型
        word_select = (word_select,)
    equalizer = torch.ones(len(values), 77) # 初始化 equalizer
    values = torch.tensor(values, dtype=torch.float32)  # 把 values 转成 tensor
    for i, word in enumerate(word_select):  # 遍历每个要调整的词
        inds = get_word_inds(text, word, tokenizer) # 找到对应的token
        equalizer[:, inds] = torch.FloatTensor(values[i])   # 赋权重
    return equalizer


### util函数用于RefinementEdit，定义了一个很小但很典型的评分参数类
class ScoreParams:
    def __init__(self, gap, match, mismatch):
        self.gap = gap
        self.match = match
        self.mismatch = mismatch
        print(">>> init ScoreParams")

    def mis_match_char(self, x, y): # 字符匹配函数
        if x != y:
            return self.mismatch
        else:
            return self.match

# 创建并初始化一个 DP 矩阵的第一行和第一列，用于表示“全是 gap 的情况”
def get_matrix(size_x, size_y, gap):
    matrix = np.zeros((size_x + 1, size_y + 1), dtype=np.int32) # 创建矩阵
    matrix[0, 1:] = (np.arange(size_y) + 1) * gap   # 初始化第一行
    matrix[1:, 0] = (np.arange(size_x) + 1) * gap   # 初始化第一列
    return matrix

# 它不是存“分数”，而是存“路径信息”
def get_traceback_matrix(size_x, size_y):
    matrix = np.zeros((size_x + 1, size_y + 1), dtype=np.int32) # 初始化矩阵
    matrix[0, 1:] = 1   # 初始化行列
    matrix[1:, 0] = 2
    matrix[0, 0] = 4    # 设置七点
    return matrix


def global_align(x, y, score):
    matrix = get_matrix(len(x), len(y), score.gap)  # 初始化分数矩阵
    trace_back = get_traceback_matrix(len(x), len(y))   # 初始化回溯矩阵
    for i in range(1, len(x) + 1):  # 外层循环：遍历 x
        for j in range(1, len(y) + 1):  # 内层循环：遍历 y
            left = matrix[i, j - 1] + score.gap # 从左边来（插入）
            up = matrix[i - 1, j] + score.gap   # 从上边来（删除）
            diag = matrix[i - 1, j - 1] + score.mis_match_char(x[i - 1], y[j - 1])  # 从对角线来（匹配/替换）
            matrix[i, j] = max(left, up, diag)  # 取最大值
            if matrix[i, j] == left:    # 如果来自左
                trace_back[i, j] = 1
            elif matrix[i, j] == up:    # 上
                trace_back[i, j] = 2
            else:   # 对角线
                trace_back[i, j] = 3
    return matrix, trace_back

# 根据 trace_back 矩阵，从右下角回溯，恢复出对齐后的两个序列，并建立 y→x 的位置映射关系
def get_aligned_sequences(x, y, trace_back):
    x_seq = []  # 初始化对齐结果
    y_seq = []
    i = len(x)  # 从右下角开始
    j = len(y)
    mapper_y_to_x = []  # 初始化映射关系
    while i > 0 or j > 0:   # 只要没到起点就不停
        if trace_back[i, j] == 3:   # 如果是对角线
            x_seq.append(x[i - 1])  # 两个字符对齐
            y_seq.append(y[j - 1])
            i = i - 1   # 向左上移动
            j = j - 1
            mapper_y_to_x.append((j, i))    # 记录点
        elif trace_back[i][j] == 1: # 来自左边（插入）
            x_seq.append("-")   # x插入gap,y保留
            y_seq.append(y[j - 1])
            j = j - 1   # 向左移
            mapper_y_to_x.append((j, -1))
        elif trace_back[i][j] == 2: # 来自上方（删除）
            x_seq.append(x[i - 1])  # x保留，y插入gap
            y_seq.append("-")
            i = i - 1   # 向上移动
        elif trace_back[i][j] == 4: # 起点，结束
            break
    mapper_y_to_x.reverse() # 反转映射
    return x_seq, y_seq, torch.tensor(mapper_y_to_x, dtype=torch.int64)

# 把两个 prompt 的 token 序列对齐，生成token 映射，哪些token是真正对齐的标志
def get_mapper(x: str, y: str, tokenizer, max_len=77):
    x_seq = tokenizer.encode(x) # 把文本转成 token id 序列
    y_seq = tokenizer.encode(y)
    score = ScoreParams(0, 1, -1)   # 定义打分规则
    matrix, trace_back = global_align(x_seq, y_seq, score)  # 计算 DP 对齐
    mapper_base = get_aligned_sequences(x_seq, y_seq, trace_back)[-1]   # 回溯得到映射
    alphas = torch.ones(max_len)    # 初始化全1
    alphas[: mapper_base.shape[0]] = mapper_base[:, 1].ne(-1).float()   # 标记有效对齐位置
    mapper = torch.zeros(max_len, dtype=torch.int64)    # 初始化 mapper
    mapper[: mapper_base.shape[0]] = mapper_base[:, 1]  # 填入对齐映射
    mapper[mapper_base.shape[0] :] = len(y_seq) + torch.arange(max_len - len(y_seq))    # 处理剩余 padding
    return mapper, alphas

# 以第一个 prompt 为基准，把它和后续每个 prompt 做 token 对齐，生成一组 mapper（映射）和 alpha（有效掩码）
def get_refinement_mapper(prompts, tokenizer, max_len=77):
    x_seq = prompts[0]  # 取基准 prompt
    mappers, alphas = [], []    # 初始化列表
    for i in range(1, len(prompts)):    # 遍历所有编辑 prompt
        mapper, alpha = get_mapper(x_seq, prompts[i], tokenizer, max_len)   # 计算单个 mapper 和 alpha
        mappers.append(mapper)  # 保存
        alphas.append(alpha)
    return torch.stack(mappers), torch.stack(alphas)
