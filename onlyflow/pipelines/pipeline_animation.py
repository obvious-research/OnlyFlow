# Adapted from https://github.com/showlab/Tune-A-Video/blob/main/tuneavideo/pipelines/pipeline_tuneavideo.py

import copy
import gc
from dataclasses import dataclass
from typing import Callable, Optional, Dict, Any, Tuple
from typing import List, Union

import PIL.Image
import numpy as np
import torch
from diffusers import AnimateDiffPipeline
from diffusers.image_processor import PipelineImageInput
from diffusers.models import AutoencoderKL
from diffusers.models.attention import FreeNoiseTransformerBlock
from diffusers.pipelines.animatediff.pipeline_animatediff import EXAMPLE_DOC_STRING
from diffusers.pipelines.free_noise_utils import AnimateDiffFreeNoiseMixin, SplitInferenceModule
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.schedulers import (
    DDIMScheduler,
    DPMSolverMultistepScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    LMSDiscreteScheduler,
    PNDMScheduler,
)
from diffusers.utils import BaseOutput
from diffusers.utils import deprecate, logging, replace_example_docstring
from einops import rearrange
from transformers import CLIPTextModel, CLIPTokenizer

from onlyflow.models.flow_adaptor import FlowEncoder
from onlyflow.models.unet import UNetMotionModel, AnimateDiffTransformer3D, \
    CrossAttnDownBlockMotion, DownBlockMotion, UpBlockMotion, CrossAttnUpBlockMotion
from ..models.attention import BasicTransformerBlock

logger = logging.get_logger(__name__)

@dataclass
class FlowCtrlPipelineOutput(BaseOutput):
    r"""
     Output class for AnimateDiff pipelines.

    Args:
         frames (`torch.Tensor`, `np.ndarray`, or List[List[PIL.Image.Image]]):
             List of video outputs - It can be a nested list of length `batch_size,` with each sub-list containing
             denoised
     PIL image sequences of length `num_frames.` It can also be a NumPy array or Torch tensor of shape
    `(batch_size, num_frames, channels, height, width)`
    """

    frames: Union[torch.Tensor, np.ndarray, List[List[PIL.Image.Image]]]


class FlowCtrlPipeline(AnimateDiffPipeline):
    model_cpu_offload_seq = "text_encoder->flow_encoder->image_encoder->unet->vae"
    _optional_components = ["feature_extractor", "image_encoder", "motion_adapter"]
    _callback_tensor_inputs = ["latents", "prompt_embeds", "negative_prompt_embeds"]

    def __init__(self,
                 vae: AutoencoderKL,
                 text_encoder: CLIPTextModel,
                 tokenizer: CLIPTokenizer,
                 unet: UNetMotionModel,
                 scheduler: Union[
                     DDIMScheduler,
                     PNDMScheduler,
                     LMSDiscreteScheduler,
                     EulerDiscreteScheduler,
                     EulerAncestralDiscreteScheduler,
                     DPMSolverMultistepScheduler],
                 flow_encoder: FlowEncoder,
                 feature_extractor=None,
                 image_encoder=None,
                 motion_adapter=None,
                 ):

        super().__init__(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            motion_adapter=motion_adapter,
            scheduler=scheduler,
            feature_extractor=feature_extractor,
            image_encoder=image_encoder,
        )
        self.register_modules(
            flow_encoder=flow_encoder
        )

    def _enable_split_inference_motion_modules_(
        self, motion_modules: List[AnimateDiffTransformer3D], spatial_split_size: int
    ) -> None:
        for motion_module in motion_modules:
            motion_module.proj_in = SplitInferenceModule(motion_module.proj_in, spatial_split_size, 0, ["input"])

            for i in range(len(motion_module.transformer_blocks)):
                motion_module.transformer_blocks[i] = SplitInferenceModule(
                    motion_module.transformer_blocks[i],
                    spatial_split_size,
                    0,
                    ["hidden_states", "encoder_hidden_states", "cross_attention_kwargs"],
                )

            motion_module.proj_out = SplitInferenceModule(motion_module.proj_out, spatial_split_size, 0, ["input"])


    def _enable_free_noise_in_block(self, block: Union[CrossAttnDownBlockMotion, DownBlockMotion, UpBlockMotion, CrossAttnUpBlockMotion]):
        r"""Helper function to enable FreeNoise in transformer blocks."""

        for motion_module in block.motion_modules:
            num_transformer_blocks = len(motion_module.transformer_blocks)

            for i in range(num_transformer_blocks):
                if isinstance(motion_module.transformer_blocks[i], FreeNoiseTransformerBlock):
                    motion_module.transformer_blocks[i].set_free_noise_properties(
                        self._free_noise_context_length,
                        self._free_noise_context_stride,
                        self._free_noise_weighting_scheme,
                    )
                else:
                    basic_transfomer_block = motion_module.transformer_blocks[i]

                    motion_module.transformer_blocks[i] = FreeNoiseTransformerBlock(
                        dim=basic_transfomer_block.dim,
                        num_attention_heads=basic_transfomer_block.num_attention_heads,
                        attention_head_dim=basic_transfomer_block.attention_head_dim,
                        dropout=basic_transfomer_block.dropout,
                        cross_attention_dim=basic_transfomer_block.cross_attention_dim,
                        activation_fn=basic_transfomer_block.activation_fn,
                        attention_bias=basic_transfomer_block.attention_bias,
                        only_cross_attention=basic_transfomer_block.only_cross_attention,
                        double_self_attention=basic_transfomer_block.double_self_attention,
                        positional_embeddings=basic_transfomer_block.positional_embeddings,
                        num_positional_embeddings=basic_transfomer_block.num_positional_embeddings,
                        context_length=self._free_noise_context_length,
                        context_stride=self._free_noise_context_stride,
                        weighting_scheme=self._free_noise_weighting_scheme,
                    ).to(device=self._execution_device, dtype=self.dtype)

                    # here i need to copy the attention processor from the basic transformer block to the free noise transformer block
                    motion_module.transformer_blocks[i].attn1 = basic_transfomer_block.attn1
                    motion_module.transformer_blocks[i].attn2 = basic_transfomer_block.attn2

                    motion_module.transformer_blocks[i].load_state_dict(
                        basic_transfomer_block.state_dict(), strict=True
                    )
                    motion_module.transformer_blocks[i].set_chunk_feed_forward(
                        basic_transfomer_block._chunk_size, basic_transfomer_block._chunk_dim
                    )

    def _disable_free_noise_in_block(self, block: Union[CrossAttnDownBlockMotion, DownBlockMotion, UpBlockMotion, CrossAttnUpBlockMotion]):
        r"""Helper function to disable FreeNoise in transformer blocks."""

        for motion_module in block.motion_modules:
            num_transformer_blocks = len(motion_module.transformer_blocks)

            for i in range(num_transformer_blocks):
                if isinstance(motion_module.transformer_blocks[i], FreeNoiseTransformerBlock):
                    free_noise_transfomer_block = motion_module.transformer_blocks[i]

                    motion_module.transformer_blocks[i] = BasicTransformerBlock(
                        dim=free_noise_transfomer_block.dim,
                        num_attention_heads=free_noise_transfomer_block.num_attention_heads,
                        attention_head_dim=free_noise_transfomer_block.attention_head_dim,
                        dropout=free_noise_transfomer_block.dropout,
                        cross_attention_dim=free_noise_transfomer_block.cross_attention_dim,
                        activation_fn=free_noise_transfomer_block.activation_fn,
                        attention_bias=free_noise_transfomer_block.attention_bias,
                        only_cross_attention=free_noise_transfomer_block.only_cross_attention,
                        double_self_attention=free_noise_transfomer_block.double_self_attention,
                        positional_embeddings=free_noise_transfomer_block.positional_embeddings,
                        num_positional_embeddings=free_noise_transfomer_block.num_positional_embeddings,
                    ).to(device=self._execution_device, dtype=self.dtype)

                    motion_module.transformer_blocks[i].load_state_dict(
                        free_noise_transfomer_block.state_dict(), strict=True
                    )
                    motion_module.transformer_blocks[i].set_chunk_feed_forward(
                        free_noise_transfomer_block._chunk_size, free_noise_transfomer_block._chunk_dim
                    )


    @torch.no_grad()
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    def __call__(
            self,
            prompt: Union[str, List[str]] = None,
            optical_flow: torch.FloatTensor = None,

            num_frames: Optional[int] = 16,
            height: Optional[int] = None,
            width: Optional[int] = None,

            num_inference_steps: int = 50,
            guidance_scale: float = 7.5,
            negative_prompt: Optional[Union[str, List[str]]] = None,
            eta: float = 0.0,
            generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
            latents: Optional[torch.Tensor] = None,

            prompt_embeds: Optional[torch.Tensor] = None,
            negative_prompt_embeds: Optional[torch.Tensor] = None,
            ip_adapter_image: Optional[PipelineImageInput] = None,
            ip_adapter_image_embeds: Optional[List[torch.Tensor]] = None,

            output_type: Optional[str] = "pt",
            return_dict: bool = True,

            callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
            callback_on_step_end_tensor_inputs: List[str] = ["latents"],

            cross_attention_kwargs: Optional[Dict[str, Any]] = None,
            motion_cross_attention_kwargs: Optional[Dict[str, Any]] = None,

            clip_skip: Optional[int] = None,
            decode_chunk_size: int = 16,

            val_scale_factor_spatial: float = 0.,
            val_scale_factor_temporal: float = 0.,

            **kwargs,
    ):
        r"""
        The call function to the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide image generation. If not defined, you need to pass `prompt_embeds`.
            height (`int`, *optional*, defaults to `self.unet.config.sample_size * self.vae_scale_factor`):
                The height in pixels of the generated video.
            width (`int`, *optional*, defaults to `self.unet.config.sample_size * self.vae_scale_factor`):
                The width in pixels of the generated video.
            num_frames (`int`, *optional*, defaults to 16):
                The number of video frames that are generated. Defaults to 16 frames which at 8 frames per seconds
                amounts to 2 seconds of video.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality videos at the
                expense of slower inference.
            guidance_scale (`float`, *optional*, defaults to 7.5):
                A higher guidance scale value encourages the model to generate images closely linked to the text
                `prompt` at the expense of lower image quality. Guidance scale is enabled when `guidance_scale > 1`.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide what to not include in image generation. If not defined, you need to
                pass `negative_prompt_embeds` instead. Ignored when not using guidance (`guidance_scale < 1`).
            eta (`float`, *optional*, defaults to 0.0):
                Corresponds to parameter eta (η) from the [DDIM](https://arxiv.org/abs/2010.02502) paper. Only applies
                to the [`~schedulers.DDIMScheduler`], and is ignored in other schedulers.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                A [`torch.Generator`](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make
                generation deterministic.
            latents (`torch.Tensor`, *optional*):
                Pre-generated noisy latents sampled from a Gaussian distribution, to be used as inputs for video
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor is generated by sampling using the supplied random `generator`. Latents should be of shape
                `(batch_size, num_channel, num_frames, height, width)`.
            prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs (prompt weighting). If not
                provided, text embeddings are generated from the `prompt` input argument.
            negative_prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs (prompt weighting). If
                not provided, `negative_prompt_embeds` are generated from the `negative_prompt` input argument.
            ip_adapter_image: (`PipelineImageInput`, *optional*):
                Optional image input to work with IP Adapters.
            ip_adapter_image_embeds (`List[torch.Tensor]`, *optional*):
                Pre-generated image embeddings for IP-Adapter. It should be a list of length same as number of
                IP-adapters. Each element should be a tensor of shape `(batch_size, num_images, emb_dim)`. It should
                contain the negative image embedding if `do_classifier_free_guidance` is set to `True`. If not
                provided, embeddings are computed from the `ip_adapter_image` input argument.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generated video. Choose between `torch.Tensor`, `PIL.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.text_to_video_synthesis.TextToVideoSDPipelineOutput`] instead
                of a plain tuple.
            cross_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the [`AttentionProcessor`] as defined in
                [`self.processor`](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            clip_skip (`int`, *optional*):
                Number of layers to be skipped from CLIP while computing the prompt embeddings. A value of 1 means that
                the output of the pre-final layer will be used for computing the prompt embeddings.
            callback_on_step_end (`Callable`, *optional*):
                A function that calls at the end of each denoising steps during the inference. The function is called
                with the following arguments: `callback_on_step_end(self: DiffusionPipeline, step: int, timestep: int,
                callback_kwargs: Dict)`. `callback_kwargs` will include a list of all tensors as specified by
                `callback_on_step_end_tensor_inputs`.
            callback_on_step_end_tensor_inputs (`List`, *optional*):
                The list of tensor inputs for the `callback_on_step_end` function. The tensors specified in the list
                will be passed as `callback_kwargs` argument. You will only be able to include variables listed in the
                `._callback_tensor_inputs` attribute of your pipeline class.
            decode_chunk_size (`int`, defaults to `16`):
                The number of frames to decode at a time when calling `decode_latents` method.

        Examples:

        Returns:
            [`~pipelines.animatediff.pipeline_output.AnimateDiffPipelineOutput`] or `tuple`:
                If `return_dict` is `True`, [`~pipelines.animatediff.pipeline_output.AnimateDiffPipelineOutput`] is
                returned, otherwise a `tuple` is returned where the first element is a list with the generated frames.
        """

        callback = kwargs.pop("callback", None)
        callback_steps = kwargs.pop("callback_steps", None)

        if callback is not None:
            deprecate(
                "callback",
                "1.0.0",
                "Passing `callback` as an input argument to `__call__` is deprecated, consider using `callback_on_step_end`",
            )
        if callback_steps is not None:
            deprecate(
                "callback_steps",
                "1.0.0",
                "Passing `callback_steps` as an input argument to `__call__` is deprecated, consider using `callback_on_step_end`",
            )

        # 0. Default height and width to unet
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor

        num_videos_per_prompt = 1

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt,
            height,
            width,
            callback_steps,
            negative_prompt,
            prompt_embeds,
            negative_prompt_embeds,
            ip_adapter_image,
            ip_adapter_image_embeds,
            callback_on_step_end_tensor_inputs,
        )

        self._guidance_scale = guidance_scale
        self._clip_skip = clip_skip
        self._cross_attention_kwargs = cross_attention_kwargs
        self._interrupt = False

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, (str, dict)):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device

        # 3. Encode input prompt
        text_encoder_lora_scale = (
            self.cross_attention_kwargs.get("scale", None) if self.cross_attention_kwargs is not None else None
        )
        if self.free_noise_enabled:
            prompt_embeds, negative_prompt_embeds = self._encode_prompt_free_noise(
                prompt=prompt,
                num_frames=num_frames,
                device=device,
                num_videos_per_prompt=num_videos_per_prompt,
                do_classifier_free_guidance=self.do_classifier_free_guidance,
                negative_prompt=negative_prompt,
                prompt_embeds=prompt_embeds,
                negative_prompt_embeds=negative_prompt_embeds,
                lora_scale=text_encoder_lora_scale,
                clip_skip=self.clip_skip,
            )
        else:
            prompt_embeds, negative_prompt_embeds = self.encode_prompt(
                prompt,
                device,
                num_videos_per_prompt,
                self.do_classifier_free_guidance,
                negative_prompt,
                prompt_embeds=prompt_embeds,
                negative_prompt_embeds=negative_prompt_embeds,
                lora_scale=text_encoder_lora_scale,
                clip_skip=self.clip_skip,
            )

            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and text embeddings into a single batch
            # to avoid doing two forward passes
            if self.do_classifier_free_guidance:
                prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])

            prompt_embeds = prompt_embeds.repeat_interleave(repeats=num_frames, dim=0)

        if ip_adapter_image is not None or ip_adapter_image_embeds is not None:
            image_embeds = self.prepare_ip_adapter_image_embeds(
                ip_adapter_image,
                ip_adapter_image_embeds,
                device,
                batch_size * num_videos_per_prompt,
                self.do_classifier_free_guidance,
            )

        # 4. Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # 5. Prepare latent variables
        num_channels_latents = self.unet.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_videos_per_prompt,
            num_channels_latents,
            num_frames,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )

        # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
        assert optical_flow.ndim == 5
        bs = optical_flow.shape[0]
        if self.free_noise_enabled:
            length = optical_flow.shape[2]
            flow_embedding_features = [
                torch.zeros((bs, length, *test_size.shape[1:]), device=self._execution_device)
                for test_size in self.flow_encoder(optical_flow[:,:,:16].to(self._execution_device))
            ]
            weight_factor = torch.zeros(length, device=self._execution_device)
            for star_idx in range(0, length, self._free_noise_context_stride):
                weight_factor[star_idx:star_idx + self._free_noise_context_length] += 1.0
                infe = self.flow_encoder(optical_flow[:,:,star_idx:star_idx + self._free_noise_context_length].to(self._execution_device))
                for flow_emb, infe_sub in zip(flow_embedding_features, infe):
                    flow_emb[:,star_idx:star_idx + self._free_noise_context_length] += rearrange(infe_sub, '(b f) c h w -> b f c h w', b=bs).to(self._execution_device)

            flow_embedding_features = [flow_emb / weight_factor[None,:,None,None,None] for flow_emb in flow_embedding_features]
            flow_embedding_features = [rearrange(x, 'b f c h w -> b c f h w') for x in flow_embedding_features]
        else:
            flow_embedding_features = self.flow_encoder(optical_flow.to(self._execution_device))  # input b c f h w into bf, c, h, w
            flow_embedding_features = [rearrange(x, '(b f) c h w -> b c f h w', b=bs).to(self._execution_device)
                                       for x in flow_embedding_features]

        del optical_flow
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()

        # 7. Add image embeds for IP-Adapter
        added_cond_kwargs = (
            {"image_embeds": image_embeds}
            if ip_adapter_image is not None or ip_adapter_image_embeds is not None
            else None
        )

        num_free_init_iters = self._free_init_num_iters if self.free_init_enabled else 1
        for free_init_iter in range(num_free_init_iters):
            if self.free_init_enabled:
                latents, timesteps = self._apply_free_init(
                    latents, free_init_iter, num_inference_steps, device, latents.dtype, generator
                )

            self._num_timesteps = len(timesteps)
            num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order

            if isinstance(flow_embedding_features[0], list):
                flow_embedding_features = [[torch.cat([x, x], dim=0) for x in flow_embedding_feature]
                                           for flow_embedding_feature in flow_embedding_features] \
                    if self.do_classifier_free_guidance else flow_embedding_features
            else:
                flow_embedding_features = [torch.cat([x, x], dim=0) for x in flow_embedding_features] \
                    if self.do_classifier_free_guidance else flow_embedding_features  # [2b c f h w]

            # 8. Denoising loop
            with self.progress_bar(total=self._num_timesteps) as progress_bar:
                for i, t in enumerate(timesteps):
                    if self.interrupt:
                        continue

                    # expand the latents if we are doing classifier free guidance
                    latent_model_input = torch.cat([latents] * 2) if self.do_classifier_free_guidance else latents
                    latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                    if added_cond_kwargs is not None:
                        added_cond_kwargs.update({"flow_embedding_features": flow_embedding_features})
                    else:
                        added_cond_kwargs = {"flow_embedding_features": flow_embedding_features}

                    if cross_attention_kwargs is not None:
                        cross_attention_kwargs.update({"flow_scale": val_scale_factor_spatial})
                    else:
                        cross_attention_kwargs = {"flow_scale": val_scale_factor_spatial}

                    if motion_cross_attention_kwargs is not None:
                        motion_cross_attention_kwargs.update({"flow_scale": val_scale_factor_temporal})
                    else:
                        motion_cross_attention_kwargs = {"flow_scale": val_scale_factor_temporal}

                    # predict the noise residual

                    noise_pred = self.unet(
                        latent_model_input,
                        t,
                        encoder_hidden_states=prompt_embeds,
                        cross_attention_kwargs=cross_attention_kwargs,
                        motion_cross_attention_kwargs=motion_cross_attention_kwargs,
                        added_cond_kwargs=added_cond_kwargs,
                    ).sample

                    # perform guidance
                    if self.do_classifier_free_guidance:
                        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                    # compute the previous noisy sample x_t -> x_t-1
                    latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample

                    if callback_on_step_end is not None:
                        callback_kwargs = {}
                        for k in callback_on_step_end_tensor_inputs:
                            callback_kwargs[k] = locals()[k]
                        callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                        latents = callback_outputs.pop("latents", latents)
                        prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                        negative_prompt_embeds = callback_outputs.pop("negative_prompt_embeds", negative_prompt_embeds)

                    # call the callback, if provided
                    if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                        progress_bar.update()
                        if callback is not None and i % callback_steps == 0:
                            callback(i, t, latents)

        # 9. Post processing
        if output_type == "latent":
            video = latents
        else:
            video_tensor = self.decode_latents(latents, decode_chunk_size)
            video = self.video_processor.postprocess_video(video=video_tensor, output_type=output_type)

        # 10. Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (video,)

        return FlowCtrlPipelineOutput(frames=video)
