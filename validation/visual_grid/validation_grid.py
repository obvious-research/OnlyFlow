import os
import shutil
import tempfile

import hydra
import omegaconf
import torch
import torchvision.transforms as T
import wids
from diffusers import StableDiffusionPipeline, DDIMScheduler
from einops import rearrange
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from torchvision.models.optical_flow import raft_large, Raft_Large_Weights
from torchvision.utils import flow_to_image

from flowctrl.data.dataset_idx import WebVidDataset, my_collate_fn
from flowctrl.models.flow_adaptor import FlowEncoder, FlowAdaptor
from flowctrl.models.unet import MotionAdapter, UNetMotionModel
from flowctrl.pipelines.pipeline_animation import FlowCtrlPipeline
from flowctrl.utils.util import setup_logger, get_video
from training.train import get_optical_flow


def validate_grid(output_dir: str,
                  launcher: str,
                  misc,
                  noise_scheduler_kwargs,
                  dataset,
                  resume: str,
                  models,
                  flow_encoder_param,
                  validation,
                  **kwargs
                  ):
    try:
        os.makedirs(output_dir, exist_ok=False)
    except FileExistsError as e:
        shutil.rmtree(output_dir)
        os.makedirs(output_dir)

    logger = setup_logger(output_dir, 0, color=True, name="FlowCtrlADValidate")
    logger.info(f"Setting up the environment")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    generator = torch.manual_seed(misc.global_seed)


    logger.info(f"Loading the models : unet")
    pipe = StableDiffusionPipeline.from_pretrained(models.model_unet.path)
    # logger.info(f"Loading the models : webvid_adapter")
    # pipe.load_lora_weights(adapter_path, adapter_name='webvid_adapter')
    # pipe.set_adapters(["webvid_adapter"], adapter_weights=[models.adapter_lora.scale])
    # logger.info(f"Loading the models : lora fusing")
    # pipe.fuse_lora(lora_scale=models.adapter_lora.scale)
    # pipe.unload_lora_weights()

    logger.info(f"Loading the models : noise scheduler")

    noise_scheduler = DDIMScheduler(**noise_scheduler_kwargs)

    logger.info(f"Loading the models : vae, text_encoder, unet")
    tokenizer = pipe.tokenizer
    text_encoder = pipe.text_encoder.eval()
    vae = pipe.vae.eval()
    base_unet = pipe.unet.eval()
    del pipe

    logger.info(f"Loading the models : motion adapter")
    motion_adapter = MotionAdapter.from_pretrained(models.model_motion_modules.path)
    unet = UNetMotionModel.from_unet2d(
        base_unet,
        motion_adapter=motion_adapter
    )

    logger.info(f"Loading the models : raft")
    raft_model = raft_large(weights=Raft_Large_Weights.DEFAULT, progress=False).to(device).eval()
    logger.info(f"Loading the models : flow encoder")
    flow_encoder = FlowEncoder(**flow_encoder_param.flow_encoder_kwargs, checkpointing=misc.checkpointing).eval()

    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet.requires_grad_(False)
    raft_model.requires_grad_(False)
    flow_encoder.requires_grad_(False)

    logger.info(f"Setting the flow adaptor")
    flow_adaptor = FlowAdaptor(unet, flow_encoder).to(device).eval()

    # init attention processor
    logger.info(f"Setting the attention processors")
    unet.set_all_attn(
        flow_channels=flow_encoder_param.flow_encoder_kwargs['channels'],
        **flow_encoder_param.attention_processor_kwargs,
    )

    logger.info(f'Building datasets')
    dataset_cache_dir = os.environ.get("JOBSCRATCH", None)
    if dataset_cache_dir is None:
        dataset_cache_dir = tempfile.mkdtemp()
    dataset_cache_dir = os.path.join(dataset_cache_dir, "dataset_cache")
    validation_dataset = WebVidDataset(
        cache_dir=dataset_cache_dir,
        val=True,
        seed=misc.global_seed,
        **dataset)

    validation_sampler = wids.DistributedChunkedSampler(
        validation_dataset, chunksize=30000, shuffle=True, seed=misc.global_seed)

    validation_dataloader = DataLoader(
        validation_dataset,
        sampler=validation_sampler,
        collate_fn=my_collate_fn,
        num_workers=1,
        prefetch_factor=1,
        multiprocessing_context='fork' if launcher == 'local' else None,
        pin_memory=True,
        drop_last=True,
        batch_size=validation.batch_size,
    )

    vae = vae.to(device)
    text_encoder = text_encoder.to(device)

    validation_pipeline = FlowCtrlPipeline(
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        unet=flow_adaptor.unet,
        flow_encoder=flow_adaptor.flow_encoder,
        scheduler=noise_scheduler,
    ).to(device)

    resume = "/data/3711799/flowctrl/checkpoint-step-15500.ckpt"

    logger.info(f"Loading from the checkpoint: {resume}")
    ckpt = torch.load(resume, map_location=device)
    flow_encoder_state_dict = ckpt['flow_encoder_state_dict']
    attention_processor_state_dict = ckpt['attention_processor_state_dict']
    pose_enc_m, pose_enc_u = flow_adaptor.flow_encoder.load_state_dict(
        flow_encoder_state_dict,
        strict=False
    )
    assert len(pose_enc_m) == 0 and len(pose_enc_u) == 0
    _, attention_processor_u = flow_adaptor.unet.load_state_dict(attention_processor_state_dict,
                                                                 strict=False)
    assert len(attention_processor_u) == 0
    logger.info(f"Loading the flow encoder and attention processor weights done.")

    for idx, (source_videos, source_texts) in enumerate(validation_dataloader):
        for idx_prompt, prompt in enumerate(
                ['a butterfly flying around', 'air balloons', 'street dancing',
                 'timelapse of a sunset', 'a woman under the pouring rain', 'a garden full of flowers']):

            source_texts = [prompt] * len(source_videos)

            val_pixel_values = source_videos.to(unet.device)  # [b, f, c, h, w]
            val_pixel_values = T.ConvertImageDtype(torch.float32)(val_pixel_values)
            video_length = val_pixel_values.shape[1] - 1
            flow_val_gt = get_optical_flow(raft_model, (val_pixel_values *2)-1, video_length,
                                           encode_chunk_size=misc.flow_chunk_size)

            flow_imgs_gt = flow_to_image(rearrange(flow_val_gt, "b c f h w -> (b f) c h w"))  # (N, 3, H, W)
            flow_imgs_gt = rearrange(flow_imgs_gt, "(b f) c h w -> b f c h w", f=video_length)

            table_vids = [val_pixel_values[:, :video_length], flow_imgs_gt]

            generator.manual_seed(misc.global_seed+idx*len(source_videos)+idx_prompt)
            generator_init_state = generator.get_state()

            for val_scale in [0.0, 0.5, 0.75, 1.0]:
                generator.set_state(generator_init_state)

                samples_flow, samples_no_flow = validation_pipeline(
                    prompt=source_texts,
                    negative_prompt=['blurry, low quality'] * len(source_texts),
                    flow_embedding=flow_val_gt,
                    output_type="pt",
                    video_length=video_length,
                    height=validation_dataset.video_size[0],
                    width=validation_dataset.video_size[1],
                    num_inference_steps=25,
                    guidance_scale=7.5,
                    generator=generator,
                    val_scale_factor_spatial=0.0,
                    val_scale_factor_temporal=val_scale,
                    return_dict=False,
                    generate_no_flow=False
                )  # [b, f, 3, h, w]

                table_vids.append(samples_flow)

            final_sample = torch.cat(table_vids, dim=-1)
            final_sample = rearrange(final_sample, "b f c h w -> c f (b h) w")
            save_path = os.path.join(output_dir, f"sample_{idx}_{idx_prompt}.gif")
            vid = get_video(final_sample.cpu(), save_path)
            with open(save_path, "wb") as f:
                f.write(vid.getbuffer())
                logger.info(f"Saved the sample to {save_path}")


@hydra.main(config_path="../../configs", config_name="default", version_base=None)
def main(cfg: omegaconf.DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    validate_grid(output_dir=output_dir, **cfg)


if __name__ == '__main__':
    main()
