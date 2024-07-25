import json
import os
import shutil
import tempfile
import urllib

import hydra
import numpy as np
import omegaconf
import torch
import torchvision.transforms as T
import wids
from diffusers import StableDiffusionPipeline, DDIMScheduler
from einops import rearrange
from huggingface_hub.constants import HF_HUB_CACHE
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

from torchmetrics.multimodal.clip_score import CLIPScore

from validation.eval.cal_fvd import calculate_fvd_preloaded
from validation.eval.cal_lpips import calculate_lpips
from validation.eval.cal_psnr import calculate_psnr
from validation.eval.cal_ssim import calculate_ssim
from validation.eval.fvd.styleganv.fvd import load_i3d_pretrained

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
    os.makedirs(output_dir, exist_ok=True)
    samples_dir = f"/lustre/fsn1/projects/rech/fkc/uhx75if/flowctrl_val_samples_spatial/{os.environ.get('SLURM_ARRAY_JOB_ID', 0)}"
    os.makedirs(samples_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "results"), exist_ok=True)


    job_array_id = os.environ.get("SLURM_ARRAY_TASK_ID", 0)
    job_task_count = os.environ.get("SLURM_ARRAY_TASK_COUNT", 1)

    print(f"Job array id: {job_array_id}")

    logger = setup_logger(output_dir, job_array_id, color=True, name="FlowCtrlADValidate")
    logger.info(f"Setting up the environment")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    generator = torch.manual_seed(misc.global_seed)

    adapter_path = os.path.join(HF_HUB_CACHE, models.adapter_lora.name)
    if not os.path.exists(adapter_path):
        urllib.request.urlretrieve(models.adapter_lora.url, adapter_path)

    logger.info(f"Loading the models : unet")
    pipe = StableDiffusionPipeline.from_pretrained(models.model_unet.path)
    logger.info(f"Loading the models : webvid_adapter")
    pipe.load_lora_weights(adapter_path, adapter_name='webvid_adapter', weight_name='')
    pipe.set_adapters(["webvid_adapter"], adapter_weights=[models.adapter_lora.scale])
    logger.info(f"Loading the models : lora fusing")
    pipe.fuse_lora(lora_scale=models.adapter_lora.scale)
    pipe.unload_lora_weights()

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
    flow_encoder = FlowEncoder(**flow_encoder_param.flow_encoder_kwargs, checkpointing=False).eval()

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
        validation_dataset,
        chunksize=30000,
        shuffle=True,
        seed=misc.global_seed,
        rank=int(job_array_id),
        num_replicas=int(job_task_count)
    )

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

    if resume is not None:
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




    clip_metric = CLIPScore(model_name_or_path="openai/clip-vit-base-patch16").to(device)

    i3d = load_i3d_pretrained(device=device)



    # create empty csv file in output_dir
    with open(os.path.join(output_dir, f"results/results_{job_array_id}.csv"), "w") as f:
        f.write("id,flow_scale,clip_score,diff_optical_flow,fvd,ssim,psnr,lpips\n")



    for idx, (source_videos, source_texts) in enumerate(validation_dataloader):
        val_pixel_values = source_videos.to(unet.device)  # [b, f, c, h, w]
        val_pixel_values = T.ConvertImageDtype(torch.float32)(val_pixel_values)
        video_length = val_pixel_values.shape[1] - 1
        flow_val_gt = get_optical_flow(raft_model, (val_pixel_values *2)-1, video_length,
                                       encode_chunk_size=misc.flow_chunk_size)

        # get a linspace of temporal scales
        for val_scale in np.linspace(0.0, 1.0, 11):

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
                val_scale_factor_spatial=val_scale,
                val_scale_factor_temporal=val_scale,
                return_dict=False,
                generate_no_flow=False
            )  # [b, f, 3, h, w]

            flow_val_gene = get_optical_flow(raft_model, (samples_flow * 2) - 1, video_length - 1,
                                           encode_chunk_size=misc.flow_chunk_size)

            for idx_intra_batch, (gen_video, prompt) in enumerate(zip(samples_flow, source_texts)):
                identifier = f"{job_array_id}_{idx}_{idx_intra_batch}"
                source_video = val_pixel_values[idx_intra_batch][:-1]
                save_path = os.path.join(samples_dir, f"sample_{identifier}_{round(val_scale, 2)}.gif")
                vid = get_video(gen_video.cpu(), save_path)
                with open(save_path, "wb") as f:
                    f.write(vid.getbuffer())
                    logger.info(f"Saved the sample to {save_path}")

                # compute FVD: Frechet Video Distance
                results = {}
                results['fvd'] = calculate_fvd_preloaded(i3d, source_video[None, ...], gen_video[None, ...], device, method='styleganv')
                results['ssim'] = calculate_ssim(source_video[None, ...].cpu(), gen_video[None, ...].cpu())
                results['psnr'] = calculate_psnr(source_video[None, ...].cpu(), gen_video[None, ...].cpu())
                results['lpips'] = calculate_lpips(source_video[None, ...], gen_video[None, ...], device)

                # compute diff optical flow
                results['diff_optical_flow'] = (flow_val_gene[idx_intra_batch] - flow_val_gt[idx_intra_batch][:, :-1]).abs().mean().item()

                # clip score
                clip_score = clip_metric((gen_video * 255).to(torch.uint8), [prompt] * gen_video.shape[0]).detach()
                results['clip_score'] = clip_score.tolist()


                # TODO later if time
                # compute Content debiased Frechet Video Distance
                # compute FVMD: Frechet Video Motion Distance

                #print(json.dumps(results, indent=4))

                with open(os.path.join(output_dir, f"results/results_{job_array_id}.csv"), "a") as f:
                    fvd = np.array(list(results['fvd']['value'].values())).tolist()
                    ssim = np.array(list(results['ssim']['value'].values())).tolist()
                    psnr = np.array(list(results['psnr']['value'].values())).tolist()
                    lpips = np.array(list(results['lpips']['value'].values())).tolist()

                    f.write(f"{identifier},{round(val_scale, 2)},{results['clip_score']},{results['diff_optical_flow']},\"{fvd}\",\"{ssim}\",\"{psnr}\",\"{lpips}\"\n")



@hydra.main(config_path="../../configs", config_name="default", version_base=None)
def main(cfg: omegaconf.DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    validate_grid(output_dir=output_dir, **cfg)


if __name__ == "__main__":
    main()
