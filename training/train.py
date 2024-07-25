from __future__ import annotations

import gc
import os
import random
import shutil
import tempfile
import time
import urllib
from contextlib import nullcontext

import hydra
import omegaconf
import torch
import torch.distributed as dist
import torch.nn.functional as F
import torchvision.transforms as T
import wandb
import wids
from diffusers import MotionAdapter, StableDiffusionPipeline, DDIMScheduler
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version
from einops import rearrange
from huggingface_hub.constants import HF_HUB_CACHE
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from torchvision.models.optical_flow import raft_large, Raft_Large_Weights
from torchvision.utils import flow_to_image

from onlyflow.data.dataset_idx import WebVidDataset, my_collate_fn
from onlyflow.models.attention_processor import FlowAdaptorAttnProcessor
from onlyflow.models.flow_adaptor import FlowAdaptor, FlowEncoder
from onlyflow.models.unet import UNetMotionModel
from onlyflow.pipelines.pipeline_animation import FlowCtrlPipeline
from onlyflow.utils.util import setup_logger, format_time, get_video
from tools.optical_flow import get_optical_flow


def encode_videos(vae, pixel_values, encode_chunk_size: int = 16):
    latents = []
    for i in range(0, pixel_values.shape[0], encode_chunk_size):
        batch_latents = pixel_values[i: i + encode_chunk_size]
        batch_latents = vae.encode(batch_latents).latent_dist.sample()
        latents.append(batch_latents)

    latents = torch.cat(latents)
    return latents


def save_model(output_dir, epoch, global_step, flow_adaptor, optimizer, logger):
    save_path = os.path.join(output_dir, f"checkpoints")
    unwrapped_flow_adaptor = flow_adaptor.module if isinstance(flow_adaptor,
                                                               torch.nn.parallel.DistributedDataParallel) else flow_adaptor
    flow_encoder_state_dict = unwrapped_flow_adaptor.flow_encoder.state_dict()
    unet = unwrapped_flow_adaptor.unet

    attention_trainable_param_names = [k for k, v in unet.named_parameters() if
                                       v.requires_grad]

    attention_processor_state_dict = {k: v for k, v in unet.state_dict().items()
                                      if k in attention_trainable_param_names}
    state_dict = {
        "epoch": epoch,
        "global_step": global_step,
        "flow_encoder_state_dict": flow_encoder_state_dict,
        "attention_processor_state_dict": attention_processor_state_dict,
        "optimizer_state_dict": optimizer.state_dict(),
    }
    torch.save(state_dict, os.path.join(save_path, f"checkpoint-step-{global_step}.ckpt"))
    logger.info(f"Saved state to {save_path} (global_step: {global_step})")


def train(
        output_dir: str,
        full_config,
        launcher: str,
        misc,
        noise_scheduler_kwargs,
        logging_args,
        profiling,
        dataset,
        resume: str,
        models,
        flow_encoder_param,
        training,
        validation,
        optimization,
        loss

):
    check_min_version("0.30.0")

    # Initialize distributed training

    if launcher == 'local':
        local_rank = global_rank = 0
        local_world_size = world_size = 1
    else:
        if launcher == 'slurm':
            import idr_torch
            dist.init_process_group("gloo" if misc.force_cpu else None, world_size=idr_torch.size, rank=idr_torch.rank)
            local_world_size = idr_torch.local_world_size
        else:
            dist.init_process_group("gloo" if misc.force_cpu else None)
            local_world_size = int(os.environ["LOCAL_WORLD_SIZE"])
        global_rank = dist.get_rank()
        local_rank = dist.get_node_local_rank()
        world_size = dist.get_world_size()

    is_main_process = global_rank == 0

    try:
        torch.set_float32_matmul_precision('high')
        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True
            import torch.backends.cudnn as cudnn
            if cudnn.is_available():
                cudnn.enabled = misc.cudnn.enabled
                cudnn.benchmark = misc.cudnn.benchmark
                cudnn.deterministic = misc.cudnn.deterministic
                cudnn.allow_tf32 = misc.cudnn.allow_tf32

        # Handle the output folder creation
        if is_main_process:
            try:
                os.makedirs(output_dir, exist_ok=False)
            except FileExistsError as e:
                shutil.rmtree(output_dir)
                os.makedirs(output_dir)
            finally:
                os.makedirs(f"{output_dir}/samples")
                os.makedirs(f"{output_dir}/sanity_check")
                os.makedirs(f"{output_dir}/checkpoints")

        run = wandb.init(
            entity=logging_args.wandb.entity,
            project=logging_args.wandb.project,
            group=logging_args.wandb.group,
            config=omegaconf.OmegaConf.to_container(
                full_config, resolve=True
            ),
            save_code=True,
            dir=output_dir
        )

        if dist.is_initialized():
            dist.barrier()

        # setup the logger
        logger = setup_logger(output_dir, global_rank, color=True, name="onlyflowAD")

        if torch.cuda.is_available() and not misc.force_cpu:
            torch.cuda.set_device(local_rank)
            assert torch.cuda.device_count() >= local_world_size
            device = torch.device("cuda")
        elif torch.backends.mps.is_available() and not misc.force_cpu:
            torch.mps.manual_seed(misc.global_seed)
            device = torch.device("mps")
        else:
            device = torch.device("cpu")

        generator = torch.manual_seed(misc.global_seed)
        generator_val = generator.clone_state()
        generator_val_init_state = generator_val.get_state()
        random.seed(misc.global_seed)

        # Load scheduler, tokenizer and models.

        adapter_path = os.path.join(HF_HUB_CACHE, models.adapter_lora.name)
        if not os.path.exists(adapter_path):
            urllib.request.urlretrieve(models.adapter_lora.url, adapter_path)

        pipe = StableDiffusionPipeline.from_pretrained(models.model_unet.path)
        pipe.load_lora_weights(adapter_path, adapter_name='webvid_adapter')
        pipe.set_adapters(["webvid_adapter"], adapter_weights=[models.adapter_lora.scale])
        pipe.fuse_lora(lora_scale=models.adapter_lora.scale)
        pipe.unload_lora_weights()

        noise_scheduler = DDIMScheduler.from_pretrained(
            models.model_unet.path,
            subfolder='scheduler',
            **noise_scheduler_kwargs
        )

        tokenizer = pipe.tokenizer
        text_encoder = pipe.text_encoder.eval()
        vae = pipe.vae.eval()
        base_unet = pipe.unet.train()
        del pipe

        motion_adapter = MotionAdapter.from_pretrained(models.model_motion_modules.path)
        unet = UNetMotionModel.from_unet2d(
            base_unet,
            motion_adapter=motion_adapter
        )

        if misc.checkpointing:
            unet.enable_gradient_checkpointing()

        raft_model = raft_large(weights=Raft_Large_Weights.DEFAULT, progress=False).to(device).eval()
        flow_encoder = FlowEncoder(**flow_encoder_param.flow_encoder_kwargs, checkpointing=misc.checkpointing).train()

        # init attention processor
        logger.info(f"Setting the attention processors")
        unet.set_all_attn(
            flow_channels=flow_encoder_param.flow_encoder_kwargs['channels'],
            **flow_encoder_param.attention_processor_kwargs,
        )

        # Freeze vae, and text_encoder
        vae.requires_grad_(False)
        text_encoder.requires_grad_(False)
        unet.requires_grad_(False)
        raft_model.requires_grad_(False)

        spatial_attn_proc_modules = torch.nn.ModuleList(
            [v for v in unet.attn_processors.values() if
             isinstance(v, FlowAdaptorAttnProcessor) and v.type == 'spatial']
        )
        temporal_attn_proc_modules = torch.nn.ModuleList(
            [v for v in unet.attn_processors.values() if
             isinstance(v, FlowAdaptorAttnProcessor) and v.type == 'temporal']
        )

        spatial_attn_proc_modules.requires_grad_(True)
        temporal_attn_proc_modules.requires_grad_(True)
        flow_encoder.requires_grad_(True)

        # Support mixed-precision training
        if misc.enable_amp.unet:
            logger.info("Mixed precision requested.")
            if not torch.amp.is_autocast_available(device.type):
                logger.warning(f"Mixed precision training is not available on {device}. Disabling it.")
                misc.enable_amp.unet = False
            else:
                scaler = torch.amp.GradScaler(device=device.type)

        if torch.cuda.is_available():
            flow_adaptor_stream = torch.cuda.Stream(device=device)
            vae_stream = torch.cuda.Stream(device=device)
            text_encoder_stream = torch.cuda.Stream(device=device)

        flow_adaptor = FlowAdaptor(unet, flow_encoder).to(device).train()

        encoder_trainable_params = list(filter(lambda p: p.requires_grad, flow_encoder.parameters()))
        encoder_trainable_param_names = [p[0] for p in
                                         filter(lambda p: p[1].requires_grad, flow_encoder.named_parameters())]
        attention_trainable_params = [v for k, v in unet.named_parameters() if
                                      v.requires_grad]
        attention_trainable_param_names = [k for k, v in unet.named_parameters() if
                                           v.requires_grad]

        trainable_params = encoder_trainable_params + attention_trainable_params
        trainable_param_names = encoder_trainable_param_names + attention_trainable_param_names

        if is_main_process:
            logger.info(f"trainable parameter number: {len(trainable_params)}")
            logger.info(f"encoder trainable number: {len(encoder_trainable_params)}")
            logger.info(f"attention processor trainable number: {len(attention_trainable_params)}")
            logger.info(f"trainable parameter names: {trainable_param_names}")
            logger.info(f"encoder trainable scale: {sum(p.numel() for p in encoder_trainable_params) / 1e6:.3f} M")
            logger.info(
                f"attention processor trainable scale: {sum(p.numel() for p in attention_trainable_params) / 1e6:.3f} M")
            logger.info(f"trainable parameter scale: {sum(p.numel() for p in trainable_params) / 1e6:.3f} M")

        optimizer = torch.optim.AdamW(
            trainable_params,
            lr=optimization.lr,
            betas=(optimization.beta1, optimization.beta2),
            weight_decay=optimization.weight_decay,
            eps=optimization.epsilon,
            fused=True
        )

        # Get the training dataset
        logger.info(f'Building datasets')
        dataset_cache_dir = os.environ.get("JOBSCRATCH", None)
        if dataset_cache_dir is None:
            dataset_cache_dir = tempfile.mkdtemp()
        dataset_cache_dir = os.path.join(dataset_cache_dir, "dataset_cache")
        train_dataset = WebVidDataset(
            cache_dir=dataset_cache_dir,
            seed=misc.global_seed,
            **dataset)
        validation_dataset = WebVidDataset(
            cache_dir=dataset_cache_dir,
            val=True,
            seed=misc.global_seed,
            **dataset)

        train_sampler = wids.DistributedChunkedSampler(
            train_dataset, chunksize=8000, shuffle=True, seed=misc.global_seed)
        validation_sampler = wids.DistributedChunkedSampler(
            validation_dataset, chunksize=8000, shuffle=True, seed=misc.global_seed)

        train_dataloader = DataLoader(
            train_dataset,
            sampler=train_sampler,
            collate_fn=my_collate_fn,
            num_workers=misc.train_dataloader.num_workers,
            prefetch_factor=misc.train_dataloader.prefetch_factor,
            multiprocessing_context='fork' if launcher == 'local' else None,
            pin_memory=True,
            drop_last=True,
            batch_size=training.batch_size,
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
            generator=generator,
            batch_size=validation.batch_size,
        )

        # Get the training iteration
        max_train_steps = training.num_epochs * len(train_dataset) // (train_dataloader.batch_size * world_size)

        # Scheduler
        lr_scheduler = get_scheduler(
            optimization.lr_scheduler,
            optimizer=optimizer,
            num_warmup_steps=optimization.lr_warmup_steps,
            num_training_steps=max_train_steps,
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

        if dist.is_initialized():
            flow_adaptor = torch.nn.parallel.DistributedDataParallel(
                flow_adaptor.to(device),
                device_ids=[local_rank],
                output_device=local_rank,
                static_graph=True,
            )

        batch_size_train = train_dataloader.batch_size * misc.gradient_accumulation_steps * world_size

        if is_main_process:
            logger.info("***** Running training *****")
            logger.info(f"  Num examples = {len(train_dataset)}")
            logger.info(
                f"  Num batch = {((len(train_dataset) // train_dataloader.batch_size) // misc.gradient_accumulation_steps)}")
            logger.info(f"    batch size = {batch_size_train}")
            logger.info(f"  Num minibatch = {len(train_dataset) // train_dataloader.batch_size}")
            logger.info(f"    minibatch size = {train_dataloader.batch_size * world_size}")
            logger.info(
                f"  Num minibatch per device = {len(train_dataset) // train_dataloader.batch_size // world_size}")
            logger.info(f"  Instantaneous minibatch size per device = {train_dataloader.batch_size}")
            logger.info(f"  Gradient Accumulation steps = {misc.gradient_accumulation_steps}")
            logger.info(f"  Num data workers = {misc.train_dataloader.num_workers}")
            logger.info(f"  Num Epochs = {training.num_epochs}")
            logger.info(f"  Num Iterations = {max_train_steps}")
            logger.info(f"  Total optimization step = {max_train_steps // misc.gradient_accumulation_steps}")
            logger.info(
                f"  Total optimization steps per epoch = {len(train_dataset) // (train_dataloader.batch_size * world_size * misc.gradient_accumulation_steps)}")

        global_step = 0
        first_epoch = 0

        if resume is not None:
            logger.info(f"Resuming the training from the checkpoint: {resume}")
            ckpt = torch.load(resume, map_location=device)
            global_step = ckpt['global_step']
            trained_iterations = global_step % ((len(train_dataset) // world_size) // train_dataloader.batch_size)
            first_epoch = int(global_step // ((len(train_dataset) // world_size) // train_dataloader.batch_size))
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            flow_encoder_state_dict = ckpt['flow_encoder_state_dict']
            attention_processor_state_dict = ckpt['attention_processor_state_dict']
            pose_enc_m, pose_enc_u = (flow_adaptor.module if isinstance(flow_adaptor,
                                                                        torch.nn.parallel.DistributedDataParallel) else flow_adaptor).flow_encoder.load_state_dict(
                flow_encoder_state_dict,
                strict=False)
            assert len(pose_enc_m) == 0 and len(pose_enc_u) == 0
            _, attention_processor_u = (flow_adaptor.module if isinstance(flow_adaptor,
                                                                          torch.nn.parallel.DistributedDataParallel) else flow_adaptor).unet.load_state_dict(
                attention_processor_state_dict,
                strict=False)
            assert len(attention_processor_u) == 0
            logger.info(f"Loading the flow encoder and attention processor weights done.")
            logger.info(f"Loading done, resuming training from the {global_step + 1}th iteration")
            lr_scheduler.last_epoch = first_epoch
        else:
            trained_iterations = 0

        # TODO: Check if this really speeds up the training
        #  also test in the inner ddp case
        if device.type == 'cuda':
            pass
            # torch._dynamo.config.verbose = True
            # torch._logging.set_logs(dynamo=logging.NOTSET)
            # vae = torch.compile(vae, disable=not misc.enable_compile, fullgraph=True)
            # text_encoder = torch.compile(text_encoder, disable=not misc.enable_compile, fullgraph=True)
            # flow_adaptor = torch.compile(flow_adaptor, disable=not misc.enable_compile, options=options)
            # raft_model = torch.compile(raft_model, disable=not misc.enable_compile)

        gc.collect()
        torch.cuda.empty_cache()

        def trace_handler(p):
            from torch.cuda._memory_viz import profile_plot
            with open(f"{output_dir}/profile_{p.step_num}_{local_rank}.html", "w") as f:
                f.write(profile_plot(p))
            p.export_chrome_trace(f"{output_dir}/trace_{p.step_num}_{local_rank}.json")

        with torch.profiler.profile(
                with_stack=profiling['with_stack'],
                profile_memory=profiling['memory'],
                record_shapes=profiling['record_shapes'],
                schedule=torch.profiler.schedule(
                    skip_first=10,
                    wait=2,
                    warmup=1,
                    active=2,
                    repeat=1),
                on_trace_ready=lambda x: trace_handler(x),
        ) if profiling['enabled'] else nullcontext() as prof:

            for epoch in range(first_epoch, training.num_epochs):

                for step, (source_videos, source_texts) in enumerate(train_dataloader, start=trained_iterations):

                    if isinstance(prof, torch.profiler.profile):
                        prof.step()

                    with torch.autograd.detect_anomaly() if misc.detect_anomaly else nullcontext():

                        iter_start_time = time.time()
                        logg_dict = {}

                        ### >>>> Training >>>> ###

                        if training.random_null_text > 0:
                            source_texts = [name if random.random() > training.random_null_text else "" for name in
                                            source_texts]

                        # Convert videos to latent space
                        pixel_values = source_videos.to(device, non_blocking=True).contiguous()  # [b, f, c, h, w]
                        pixel_values = (T.ConvertImageDtype(torch.float32)(pixel_values) * 2) - 1

                        # needed not to consider the last frame (which is only used to compute the optical flow)
                        video_length = pixel_values.shape[1] - 1

                        with torch.no_grad():
                            with torch.autocast(device_type=device.type) if misc.enable_amp.vae else nullcontext():

                                # encode the videos
                                if device.type == 'cuda':
                                    vae_stream.wait_stream(torch.cuda.default_stream(device))
                                with torch.cuda.stream(vae_stream) if device.type == 'cuda' else nullcontext():
                                    vae_pixel_values = rearrange(pixel_values[:, :-1], "b f c h w -> (b f) c h w")
                                    latents = encode_videos(vae, vae_pixel_values,
                                                            encode_chunk_size=misc.vae_chunk_size)
                                    latents = rearrange(latents, "(b f) c h w -> b c f h w", f=video_length)
                                    latents = latents * vae.config.scaling_factor

                                    vae_pixel_values.record_stream(vae_stream)
                                    latents.record_stream(vae_stream)
                                    pixel_values.record_stream(vae_stream)

                                # get the flow embedding
                                if device.type == 'cuda':
                                    flow_adaptor_stream.wait_stream(torch.cuda.default_stream(device))
                                with torch.cuda.stream(flow_adaptor_stream) if device.type == 'cuda' else nullcontext():
                                    flow_embedding = get_optical_flow(raft_model, pixel_values, video_length,
                                                                      encode_chunk_size=misc.flow_chunk_size)
                                    pixel_values.record_stream(flow_adaptor_stream)
                                    flow_embedding.record_stream(flow_adaptor_stream)

                        # Sample noise that we'll add to the latents
                        noise = torch.randn_like(latents)  # [b, c, f, h, w]
                        bsz = latents.shape[0]

                        # Sample a random timestep for each video
                        timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,),
                                                  device=latents.device).long()

                        # Add noise to the latents according to the noise magnitude at each timestep
                        # (this is the forward diffusion process)
                        noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)  # [b, c, f, h, w]

                        # Get the text embedding for conditioning
                        with torch.no_grad():

                            if device.type == 'cuda':
                                text_encoder_stream.wait_stream(torch.cuda.default_stream(device))
                            with torch.cuda.stream(text_encoder_stream) if device.type == 'cuda' else nullcontext():
                                prompt_ids = tokenizer(
                                    source_texts, max_length=tokenizer.model_max_length, padding="max_length",
                                    truncation=True,
                                    return_tensors="pt"
                                ).input_ids.to(latents.device, non_blocking=True)
                                prompt_ids.record_stream(text_encoder_stream)
                                encoder_hidden_states = text_encoder(prompt_ids)[0]  # b l c
                                encoder_hidden_states.record_stream(text_encoder_stream)

                        # Predict the noise residual and compute loss
                        # Mixed-precision training
                        with torch.autocast(device_type=device.type) if misc.enable_amp.unet else nullcontext():
                            torch.cuda.default_stream(device).wait_stream(vae_stream)
                            torch.cuda.default_stream(device).wait_stream(flow_adaptor_stream)
                            torch.cuda.default_stream(device).wait_stream(text_encoder_stream)

                            model_pred = flow_adaptor(noisy_latents,
                                                      timesteps,
                                                      encoder_hidden_states=encoder_hidden_states,
                                                      flow_embedding=flow_embedding)  # [b c f h w]

                            # Get the target for loss depending on the prediction type
                            if noise_scheduler.config.prediction_type == "epsilon":
                                target = noise
                            elif noise_scheduler.config.prediction_type == "v_prediction":
                                target = noise_scheduler.get_velocity(latents, noise, timesteps)
                            else:
                                raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

                            loss = F.mse_loss(model_pred, target, reduction="mean")
                            logg_dict.update({
                                "train/loss": loss.item(),
                            })
                            loss = loss / misc.gradient_accumulation_steps

                        # Backpropagation
                        if misc.enable_amp.unet:
                            scaler.scale(loss).backward()
                        else:
                            loss.backward()

                    # optimize
                    if (global_step + 1) % misc.gradient_accumulation_steps == 0:

                        if misc.enable_amp.unet:
                            scaler.unscale_(optimizer)

                        grad_norm = torch.nn.utils.clip_grad_norm_(
                            filter(lambda p: p.requires_grad, flow_adaptor.parameters()),
                            optimization.max_grad_norm)

                        if misc.enable_amp.unet:
                            scaler.step(optimizer)
                            scaler.update()
                        else:
                            optimizer.step()

                        lr_scheduler.step()
                        optimizer.zero_grad(set_to_none=True)

                        logg_dict.update({
                            "train/grad_norm": grad_norm.item(),
                        })
                    iter_end_time = time.time()

                    logg_dict.update({
                        "train/lr": lr_scheduler.get_last_lr()[0],
                        "train/iter_time": iter_end_time - iter_start_time,
                    })

                    # Save checkpoint
                    if is_main_process and (global_step % logging_args.checkpointing_every.steps == 0):
                        save_model(
                            output_dir,
                            epoch,
                            global_step,
                            flow_adaptor,
                            optimizer,
                            logger
                        )

                    # Periodically validation
                    if (global_step == 0 and validation.first) or (global_step > 0 and validation.every_steps != 0 and (
                            global_step % validation.every_steps == 0)):

                        generator_val.set_state(generator_val_init_state)
                        flow_adaptor.eval()

                        for idx, (source_videos, source_texts) in enumerate(validation_dataloader):
                            val_pixel_values = source_videos.to(unet.device) # [b, f, c, h, w]
                            val_pixel_values = T.ConvertImageDtype(torch.float32)(val_pixel_values)
                            video_length = val_pixel_values.shape[1] - 1
                            flow_val_gt = get_optical_flow(raft_model, (val_pixel_values * 2) - 1, video_length,
                                                           encode_chunk_size=misc.flow_chunk_size)
                            # (B, 2, F, H, W)

                            samples_flow, samples_no_flow = validation_pipeline(
                                prompt=source_texts,
                                flow_embedding=flow_val_gt,
                                output_type="pt",
                                video_length=video_length,
                                height=validation_dataset.video_size[0],
                                width=validation_dataset.video_size[1],
                                num_inference_steps=30,
                                guidance_scale=7.0,
                                generator=generator_val,
                                val_scale_factor_spatial=1.0,
                                val_scale_factor_temporal=1.0,
                                return_dict=False,
                                generate_no_flow=True
                            )  # [b, f, 3, h, w]

                            flow_adaptor.train()

                            flow_imgs_gt = []
                            for flow in flow_val_gt:
                                flow_imgs_gt.append(flow_to_image(rearrange(flow, "c f h w -> f c h w")))
                            flow_imgs_gt = torch.stack(flow_imgs_gt, dim=0)

                            # compute the flow embedding on the generated samples
                            flow_val_gener = get_optical_flow(raft_model, (samples_flow*2)-1, video_length - 1,
                                                              encode_chunk_size=misc.flow_chunk_size)

                            # compute the difference between the generated flow and the ground truth flow
                            diff = flow_val_gener - flow_val_gt[:, :, :-1]
                            diff = diff.abs().mean()

                            logg_dict.update({
                                "validation/flow_diff": diff.item(),
                            })

                            flow_val_gener = torch.cat([flow_val_gener, flow_val_gener[:, :, -1:]], dim=2)

                            flow_imgs_gener = []
                            for flow in flow_val_gener:
                                flow_imgs_gener.append(flow_to_image(rearrange(flow, "c f h w -> f c h w")))
                            flow_imgs_gener = torch.stack(flow_imgs_gener, dim=0)

                            sample_to_save = torch.cat(
                                [flow_imgs_gener, samples_flow, samples_no_flow, flow_imgs_gt,
                                 val_pixel_values[:, :video_length]],
                                dim=3)  # [b, f, 3, 5h, w]

                            sample_to_save = rearrange(sample_to_save, "b f c h w -> c f h (b w)")
                            save_path = f"{output_dir}/samples/sample-{global_step}/rank-{global_rank}-{idx}.gif"
                            vids = get_video(sample_to_save.cpu(), save_path)
                            os.makedirs(os.path.dirname(save_path), exist_ok=True)
                            with open(save_path, "wb") as outfile:
                                outfile.write(vids.getbuffer())
                            logger.info(f"Saved samples to {save_path}")
                            break

                    run.log(logg_dict)

                    if (global_step % logging_args.interval) == 0 or global_step == 0:
                        rank_loss = loss.detach()
                        if dist.is_initialized():
                            global_loss = rank_loss.clone()
                            torch.distributed.all_reduce(global_loss, op=dist.ReduceOp.AVG)
                        else:
                            global_loss = rank_loss

                        msg = (
                                f"Iter: {global_step}/{max_train_steps}, "
                                f"Loss: {global_loss.item(): .4f}, " +
                                str(f"Rank loss: {rank_loss.item(): .4f}, " if dist.is_initialized() else "") +
                                f"lr: {lr_scheduler.get_last_lr()}, "
                                f"Iter time: {format_time(iter_end_time - iter_start_time)}, "
                                f"ETA: {format_time((iter_end_time - iter_start_time) * (max_train_steps - global_step))}")
                        logger.info(msg)

                    global_step += 1


    finally:
        try:
            shutil.rmtree(dataset_cache_dir)
        except UnboundLocalError as e:
            pass
        if dist.is_initialized():
            dist.destroy_process_group()

        wandb.finish()


@hydra.main(config_path="../configs", config_name="default", version_base=None)
def main(cfg: omegaconf.DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    train(output_dir=output_dir, full_config=cfg, **cfg)


if __name__ == "__main__":
    main()
