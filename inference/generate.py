import gc

import torch
import torchvision
from huggingface_hub import hf_hub_download

from diffusers import MotionAdapter, DDIMScheduler, StableDiffusionPipeline, AnimateDiffPipeline
from diffusers.utils import export_to_video, load_image
from torchvision.models.optical_flow import raft_large, Raft_Large_Weights

import torchvision.transforms as T

from models.flow_adaptor import FlowAdaptor, FlowEncoder
from models.unet import UNetMotionModel
from pipelines.pipeline_animation import FlowCtrlPipeline
from tools.optical_flow import get_optical_flow

device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

torch.backends.cuda.enable_cudnn_sdp(False)

# load SD 1.5 based finetuned model
model_id = "stable-diffusion-v1-5/stable-diffusion-v1-5"
scheduler = DDIMScheduler.from_pretrained(
    model_id,
    subfolder="scheduler",
    clip_sample=False,
    timestep_spacing="linspace",
    beta_schedule="linear",
    steps_offset=1,
    local_files_only=True
)

base_pipe = StableDiffusionPipeline.from_pretrained(model_id, local_files_only=True)

base_pipe.load_ip_adapter('h94/IP-Adapter', subfolder='models', weight_name='ip-adapter_sd15.bin', local_files_only=True)
base_pipe.set_ip_adapter_scale(0.5)

# Load models and pipeline
motion_adapter = MotionAdapter.from_pretrained("guoyww/animatediff-motion-adapter-v1-5-3", local_files_only=True)

unet = UNetMotionModel.from_unet2d(
            base_pipe.unet,
            motion_adapter=motion_adapter
)

unet.set_all_attn(
            flow_channels=[ 320, 640, 1280, 1280 ],
            add_spatial=False,
            add_temporal=True,
            encoder_only=False,
            query_condition=True,
            key_value_condition=True,
            flow_scale=1.0,
        )

flow_encoder = FlowEncoder(
            downscale_factor= 8,
            channels = [320, 640, 1280, 1280],
            nums_rb= 2,
            ksize= 1,
            sk = True,
            use_conv = False,
            compression_factor = 1,
            temporal_attention_nhead = 8,
            positional_embeddings = "sinusoidal",
            num_positional_embeddings= 16,
            checkpointing=False
        )

flow_adaptor = FlowAdaptor(unet, flow_encoder).eval().requires_grad_(False)

pipe = FlowCtrlPipeline(
        vae=base_pipe.vae,
        text_encoder=base_pipe.text_encoder,
        tokenizer=base_pipe.tokenizer,
        unet=flow_adaptor.unet,
        motion_adapter=motion_adapter,
        flow_encoder=flow_adaptor.flow_encoder,
        image_encoder=base_pipe.image_encoder,
        feature_extractor=base_pipe.feature_extractor,
        scheduler=scheduler,
)

ckpt_path = hf_hub_download('obvious-research/onlyflow', 'weights_fp16.ckpt')

ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=True)
flow_encoder_state_dict = ckpt['flow_encoder_state_dict']
attention_processor_state_dict = ckpt['attention_processor_state_dict']
pose_enc_m, pose_enc_u = flow_adaptor.flow_encoder.load_state_dict(
    flow_encoder_state_dict,
    strict=False,
)
assert len(pose_enc_m) == 0 and len(pose_enc_u) == 0
_, attention_processor_u = flow_adaptor.unet.load_state_dict(
    attention_processor_state_dict,
    strict=False,
)
assert len(attention_processor_u) == 0

del ckpt, base_pipe, flow_encoder_state_dict, attention_processor_state_dict, pose_enc_m, pose_enc_u, attention_processor_u

gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

# Can be a single prompt, or a dictionary with frame timesteps
positive_prompt = "Soft sand dune with pastel rainbow stripes under a clear blue sky, minimal and dreamy landscape"
negative_prompt = "bad quality, worst quality"

width = 512
height = 512
video_length = 40

temporal_downscale = 1
guidance_scale = 7.5
num_inference_steps = 30

context_length = 16
context_stride = 12

# Memory optim settings:

# requirements:
# 1. width % 64 == 0
# 2. height % 64 == 0
# 3. chunk_size has to divide temporal_split_size
# 4. chunk_size has to divide spatial_split_size
# 5. chunk_size has to divide (2 * (width/64) * (height/64))
# 6. if (2*video_length) % temporal_split_size != 0, chunk_size has to divide (2*video_length) as well

pipe.enable_model_cpu_offload()
pipe.enable_vae_slicing()
pipe.enable_vae_tiling()
pipe.enable_free_noise(
    context_length=16,
    context_stride=12,
)
pipe.enable_free_noise_split_inference(
    temporal_split_size=16,
    spatial_split_size=256
)
pipe.unet.enable_forward_chunking(16)


# Load RAFT model
raft_model = raft_large(weights=Raft_Large_Weights.DEFAULT, progress=False).to(device).eval().requires_grad_(False)

# Load video
video = torchvision.io.read_video("input.mp4", output_format='TCHW', pts_unit='sec')[0]

num_sup_frames = (video_length-context_length) % context_stride
if num_sup_frames != 0:
    print(f"Warning: video length ({video_length}) and context stride ({context_stride}) are not well compatible, {num_sup_frames} frames will be rendered without with much consistency")

if video_length * temporal_downscale > video.shape[0]:
    raise ValueError("Video is too short for the specified length and downscale factor")
video = video[:temporal_downscale*video_length+1:temporal_downscale]

# Preprocess video
video = T.CenterCrop((min(video.shape[2], video.shape[3])))(video)

# Resize video
video = T.Resize((height, width))(video).unsqueeze(0).contiguous()

# Normalize videos
video = T.ConvertImageDtype(torch.float32)(video).to(device)

print("Video shape:", video.shape)

optical_flow = get_optical_flow(
    raft_model,
    pixel_values=(video * 2) - 1,
    video_length=video_length,
    encode_chunk_size=16
).to('cpu')

del raft_model, video
if torch.cuda.is_available():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()


image = load_image("image.jpeg")

print("optical flow shape:", optical_flow.shape)

output = pipe(
    prompt=positive_prompt,
    negative_prompt=negative_prompt,
    optical_flow=optical_flow,
    width=width,
    height=height,
    num_frames=video_length,
    guidance_scale=guidance_scale,
    num_inference_steps=num_inference_steps,
    val_scale_factor_temporal=0.75,
    generator=torch.Generator("cpu").manual_seed(0),
    ip_adapter_image=image,
    output_type="pil",
    motion_cross_attention_kwargs={'attn_scale': 1.0, 'attn_scale_flow': 1.0},
)

# Save video
frames = output.frames[0]
export_to_video(frames, "output.mp4", fps=8)
export_to_video(frames, "output.gif", fps=8)
