import torch
from einops import rearrange

@torch.no_grad()
def get_optical_flow(raft_model, pixel_values, video_length, encode_chunk_size=48, num_flow_updates=14):
    imgs_1 = pixel_values[:, :-1]
    imgs_2 = pixel_values[:, 1:]
    imgs_1 = rearrange(imgs_1, "b f c h w -> (b f) c h w")
    imgs_2 = rearrange(imgs_2, "b f c h w -> (b f) c h w")

    flow_embedding = []

    for i in range(0, imgs_1.shape[0], encode_chunk_size):
        imgs_1_chunk = imgs_1[i:i + encode_chunk_size]
        imgs_2_chunk = imgs_2[i:i + encode_chunk_size]
        flow_embedding_chunk = raft_model(imgs_1_chunk, imgs_2_chunk, num_flow_updates)[-1]
        flow_embedding.append(flow_embedding_chunk)

    flow_embedding = torch.cat(flow_embedding).contiguous()
    flow_embedding = rearrange(flow_embedding, "(b f) c h w -> b c f h w", f=video_length)

    return flow_embedding