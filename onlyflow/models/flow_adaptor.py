import math

import torch
import torch.nn as nn
from einops import rearrange
from torch.utils import checkpoint

from onlyflow.models.attention import BasicTransformerBlock


def get_parameter_dtype(parameter: torch.nn.Module):
    params = tuple(parameter.parameters())
    if len(params) > 0:
        return params[0].dtype

    buffers = tuple(parameter.buffers())
    if len(buffers) > 0:
        return buffers[0].dtype


def conv_nd(dims, *args, **kwargs):
    """
    Create a 1D, 2D, or 3D convolution module.
    """
    if dims == 1:
        return nn.Conv1d(*args, **kwargs)
    elif dims == 2:
        return nn.Conv2d(*args, **kwargs)
    elif dims == 3:
        return nn.Conv3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")


def avg_pool_nd(dims, *args, **kwargs):
    """
    Create a 1D, 2D, or 3D average pooling module.
    """
    if dims == 1:
        return nn.AvgPool1d(*args, **kwargs)
    elif dims == 2:
        return nn.AvgPool2d(*args, **kwargs)
    elif dims == 3:
        return nn.AvgPool3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")


class FlowAdaptor(nn.Module):
    def __init__(self, unet, flow_encoder, ckpt_act=True):
        super().__init__()
        self.unet = unet
        self.flow_encoder = flow_encoder
        self.ckpt_act = ckpt_act

    def forward(self, noisy_latents, timesteps, encoder_hidden_states, flow_embedding):
        assert flow_embedding.ndim == 5
        bs = flow_embedding.shape[0]  # b c f h w
        flow_embedding_features = self.flow_encoder(flow_embedding)  # flow_embedding b f c h w
        flow_embedding_features = [rearrange(x, '(b f) c h w -> b c f h w', b=bs)
                                   for x in flow_embedding_features]

        added_cond_kwargs = {'flow_embedding_features': flow_embedding_features}

        noise_pred = self.unet(noisy_latents,
                               timesteps,
                               encoder_hidden_states,
                               added_cond_kwargs=added_cond_kwargs,
                               )

        return noise_pred.sample


class Downsample(nn.Module):
    """
    A downsampling layer with an optional convolution.
    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 downsampling occurs in the inner-two dimensions.
    """

    def __init__(self, channels, use_conv, dims=2, out_channels=None, padding=1):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        stride = 2 if dims != 3 else (1, 2, 2)
        if use_conv:
            self.op = conv_nd(dims, self.channels, self.out_channels, 3, stride=stride, padding=padding)
        else:
            assert self.channels == self.out_channels
            self.op = avg_pool_nd(dims, kernel_size=stride, stride=stride)

    def forward(self, x):
        assert x.shape[1] == self.channels
        return self.op(x)


class ResnetBlock(nn.Module):

    def __init__(self, in_c, out_c, down, ksize=3, sk=False, use_conv=True):
        super().__init__()
        ps = ksize // 2
        if in_c != out_c or sk == False:
            self.in_conv = nn.Conv2d(in_c, out_c, ksize, 1, ps)
        else:
            self.in_conv = None
        self.block1 = nn.Conv2d(out_c, out_c, 3, 1, 1)
        self.act = nn.ReLU()
        self.block2 = nn.Conv2d(out_c, out_c, ksize, 1, ps)
        if not sk:
            self.skep = nn.Conv2d(in_c, out_c, ksize, 1, ps)
        else:
            self.skep = None

        self.down = down
        if self.down:
            self.down_opt = Downsample(in_c, use_conv=use_conv)

    def forward(self, x):
        if self.down:
            x = self.down_opt(x)
        if self.in_conv is not None:  # edit
            x = self.in_conv(x)

        h = self.block1(x)
        h = self.act(h)
        h = self.block2(h)
        if self.skep is not None:
            return h + self.skep(x)
        else:
            return h + x


class PositionalEncoding(nn.Module):
    def __init__(
            self,
            d_model,
            dropout=0.,
            max_len=32,
    ):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2, ...] = torch.sin(position * div_term)
        pe[0, :, 1::2, ...] = torch.cos(position * div_term)
        pe.unsqueeze_(-1).unsqueeze_(-1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), ...]
        return self.dropout(x)


class FlowEncoder(nn.Module):

    def __init__(self,
                 downscale_factor,
                 channels=None,
                 nums_rb=3,
                 ksize=3,
                 sk=False,
                 use_conv=True,
                 compression_factor=1,
                 temporal_attention_nhead=8,
                 positional_embeddings=None,
                 num_positional_embeddings=16,
                 rescale_output_factor=1.0,
                 checkpointing=False):
        super(FlowEncoder, self).__init__()
        if channels is None:
            channels = [320, 640, 1280, 1280]

        self.checkpointing = checkpointing
        self.unshuffle = nn.PixelUnshuffle(downscale_factor)
        self.channels = channels
        self.nums_rb = nums_rb
        self.encoder_down_conv_blocks = nn.ModuleList()
        self.encoder_down_attention_blocks = nn.ModuleList()
        for i in range(len(channels)):
            conv_layers = nn.ModuleList()
            temporal_attention_layers = nn.ModuleList()
            for j in range(nums_rb):
                if j == 0 and i != 0:
                    in_dim = channels[i - 1]
                    out_dim = int(channels[i] / compression_factor)
                    conv_layer = ResnetBlock(in_dim, out_dim, down=True, ksize=ksize, sk=sk, use_conv=use_conv)
                elif j == 0:
                    in_dim = channels[0]
                    out_dim = int(channels[i] / compression_factor)
                    conv_layer = ResnetBlock(in_dim, out_dim, down=False, ksize=ksize, sk=sk, use_conv=use_conv)
                elif j == nums_rb - 1:
                    in_dim = channels[i] / compression_factor
                    out_dim = channels[i]
                    conv_layer = ResnetBlock(in_dim, out_dim, down=False, ksize=ksize, sk=sk, use_conv=use_conv)
                else:
                    in_dim = int(channels[i] / compression_factor)
                    out_dim = int(channels[i] / compression_factor)
                    conv_layer = ResnetBlock(in_dim, out_dim, down=False, ksize=ksize, sk=sk, use_conv=use_conv)
                temporal_attention_layer = BasicTransformerBlock(
                    dim=out_dim,
                    num_attention_heads=temporal_attention_nhead,
                    attention_head_dim=int(out_dim / temporal_attention_nhead),
                    dropout=0.0,
                    positional_embeddings=positional_embeddings,
                    num_positional_embeddings=num_positional_embeddings
                )
                conv_layers.append(conv_layer)
                temporal_attention_layers.append(temporal_attention_layer)
            self.encoder_down_conv_blocks.append(conv_layers)
            self.encoder_down_attention_blocks.append(temporal_attention_layers)

        self.encoder_conv_in = nn.Conv2d(2 * (downscale_factor ** 2), channels[0], 3, 1, 1)

    @property
    def dtype(self) -> torch.dtype:
        """
        `torch.dtype`: The dtype of the module (assuming that all the module parameters have the same dtype).
        """
        return get_parameter_dtype(self)

    def forward(self, x):
        # unshuffle
        bs = x.shape[0]
        x = rearrange(x, "b c f h w -> (b f) c h w")
        x = self.unshuffle(x)
        # extract features
        features = []
        x = self.encoder_conv_in(x)
        for i, (res_block, attention_block) in enumerate(
                zip(self.encoder_down_conv_blocks, self.encoder_down_attention_blocks)):
            for j, (res_layer, attention_layer) in enumerate(zip(res_block, attention_block)):
                if self.checkpointing:
                    x = checkpoint.checkpoint(res_layer, x, use_reentrant=False)
                else:
                    x = res_layer(x)
                h, w = x.shape[-2:]
                x = rearrange(x, '(b f) c h w -> (b h w) f c', b=bs)
                if self.checkpointing:
                    x = checkpoint.checkpoint(attention_layer, x, use_reentrant=False)
                else:
                    x = attention_layer(x)
                x = rearrange(x, '(b h w) f c -> (b f) c h w', h=h, w=w)
            features.append(x)
        return features
