import logging
from collections import OrderedDict
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor
from torch.nn import LayerNorm


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])

    def forward(self, x: torch.Tensor):
        return self.resblocks(x)


class VisionEncoder(nn.Module):
    """
    根据CLIP的VisionTransformer修改而来

    修改：
    1.适配高光谱
    2.返回中间层特征的视觉部分
    3.最后投影改为包含类别和视觉全部特征，并全部返回
    """

    def __init__(self,
                 in_channels,
                 image_size,
                 patch_size,
                 hidden_dim=768,
                 layers=12,
                 heads=12,
                 output_dim=512,
                 out_indices=(3, 5, 7, 11),
                 pretrained: str = 'pretrained/ViT-B-16.pt'):
        super().__init__()

        self.image_size = image_size
        self.output_dim = output_dim
        self.out_indices = out_indices
        self.pretrained = pretrained
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=hidden_dim, kernel_size=patch_size, stride=patch_size, bias=False)

        scale = hidden_dim ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(hidden_dim))
        self.positional_embedding = nn.Parameter(scale * torch.randn((image_size // patch_size) ** 2 + 1, hidden_dim))
        self.ln_pre = LayerNorm(hidden_dim)

        self.transformer = Transformer(hidden_dim, layers, heads)

        self.ln_post = LayerNorm(hidden_dim)
        self.proj = nn.Parameter(scale * torch.randn(hidden_dim, output_dim))

    def forward(self, x: torch.Tensor):
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        img = x.clone()

        b, c, h, w = x.shape
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        features = [img]
        for i, resblock in enumerate(self.transformer.resblocks):
            x = resblock(x)
            if i in self.out_indices:
                x_feat = rearrange(x[1:, :, :], '(h w) b c -> b c h w', h=h, w=w)
                features.append(x_feat.contiguous())
        x = x.permute(1, 0, 2)  # LND -> NLD

        x = self.ln_post(x)

        if self.proj is not None:
            x = x @ self.proj
        cls_feat = x[:, 0]
        features.append(cls_feat.contiguous())
        return features


class TextEncoder(nn.Module):
    """
    根据CLIP的encode_text函数修改而来
    """

    def __init__(self,
                 context_length=77,
                 vocab_size=49408,
                 hidden_dim=512,
                 layers=12,
                 heads=8,
                 output_dim=512,
                 pretrained: str = 'pretrained/ViT-B-16.pt'):
        super().__init__()

        self.context_length = context_length
        self.pretrained = pretrained
        self.token_embedding = nn.Embedding(vocab_size, hidden_dim)
        self.positional_embedding = nn.Parameter(torch.empty(self.context_length, hidden_dim))
        self.transformer = Transformer(
            width=hidden_dim,
            layers=layers,
            heads=heads,
            attn_mask=self.build_attention_mask()
        )
        self.ln_final = LayerNorm(hidden_dim)
        self.text_projection = nn.Parameter(torch.empty(hidden_dim, output_dim))

    def build_attention_mask(self):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask

    def forward(self, text):
        x = self.token_embedding(text)  # [batch_size, n_ctx, d_model]

        x = x + self.positional_embedding
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection

        return x


class ResidualCrossAttentionBlock_old(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()

        self.text_img_attn = nn.MultiheadAttention(d_model, n_head)
        self.img_text_attn = nn.MultiheadAttention(d_model, n_head)
        self.text_attn = nn.MultiheadAttention(d_model, n_head)
        self.img_attn = nn.MultiheadAttention(d_model, n_head)
        self.text_ln_1 = LayerNorm(d_model)
        self.img_ln_1 = LayerNorm(d_model)
        self.text_mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.img_mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.text_ln_2 = LayerNorm(d_model)
        self.img_ln_2 = LayerNorm(d_model)
        self.text_ln_3 = LayerNorm(d_model)
        self.img_ln_3 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def forward(self, img: torch.Tensor, text: torch.Tensor):
        text_ln = self.text_ln_1(text)
        img_ln = self.img_ln_1(img)
        text = text + self.text_img_attn(text_ln, img_ln, img_ln, need_weights=False)[0]
        img = img + self.img_text_attn(img_ln, text_ln, text_ln, need_weights=False)[0]
        q = k = v = self.text_ln_2(text)
        text = text + self.text_attn(q, k, v, need_weights=False)[0]
        q = k = v = self.img_ln_2(img)
        img = img + self.img_attn(q, k, v, need_weights=False)[0]
        text = text + self.text_mlp(self.text_ln_3(text))
        img = img + self.img_mlp(self.img_ln_3(img))
        return img, text


class ResidualCrossAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, q: torch.Tensor, kv: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=q.dtype, device=q.device) if self.attn_mask is not None else None
        return self.attn(q, kv, kv, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, q: torch.Tensor, kv: torch.Tensor):
        q = self.ln_1(q)
        kv = self.ln_1(kv)
        q = q + self.attention(q, kv)
        q = q + self.mlp(self.ln_2(q))
        return q


class CrossTransformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, q: str, kv: str, attn_mask: torch.Tensor = None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.q = q
        self.kv = kv
        self.resblocks = nn.ModuleList([ResidualCrossAttentionBlock(width, heads, attn_mask) for _ in range(layers)])
        self.norm = nn.LayerNorm(width)

    def forward(self, img: torch.Tensor, text: torch.Tensor):
        """

        :param img: (B 512 5 5)
        :param text: (B K 512)
        :return:
        """
        img = img.reshape(img.shape[0], img.shape[1], -1)  # B 512 25 NDL
        img = img.permute(2, 0, 1)  # NDL -> LND
        text = text.permute(1, 0, 2)  # NLD -> LND
        q, kv = (img, text) if self.q == 'img' and self.kv == 'text' else (text, img)
        for block in self.resblocks:
            q = block(q, kv)
        return self.norm(q)


class FPN(nn.Module):
    def __init__(self,
                 in_channels: List[int],
                 out_channels: int,
                 upsample_mode='bilinear'):
        super().__init__()
        # 构建 FPN 的上采样和 1x1 卷积层
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(in_ch, out_channels, kernel_size=1)
            for in_ch in in_channels
        ])
        self.fpn_convs = nn.ModuleList([
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
            for _ in in_channels
        ])
        self.upsample_mode = upsample_mode

    def forward(self, features):
        num_feat = len(features)
        # 调整通道
        lateral_feats = [
            lateral_conv(f)
            for lateral_conv, f in zip(self.lateral_convs, features)
        ]

        # 从底层特征图开始逐层上采样和融合
        for i in range(num_feat - 1, 0, -1):
            prev_shape = lateral_feats[i - 1].shape[2:]
            if lateral_feats[i].shape[2:] != prev_shape:
                upsample = nn.Upsample(size=prev_shape, mode=self.upsample_mode)
                lateral_feats[i] = upsample(lateral_feats[i])
            lateral_feats[i - 1] += lateral_feats[i]

        # 3*3卷积进一步融合，size不变
        outs = [
            fpn_conv(f)
            for fpn_conv, f in zip(self.fpn_convs, lateral_feats)
        ]

        # 返回处理后的特征图
        return outs


class FeatureFusion(nn.Module):
    def __init__(self,
                 k: int,
                 hidden_dim: int,
                 layers: int,
                 heads: int,
                 ):
        super().__init__()
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.k = k
        self.cross_tf = CrossTransformer(width=hidden_dim, layers=layers, heads=heads, q='text', kv='img')
        self.alpha = nn.Parameter(torch.full([], 0.5))

    def similarity(self, img, text):
        """
        计算相似度
        :param img: (B,512)
        :param text: (57,512)
        :return: sim (B,K), idx (B,K)
        """
        text = text / text.norm(dim=1, keepdim=True)
        img = img / img.norm(dim=1, keepdim=True)

        logit_scale = self.logit_scale.exp()
        sim = logit_scale * img @ text.t()  # B,512 * 512,57 = B,57
        sim, idx = sim.topk(self.k, dim=1)  # B,K B,K
        return sim, idx

    def kl_div(self, img, text):
        """
        计算KL散度
        :param img: (B,512)
        :param text: (B,k,512)
        :return: kl: (B K)
        """
        img = img.unsqueeze(1)  # B 1 512

        # 归一化为概率分布（softmax / log_softmax）
        img_prob = F.softmax(img, dim=-1)
        text_log_prob = F.log_softmax(text, dim=-1)

        # KL 散度：KL(text_k || img)
        kl = F.kl_div(text_log_prob, img_prob, reduction='none').mean(dim=-1)  # (B, K)
        return kl

    def forward(self, img: List[Tensor], text: Tensor):
        """

        :param img: [(B 512 5 5), (B 512)]
        :param text: (57 512)
        :return:
        """
        img_vis, img_cls = img
        sim, idx = self.similarity(img_cls, text)
        text = text[idx]  # B K 512
        kl = self.kl_div(img_cls, text)
        sim = F.softmax(sim, dim=-1)
        kl = F.softmax(-kl, dim=-1)
        cross_text = self.cross_tf(img_vis, text)  # (B 512 5 5) (B K 512)
        cross_text = cross_text.permute(1, 0, 2)  # LND -> NLD
        text = (self.alpha * sim + (1 - self.alpha) * kl).unsqueeze(-1) * text + cross_text + text
        score_map = torch.einsum('bchw,bkc->bkhw', img_vis, text)  # B,512,5,5 * B,k,512 = B,k,5,5
        img_vis = torch.cat((img_vis, score_map), dim=1)
        return img_vis, text


class GaussianFreqSub(nn.Module):
    def __init__(self, in_channels, hidden_dim, sigma_ratio=0.2):
        """
        sigma_ratio: 与图像尺寸相关的sigma比率，用于确定高斯滤波器sigma的比例值，越大则低通衰减更慢(带宽更宽)。
                     典型取值可在[0.05 ~ 0.5]之间，根据实际任务调节。
        learnable_gating: 是否使用可学习 gating, 如果=False则默认为常数 0.5
        """
        super().__init__()
        self.sigma_ratio = sigma_ratio
        self.conv = nn.Conv2d(in_channels, hidden_dim, kernel_size=1)
        # 这里用一个标量做示例，也可以改成更精细的可学习映射
        self.gating_scale = nn.Parameter(torch.zeros(1, hidden_dim, 1, 1))

    def forward(self, t1, t2):
        """
        x: (B, C, H, W) 的实数张量 (图像或特征图)
        return: (B, C, H, W), 与 x 同大小
        """
        B, C, H, W = t1.shape

        t1 = self.conv(t1)
        t2 = self.conv(t2)

        # 1) 计算 FFT (默认对最后两个空间维度)
        t1_freq = torch.fft.fft2(t1, dim=(-2, -1))  # (B, C, H, W), 复数
        t2_freq = torch.fft.fft2(t2, dim=(-2, -1))

        # 2) 构建高斯型低通核 (Gaussian LPF)
        # 先计算每个像素 (u,v) 到中心 (H/2, W/2) 的距离
        # 为方便与以往一致，u,v 取 [-H/2, H/2), [-W/2, W/2)
        # dist(u,v)^2 = (u^2 + v^2)
        u = torch.arange(H, device=t1.device).view(H, 1).expand(H, W) - (H / 2)
        v = torch.arange(W, device=t1.device).view(1, W).expand(H, W) - (W / 2)
        dist_sq = (u ** 2 + v ** 2)

        # sigma 的绝对值 => sigma_ratio * (图像中心半径),可自行设计
        # 例如 min(H/2, W/2) 可以视作最大可用半径
        max_radius = min(H / 2, W / 2)
        sigma = self.sigma_ratio * max_radius

        # 高斯滤波器: LPF(u,v) = exp( - dist^2 / (2 sigma^2) )
        # dist_sq 和 sigma^2 都是标量/张量 => 得到 (H,W) mask
        gaussian_LPF = torch.exp(- dist_sq / (2.0 * sigma ** 2))  # (H, W)

        # 适配 batch/channel 维度
        gaussian_LPF = gaussian_LPF.view(1, 1, H, W)  # 广播到 (B,C,H,W) 时自动扩展

        # 3) 频域点乘得到低通/高通分量
        t1_freq_low = t1_freq * gaussian_LPF
        t1_freq_high = t1_freq * (1 - gaussian_LPF)
        t2_freq_low = t2_freq * gaussian_LPF
        t2_freq_high = t2_freq * (1 - gaussian_LPF)

        sub_freq_low = abs(t1_freq_low - t2_freq_low)
        sub_freq_high = abs(t1_freq_high - t2_freq_high)

        # 4) 简单 gating 融合
        gating_val = torch.sigmoid(self.gating_scale)
        sub_freq = gating_val * sub_freq_low + (1.0 - gating_val) * sub_freq_high

        # 5) 逆变换回时域
        y = torch.fft.ifft2(sub_freq, dim=(-2, -1)).real
        return y


class ChangeAware(nn.Module):
    def __init__(self,
                 in_channels: int,
                 last_in_channels: int,
                 num_feature: int = 5):
        super().__init__()
        self.num_feature = num_feature
        self.last_conv = nn.Conv2d(last_in_channels, in_channels, kernel_size=1)
        self.conv = nn.ModuleList([nn.Conv2d(in_channels, in_channels // 2, kernel_size=1) for _ in range(num_feature)])
        self.fpn = FPN(in_channels=[in_channels * 2] * (num_feature - 1) + [last_in_channels * 2], out_channels=in_channels // 2)
        self.freq = GaussianFreqSub(in_channels=in_channels, hidden_dim=in_channels // 2)

    def forward(self, t1, t2):
        """

        :param t1: [B 512 5 5]*5
        :param t2: [B 512 5 5]*5
        :return:
        """
        fpn = self.fpn([torch.cat([t1, t2], dim=1) for t1, t2 in zip(t1, t2)])
        features = []
        for idx, (t1, t2) in enumerate(zip(t1, t2)):
            if idx == self.num_feature - 1:
                t1 = self.last_conv(t1)
                t2 = self.last_conv(t2)
            sub = self.conv[idx](torch.abs(t1 - t2))
            cos = 1 - F.cosine_similarity(t1, t2).unsqueeze(1)
            fpn_cos = cos * fpn[idx]
            freq = self.freq(t1, t2)
            features.append(torch.cat([sub, fpn_cos, fpn[idx], freq], dim=1))
        return features


class TextChannelAttention(nn.Module):
    def __init__(self, hidden_dim: int, layers: int, heads: int, reduction: int = 16, text_dim=512):
        super().__init__()
        self.linear = nn.Linear(text_dim, hidden_dim)
        self.cross_tf = CrossTransformer(width=hidden_dim, layers=layers, heads=heads, q='img', kv='text')
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // reduction, hidden_dim, bias=False),
            nn.Sigmoid()
        )

    def forward(self, img_list: List[Tensor], text: Tensor):
        b, c, _, _ = img_list[0].shape
        text = self.linear(text)
        new_img = []
        for img in img_list:
            cross_img = self.cross_tf(img, text)
            cross_img = cross_img.permute(1, 2, 0)  # LND->NDL
            cross_img = self.pool(cross_img).view(b, c)
            att = self.fc(cross_img).view(b, c, 1, 1)
            new_img.append(img * att.expand_as(img) + img)
        new_img = torch.cat(new_img, dim=1)  # [B 2048 5 5]*5
        return new_img


class Decoder(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 4, kernel_size=3, padding=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_channels // 4, in_channels // 8, kernel_size=3, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_channels // 8, in_channels // 16, kernel_size=3, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_channels // 16, in_channels // 32, kernel_size=1, bias=False)
        )
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(in_channels // 32, 2, bias=False)

    def forward(self, x: List[Tensor]):
        x = self.fc(self.flatten(self.conv(x)))
        return x


class PromptHSICD(nn.Module):
    def __init__(self,
                 in_channels,
                 image_size,
                 patch_size,
                 # VisionEncoder
                 vis_hidden_dim=768,
                 vis_output_dim=512,
                 # FeatureFusion
                 k=10,
                 fusion_hidden_dim=512,
                 fusion_layer=3,
                 fusion_heads=8,
                 # TextChannelAttention
                 att_layer: int = 1,
                 att_heads: int = 8,
                 vis_dim=512,
                 pretrained: str = 'pretrained/ViT-B-16.pt'):
        super().__init__()

        self.pretrained = pretrained

        self.text_encoder = TextEncoder()
        self.vision_encoder = VisionEncoder(in_channels=in_channels, image_size=image_size, patch_size=patch_size, hidden_dim=vis_hidden_dim, output_dim=vis_output_dim)
        self.neck = FPN(in_channels=[vis_hidden_dim] * 5, out_channels=vis_output_dim)
        self.fusion = FeatureFusion(
            k=k,
            hidden_dim=fusion_hidden_dim,
            layers=fusion_layer,
            heads=fusion_heads
        )
        self.change_aware = ChangeAware(in_channels=vis_output_dim, last_in_channels=vis_output_dim + k)
        self.text_channel_att = TextChannelAttention(hidden_dim=(vis_output_dim // 2) * 4, layers=att_layer, heads=att_heads)
        self.decoder = Decoder(in_channels=(vis_output_dim // 2) * 4 * 5)

        self.initialize_weights()

    def initialize_weights(self):
        vis_state_dict = {}
        text_state_dict = {}
        if self.pretrained == 'pretrained/ViT-B-16.pt':
            checkpoint = torch.jit.load(self.pretrained, map_location='cpu').float().state_dict()
            for k in checkpoint.keys():
                if k.startswith('visual.'):
                    if k in ['visual.positional_embedding', 'visual.conv1.weight']:
                        continue
                    new_k = k.replace('visual.', '')
                    vis_state_dict[new_k] = checkpoint[k]
                else:
                    if k in ['logit_scale', 'input_resolution', 'context_length', 'vocab_size']:
                        continue
                    text_state_dict[k] = checkpoint[k]
        else:
            checkpoint = torch.load(self.pretrained, map_location='cpu', weights_only=False)['model']
            for k in checkpoint.keys():
                if k.startswith('vision_encoder.'):
                    if k in ['vision_encoder.positional_embedding', 'vision_encoder.conv1.weight']:
                        continue
                    new_k = k.replace('vision_encoder.', '')
                    vis_state_dict[new_k] = checkpoint[k]
                else:
                    if k in ['logit_scale']:
                        continue
                    new_k = k.replace('text_encoder.', '')
                    text_state_dict[new_k] = checkpoint[k]

        u, w = self.text_encoder.load_state_dict(text_state_dict, False)
        logging.info(f'{u} {w} are misaligned params in text_encoder')
        u, w = self.vision_encoder.load_state_dict(vis_state_dict, False)
        logging.info(f'{u} {w} are misaligned params in vision_encoder')

        for module in [self.fusion, self.text_channel_att]:
            for m in module.modules():
                if isinstance(m, CrossTransformer):
                    proj_std = (m.width ** -0.5) * ((2 * m.layers) ** -0.5)
                    attn_std = m.width ** -0.5
                    fc_std = (2 * m.width) ** -0.5
                    for block in m.resblocks:
                        for module in block.modules():
                            if isinstance(module, nn.MultiheadAttention):
                                nn.init.normal_(module.in_proj_weight, std=attn_std)
                                nn.init.normal_(module.out_proj.weight, std=proj_std)
                            elif isinstance(module, nn.Sequential):
                                nn.init.normal_(module.c_fc.weight, std=fc_std)
                                nn.init.normal_(module.c_proj.weight, std=proj_std)

        self.text_encoder.requires_grad_(False)
        self.vision_encoder.requires_grad_(False)
        self.vision_encoder.positional_embedding.requires_grad_(True)
        self.vision_encoder.conv1.requires_grad_(True)

    def forward(self, t1, t2, text):  # t1 t2 (B,100,5,5) text (56,77)
        text = self.text_encoder(text)  # 57 512
        t1_feats = self.vision_encoder(t1)  # [B 768 5 5]*5 [B 512]
        t2_feats = self.vision_encoder(t2)

        t1_vis = self.neck(t1_feats[:-1])  # [B 512 5 5]*5
        t2_vis = self.neck(t2_feats[:-1])
        t1_cls = t1_feats[-1]  # B 512
        t2_cls = t2_feats[-1]

        t1_vis[-1], t1_text = self.fusion([t1_vis[-1], t1_cls], text)  # (B 512+k 5 5) (B K 512)
        t2_vis[-1], t2_text = self.fusion([t2_vis[-1], t2_cls], text)

        dif_feats = self.change_aware(t1_vis, t2_vis)  # B 1024 5 5

        text = torch.cat([t1_text, t2_text], dim=1)  # B K*2 512
        dif_feats = self.text_channel_att(dif_feats, text)

        out = self.decoder(dif_feats)
        return out
