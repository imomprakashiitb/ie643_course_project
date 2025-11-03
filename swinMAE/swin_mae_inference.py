import sys
sys.path.insert(0, '/content/drive/MyDrive/ie643_course_project_24M1644')
import os
import torch
import torch.nn as nn
import numpy as np
from functools import partial
from einops import rearrange
from swin_unet import PatchEmbedding, BasicBlock, PatchExpanding, BasicBlockUp
from utils.pos_embed import get_2d_sincos_pos_embed


class SwinMAE(nn.Module):
    """
    Masked Autoencoder with Swin Transformer backbone (CUDA-safe version)
    """

    def __init__(self, img_size=224, patch_size=4, mask_ratio=0.25, in_chans=3,
                 decoder_embed_dim=768, norm_pix_loss=False,
                 depths=(2, 2, 2, 2), embed_dim=96, num_heads=(3, 6, 12, 24),
                 window_size=7, qkv_bias=True, mlp_ratio=4.,
                 drop_path_rate=0.1, drop_rate=0., attn_drop_rate=0.,
                 norm_layer=partial(nn.LayerNorm, eps=1e-6), patch_norm=True):
        super().__init__()
        self.mask_ratio = mask_ratio
        assert img_size % patch_size == 0
        self.num_patches = (img_size // patch_size) ** 2
        self.patch_size = patch_size
        self.norm_pix_loss = norm_pix_loss
        self.depths = depths
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.mlp_ratio = mlp_ratio
        self.qkv_bias = qkv_bias
        self.drop_path = drop_path_rate
        self.drop_rate = drop_rate
        self.attn_drop_rate = attn_drop_rate
        self.norm_layer = norm_layer

        # Encoder
        self.patch_embed = PatchEmbedding(patch_size=patch_size, in_c=in_chans, embed_dim=embed_dim,
                                          norm_layer=norm_layer if patch_norm else None)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim), requires_grad=False)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.layers = self.build_layers()

        # Decoder
        self.first_patch_expanding = PatchExpanding(dim=decoder_embed_dim, norm_layer=norm_layer)
        self.layers_up = self.build_layers_up()
        self.norm_up = norm_layer(embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim // 8, patch_size ** 2 * in_chans, bias=True)

        self.initialize_weights()

    def initialize_weights(self):
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.num_patches ** 0.5), cls_token=False)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
        nn.init.normal_(self.mask_token, std=.02)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    # ---- Patch operations ----
    def patchify(self, imgs):
        p = self.patch_size
        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(imgs.shape[0], h * w, p ** 2 * 3)
        return x

    def unpatchify(self, x):
        p = self.patch_size
        h = w = int(x.shape[1] ** 0.5)
        x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(x.shape[0], 3, h * p, h * p)
        return imgs

    # ---- Window masking ----
    def window_masking(self, x, r=4, remove=False, mask_len_sparse=False):
        x = rearrange(x, 'B H W C -> B (H W) C')
        B, L, D = x.shape
        device = x.device
        d = int(L ** 0.5 // r)

        noise = torch.rand(B, d ** 2, device=device)
        sparse_shuffle = torch.argsort(noise, dim=1)
        sparse_restore = torch.argsort(sparse_shuffle, dim=1)
        sparse_keep = sparse_shuffle[:, :int(d ** 2 * (1 - self.mask_ratio))]

        index_keep_part = (torch.div(sparse_keep, d, rounding_mode='floor') * d * r ** 2 +
                           sparse_keep % d * r).long().to(device)
        index_keep = index_keep_part
        for i in range(r):
            for j in range(r):
                if i == 0 and j == 0:
                    continue
                offset = (int(L ** 0.5) * i + j)
                index_keep = torch.cat([index_keep, (index_keep_part + offset)], dim=1)

        index_all = torch.arange(L, device=device).unsqueeze(0).repeat(B, 1)
        index_mask = torch.zeros((B, L - index_keep.shape[-1]), dtype=torch.long, device=device)
        for i in range(B):
            diff = torch.tensor(
                np.setdiff1d(index_all[i].cpu().numpy(), index_keep[i].cpu().numpy(), assume_unique=True),
                device=device, dtype=torch.long
            )
            index_mask[i, :diff.shape[0]] = diff

        index_shuffle = torch.cat([index_keep, index_mask], dim=1)
        index_restore = torch.argsort(index_shuffle, dim=1)

        mask = torch.ones([B, L], device=device)
        mask[:, :index_keep.shape[-1]] = 0
        mask = torch.gather(mask, dim=1, index=index_restore)

        if remove:
            x_masked = torch.gather(x, dim=1, index=index_keep.unsqueeze(-1).repeat(1, 1, D))
            x_masked = rearrange(x_masked, 'B (H W) C -> B H W C', H=int(x_masked.shape[1] ** 0.5))
            return x_masked, mask, sparse_restore
        else:
            x_masked = torch.clone(x)
            for i in range(B):
                x_masked[i, index_mask[i], :] = self.mask_token.to(device)
            x_masked = rearrange(x_masked, 'B (H W) C -> B H W C', H=int(x_masked.shape[1] ** 0.5))
            return x_masked, mask

    def window_masking_(self, x, window_arr, r=4, remove=False, mask_len_sparse=False, index=27):
        """
        Device-safe variant used in inference with a fixed window_arr.
        """
        x = rearrange(x, 'B H W C -> B (H W) C')
        B, L, D = x.shape
        device = x.device
        d = int(L ** 0.5 // r)

        noise = torch.rand(B, d ** 2, device=device)
        sparse_shuffle = torch.argsort(noise, dim=1)
        sparse_restore = torch.argsort(sparse_shuffle, dim=1)
        sparse_keep = sparse_shuffle[:, :int(d ** 2 * (1 - self.mask_ratio))]

        # window_arr exclusion on GPU
        arr = list(range(0, 196))
        arr2 = sorted(list(set(arr) - set(window_arr)))
        sparse_keep = torch.tensor([arr2], dtype=torch.long, device=device).repeat(B, 1)

        index_keep_part = (torch.div(sparse_keep, d, rounding_mode='floor') * d * r ** 2 +
                           sparse_keep % d * r).long().to(device)
        index_keep = index_keep_part
        for i in range(r):
            for j in range(r):
                if i == 0 and j == 0:
                    continue
                offset = (int(L ** 0.5) * i + j)
                index_keep = torch.cat([index_keep, (index_keep_part + offset)], dim=1)

        index_all = torch.arange(L, device=device).unsqueeze(0).repeat(B, 1)
        index_mask = torch.zeros((B, L - index_keep.shape[-1]), dtype=torch.long, device=device)
        for i in range(B):
            diff = torch.tensor(
                np.setdiff1d(index_all[i].cpu().numpy(), index_keep[i].cpu().numpy(), assume_unique=True),
                device=device, dtype=torch.long
            )
            index_mask[i, :diff.shape[0]] = diff

        index_shuffle = torch.cat([index_keep, index_mask], dim=1)
        index_restore = torch.argsort(index_shuffle, dim=1)

        mask = torch.ones([B, L], device=device)
        mask[:, :index_keep.shape[-1]] = 0
        mask = torch.gather(mask, dim=1, index=index_restore)

        if remove:
            x_masked = torch.gather(x, dim=1, index=index_keep.unsqueeze(-1).repeat(1, 1, D))
            x_masked = rearrange(x_masked, 'B (H W) C -> B H W C', H=int(x_masked.shape[1] ** 0.5))
            return x_masked, mask, sparse_restore
        else:
            x_masked = torch.clone(x)
            for i in range(B):
                x_masked[i, index_mask[i], :] = self.mask_token.to(device)
            x_masked = rearrange(x_masked, 'B (H W) C -> B H W C', H=int(x_masked.shape[1] ** 0.5))
            return x_masked, mask

    # ---- Network construction ----
    def build_layers(self):
        layers = nn.ModuleList()
        for i in range(len(self.depths)):
            layers.append(
                BasicBlock(
                    index=i,
                    depths=self.depths,
                    embed_dim=self.embed_dim,
                    num_heads=self.num_heads,
                    drop_path=self.drop_path,
                    window_size=self.window_size,
                    mlp_ratio=self.mlp_ratio,
                    qkv_bias=self.qkv_bias,
                    drop_rate=self.drop_rate,
                    attn_drop_rate=self.attn_drop_rate,
                    norm_layer=self.norm_layer,
                    patch_merging=(i < len(self.depths) - 1)
                )
            )
        return layers

    def build_layers_up(self):
        layers_up = nn.ModuleList()
        for i in range(len(self.depths) - 1):
            layers_up.append(
                BasicBlockUp(
                    index=i,
                    depths=self.depths,
                    embed_dim=self.embed_dim,
                    num_heads=self.num_heads,
                    drop_path=self.drop_path,
                    window_size=self.window_size,
                    mlp_ratio=self.mlp_ratio,
                    qkv_bias=self.qkv_bias,
                    drop_rate=self.drop_rate,
                    attn_drop_rate=self.attn_drop_rate,
                    patch_expanding=(i < len(self.depths) - 2),
                    norm_layer=self.norm_layer
                )
            )
        return layers_up

    # ---- Forward passes ----
    def forward_encoder(self, x, window_arr):
        x = self.patch_embed(x)
        x, mask = self.window_masking(x, remove=False, mask_len_sparse=False)
        for layer in self.layers:
            x = layer(x)
        return x, mask

    def forward_decoder(self, x):
        x = self.first_patch_expanding(x)
        for layer in self.layers_up:
            x = layer(x)
        x = self.norm_up(x)
        x = rearrange(x, 'B H W C -> B (H W) C')
        x = self.decoder_pred(x)
        return x

    def forward_loss(self, imgs, pred, mask):
        target = self.patchify(imgs)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6) ** 0.5
        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)
        loss = (loss * mask).sum() / mask.sum()
        return loss

    def forward(self, x, window_arr):
        latent, mask = self.forward_encoder(x, window_arr)
        pred = self.forward_decoder(latent)
        loss = self.forward_loss(x, pred, mask)
        return loss, pred, mask


def swin_mae(**kwargs):
    model = SwinMAE(
        img_size=224, patch_size=4, in_chans=3,
        decoder_embed_dim=768,
        depths=(2, 2, 2, 2), embed_dim=96, num_heads=(3, 6, 12, 24),
        window_size=7, qkv_bias=True, mlp_ratio=4,
        drop_path_rate=0.1, drop_rate=0, attn_drop_rate=0,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model