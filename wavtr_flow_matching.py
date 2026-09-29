"""WavTR-Flow: the WavTRGAN generator trained with conditional flow matching.

Script version of wavtr-flow-matching.ipynb (same code, one "# %%" block per notebook cell).
Run it as a notebook (Kaggle / Jupyter) or cell by cell in VS Code / Spyder."""

# %% [markdown]
# # WavTR-Flow: flow matching for the WavTRGAN generator
#
# This notebook keeps the WavTRGAN generator (the wavelet-query TransUNet) and trains it as a **conditional flow-matching** model instead of a one-shot GAN generator, for **blind** inpainting: the network only sees the corrupted image and is never given the mask of the corrupted region.
#
# **How it works**
# - Images are mapped to [-1, 1]. For a training pair (corrupted input `y`, ground truth `x1`), draw noise `x0 ~ N(0, I)` and a time `t` in [0, 1], and form `x_t = t*x1 + (1 - t)*x0`.
# - The generator receives `[y, x_t]` and `t` and predicts the clean image `x1_hat`. The velocity is `v_hat = (x1_hat - x_t) / (1 - t)` and the loss is the flow-matching loss `||v_hat - (x1 - x0)||^2`. Predicting the clean image (x-prediction) instead of the velocity suits this architecture: it never has to reproduce the noise through its thin full-resolution output path. `opt.pred = "v"` switches to plain velocity prediction.
# - Optionally, the WavTRGAN losses (L1, VGG perceptual, style, PatchGAN) are also applied to the one-step estimate `x1_hat`, weighted by `t` so they act mostly where `x1_hat` is already sharp. Set their weights to 0 for pure flow matching.
# - Sampling integrates `dx/dt = v_hat(x, t, y)` from `x0 ~ N(0, I)` at `t = 0` to `t = 1` with an EMA copy of the generator: by default Heun with 12 steps (23 generator calls) on a time grid that takes smaller steps near the image, or Euler.
# - Mask prediction (`opt.mask_head`): a second output head predicts where `y` is corrupted, as in blind-inpainting networks such as VCNet. Its training target is computed from the training pair (the pixels where `y` and `x1` differ) and is never an input of the network, so the model stays blind. At test time the input pixels that the network is confident are uncorrupted are kept unchanged (`opt.composite`, `opt.composite_threshold`).
#
# **Changes to the generator** (the wavelet SSL, the 3-branch ResNetV2, the 3 transformer blocks and the TransUNet decoder are kept)
# - Input: 6 channels `[y, x_t]` instead of 3; the SSL convolutions, the 1x1 fusion conv and the ResNet root are widened accordingly.
# - Time conditioning: sinusoidal + MLP embedding of `t`, injected with zero-initialised FiLM in every ResNet bottleneck and decoder block and adaLN in every transformer block, so every block starts out as the original block.
# - Decoder BatchNorm -> GroupNorm: batch statistics would mix noise levels and differ between training and sampling.
# - `opt.full_res_skip`: an extra 256x256 skip from the input into the last decoder block, which had none.
# - The DWT/IDWT are built once in `__init__`. The old `SSL.forward` created `DWTInverse` on the CPU, which is the `torch.cuda.FloatTensor` / `torch.FloatTensor` error of the old evaluation cell.
# - `opt.mask_head`: a 3x3 convolution next to the image head that outputs the logits of the corrupted-region mask.
# - `opt.init_from_gan`: optional warm start from a trained WavTRGAN or WavTR-Flow checkpoint.
#
# The dataset layout, 286->256 random crops, direction `b2a`, the PatchGAN discriminator and the loss definitions are the same as in WavTRGAN.

# %%
# %pip install -q pytorch_wavelets PyWavelets torch-fidelity lpips   (notebook only; in a terminal: pip install pytorch_wavelets PyWavelets torch-fidelity)

# %% [markdown]
# # Configuration

# %%
import copy
import math
import os
import random
import time
from types import SimpleNamespace

import numpy as np
import torch

opt = SimpleNamespace(
    # data: <root_path>/train/{a,b} and <root_path>/test/{a,b}, as in WavTRGAN
    root_path="/kaggle/input/celeba-hq-img-full-50/CelebA-HQ-img",
    direction="b2a",            # b2a: input = folder b (corrupted), target = folder a
    img_size=256,
    batch_size=4,
    test_batch_size=8,
    threads=4,
    seed=123,

    # optimisation (WavTRGAN schedule: constant lr for niter epochs, then linear decay over niter_decay epochs)
    epoch_count=1,
    niter=100,
    niter_decay=50,
    max_epochs_per_run=0,       # > 0: stop after this many epochs (Kaggle sessions end after 12 h; continue with opt.resume)
    lr_policy="lambda",
    lr_decay_iters=50,
    lr_g=1e-4,
    lr_d=1e-4,
    beta1_d=0.5,
    grad_clip=1.0,
    ema_decay=0.999,

    # flow matching
    pred="x",                   # "x": the generator predicts the clean image (recommended); "v": the velocity
    t_sampling="uniform",       # "uniform" or "logit_normal" (rarely visits t near 1, where fine detail is formed)
    t_mean=0.0,                 # logit-normal location; < 0 spends more training on noisy t
    t_std=1.0,
    t_eps=0.05,                 # training: 1 - t is clipped to >= t_eps when x1_hat is turned into a velocity
    sample_steps=12,            # ODE steps at sampling time
    solver="heun",              # "heun" (2 generator calls per step, the last step 1) or "euler" (1 call per step)
    sample_power=1.5,           # time grid t_i = 1 - (1 - i/N)^power: > 1 takes smaller steps near the image end

    # losses: flow matching + WavTRGAN losses on the one-step estimate x1_hat (a weight of 0 disables the term)
    lambda_fm=1.0,
    lambda_l1=1.0,
    lambda_per=0.1,
    lambda_style=250.0,
    lambda_gan=0.1,
    lambda_gp=0.0,              # gradient penalty on D; off: spectral normalization already bounds D (see the losses cell)
    d_norm="none",              # discriminator normalization: "none" = SN-PatchGAN (recommended), "batch" = WavTRGAN
    d_init="normal",            # discriminator initialization N(0, 0.02) as in pix2pix ("xavier" * 0.02 = WavTRGAN)
    aux_weighting="t",          # per-sample weight of the x1_hat losses: "t", "t2" or "none"

    # mask prediction (blind: the network never receives the mask, it learns to predict it)
    mask_head=True,             # extra output head that predicts the corrupted region
    lambda_mask=1.0,            # weight of its binary cross-entropy loss
    mask_gt_threshold=0.05,     # its training target: pixels where input and ground truth differ by more than this
    composite=True,             # test time: keep the input pixels that the network predicts as uncorrupted
    composite_threshold=0.2,    # ... i.e. where the predicted mask is below this (< 0.5: when unsure, regenerate)

    # generator
    time_emb_dim=512,
    full_res_skip=True,
    use_wavelet=True,           # False: ablation without the wavelet query (the three branches see the input only)

    # checkpoints / logging
    init_from_gan="",           # WavTRGAN or WavTR-Flow weights, e.g. "/kaggle/input/archive-3/netG_model_epoch_14.pth"
    resume="",                  # e.g. "checkpoint/flow_latest.pth" (models + optimizers of the last epoch)
    save_every=1,               # every N epochs also keep checkpoint/flow_ema_epoch_N.pth (EMA generator, ~0.2 GB)
    checkpoint_dir="checkpoint",
    sample_dir="samples",
    log_every=50,
    max_iters_per_epoch=0,      # > 0 ends every epoch early (quick tests)
)

random.seed(opt.seed)
np.random.seed(opt.seed)
torch.manual_seed(opt.seed)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True
os.makedirs(opt.checkpoint_dir, exist_ok=True)
os.makedirs(opt.sample_dir, exist_ok=True)
print(device)
print(opt)

# %% [markdown]
# # Data

# %%
from os import listdir
from os.path import join

import matplotlib.pyplot as plt
import torch.utils.data as data
import torchvision.transforms.functional as TF
from PIL import Image


def is_image_file(filename):
    return filename.lower().endswith((".png", ".jpg", ".jpeg"))


class DatasetFromFolder(data.Dataset):
    """Pairs <dir>/a/<name> and <dir>/b/<name> as tensors in [0, 1].
    augment=True (training): resize to 286, random 256 crop, random horizontal flip, as in WavTRGAN.
    augment=False (evaluation): plain resize to 256, so every run is scored on the same images."""

    def __init__(self, image_dir, direction, augment=True, img_size=256, load_size=286):
        super().__init__()
        self.direction = direction
        self.augment = augment
        self.img_size = img_size
        self.load_size = load_size
        self.a_path = join(image_dir, "a")
        self.b_path = join(image_dir, "b")
        self.image_filenames = sorted(x for x in listdir(self.a_path) if is_image_file(x))

    def __getitem__(self, index):
        name = self.image_filenames[index]
        a = Image.open(join(self.a_path, name)).convert("RGB")
        b = Image.open(join(self.b_path, name)).convert("RGB")
        if self.augment:
            size = (self.load_size, self.load_size)
            a = TF.to_tensor(a.resize(size, Image.BICUBIC))
            b = TF.to_tensor(b.resize(size, Image.BICUBIC))
            w_offset = random.randint(0, max(0, self.load_size - self.img_size - 1))
            h_offset = random.randint(0, max(0, self.load_size - self.img_size - 1))
            a = a[:, h_offset:h_offset + self.img_size, w_offset:w_offset + self.img_size]
            b = b[:, h_offset:h_offset + self.img_size, w_offset:w_offset + self.img_size]
            if random.random() < 0.5:
                a, b = a.flip(2), b.flip(2)
        else:
            size = (self.img_size, self.img_size)
            a = TF.to_tensor(a.resize(size, Image.BICUBIC))
            b = TF.to_tensor(b.resize(size, Image.BICUBIC))
        return (a, b) if self.direction == "a2b" else (b, a)

    def __len__(self):
        return len(self.image_filenames)


def get_training_set(root_dir, direction):
    return DatasetFromFolder(join(root_dir, "train"), direction, augment=True)


def get_test_set(root_dir, direction):
    return DatasetFromFolder(join(root_dir, "test"), direction, augment=False)


def corruption_mask(corrupted, clean):
    """True corrupted region of a pair, (B, 1, H, W) in {0, 1}: pixels where the input differs from the ground truth
    by more than opt.mask_gt_threshold. Only a training target and an evaluation reference, never a network input."""
    return ((corrupted - clean).abs().amax(dim=1, keepdim=True) > opt.mask_gt_threshold).float()


def save_images(prediction, test_input, target, epoch, mask=None, max_rows=4):
    """Rows of (input, ground truth, prediction[, predicted mask]); all tensors in [0, 1]."""
    rows = min(max_rows, prediction.size(0))
    columns = [test_input, target, prediction] + ([mask.expand_as(prediction)] if mask is not None else [])
    fig, axes = plt.subplots(rows, len(columns), figsize=(4 * len(columns), 4 * rows), squeeze=False)
    titles = ["Input Image", "Ground Truth", "Prediction Image", "Predicted Mask"]
    for i in range(rows):
        for j, img in enumerate(columns):
            axes[i, j].imshow(img[i].detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy())
            axes[i, j].set_title(titles[j] if i == 0 else "")
            axes[i, j].axis("off")
    plt.tight_layout()
    plt.savefig(join(opt.sample_dir, f"epoch_{epoch}.jpg"))
    plt.close(fig)

# %% [markdown]
# # WavTR-Flow generator
#
# The WavTRGAN generator with a 6-channel input `[y, x_t]`, time conditioning and an optional mask head. Module names are unchanged, so WavTRGAN weights can be loaded (see the warm-start section).

# %%
from collections import OrderedDict

import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Conv2d, Dropout, LayerNorm, Linear, Softmax
from torch.nn.modules.utils import _pair
from pytorch_wavelets import DWTForward, DWTInverse


##################################################
################ Time conditioning ###############
##################################################

def timestep_embedding(t, dim, max_period=10000):
    """Sinusoidal embedding of t in [0, 1] (scaled by 1000, as in DDPM/DiT)."""
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(half, dtype=torch.float32, device=t.device) / half)
    args = 1000.0 * t.float()[:, None] * freqs[None]
    return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)


class TimeEmbedding(nn.Module):
    def __init__(self, emb_dim=512, freq_dim=256):
        super().__init__()
        self.freq_dim = freq_dim
        self.mlp = nn.Sequential(nn.Linear(freq_dim, emb_dim), nn.SiLU(), nn.Linear(emb_dim, emb_dim))

    def forward(self, t):
        return self.mlp(timestep_embedding(t, self.freq_dim))


class FiLM(nn.Module):
    """Time-dependent scale/shift of a normalised feature map. Zero-initialised: starts as the identity."""

    def __init__(self, emb_dim, channels):
        super().__init__()
        self.proj = nn.Sequential(nn.SiLU(), nn.Linear(emb_dim, 2 * channels))
        nn.init.zeros_(self.proj[1].weight)
        nn.init.zeros_(self.proj[1].bias)

    def forward(self, x, emb):
        scale, shift = self.proj(emb)[:, :, None, None].chunk(2, dim=1)
        return x * (1 + scale) + shift


##################################################
################# ResNet Network #################
##################################################

class StdConv2d(nn.Conv2d):

    def forward(self, x):
        w = self.weight
        v, m = torch.var_mean(w, dim=[1, 2, 3], keepdim=True, unbiased=False)
        w = (w - m) / torch.sqrt(v + 1e-5)
        return F.conv2d(x, w, self.bias, self.stride, self.padding, self.dilation, self.groups)


def conv3x3(cin, cout, stride=1, groups=1, bias=False):
    return StdConv2d(cin, cout, kernel_size=3, stride=stride, padding=1, bias=bias, groups=groups)


def conv1x1(cin, cout, stride=1, bias=False):
    return StdConv2d(cin, cout, kernel_size=1, stride=stride, padding=0, bias=bias)


class PreActBottleneck(nn.Module):
    """Pre-activation (v2) bottleneck block, with time FiLM after the middle GroupNorm."""

    def __init__(self, cin, cout=None, cmid=None, stride=1, emb_dim=None):
        super().__init__()
        cout = cout or cin
        cmid = cmid or cout // 4

        self.gn1 = nn.GroupNorm(32, cmid, eps=1e-6)
        self.conv1 = conv1x1(cin, cmid, bias=False)
        self.gn2 = nn.GroupNorm(32, cmid, eps=1e-6)
        self.conv2 = conv3x3(cmid, cmid, stride, bias=False)
        self.gn3 = nn.GroupNorm(32, cout, eps=1e-6)
        self.conv3 = conv1x1(cmid, cout, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.film = FiLM(emb_dim, cmid) if emb_dim else None

        if stride != 1 or cin != cout:
            self.downsample = conv1x1(cin, cout, stride, bias=False)
            self.gn_proj = nn.GroupNorm(cout, cout)

    def forward(self, x, emb=None):
        residual = x
        if hasattr(self, "downsample"):
            residual = self.gn_proj(self.downsample(x))

        y = self.relu(self.gn1(self.conv1(x)))
        y = self.gn2(self.conv2(y))
        if self.film is not None:
            y = self.film(y, emb)
        y = self.relu(y)
        y = self.gn3(self.conv3(y))
        return self.relu(residual + y)


class ResNetV2(nn.Module):
    """Pre-activation (v2) ResNet of WavTRGAN with a configurable number of input channels."""

    def __init__(self, block_units, width_factor, in_channels=3, emb_dim=None):
        super().__init__()
        width = int(64 * width_factor)
        self.width = width

        def unit(cin, cout, cmid, stride=1):
            return PreActBottleneck(cin=cin, cout=cout, cmid=cmid, stride=stride, emb_dim=emb_dim)

        self.root = nn.Sequential(OrderedDict([
            ("conv", StdConv2d(in_channels, width, kernel_size=7, stride=2, bias=False, padding=3)),
            ("gn", nn.GroupNorm(32, width, eps=1e-6)),
            ("relu", nn.ReLU(inplace=True)),
        ]))
        self.body = nn.Sequential(OrderedDict([
            ("block1", nn.Sequential(OrderedDict(
                [("unit1", unit(width, width * 4, width))] +
                [(f"unit{i:d}", unit(width * 4, width * 4, width)) for i in range(2, block_units[0] + 1)]))),
            ("block2", nn.Sequential(OrderedDict(
                [("unit1", unit(width * 4, width * 8, width * 2, stride=2))] +
                [(f"unit{i:d}", unit(width * 8, width * 8, width * 2)) for i in range(2, block_units[1] + 1)]))),
            ("block3", nn.Sequential(OrderedDict(
                [("unit1", unit(width * 8, width * 16, width * 4, stride=2))] +
                [(f"unit{i:d}", unit(width * 16, width * 16, width * 4)) for i in range(2, block_units[2] + 1)]))),
        ]))

    @staticmethod
    def _run(block, x, emb):
        for unit in block:
            x = unit(x, emb)
        return x

    def forward(self, x, emb=None):
        features = []
        b, c, in_size, _ = x.size()
        x = self.root(x)
        features.append(x)
        x = F.max_pool2d(x, kernel_size=3, stride=2, padding=0)
        for i in range(len(self.body) - 1):
            x = self._run(self.body[i], x, emb)
            right_size = int(in_size / 4 / (i + 1))
            if x.size(2) != right_size:
                pad = right_size - x.size(2)
                assert 0 < pad < 3, "x {} should {}".format(x.size(), right_size)
                feat = x.new_zeros((b, x.size(1), right_size, right_size))
                feat[:, :, :x.size(2), :x.size(3)] = x
            else:
                feat = x
            features.append(feat)
        x = self._run(self.body[-1], x, emb)
        return x, features[::-1]


##################################################
################# Wavelet query ##################
##################################################

class SSL(nn.Module):
    """Wavelet module of WavTRGAN. The DWT/IDWT hold filter buffers, so they are created here (and follow
    model.to(device)) instead of in forward."""

    def __init__(self, channels, wave="db3"):
        super().__init__()
        self.dwt = DWTForward(J=1, mode="zero", wave=wave)
        self.idwt = DWTInverse(mode="zero", wave=wave)
        self.conv_approx = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.conv_horiz = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.conv_vert = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.conv_diag = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        yl, yh = self.dwt(x)
        yh = yh[0]
        approx = self.conv_approx(yl)
        horiz = self.conv_horiz(yh[:, :, 0])   # horizontal details
        vert = self.conv_vert(yh[:, :, 1])     # vertical details
        diag = self.conv_diag(yh[:, :, 2])     # diagonal details
        zeros = torch.zeros_like(horiz)
        rec_horiz = self.idwt((approx, [torch.stack((horiz, zeros, zeros), dim=2)]))
        rec_vert = self.idwt((approx, [torch.stack((zeros, vert, zeros), dim=2)]))
        rec_diag = self.idwt((approx, [torch.stack((zeros, zeros, diag), dim=2)]))
        return rec_horiz, rec_vert, rec_diag


##################################################
############### Vision Transformer ###############
##################################################

class Attention(nn.Module):
    def __init__(self, vis):
        super().__init__()
        self.vis = vis
        self.num_attention_heads = 12
        self.attention_head_size = int(768 / self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = Linear(768, self.all_head_size)
        self.key = Linear(768, self.all_head_size)
        self.value = Linear(768, self.all_head_size)
        self.out = Linear(768, 768)
        self.attn_dropout = Dropout(0.0)
        self.proj_dropout = Dropout(0.1)
        self.softmax = Softmax(dim=-1)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        return x.view(*new_x_shape).permute(0, 2, 1, 3)

    def forward(self, hidden_states):
        query_layer = self.transpose_for_scores(self.query(hidden_states))
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = self.softmax(attention_scores)
        weights = attention_probs if self.vis else None
        attention_probs = self.attn_dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        context_layer = context_layer.view(*context_layer.size()[:-2], self.all_head_size)
        attention_output = self.proj_dropout(self.out(context_layer))
        return attention_output, weights


class Mlp(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = Linear(768, 3072)
        self.fc2 = Linear(3072, 768)
        self.act_fn = F.gelu
        self.dropout = Dropout(0.1)

        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, x):
        x = self.dropout(self.act_fn(self.fc1(x)))
        return self.dropout(self.fc2(x))


class Embeddings(nn.Module):
    """Wavelet + hybrid ResNet patch embeddings of WavTRGAN, for the 6-channel input [y, x_t]."""

    def __init__(self, img_size, in_channels=6, emb_dim=None):
        super().__init__()
        self.num_layers = (3, 4, 9)
        self.width_factor = 1
        img_size = _pair(img_size)

        grid_size = (16, 16)
        patch_size = (img_size[0] // 16 // grid_size[0], img_size[1] // 16 // grid_size[1])
        patch_size_real = (patch_size[0] * 16, patch_size[1] * 16)
        n_patches = (img_size[0] // patch_size_real[0]) * (img_size[1] // patch_size_real[1])

        self.hybrid_model = ResNetV2(self.num_layers, self.width_factor, in_channels=in_channels, emb_dim=emb_dim)
        self.patch_embeddings = Conv2d(self.hybrid_model.width * 16, 768, kernel_size=patch_size, stride=patch_size)
        self.position_embeddings = nn.Parameter(torch.zeros(1, n_patches, 768))
        self.dropout = Dropout(0.1)

        self.wavelet = SSL(in_channels)
        self.conv_layer = nn.Conv2d(2 * in_channels, in_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x, emb=None, use_wavelet=True):
        if use_wavelet:
            x_h, x_v, x_d = self.wavelet(x)
        else:   # no wavelet contribution
            x_h = x_v = x_d = torch.zeros_like(x)

        # horizontal / vertical / diagonal branches share one ResNet; the skips come from the last branch
        f_h, _ = self.hybrid_model(self.conv_layer(torch.cat((x, x_h), 1)), emb)
        f_v, _ = self.hybrid_model(self.conv_layer(torch.cat((x, x_v), 1)), emb)
        f_d, features = self.hybrid_model(self.conv_layer(torch.cat((x, x_d), 1)), emb)

        x = self.patch_embeddings(f_h + f_v + f_d)   # (B, hidden, 16, 16)
        x = x.flatten(2).transpose(-1, -2)            # (B, n_patches, hidden)
        embeddings = self.dropout(x + self.position_embeddings)
        return embeddings, features


class Block(nn.Module):
    """Pre-LN transformer block with adaLN time modulation (shift/scale of both norms, gate of both
    residual branches), zero-initialised so the block starts out as the original WavTRGAN block."""

    def __init__(self, vis, emb_dim=None):
        super().__init__()
        self.hidden_size = 768
        self.attention_norm = LayerNorm(768, eps=1e-6)
        self.ffn_norm = LayerNorm(768, eps=1e-6)
        self.ffn = Mlp()
        self.attn = Attention(vis)
        self.adaLN = None
        if emb_dim:
            self.adaLN = nn.Sequential(nn.SiLU(), nn.Linear(emb_dim, 6 * 768))
            nn.init.zeros_(self.adaLN[1].weight)
            nn.init.zeros_(self.adaLN[1].bias)

    def forward(self, x, emb=None):
        if self.adaLN is None:
            shift_a = scale_a = gate_a = shift_m = scale_m = gate_m = 0.0
        else:
            shift_a, scale_a, gate_a, shift_m, scale_m, gate_m = self.adaLN(emb)[:, None, :].chunk(6, dim=-1)

        h = x
        x, weights = self.attn(self.attention_norm(x) * (1 + scale_a) + shift_a)
        x = h + (1 + gate_a) * x

        h = x
        x = self.ffn(self.ffn_norm(x) * (1 + scale_m) + shift_m)
        x = h + (1 + gate_m) * x
        return x, weights


class Encoder(nn.Module):
    def __init__(self, vis, emb_dim=None, num_blocks=3):
        super().__init__()
        self.vis = vis
        self.layer = nn.ModuleList([Block(vis, emb_dim) for _ in range(num_blocks)])
        self.encoder_norm = LayerNorm(768, eps=1e-6)

    def forward(self, hidden_states, emb=None):
        attn_weights = []
        for layer_block in self.layer:
            hidden_states, weights = layer_block(hidden_states, emb)
            if self.vis:
                attn_weights.append(weights)
        return self.encoder_norm(hidden_states), attn_weights


class Transformer(nn.Module):
    def __init__(self, img_size, vis, in_channels=6, emb_dim=None):
        super().__init__()
        self.embeddings = Embeddings(img_size, in_channels=in_channels, emb_dim=emb_dim)
        self.encoder = Encoder(vis, emb_dim)

    def forward(self, x, emb=None, use_wavelet=True):
        embedding_output, features = self.embeddings(x, emb, use_wavelet=use_wavelet)
        encoded, attn_weights = self.encoder(embedding_output, emb)   # (B, n_patch, hidden)
        return encoded, attn_weights, features


##################################################
#################### Decoder #####################
##################################################

class Conv2dReLU(nn.Sequential):
    """Conv -> GroupNorm -> (time FiLM) -> ReLU. GroupNorm replaces WavTRGAN's BatchNorm: batch statistics
    would mix samples at different noise levels t and differ between training and sampling."""

    def __init__(self, in_channels, out_channels, kernel_size, padding=0, stride=1, emb_dim=None):
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=False),
            nn.GroupNorm(min(32, out_channels // 4), out_channels),
            nn.ReLU(inplace=True),
        )
        self.film = FiLM(emb_dim, out_channels) if emb_dim else None

    def forward(self, x, emb=None):
        x = self[1](self[0](x))
        if self.film is not None:
            x = self.film(x, emb)
        return self[2](x)


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, skip_channels=0, emb_dim=None):
        super().__init__()
        self.conv1 = Conv2dReLU(in_channels + skip_channels, out_channels, kernel_size=3, padding=1, emb_dim=emb_dim)
        self.conv2 = Conv2dReLU(out_channels, out_channels, kernel_size=3, padding=1)
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)

    def forward(self, x, skip=None, emb=None):
        x = self.up(x)
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        x = self.conv1(x, emb)
        return self.conv2(x)


class SegmentationHead(nn.Sequential):

    def __init__(self, in_channels, out_channels, kernel_size=3, upsampling=1):
        conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        upsampling = nn.UpsamplingBilinear2d(scale_factor=upsampling) if upsampling > 1 else nn.Identity()
        super().__init__(conv2d, upsampling)


class DecoderCup(nn.Module):
    def __init__(self, emb_dim=None, full_res_skip_channels=0):
        super().__init__()
        head_channels = 512
        self.conv_more = Conv2dReLU(768, head_channels, kernel_size=3, padding=1, emb_dim=emb_dim)
        decoder_channels = (256, 128, 64, 16)
        in_channels = [head_channels] + list(decoder_channels[:-1])
        skip_channels = [512, 256, 64, full_res_skip_channels]   # the last skip is the optional 256x256 one
        self.blocks = nn.ModuleList([
            DecoderBlock(in_ch, out_ch, sk_ch, emb_dim)
            for in_ch, out_ch, sk_ch in zip(in_channels, decoder_channels, skip_channels)
        ])

    def forward(self, hidden_states, features, emb=None):
        B, n_patch, hidden = hidden_states.size()   # (B, n_patch, hidden) -> (B, hidden, h, w)
        h = w = math.isqrt(n_patch)
        x = hidden_states.permute(0, 2, 1).contiguous().view(B, hidden, h, w)
        x = self.conv_more(x, emb)
        for i, decoder_block in enumerate(self.blocks):
            skip = features[i] if i < len(features) else None
            x = decoder_block(x, skip=skip, emb=emb)
        return x


##################################################
############ WavTR-Flow (the generator) ##########
##################################################

class WavTRFlow(nn.Module):
    """WavTRGAN generator as a flow-matching network: out = G(x_t, t, y).

    x_t: state on the path from noise (t = 0) to the image (t = 1), in [-1, 1] scale; t: (B,) in [0, 1];
    y: the corrupted image in [-1, 1] (no mask: blind inpainting). The output is the clean-image estimate x1_hat
    when opt.pred == "x", or the velocity when opt.pred == "v" (see model_velocity). With mask_head=True,
    return_mask=True also returns the logits of the predicted corrupted-region mask."""

    def __init__(self, img_size=256, img_channels=3, vis=False, emb_dim=512, full_res_skip=True, mask_head=False):
        super().__init__()
        in_channels = 2 * img_channels
        self.time_embed = TimeEmbedding(emb_dim)
        self.transformer = Transformer(img_size, vis, in_channels=in_channels, emb_dim=emb_dim)
        self.stem = Conv2dReLU(in_channels, 16, kernel_size=3, padding=1, emb_dim=emb_dim) if full_res_skip else None
        self.decoder = DecoderCup(emb_dim=emb_dim, full_res_skip_channels=16 if full_res_skip else 0)
        self.segmentation_head = SegmentationHead(in_channels=16, out_channels=img_channels, kernel_size=3)
        nn.init.zeros_(self.segmentation_head[0].weight)
        nn.init.zeros_(self.segmentation_head[0].bias)
        self.mask_head = SegmentationHead(in_channels=16, out_channels=1, kernel_size=3) if mask_head else None

    def forward(self, x_t, t, cond, return_attn=False, use_wavelet=True, return_mask=False):
        emb = self.time_embed(t)
        x = torch.cat((cond, x_t), dim=1)
        hidden, attn_weights, features = self.transformer(x, emb, use_wavelet=use_wavelet)
        if self.stem is not None:
            features = list(features) + [self.stem(x, emb)]
        feat = self.decoder(hidden, features, emb)
        out = self.segmentation_head(feat)
        outputs = (out,)
        if return_mask:
            outputs += (self.mask_head(feat) if self.mask_head is not None else None,)
        if return_attn:
            outputs += (attn_weights,)
        return outputs if len(outputs) > 1 else out


def build_generator(vis=False):
    return WavTRFlow(img_size=opt.img_size, vis=vis, emb_dim=opt.time_emb_dim, full_res_skip=opt.full_res_skip,
                     mask_head=opt.mask_head)

# %% [markdown]
# # Flow matching: training loss, ODE sampler, EMA
#
# Time runs from noise (`t = 0`) to data (`t = 1`) on the straight path `x_t = t*x1 + (1 - t)*x0`, whose velocity is `x1 - x0`.

# %%
def to_model_range(x):   # [0, 1] -> [-1, 1]
    return x * 2 - 1


def to_image_range(x):   # [-1, 1] -> [0, 1]
    return ((x + 1) / 2).clamp(0, 1)


def sample_t(batch_size, device):
    if opt.t_sampling == "logit_normal":
        return torch.sigmoid(opt.t_mean + opt.t_std * torch.randn(batch_size, device=device))
    return torch.rand(batch_size, device=device)


def model_velocity(net, x_t, t, cond, t_eps=0.0):
    """One generator call -> (velocity v_hat, clean-image estimate x1_hat, mask logits or None)."""
    out, mask_logits = net(x_t, t, cond, return_mask=True, use_wavelet=opt.use_wavelet)
    one_minus_t = (1 - t).view(-1, 1, 1, 1)
    if opt.pred == "v":
        return out, x_t + one_minus_t * out, mask_logits
    return (out - x_t) / one_minus_t.clamp_min(max(t_eps, 1e-4)), out, mask_logits


def flow_matching_loss(net, x1, cond):
    """Conditional flow matching on x_t = t*x1 + (1-t)*x0 with target velocity x1 - x0.
    Returns the loss, the sampled t, the one-step estimate x1_hat (used by the auxiliary losses) and the
    mask logits (None without a mask head)."""
    x0 = torch.randn_like(x1)
    t = sample_t(x1.size(0), x1.device)
    tt = t.view(-1, 1, 1, 1)
    x_t = tt * x1 + (1 - tt) * x0
    v_hat, x1_hat, mask_logits = model_velocity(net, x_t, t, cond, opt.t_eps)
    if opt.pred == "v":
        v_target = x1 - x0
    else:   # same clipped denominator as v_hat, i.e. the loss is ||x1_hat - x1||^2 / max(1 - t, t_eps)^2
        v_target = (x1 - x_t) / (1 - tt).clamp_min(opt.t_eps)
    return F.mse_loss(v_hat, v_target), t, x1_hat, mask_logits


@torch.no_grad()
def sample_flow(net, cond, steps=None, solver=None, noise=None):
    """Integrates dx/dt = v_hat(x, t, y) from x0 ~ N(0, I) at t = 0 to t = 1. cond and the result are in [-1, 1].
    Also returns the mask probabilities of the last generator call (None without a mask head)."""
    steps = steps or opt.sample_steps
    solver = solver or opt.solver
    x = torch.randn_like(cond) if noise is None else noise
    ts = 1 - (1 - torch.linspace(0, 1, steps + 1, device=cond.device)) ** opt.sample_power
    for i in range(steps):
        dt = ts[i + 1] - ts[i]
        v0, _, mask_logits = model_velocity(net, x, ts[i].expand(x.size(0)), cond)
        if solver == "heun" and i < steps - 1:   # the last step stays Euler: x-prediction has no velocity at t = 1
            v1, _, _ = model_velocity(net, x + dt * v0, ts[i + 1].expand(x.size(0)), cond)
            x = x + dt * (v0 + v1) / 2
        else:
            x = x + dt * v0
    return x, (None if mask_logits is None else torch.sigmoid(mask_logits))


@torch.no_grad()
def inpaint(net, cond, **kwargs):
    """Blind inpainting of the corrupted images cond (in [-1, 1]) -> (result in [-1, 1], predicted mask or None).
    With opt.composite, the input pixels that the network is confident are uncorrupted (predicted mask below
    opt.composite_threshold) are kept; all other pixels, grown by one pixel, come from the ODE sample. Copying a
    corrupted pixel is worse than regenerating a clean one, which the network is trained to reproduce, so the
    threshold is below 0.5; an untrained mask head (about 0.5 everywhere) leaves the ODE sample unchanged."""
    x, mask = sample_flow(net, cond, **kwargs)
    if mask is not None and opt.composite:
        hole = F.max_pool2d((mask > opt.composite_threshold).float(), kernel_size=3, stride=1, padding=1)
        x = hole * x + (1 - hole) * cond
    return x, mask


@torch.no_grad()
def update_ema(ema_model, model, decay):
    for p_ema, p in zip(ema_model.parameters(), model.parameters()):
        p_ema.lerp_(p, 1 - decay)
    for b_ema, b in zip(ema_model.buffers(), model.buffers()):
        b_ema.copy_(b)


def aux_weight(t):
    """Per-sample weight of the losses on x1_hat: small for noisy t, where x1_hat is necessarily blurry."""
    if opt.aux_weighting == "t":
        return t
    if opt.aux_weighting == "t2":
        return t ** 2
    return torch.ones_like(t)

# %% [markdown]
# # Pix2Pix discriminator and auxiliary losses (from WavTRGAN)
#
# Only built when their weights are > 0. The perceptual and style losses are the WavTRGAN ones, but return one value per sample so they can be weighted by `t`.
#
# The discriminator is the WavTRGAN PatchGAN with spectral normalization, but by default without BatchNorm (`opt.d_norm = "none"`, the SN-PatchGAN used in inpainting), initialized with N(0, 0.02) as in pix2pix, and without gradient penalty. With the earlier settings (WavTRGAN's BatchNorm and xavier * 0.02 initialization, plus a gradient penalty of weight 10) it never learned to separate real and generated images: its loss stayed at 0.5, the value of a discriminator that outputs 0.5 for everything. The penalty is computed on the sum of the ~900 patch outputs and lets the scores of real and generated patches differ by only a few hundredths; without it, the tiny initialization made the discriminator unstable, and with BatchNorm it learned much more slowly.

# %%
import functools

import torchvision
from torch.nn import init
from torch.optim import lr_scheduler


def get_norm_layer(norm_type="instance"):
    if norm_type == "batch":
        return functools.partial(nn.BatchNorm2d, affine=True)
    if norm_type == "instance":
        return functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    if norm_type == "none":
        return None
    raise NotImplementedError("normalization layer [%s] is not found" % norm_type)


def get_scheduler(optimizer, opt):
    if opt.lr_policy == "lambda":
        def lambda_rule(epoch):
            return 1.0 - max(0, epoch + opt.epoch_count - opt.niter) / float(opt.niter_decay + 1)
        return lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    if opt.lr_policy == "step":
        return lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    if opt.lr_policy == "cosine":
        return lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.niter, eta_min=0)
    raise NotImplementedError("learning rate policy [%s] is not implemented" % opt.lr_policy)


# update learning rate (called once every epoch)
def update_learning_rate(scheduler, optimizer):
    scheduler.step()
    print("learning rate = %.7f" % optimizer.param_groups[0]["lr"])


def init_weights(net, init_type="normal", gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, "weight") and (classname.find("Conv") != -1 or classname.find("Linear") != -1):
            if init_type == "normal":
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == "xavier":
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == "kaiming":
                init.kaiming_normal_(m.weight.data, a=0, mode="fan_in")
            elif init_type == "orthogonal":
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError("initialization method [%s] is not implemented" % init_type)
            if hasattr(m, "bias") and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find("BatchNorm2d") != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)

    print("initialize network with %s" % init_type)
    net.apply(init_func)


def init_net(net, init_type="normal", init_gain=0.02, gpu_id="cuda:0"):
    net.to(gpu_id)
    init_weights(net, init_type, gain=init_gain)
    return net


def define_D(input_nc, ndf, netD, n_layers_D=3, norm="batch", use_sigmoid=False,
             init_type="normal", init_gain=0.02, gpu_id="cuda:0"):
    norm_layer = get_norm_layer(norm_type=norm)
    if netD == "basic":
        net = NLayerDiscriminator(input_nc, ndf, n_layers=3, norm_layer=norm_layer, use_sigmoid=use_sigmoid)
    elif netD == "n_layers":
        net = NLayerDiscriminator(input_nc, ndf, n_layers_D, norm_layer=norm_layer, use_sigmoid=use_sigmoid)
    else:
        raise NotImplementedError("Discriminator model name [%s] is not recognized" % netD)
    return init_net(net, init_type, init_gain, gpu_id)


# Defines the PatchGAN discriminator with the specified arguments.
class NLayerDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, use_sigmoid=False):
        super().__init__()
        if norm_layer is None:   # no normalization layers: spectral normalization alone (SN-PatchGAN)
            use_bias = True
        elif type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        def norm(channels):
            return [norm_layer(channels)] if norm_layer is not None else []

        kw = 4
        padw = 1
        sequence = [
            nn.utils.spectral_norm(nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw)),
            nn.LeakyReLU(0.2, True)
        ]

        nf_mult = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.utils.spectral_norm(nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                                                 kernel_size=kw, stride=2, padding=padw, bias=use_bias)),
                *norm(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.utils.spectral_norm(nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                                             kernel_size=kw, stride=1, padding=padw, bias=use_bias)),
            *norm(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]
        sequence += [nn.utils.spectral_norm(nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw))]
        if use_sigmoid:
            sequence += [nn.Sigmoid()]
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        return self.model(input)


class GANLoss(nn.Module):
    def __init__(self, use_lsgan=True, target_real_label=1.0, target_fake_label=0.0):
        super().__init__()
        self.register_buffer("real_label", torch.tensor(target_real_label))
        self.register_buffer("fake_label", torch.tensor(target_fake_label))
        self.loss = nn.MSELoss() if use_lsgan else nn.BCELoss()

    def __call__(self, input, target_is_real):
        target_tensor = self.real_label if target_is_real else self.fake_label
        return self.loss(input, target_tensor.expand_as(input))


def gradient_penalty(D, real_ab, fake_ab):
    """Gradient penalty on random interpolations between real and fake (input, output) pairs.
    WavTRGAN's version drew alpha from N(0, 1) instead of U(0, 1) and fed image-only pairs to D.
    Off by default (opt.lambda_gp = 0): on the sum of the PatchGAN outputs it is so strong that D cannot
    separate real and generated images."""
    alpha = torch.rand(real_ab.size(0), 1, 1, 1, device=real_ab.device)
    interpolates = (alpha * real_ab + (1 - alpha) * fake_ab).requires_grad_(True)
    gradients = torch.autograd.grad(D(interpolates).sum(), interpolates, create_graph=True)[0]
    return ((gradients.flatten(1).norm(2, dim=1) - 1) ** 2).mean()


def set_requires_grad(net, requires_grad):
    for p in net.parameters():
        p.requires_grad_(requires_grad)


class VGGPerceptualLoss(nn.Module):
    """VGG16 perceptual loss of WavTRGAN (4 blocks, MSE), one value per sample. Inputs in [0, 1]."""

    def __init__(self, resize=True):
        super().__init__()
        vgg = torchvision.models.vgg16(weights=torchvision.models.VGG16_Weights.IMAGENET1K_V1).features.eval()
        self.blocks = nn.ModuleList([vgg[:4], vgg[4:9], vgg[9:16], vgg[16:23]])
        for p in self.parameters():
            p.requires_grad_(False)
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        self.resize = resize

    def _prep(self, x):
        x = (x - self.mean) / self.std
        if self.resize:
            x = F.interpolate(x, mode="bilinear", size=(224, 224), align_corners=False)
        return x

    def forward(self, input, target):
        x, y = self._prep(input), self._prep(target)
        loss = 0.0
        for block in self.blocks:
            x = block(x)
            with torch.no_grad():
                y = block(y)
            loss = loss + ((x - y) ** 2).mean(dim=(1, 2, 3))
        return loss / len(self.blocks)


class StyleLoss(nn.Module):
    """Gram-matrix style loss of WavTRGAN (VGG19 relu2_2, relu3_4, relu4_4, relu5_2; L1), one value per sample."""

    def __init__(self):
        super().__init__()
        vgg = torchvision.models.vgg19(weights=torchvision.models.VGG19_Weights.IMAGENET1K_V1).features.eval()
        self.slices = nn.ModuleList([vgg[:9], vgg[9:18], vgg[18:27], vgg[27:32]])
        for p in self.parameters():
            p.requires_grad_(False)

    @staticmethod
    def compute_gram(x):
        b, ch, h, w = x.size()
        f = x.reshape(b, ch, h * w)
        return f.bmm(f.transpose(1, 2)) / (h * w * ch)

    def forward(self, x, y):
        style_loss = 0.0
        for s in self.slices:
            x = s(x)
            with torch.no_grad():
                y = s(y)
            style_loss = style_loss + (self.compute_gram(x) - self.compute_gram(y)).abs().mean(dim=(1, 2))
        return style_loss

# %% [markdown]
# # Warm start from a WavTRGAN checkpoint (optional)
#
# Set `opt.init_from_gan` to a checkpoint saved by the old notebook (`torch.save(net_g, ...)`) or to a WavTR-Flow checkpoint. Every tensor whose name and shape still match is copied: the ResNet body, the transformer, the decoder convolutions and the GroupNorm affine parameters (taken from BatchNorm). For the input layers that grew from 3 to 6 channels, only the overlapping slice is copied. The new time-conditioning layers are zero-initialised, so they start as no-ops. The old output head is rescaled from [0, 1] to [-1, 1] for x-prediction. The mask head is new in both cases.

# %%
import pickle
import types


class _LenientUnpickler(pickle.Unpickler):
    """Classes that no longer exist (e.g. WavTRGAN's VisionTransformer) become empty nn.Modules,
    which is enough to read the weights of a model saved with torch.save(net_g)."""

    def find_class(self, module, name):
        try:
            return super().find_class(module, name)
        except (AttributeError, ImportError):
            return type(name, (nn.Module,), {})


_lenient_pickle = types.ModuleType("lenient_pickle")
_lenient_pickle.Unpickler = _LenientUnpickler
_lenient_pickle.load = pickle.load


def load_generator_state(path):
    """State dict from a whole pickled model (torch.save(net_g)), a state dict or a WavTR-Flow checkpoint."""
    obj = torch.load(path, map_location="cpu", weights_only=False, pickle_module=_lenient_pickle)
    if isinstance(obj, nn.Module):
        obj = obj.state_dict()
    elif isinstance(obj, dict) and "ema_g" in obj:
        obj = obj["ema_g"]
    return {k[len("module."):] if k.startswith("module.") else k: v for k, v in obj.items()}


@torch.no_grad()
def load_matching(model, state):
    own = model.state_dict()
    copied, partial = 0, []
    for k, v in state.items():
        if k not in own or own[k].dim() != v.dim():
            continue
        if own[k].shape == v.shape:
            own[k].copy_(v)
            copied += 1
        else:   # a layer that grew: copy the overlapping slice
            idx = tuple(slice(0, min(a, b)) for a, b in zip(own[k].shape, v.shape))
            own[k][idx].copy_(v[idx])
            partial.append(k)
    new = [k for k in own if k not in state]
    print("warm start: %d tensors copied, %d partially copied %s, %d new" % (copied, len(partial), partial, len(new)))


def warm_start_from_gan(net, path):
    state = load_generator_state(path)
    from_gan = not any(k.startswith("time_embed.") for k in state)
    head = {k: state.pop(k) for k in list(state) if k.startswith("segmentation_head.")} if from_gan else {}
    load_matching(net, state)
    if opt.pred == "x" and head:   # the WavTRGAN head outputs the image in [0, 1]; x-prediction works in [-1, 1]
        with torch.no_grad():
            net.segmentation_head[0].weight.copy_(2 * head["segmentation_head.0.weight"])
            net.segmentation_head[0].bias.copy_(2 * head["segmentation_head.0.bias"] - 1)

# %% [markdown]
# # Train
#
# Checkpoints: `checkpoint/flow_latest.pth` (everything needed to resume, overwritten every epoch) and `checkpoint/flow_ema_epoch_N.pth` (EMA generator, for evaluation). On Kaggle, set `opt.max_epochs_per_run` so that a run finishes within the 12-hour limit, then continue in a new session with `opt.resume`.
#
# To add the mask head to a run trained without it, warm-start instead of resuming: `opt.init_from_gan = ".../checkpoint/flow_latest.pth"`, `opt.resume = ""`, and `opt.epoch_count` = the next epoch (e.g. 8) to keep the epoch numbering and the learning-rate schedule. The image generator (the EMA weights of that run) and the discriminator continue from the checkpoint; the mask head and the optimizer states start fresh. Before training, `samples/true_masks.jpg` shows the training target of the mask head for the preview images: it should cover the corrupted strokes.
#
# Checkpoints saved before the discriminator fix can be resumed as usual: their discriminator, which had not learned to separate real and generated images, is replaced by a new one (see the losses cell), and everything else continues. With a working discriminator the D loss (`Loss_D`) stays below 0.5.

# %%
from torch.utils.data import DataLoader
from torchvision.utils import save_image

print("===> Loading datasets")
train_set = get_training_set(opt.root_path, opt.direction)
test_set = get_test_set(opt.root_path, opt.direction)
training_data_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batch_size,
                                  shuffle=True, drop_last=True)
testing_data_loader = DataLoader(dataset=test_set, num_workers=opt.threads, batch_size=opt.test_batch_size,
                                 shuffle=False)
print(len(train_set), "training pairs,", len(test_set), "test pairs")
# the same test images and noise are previewed after every epoch
fixed_a, fixed_b = next(iter(testing_data_loader))
fixed_noise = torch.randn(fixed_a.shape, generator=torch.Generator().manual_seed(opt.seed)).to(device)
if opt.mask_head:   # training target of the mask head for the preview images: it should cover the corruption
    true_masks = corruption_mask(fixed_a, fixed_b)
    save_image(torch.cat((fixed_a, true_masks.expand_as(fixed_a))), os.path.join(opt.sample_dir, "true_masks.jpg"),
               nrow=fixed_a.size(0))
    print("true corruption mask: %.1f%% of the preview pixels" % (100 * true_masks.mean().item()))

print("===> Building models")
net_g = build_generator().to(device)
if opt.init_from_gan and not opt.resume:
    warm_start_from_gan(net_g, opt.init_from_gan)
ema_g = copy.deepcopy(net_g).eval().requires_grad_(False)
optimizer_g = torch.optim.AdamW(net_g.parameters(), lr=opt.lr_g, betas=(0.9, 0.99), weight_decay=0.0)
print("generator: %.1fM parameters" % (sum(p.numel() for p in net_g.parameters()) / 1e6))

def same_discriminator(ckpt):
    """True if the checkpoint's discriminator has the current settings. Discriminators with the earlier settings
    never learned to separate real and generated images (D loss stuck at 0.5), so a new one is trained instead."""
    cfg = ckpt.get("opt", {})
    same = cfg.get("d_norm", "batch") == opt.d_norm and cfg.get("d_init", "xavier") == opt.d_init
    if not same:
        print("===> the checkpoint's discriminator has the earlier settings: training a new discriminator")
    return same


use_gan = opt.lambda_gan > 0
if use_gan:
    net_d = define_D(6, 64, "basic", norm=opt.d_norm, init_type=opt.d_init, init_gain=0.02, gpu_id=device)
    optimizer_d = torch.optim.Adam(net_d.parameters(), lr=opt.lr_d, betas=(opt.beta1_d, 0.999))
    criterionGAN = GANLoss().to(device)
    if opt.init_from_gan and not opt.resume:   # a WavTR-Flow checkpoint also holds its discriminator
        src = torch.load(opt.init_from_gan, map_location="cpu", weights_only=False, pickle_module=_lenient_pickle)
        if isinstance(src, dict) and "net_d" in src and same_discriminator(src):
            net_d.load_state_dict(src["net_d"])
            print("warm start: discriminator loaded")
        del src
per_loss = VGGPerceptualLoss().to(device) if opt.lambda_per > 0 else None
style_loss = StyleLoss().to(device) if opt.lambda_style > 0 else None
use_aux = use_gan or opt.lambda_l1 > 0 or per_loss is not None or style_loss is not None

global_step = 0
if opt.resume:
    ckpt = torch.load(opt.resume, map_location="cpu", weights_only=False)
    if ckpt["opt"].get("mask_head", False) != opt.mask_head:
        raise ValueError("opt.mask_head differs from the checkpoint: warm-start with opt.init_from_gan = opt.resume "
                         "and opt.resume = '' instead of resuming")
    net_g.load_state_dict(ckpt["net_g"])
    ema_g.load_state_dict(ckpt["ema_g"])
    optimizer_g.load_state_dict(ckpt["optimizer_g"])
    if use_gan and "net_d" in ckpt and same_discriminator(ckpt):
        net_d.load_state_dict(ckpt["net_d"])
        optimizer_d.load_state_dict(ckpt["optimizer_d"])
    opt.epoch_count = ckpt["epoch"] + 1
    global_step = ckpt["global_step"]
    del ckpt
    print("===> Resumed from %s, starting at epoch %d" % (opt.resume, opt.epoch_count))

# created after resuming: the lambda policy offsets the epoch by opt.epoch_count
net_g_scheduler = get_scheduler(optimizer_g, opt)
net_d_scheduler = get_scheduler(optimizer_d, opt) if use_gan else None
history = {"fm": [], "aux": [], "d": [], "mask": []}

last_epoch = opt.niter + opt.niter_decay
if opt.max_epochs_per_run:
    last_epoch = min(last_epoch, opt.epoch_count + opt.max_epochs_per_run - 1)
for epoch in range(opt.epoch_count, last_epoch + 1):
    net_g.train()
    start = time.time()
    for iteration, batch in enumerate(training_data_loader, 1):
        # forward: flow-matching loss and one-step estimate x1_hat
        real_a, real_b = batch[0].to(device), batch[1].to(device)   # [0, 1]; real_a = corrupted input
        loss_fm, t, x1_hat, mask_logits = flow_matching_loss(net_g, to_model_range(real_b), to_model_range(real_a))
        fake_b = (x1_hat + 1) / 2                                   # x1_hat in [0, 1]

        ######################
        # (1) Update D network
        ######################
        if use_gan:
            optimizer_d.zero_grad(set_to_none=True)
            real_ab = torch.cat((real_a, real_b), 1)
            fake_ab = torch.cat((real_a, fake_b.detach()), 1)
            loss_d = criterionGAN(net_d(fake_ab), False) + criterionGAN(net_d(real_ab), True)
            if opt.lambda_gp > 0:
                loss_d = loss_d + opt.lambda_gp * gradient_penalty(net_d, real_ab, fake_ab)
            loss_d.backward()
            optimizer_d.step()
            history["d"].append(loss_d.item())

        ######################
        # (2) Update G network
        ######################
        optimizer_g.zero_grad(set_to_none=True)
        loss_g = opt.lambda_fm * loss_fm
        if mask_logits is not None:   # the true mask is only the target of the mask head, never an input
            loss_mask = F.binary_cross_entropy_with_logits(mask_logits, corruption_mask(real_a, real_b))
            loss_g = loss_g + opt.lambda_mask * loss_mask
            history["mask"].append(loss_mask.item())
        if use_aux:
            aux = torch.zeros_like(t)   # per-sample WavTRGAN losses on x1_hat
            if opt.lambda_l1 > 0:
                aux = aux + opt.lambda_l1 * (fake_b - real_b).abs().mean(dim=(1, 2, 3))
            if per_loss is not None:
                aux = aux + opt.lambda_per * per_loss(fake_b, real_b)
            if style_loss is not None:
                aux = aux + opt.lambda_style * style_loss(fake_b, real_b)
            if use_gan:   # G(A) should fake the discriminator
                set_requires_grad(net_d, False)
                pred_fake = net_d(torch.cat((real_a, fake_b), 1))
                set_requires_grad(net_d, True)
                aux = aux + opt.lambda_gan * ((pred_fake - 1) ** 2).mean(dim=(1, 2, 3))
            loss_aux = (aux_weight(t) * aux).mean()
            loss_g = loss_g + loss_aux
            history["aux"].append(loss_aux.item())
        loss_g.backward()
        if opt.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(net_g.parameters(), opt.grad_clip)
        optimizer_g.step()
        update_ema(ema_g, net_g, min(opt.ema_decay, (1 + global_step) / (10 + global_step)))
        global_step += 1
        history["fm"].append(loss_fm.item())

        if iteration % opt.log_every == 0 or iteration == 1:
            msg = "===> Epoch[{}]({}/{}): Loss_FM: {:.4f}".format(epoch, iteration, len(training_data_loader), loss_fm.item())
            if use_aux:
                msg += " Loss_aux: {:.4f}".format(loss_aux.item())
            if use_gan:
                msg += " Loss_D: {:.4f}".format(loss_d.item())
            if mask_logits is not None:
                msg += " Loss_mask: {:.4f}".format(loss_mask.item())
            print(msg)
        if opt.max_iters_per_epoch and iteration >= opt.max_iters_per_epoch:
            break

    update_learning_rate(net_g_scheduler, optimizer_g)
    if use_gan:
        update_learning_rate(net_d_scheduler, optimizer_d)
    print("===> Epoch {} took {:.1f} min".format(epoch, (time.time() - start) / 60))

    # preview: EMA generator, fixed test images and noise
    prediction, pred_mask = inpaint(ema_g, to_model_range(fixed_a.to(device)), noise=fixed_noise)
    save_images(to_image_range(prediction), fixed_a, fixed_b, epoch, mask=pred_mask)

    # checkpoints (state dicts): the EMA generator every save_every epochs for evaluation, and
    # flow_latest.pth with everything needed by opt.resume (~0.9 GB, overwritten every epoch)
    ckpt = {"epoch": epoch, "global_step": global_step, "opt": dict(vars(opt)), "ema_g": ema_g.state_dict()}
    if epoch % opt.save_every == 0:
        torch.save(ckpt, os.path.join(opt.checkpoint_dir, "flow_ema_epoch_{}.pth".format(epoch)))
    ckpt.update(net_g=net_g.state_dict(), optimizer_g=optimizer_g.state_dict())
    if use_gan:
        ckpt.update(net_d=net_d.state_dict(), optimizer_d=optimizer_d.state_dict())
    torch.save(ckpt, os.path.join(opt.checkpoint_dir, "flow_latest.pth"))

# %%
def smooth(values, k=100):
    values = np.asarray(values)
    return values if len(values) < k else np.convolve(values, np.ones(k) / k, mode="valid")


plt.figure(figsize=(10, 5))
plt.title("Training losses (moving average)")
for name, values in history.items():
    if values:
        plt.plot(smooth(values), label=name)
plt.xlabel("iteration")
plt.yscale("log")
plt.legend()
plt.show()

# %% [markdown]
# # Generate test images
#
# Samples the test set with the EMA generator (`opt.sample_steps` ODE steps; `sample_steps=1` gives the one-step estimate from pure noise). The noise is seeded and the test images are resized to 256 without the random crop/flip of the old evaluation cell, so repeated runs are scored on the same images. `EVAL_CHECKPOINT` loads a saved checkpoint instead of the generator trained above. With the mask head, the predicted masks are saved to `pred_masks_flow` and compared with the true corrupted region of each test pair (pixel precision, recall, F1 and IoU); the true mask is used only for this score. The corrupted inputs are saved to `input_images_flow`: to score another method on exactly the same images, save its outputs for these inputs under the same file names and point `FAKE_DIR` to that folder in the metric cells below. The sampling time per image is measured with batches of `opt.test_batch_size`.

# %%
from torch.utils.data import DataLoader
from torchvision.utils import save_image

EVAL_CHECKPOINT = ""   # e.g. "checkpoint/flow_ema_epoch_40.pth"; empty: the EMA generator trained above
REAL_DIR, FAKE_DIR, MASK_DIR, INPUT_DIR = "Real_images_flow", "fake_images_flow", "pred_masks_flow", "input_images_flow"
MAX_EVAL_IMAGES = 2000


def load_flow_generator(path):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    cfg = ckpt.get("opt", {})
    for key in ("pred", "img_size", "time_emb_dim", "full_res_skip"):   # build and sample it as it was trained
        setattr(opt, key, cfg.get(key, getattr(opt, key)))
    opt.mask_head = cfg.get("mask_head", False)   # checkpoints from before the mask head have none
    opt.use_wavelet = cfg.get("use_wavelet", True)
    net = build_generator().to(device)
    net.load_state_dict(ckpt["ema_g"])
    return net.eval()


eval_g = load_flow_generator(EVAL_CHECKPOINT) if EVAL_CHECKPOINT else ema_g.eval()
for folder in (REAL_DIR, FAKE_DIR, MASK_DIR, INPUT_DIR):
    os.makedirs(folder, exist_ok=True)
eval_loader = DataLoader(get_test_set(opt.root_path, opt.direction), batch_size=opt.test_batch_size,
                         shuffle=False, num_workers=2)
noise_gen = torch.Generator(device=device).manual_seed(opt.seed)

n, tp, fp, fn = 0, 0, 0, 0
start, sample_time = time.time(), 0.0
for real_a, real_b in eval_loader:
    real_a, real_b = real_a[:MAX_EVAL_IMAGES - n], real_b[:MAX_EVAL_IMAGES - n]
    noise = torch.randn(real_a.shape, device=device, generator=noise_gen)
    tic = time.time()
    fake_b, mask = inpaint(eval_g, to_model_range(real_a.to(device)), noise=noise)
    if device.type == "cuda":
        torch.cuda.synchronize()
    sample_time += time.time() - tic
    fake_b = to_image_range(fake_b).cpu()
    if mask is not None:   # predicted vs true corrupted region (the true mask comes from the test pair)
        mask = mask.cpu()
        pred_m, true_m = mask > 0.5, corruption_mask(real_a, real_b) > 0.5
        tp += (pred_m & true_m).sum().item()
        fp += (pred_m & ~true_m).sum().item()
        fn += (~pred_m & true_m).sum().item()
    for i in range(fake_b.size(0)):
        save_image(real_b[i], os.path.join(REAL_DIR, f"{n}.png"))
        save_image(fake_b[i], os.path.join(FAKE_DIR, f"{n}.png"))
        save_image(real_a[i], os.path.join(INPUT_DIR, f"{n}.png"))
        if mask is not None:
            save_image(mask[i], os.path.join(MASK_DIR, f"{n}.png"))
        n += 1
    if n == MAX_EVAL_IMAGES:
        break
nfe = 2 * opt.sample_steps - 1 if opt.solver == "heun" else opt.sample_steps
print("%d images in %.1f min; sampling: %.3f s per image, %d generator calls (%d %s steps)" % (
    n, (time.time() - start) / 60, sample_time / max(n, 1), nfe, opt.sample_steps, opt.solver))
if tp + fp + fn > 0:
    print("predicted vs true mask: precision %.4f  recall %.4f  F1 %.4f  IoU %.4f" % (
        tp / max(1, tp + fp), tp / max(1, tp + fn), 2 * tp / (2 * tp + fp + fn), tp / (tp + fp + fn)))

# %% [markdown]
# # Compute FID / IS

# %%
import torch_fidelity

metrics_dict = torch_fidelity.calculate_metrics(
    input1=FAKE_DIR,
    input2=REAL_DIR,
    cuda=torch.cuda.is_available(),
    isc=True,
    fid=True,
    kid=False,
)
print(metrics_dict)

# %% [markdown]
# # Compute PSNR / SSIM / LPIPS
#
# Overall and grouped by the fraction of corrupted pixels of each test image (computed from `INPUT_DIR` and `REAL_DIR`). LPIPS uses AlexNet features (lower is better). The PSNR cell of the old notebook subtracted `uint8` images (`cv2.imread`) directly. `img1 - img2` then wraps around modulo 256, so large errors are counted as small ones and the PSNR comes out too high (e.g. 33.9 dB instead of 13.7 dB for a badly filled 128x128 hole). The images are converted to float here. For a like-for-like comparison, run this cell on the old WavTRGAN outputs as well (point `REAL_DIR` / `FAKE_DIR` to `Real_images1` / `fake_images1`).

# %%
import lpips
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

GRAYSCALE = False   # True: PSNR/SSIM on luminance only, like the old notebook's cv2 grayscale conversion
RATIO_EDGES = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 1.0]   # bins of the corrupted fraction for the per-bin results

lpips_fn = lpips.LPIPS(net="alex", verbose=False).to(device).eval()


def load_rgb01(folder, name):
    return TF.to_tensor(Image.open(os.path.join(folder, name)).convert("RGB"))[None]


psnr_list, ssim_list, lpips_list, ratio_list = [], [], [], []
names = sorted((f for f in os.listdir(FAKE_DIR) if is_image_file(f)), key=lambda f: int(os.path.splitext(f)[0]))
with torch.no_grad():
    for name in names:
        mode = "L" if GRAYSCALE else "RGB"
        gt = np.asarray(Image.open(os.path.join(REAL_DIR, name)).convert(mode), dtype=np.float64)
        pr = np.asarray(Image.open(os.path.join(FAKE_DIR, name)).convert(mode), dtype=np.float64)
        psnr_list.append(peak_signal_noise_ratio(gt, pr, data_range=255))
        ssim_list.append(structural_similarity(gt, pr, data_range=255, channel_axis=None if GRAYSCALE else 2))
        gt_t, pr_t = load_rgb01(REAL_DIR, name), load_rgb01(FAKE_DIR, name)
        lpips_list.append(lpips_fn(pr_t.to(device) * 2 - 1, gt_t.to(device) * 2 - 1).item())
        if os.path.exists(os.path.join(INPUT_DIR, name)):
            ratio_list.append(corruption_mask(load_rgb01(INPUT_DIR, name), gt_t).mean().item())
print("PSNR: %.3f dB   SSIM: %.4f   LPIPS: %.4f   (%d images)" % (
    np.mean(psnr_list), np.mean(ssim_list), np.mean(lpips_list), len(names)))
if len(ratio_list) == len(names):
    ratios, metrics = np.array(ratio_list), np.array([psnr_list, ssim_list, lpips_list])
    for lo, hi in zip(RATIO_EDGES[:-1], RATIO_EDGES[1:]):
        idx = (ratios > lo) & (ratios <= hi)
        if idx.any():
            p, s, l = metrics[:, idx].mean(axis=1)
            print("corrupted %3d-%3d%%: PSNR %.3f dB   SSIM %.4f   LPIPS %.4f   (%d images)" % (
                100 * lo, 100 * hi, p, s, l, idx.sum()))

# %% [markdown]
# # Diversity (optional)
#
# Draws `DIV_SAMPLES` completions of each of the first `DIV_IMAGES` test images and reports the mean pairwise LPIPS between them (higher = more diverse; 0 for a deterministic method). With the mask-guided output, the pixels kept from the input are identical in all samples, so only the corrupted region contributes.

# %%
DIV_IMAGES, DIV_SAMPLES = 100, 5

div_scores, n = [], 0
div_gen = torch.Generator(device=device).manual_seed(opt.seed + 1)
div_loader = DataLoader(get_test_set(opt.root_path, opt.direction), batch_size=opt.test_batch_size, shuffle=False)
with torch.no_grad():
    for real_a, _ in div_loader:
        cond = to_model_range(real_a[:DIV_IMAGES - n].to(device))
        samples = [inpaint(eval_g, cond, noise=torch.randn(cond.shape, device=device, generator=div_gen))[0]
                   for _ in range(DIV_SAMPLES)]
        for i in range(DIV_SAMPLES):
            for j in range(i + 1, DIV_SAMPLES):
                div_scores.append(lpips_fn(samples[i].clamp(-1, 1), samples[j].clamp(-1, 1)).flatten().cpu())
        n += cond.size(0)
        if n == DIV_IMAGES:
            break
print("diversity: mean pairwise LPIPS %.4f (%d samples per image, %d images)" % (torch.cat(div_scores).mean(), DIV_SAMPLES, n))

# %% [markdown]
# # Visualize the wavelet impact on attention (optional)
#
# Same analysis as in the old notebook: the attention of the last transformer block with and without the wavelet branch, here at a chosen flow time `T_VIS`.

# %%
def attn_to_map(attn_layer):
    A = attn_layer.mean(dim=1)       # average heads -> (B, N, N)
    received = A.mean(dim=1)         # average over queries -> (B, N)
    B, N = received.shape
    g = math.isqrt(N)
    return received.view(B, 1, g, g)


def attn_impact_map(attn_on, attn_off, layer_idx=-1, out_hw=(256, 256)):
    m_on = F.interpolate(attn_to_map(attn_on[layer_idx]), size=out_hw, mode="bilinear", align_corners=False)
    m_off = F.interpolate(attn_to_map(attn_off[layer_idx]), size=out_hw, mode="bilinear", align_corners=False)
    d = (m_on - m_off).abs()[0, 0].cpu().numpy()
    return (d - d.min()) / (d.max() - d.min() + 1e-8)


T_VIS = 0.5
vis_g = build_generator(vis=True).to(device)
vis_g.load_state_dict(eval_g.state_dict())
vis_g.eval()
with torch.no_grad():
    inp, gt = (x[None].to(device) for x in eval_loader.dataset[0])   # first test pair
    cond = to_model_range(inp)
    t = torch.full((1,), T_VIS, device=device)
    x_t = T_VIS * to_model_range(gt) + (1 - T_VIS) * torch.randn_like(cond)
    _, attn_on = vis_g(x_t, t, cond, return_attn=True, use_wavelet=True)
    _, attn_off = vis_g(x_t, t, cond, return_attn=True, use_wavelet=False)
    pred = to_image_range(inpaint(vis_g, cond)[0])

impact = attn_impact_map(attn_on, attn_off, out_hw=tuple(inp.shape[-2:]))
fig, ax = plt.subplots(1, 3, figsize=(15, 5))
ax[0].imshow(inp[0].permute(1, 2, 0).cpu().numpy())
ax[0].set_title("Input (corrupted)")
ax[1].imshow(pred[0].permute(1, 2, 0).cpu().numpy())
ax[1].set_title("Generated (inpainted)")
ax[2].imshow(inp[0].permute(1, 2, 0).cpu().numpy())
ax[2].imshow(impact, cmap="hot", alpha=0.55)
ax[2].set_title("|Attn(ON) - Attn(OFF)| overlay, t = %.2f" % T_VIS)
for a in ax:
    a.axis("off")
fig.suptitle("Wavelet impact on attention", fontsize=14)
plt.tight_layout()
plt.savefig("./wavelet_attention_impact.png", dpi=300, bbox_inches="tight")
plt.show()

# %% [markdown]
# # Precision-recall curves
#
# The PR-curve script of the old notebook works unchanged on these outputs: set `GT_FOLDER = REAL_DIR`, `GENERATED_FOLDER = FAKE_DIR` and add `'WavTR-Flow (Ours)': FAKE_DIR` to `METHODS`.
