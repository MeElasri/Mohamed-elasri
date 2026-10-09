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
# The dataset layout, 286->256 random crops, direction `b2a`, the PatchGAN discriminator and the loss definitions are the same as in WavTRGAN. Other corruptions (free-form strokes, boxes, a center square or mask files, filled with white, a random color, noise or a patch of another image) can be generated with `make_corrupted_dataset.py`; its `mask` folder gives the exact corrupted region of every image.

# %%
# %pip install -q pytorch_wavelets PyWavelets torch-fidelity lpips einops   (notebook only; in a terminal: pip install pytorch_wavelets PyWavelets torch-fidelity)

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
    root_path="/kaggle/input/celeba-hq-img-full-50/CelebA-HQ-img",   # or the output of make_corrupted_dataset.py
    direction="b2a",            # b2a: input = folder b (corrupted), target = folder a
    img_size=256,
    batch_size=4,
    test_batch_size=8,
    threads=4,
    seed=123,

    # optimisation: constant lr for niter epochs, then linear decay over niter_decay epochs (WavTRGAN: 100 + 50)
    epoch_count=1,
    niter=50,
    niter_decay=30,
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

    # ablation study: one switch per run (see ABLATIONS below), "" = normal training
    ablation="",
    ablation_epochs=20,         # short schedule of every ablation run (3/4 constant lr, 1/4 decay) ...
    ablation_iters_per_epoch=1750,   # ... of this many iterations each (1750 x 20 = 5 epochs of 28k images at batch 4)
)

# Ablation runs change one setting of the full model, all with the same short schedule, and keep their checkpoints
# and samples in their own folders. "full" is the reference: compare every variant with it, not with the main run.
# Use the same opt.init_from_gan for all of them.
ABLATIONS = {
    "full": {},
    "pure_fm": dict(lambda_l1=0.0, lambda_per=0.0, lambda_style=0.0, lambda_gan=0.0, mask_head=False, composite=False),
    "no_mask_head": dict(mask_head=False, composite=False),
    "no_wavelet": dict(use_wavelet=False),
    "v_pred": dict(pred="v"),
    "no_time_weight": dict(aux_weighting="none"),
    "no_full_res_skip": dict(full_res_skip=False),
}
if opt.ablation:
    for key, value in ABLATIONS[opt.ablation].items():
        setattr(opt, key, value)
    opt.niter = opt.ablation_epochs - opt.ablation_epochs // 4
    opt.niter_decay = opt.ablation_epochs // 4
    opt.max_iters_per_epoch = opt.ablation_iters_per_epoch
    opt.checkpoint_dir, opt.sample_dir = "checkpoint_" + opt.ablation, "samples_" + opt.ablation

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
    """Pairs <dir>/a/<name> and <dir>/b/<name> as tensors in [0, 1], plus the true corrupted region (1, H, W).
    The true region is the exact mask <dir>/mask/<stem>.png when the dataset has one (make_corrupted_dataset.py),
    grown by the pixels where input and ground truth differ, which covers the resampling blur at its border;
    otherwise it is only the latter. It is a training target and an evaluation reference, never a network input.
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
        self.mask_path = join(image_dir, "mask")
        self.has_masks = os.path.isdir(self.mask_path)
        self.image_filenames = sorted(x for x in listdir(self.a_path) if is_image_file(x))

    def transform(self, images):
        """The same resize, crop and flip for all images of a sample; masks (mode "L") are resized bilinearly."""
        filters = [Image.BILINEAR if im.mode == "L" else Image.BICUBIC for im in images]
        if not self.augment:
            size = (self.img_size, self.img_size)
            return [TF.to_tensor(im.resize(size, f)) for im, f in zip(images, filters)]
        size = (self.load_size, self.load_size)
        out = [TF.to_tensor(im.resize(size, f)) for im, f in zip(images, filters)]
        w_offset = random.randint(0, max(0, self.load_size - self.img_size - 1))
        h_offset = random.randint(0, max(0, self.load_size - self.img_size - 1))
        out = [t[:, h_offset:h_offset + self.img_size, w_offset:w_offset + self.img_size] for t in out]
        if random.random() < 0.5:
            out = [t.flip(2) for t in out]
        return out

    def __getitem__(self, index):
        name = self.image_filenames[index]
        images = [Image.open(join(self.a_path, name)).convert("RGB"), Image.open(join(self.b_path, name)).convert("RGB")]
        if self.has_masks:
            images.append(Image.open(join(self.mask_path, os.path.splitext(name)[0] + ".png")).convert("L"))
        a, b, *exact = self.transform(images)
        x, y = (a, b) if self.direction == "a2b" else (b, a)
        true_mask = corruption_mask(x[None], y[None])[0]
        if exact:   # any pixel the resized exact mask touches
            true_mask = torch.maximum(true_mask, (exact[0] > 0).float())
        return x, y, true_mask

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
# **Ablation runs:** set `opt.ablation` in the configuration to one of the keys of `ABLATIONS` and train as usual; each run uses the short schedule `opt.ablation_epochs` x `opt.ablation_iters_per_epoch` and its own `checkpoint_<name>` and `samples_<name>` folders, and is continued with `opt.resume = "checkpoint_<name>/flow_latest.pth"`. Train `"full"` (the reference) first, then `"pure_fm"`, `"no_mask_head"` and `"no_wavelet"`; `"v_pred"`, `"no_time_weight"` and `"no_full_res_skip"` if time allows. Rows (d) and (e) of the ablation table come from the `"full"` run, evaluated with `opt.composite = False` and with `opt.composite_threshold = 0.5`.
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
fixed_a, fixed_b, fixed_m = next(iter(testing_data_loader))
fixed_noise = torch.randn(fixed_a.shape, generator=torch.Generator().manual_seed(opt.seed)).to(device)
if opt.mask_head:   # training target of the mask head for the preview images: it should cover the corruption
    true_masks = fixed_m
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
        real_a, real_b, true_m = (x.to(device) for x in batch)   # [0, 1]; real_a = corrupted input
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
            loss_mask = F.binary_cross_entropy_with_logits(mask_logits, true_m)
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
# Samples the test set with the EMA generator (`opt.sample_steps` ODE steps; `sample_steps=1` gives the one-step estimate from pure noise). The noise is seeded and the test images are resized to 256 without the random crop/flip of the old evaluation cell, so repeated runs are scored on the same images. `EVAL_CHECKPOINT` loads a saved checkpoint instead of the generator trained above. With the mask head, the predicted masks are saved to `pred_masks_flow` and compared with the true corrupted region of each test pair (pixel precision, recall, F1 and IoU); the true mask is used only for this score. The corrupted inputs are saved to `input_images_flow` and their true masks to `true_masks_flow`: to score another method on exactly the same images, save its outputs for these inputs under the same file names and point `FAKE_DIR` to that folder in the metric cells below. The sampling time per image is measured with batches of `opt.test_batch_size`.

# %%
from torch.utils.data import DataLoader
from torchvision.utils import save_image

EVAL_CHECKPOINT = ""   # e.g. "checkpoint/flow_ema_epoch_40.pth"; empty: the EMA generator trained above
REAL_DIR, FAKE_DIR, MASK_DIR, INPUT_DIR = "Real_images_flow", "fake_images_flow", "pred_masks_flow", "input_images_flow"
TRUE_MASK_DIR = "true_masks_flow"
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
for folder in (REAL_DIR, FAKE_DIR, MASK_DIR, INPUT_DIR, TRUE_MASK_DIR):
    os.makedirs(folder, exist_ok=True)
eval_loader = DataLoader(get_test_set(opt.root_path, opt.direction), batch_size=opt.test_batch_size,
                         shuffle=False, num_workers=2)
noise_gen = torch.Generator(device=device).manual_seed(opt.seed)

n, tp, fp, fn = 0, 0, 0, 0
start, sample_time = time.time(), 0.0
for real_a, real_b, true_mask in eval_loader:
    real_a, real_b, true_mask = real_a[:MAX_EVAL_IMAGES - n], real_b[:MAX_EVAL_IMAGES - n], true_mask[:MAX_EVAL_IMAGES - n]
    noise = torch.randn(real_a.shape, device=device, generator=noise_gen)
    tic = time.time()
    fake_b, mask = inpaint(eval_g, to_model_range(real_a.to(device)), noise=noise)
    if device.type == "cuda":
        torch.cuda.synchronize()
    sample_time += time.time() - tic
    fake_b = to_image_range(fake_b).cpu()
    if mask is not None:   # predicted vs true corrupted region (the true mask comes from the test pair)
        mask = mask.cpu()
        pred_m, true_m = mask > 0.5, true_mask > 0.5
        tp += (pred_m & true_m).sum().item()
        fp += (pred_m & ~true_m).sum().item()
        fn += (~pred_m & true_m).sum().item()
    for i in range(fake_b.size(0)):
        save_image(real_b[i], os.path.join(REAL_DIR, f"{n}.png"))
        save_image(fake_b[i], os.path.join(FAKE_DIR, f"{n}.png"))
        save_image(real_a[i], os.path.join(INPUT_DIR, f"{n}.png"))
        save_image(true_mask[i], os.path.join(TRUE_MASK_DIR, f"{n}.png"))
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
# Overall and grouped by the fraction of corrupted pixels of each test image (from the true masks in `TRUE_MASK_DIR`; with the exact masks of `make_corrupted_dataset.py` this is the real hole size). LPIPS uses AlexNet features (lower is better). The PSNR cell of the old notebook subtracted `uint8` images (`cv2.imread`) directly. `img1 - img2` then wraps around modulo 256, so large errors are counted as small ones and the PSNR comes out too high (e.g. 33.9 dB instead of 13.7 dB for a badly filled 128x128 hole). The images are converted to float here. For a like-for-like comparison, run this cell on the old WavTRGAN outputs as well (point `REAL_DIR` / `FAKE_DIR` to `Real_images1` / `fake_images1`).

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
        if os.path.exists(os.path.join(TRUE_MASK_DIR, name)):
            ratio_list.append(TF.to_tensor(Image.open(os.path.join(TRUE_MASK_DIR, name)).convert("L")).mean().item())
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
    for real_a, *_ in div_loader:
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
    inp, gt = (x[None].to(device) for x in eval_loader.dataset[0][:2])   # first test pair
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

# %% [markdown]
# # Score checkpoints and samplers
#
# Scores saved EMA checkpoints on the first `SCORE_IMAGES` test images, with the seeded noise of the evaluation cell, so that all rows are computed on the same images and can be compared with each other. Two uses:
# - **Training curve:** `SCORE_CHECKPOINTS = [f"{opt.checkpoint_dir}/flow_ema_epoch_{e}.pth" for e in range(40, 81, 5)]` with 500 images. Use it to follow training, not to pick the epoch you report: report the last checkpoint.
# - **Sampler table:** one checkpoint, `SCORE_IMAGES = 2000` and `SCORE_SAMPLERS = [(1, "euler"), (12, "euler"), (4, "heun"), (8, "heun"), (12, "heun"), (20, "heun")]`. `(1, "euler")` is the one-step estimate from pure noise, i.e. the posterior mean predicted at t = 0: it gives the highest PSNR / SSIM, the full sampler the best FID / LPIPS.
#
# FID from 500 images is biased upwards; compare it only with other 500-image FIDs. The rows are also written to `SCORE_CSV`. The functions defined here (`score_folder`, `sample_test_images`) are used by the baselines below as well.

# %%
import csv
import shutil

import lpips
import torch_fidelity
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from torch.utils.data import DataLoader
from torchvision.utils import save_image

SCORE_CHECKPOINTS = []      # e.g. [f"{opt.checkpoint_dir}/flow_ema_epoch_{e}.pth" for e in range(40, 81, 5)]
SCORE_SAMPLERS = [(opt.sample_steps, opt.solver)]   # (steps, solver) pairs, e.g. [(1, "euler"), (12, "heun")]
SCORE_IMAGES = 500          # the first N test images, the same for every row
SCORE_FID = True
SCORE_CSV = "checkpoint_scores.csv"
RATIO_EDGES = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 1.0]   # groups of the corrupted fraction

if "lpips_fn" not in globals():
    lpips_fn = lpips.LPIPS(net="alex", verbose=False).to(device).eval()

if "load_flow_generator" not in globals():   # defined in the evaluation cell; repeated so that this cell runs on its own
    def load_flow_generator(path):
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        cfg = ckpt.get("opt", {})
        for key in ("pred", "img_size", "time_emb_dim", "full_res_skip"):
            setattr(opt, key, cfg.get(key, getattr(opt, key)))
        opt.mask_head = cfg.get("mask_head", False)
        opt.use_wavelet = cfg.get("use_wavelet", True)
        net = build_generator().to(device)
        net.load_state_dict(ckpt["ema_g"])
        return net.eval()


def image_names(folder):
    return sorted((f for f in os.listdir(folder) if is_image_file(f)), key=lambda f: int(os.path.splitext(f)[0]))


def fresh_dir(folder):
    shutil.rmtree(folder, ignore_errors=True)
    os.makedirs(folder)


def corrupted_fraction(name, real_dir, true_mask_dir=None, input_dir=None):
    """Fraction of corrupted pixels of one test image: from its saved true mask if there is one, otherwise from the
    pixels where the corrupted input and the ground truth differ (corruption_mask). None if neither is available."""
    if true_mask_dir and os.path.exists(os.path.join(true_mask_dir, name)):
        return TF.to_tensor(Image.open(os.path.join(true_mask_dir, name)).convert("L")).mean().item()
    if input_dir and os.path.exists(os.path.join(input_dir, name)):
        load = lambda folder: TF.to_tensor(Image.open(os.path.join(folder, name)).convert("RGB"))[None]
        return corruption_mask(load(input_dir), load(real_dir)).mean().item()
    return None


def score_folder(fake_dir, real_dir=None, true_mask_dir=None, input_dir=None, with_fid=True, verbose=True):
    """PSNR / SSIM on the saved 8-bit RGB images (as in the metric cell), LPIPS and FID of the images in fake_dir
    against the ground truths of the same names in real_dir; with true masks or corrupted inputs also per group of
    corrupted fraction. Returns the mean values."""
    real_dir = real_dir or REAL_DIR
    names = image_names(fake_dir)
    psnr, ssim, lp, ratios = [], [], [], []
    with torch.no_grad():
        for name in names:
            gt_img = Image.open(os.path.join(real_dir, name)).convert("RGB")
            pr_img = Image.open(os.path.join(fake_dir, name)).convert("RGB")
            gt, pr = np.asarray(gt_img, dtype=np.float64), np.asarray(pr_img, dtype=np.float64)
            psnr.append(peak_signal_noise_ratio(gt, pr, data_range=255))
            ssim.append(structural_similarity(gt, pr, data_range=255, channel_axis=2))
            gt_t, pr_t = (TF.to_tensor(im)[None].to(device) * 2 - 1 for im in (gt_img, pr_img))
            lp.append(lpips_fn(pr_t, gt_t).item())
            ratios.append(corrupted_fraction(name, real_dir, true_mask_dir, input_dir))
    result = {"psnr": float(np.mean(psnr)), "ssim": float(np.mean(ssim)), "lpips": float(np.mean(lp)),
              "images": len(names)}
    if with_fid:   # FID needs at least two images (a covariance); it is only meaningful for many more
        result["fid"] = float("nan") if len(names) < 2 else torch_fidelity.calculate_metrics(
            input1=fake_dir, input2=real_dir, cuda=torch.cuda.is_available(), fid=True, isc=False, kid=False,
            verbose=False)["frechet_inception_distance"]
    if verbose:
        fid = "FID: %.3f   " % result["fid"] if with_fid else ""
        print("PSNR: %.3f dB   SSIM: %.4f   LPIPS: %.4f   %s(%d images)" % (
            result["psnr"], result["ssim"], result["lpips"], fid, len(names)))
        if names and None not in ratios:
            r, values = np.array(ratios), np.array([psnr, ssim, lp])
            for lo, hi in zip(RATIO_EDGES[:-1], RATIO_EDGES[1:]):
                idx = (r > lo) & (r <= hi)
                if idx.any():
                    p, s, l = values[:, idx].mean(axis=1)
                    print("corrupted %3d-%3d%%: PSNR %.3f dB   SSIM %.4f   LPIPS %.4f   (%d images)" % (
                        100 * lo, 100 * hi, p, s, l, idx.sum()))
    return result


def sample_test_images(net, out_dir, max_images, steps=None, solver=None, real_dir=None, true_mask_dir=None):
    """Inpaints the first max_images test images into out_dir (<index>.png, seeded noise as in the evaluation cell)
    and optionally saves their ground truths and true masks. Returns (images, seconds per image, mask F1 or None)."""
    for folder in (out_dir, real_dir, true_mask_dir):
        if folder:
            fresh_dir(folder)
    loader = DataLoader(get_test_set(opt.root_path, opt.direction), batch_size=opt.test_batch_size,
                        shuffle=False, num_workers=2)
    noise_gen = torch.Generator(device=device).manual_seed(opt.seed)
    n, tp, fp, fn, elapsed = 0, 0, 0, 0, 0.0
    with torch.no_grad():
        for batch in loader:
            real_a, real_b = batch[0][:max_images - n], batch[1][:max_images - n]
            true_m = batch[2][:max_images - n] if len(batch) > 2 else corruption_mask(real_a, real_b)
            noise = torch.randn(real_a.shape, device=device, generator=noise_gen)
            tic = time.time()
            fake_b, mask = inpaint(net, to_model_range(real_a.to(device)), noise=noise, steps=steps, solver=solver)
            if device.type == "cuda":
                torch.cuda.synchronize()
            elapsed += time.time() - tic
            fake_b = to_image_range(fake_b).cpu()
            if mask is not None:
                pred_m, true_b = mask.cpu() > 0.5, true_m > 0.5
                tp += (pred_m & true_b).sum().item()
                fp += (pred_m & ~true_b).sum().item()
                fn += (~pred_m & true_b).sum().item()
            for i in range(fake_b.size(0)):
                save_image(fake_b[i], os.path.join(out_dir, f"{n}.png"))
                if real_dir:
                    save_image(real_b[i], os.path.join(real_dir, f"{n}.png"))
                if true_mask_dir:
                    save_image(true_m[i], os.path.join(true_mask_dir, f"{n}.png"))
                n += 1
            if n >= max_images:
                break
    f1 = 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else None
    return n, elapsed / max(n, 1), f1


def score_checkpoints(paths, samplers, max_images, with_fid=True, csv_path=None):
    rows = []
    for path in paths:
        net = load_flow_generator(path)
        for steps, solver in samplers:
            n, sec, f1 = sample_test_images(net, "score_fake", max_images, steps, solver,
                                            real_dir="score_real", true_mask_dir="score_true_masks")
            m = score_folder("score_fake", "score_real", true_mask_dir="score_true_masks", with_fid=with_fid,
                             verbose=False)
            row = dict(checkpoint=os.path.basename(path), steps=steps, solver=solver,
                       nfe=2 * steps - 1 if solver == "heun" else steps, sec_per_image=round(sec, 4), images=n,
                       psnr=round(m["psnr"], 3), ssim=round(m["ssim"], 4), lpips=round(m["lpips"], 4),
                       fid=round(m["fid"], 3) if with_fid else None, mask_f1=None if f1 is None else round(f1, 4))
            rows.append(row)
            print("%-28s %5s N=%-3d NFE=%-3d %.3f s/img  PSNR %.3f  SSIM %.4f  LPIPS %.4f  FID %s  mask F1 %s" % (
                row["checkpoint"], solver, steps, row["nfe"], sec, row["psnr"], row["ssim"], row["lpips"],
                row["fid"], row["mask_f1"]))
        del net
        if device.type == "cuda":
            torch.cuda.empty_cache()
    if csv_path and rows:
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    return rows


if SCORE_CHECKPOINTS:
    score_rows = score_checkpoints(SCORE_CHECKPOINTS, SCORE_SAMPLERS, SCORE_IMAGES, SCORE_FID, SCORE_CSV)

# %% [markdown]
# # Baselines on the same test images
#
# The evaluation cell saved the corrupted test inputs to `INPUT_DIR` and their ground truths to `REAL_DIR`. Each baseline below is run on these inputs, writes its results under the same file names to its own folder, and is scored with `score_folder`, exactly like WavTR-Flow. Run the evaluation cell and the cell above first.
#
# For methods with their own code (e.g. CAML, TransCNN-HAE, OmniWavNet), run their test script on the images of `INPUT_DIR`, put the results into a folder under the same file names and call `score_folder(folder, REAL_DIR, input_dir=INPUT_DIR)`.

# %%
def run_on_inputs(fn, out_dir, batch_size=None):
    """fn: corrupted inputs (B, 3, H, W) in [0, 1] on the device -> results in [0, 1]. Applies fn to all images of
    INPUT_DIR, writes the results to out_dir under the same names and returns the time per image in seconds."""
    fresh_dir(out_dir)
    names = image_names(INPUT_DIR)
    batch_size = batch_size or opt.test_batch_size
    elapsed = 0.0
    with torch.no_grad():
        for i in range(0, len(names), batch_size):
            chunk = names[i:i + batch_size]
            x = torch.stack([TF.to_tensor(Image.open(os.path.join(INPUT_DIR, name)).convert("RGB")) for name in chunk])
            x = x.to(device)
            tic = time.time()
            y = fn(x).clamp(0, 1)
            if device.type == "cuda":
                torch.cuda.synchronize()
            elapsed += time.time() - tic
            for name, img in zip(chunk, y.cpu()):
                save_image(img, os.path.join(out_dir, name))
    return elapsed / max(len(names), 1)

# %% [markdown]
# ## WavTRGAN
#
# `WAVTRGAN_SOURCE` is the generator cell of the WavTRGAN notebook, unchanged. It is executed in a module of its own because several of its class names also exist in this notebook. Two lines are adapted when it is loaded so that it runs on any device: the DWT is created on the device of the input instead of with `.cuda()`, and so is the inverse DWT, which the old code created on the CPU (the `torch.cuda.FloatTensor` / `torch.FloatTensor` error of the old evaluation cell). Neither has weights. The checkpoint is loaded strictly, so it must match this architecture. WavTRGAN takes and returns images in [0, 1] and has no mask-guided output.

# %%
import types

WAVTRGAN_WEIGHTS = ""   # the trained WavTRGAN generator, e.g. "/kaggle/input/archive-3/netG_model_epoch_14.pth"
WAVTRGAN_DIR = "fake_images_wavtrgan"

WAVTRGAN_SOURCE = r'''
import math


from collections import OrderedDict
from pytorch_wavelets import DWTForward, DWTInverse
import pywt
import torch
import torch.nn as nn
import torch.nn.functional as F

import copy
#import logging
import math

from os.path import join as pjoin

import torch
import torch.nn as nn
import numpy as np

from torch.nn import CrossEntropyLoss, Dropout, Softmax, Linear, Conv2d, LayerNorm
from torch.nn.modules.utils import _pair
from scipy import ndimage

######################Wavelets_function############################
##################################################################



###########################################
#########Resnet Network###################
#########################################

class StdConv2d(nn.Conv2d):

    def forward(self, x):
        w = self.weight
        v, m = torch.var_mean(w, dim=[1, 2, 3], keepdim=True, unbiased=False)
        w = (w - m) / torch.sqrt(v + 1e-5)
        return F.conv2d(x, w, self.bias, self.stride, self.padding,
                        self.dilation, self.groups)


def conv3x3(cin, cout, stride=1, groups=1, bias=False):
    return StdConv2d(cin, cout, kernel_size=3, stride=stride,
                     padding=1, bias=bias, groups=groups)


def conv1x1(cin, cout, stride=1, bias=False):
    return StdConv2d(cin, cout, kernel_size=1, stride=stride,
                     padding=0, bias=bias)


class PreActBottleneck(nn.Module):
    """Pre-activation (v2) bottleneck block.
    """

    def __init__(self, cin, cout=None, cmid=None, stride=1):
        super().__init__()
        cout = cout or cin
        cmid = cmid or cout//4

        self.gn1 = nn.GroupNorm(32, cmid, eps=1e-6)
        self.conv1 = conv1x1(cin, cmid, bias=False)
        self.gn2 = nn.GroupNorm(32, cmid, eps=1e-6)
        self.conv2 = conv3x3(cmid, cmid, stride, bias=False)  # Original code has it on conv1!!
        self.gn3 = nn.GroupNorm(32, cout, eps=1e-6)
        self.conv3 = conv1x1(cmid, cout, bias=False)
        self.relu = nn.ReLU(inplace=True)

        if (stride != 1 or cin != cout):
            # Projection also with pre-activation according to paper.
            self.downsample = conv1x1(cin, cout, stride, bias=False)
            self.gn_proj = nn.GroupNorm(cout, cout)

    def forward(self, x):

        # Residual branch
        residual = x
        if hasattr(self, 'downsample'):
            residual = self.downsample(x)
            residual = self.gn_proj(residual)

        # Unit's branch
        y = self.relu(self.gn1(self.conv1(x)))
        y = self.relu(self.gn2(self.conv2(y)))
        y = self.gn3(self.conv3(y))

        y = self.relu(residual + y)
        return y

class ResNetV2(nn.Module):
    """Implementation of Pre-activation (v2) ResNet mode."""

    def __init__(self, block_units, width_factor):
        super().__init__()
        width = int(64 * width_factor)
        self.width = width

        self.root = nn.Sequential(OrderedDict([
            ('conv', StdConv2d(3, width, kernel_size=7, stride=2, bias=False, padding=3)),
            ('gn',nn.GroupNorm(32, width, eps=1e-6)),
            ('relu', nn.ReLU(inplace=True)),
            #('pool', nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        ]))

        self.body = nn.Sequential(OrderedDict([
            ('block1', nn.Sequential(OrderedDict(
                [('unit1', PreActBottleneck(cin=width, cout=width*4, cmid=width))] +
                [(f'unit{i:d}', PreActBottleneck(cin=width*4, cout=width*4, cmid=width)) for i in range(2, block_units[0] + 1)],
                ))),
            ('block2', nn.Sequential(OrderedDict(
                [('unit1', PreActBottleneck(cin=width*4, cout=width*8, cmid=width*2, stride=2))] +
                [(f'unit{i:d}', PreActBottleneck(cin=width*8, cout=width*8, cmid=width*2)) for i in range(2, block_units[1] + 1)],
                ))),
            ('block3', nn.Sequential(OrderedDict(
                [('unit1', PreActBottleneck(cin=width*8, cout=width*16, cmid=width*4, stride=2))] +
                [(f'unit{i:d}', PreActBottleneck(cin=width*16, cout=width*16, cmid=width*4)) for i in range(2, block_units[2] + 1)],
                ))),
            
        ]))

    def forward(self, x):
        features = []
        b, c, in_size, _ = x.size()
        x = self.root(x)
        features.append(x)
        x = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)(x)
        for i in range(len(self.body)-1):
            x = self.body[i](x)
            right_size = int(in_size / 4 / (i+1))
            if x.size()[2] != right_size:
                pad = right_size - x.size()[2]
                assert pad < 3 and pad > 0, "x {} should {}".format(x.size(), right_size)
                feat = torch.zeros((b, x.size()[1], right_size, right_size), device=x.device)
                feat[:, :, 0:x.size()[2], 0:x.size()[3]] = x[:]
            else:
                feat = x
            features.append(feat)
        x = self.body[-1](x)
        return x, features[::-1]
    
#####################################################
############Vision Transformer Net###################
####################################################
################ Wavlet_Suery#######################


class SSL(nn.Module):
    def __init__(self, channels):
        super(SSL, self).__init__()

        # Convolutional layers for processing wavelet components
        self.conv_approx = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.conv_horiz = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.conv_vert = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.conv_diag = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        # Apply DWT to decompose the input image
        dwt = DWTForward(J=1, mode='zero', wave='db3').cuda()
        yl, yh = dwt(x)

        # Extract detail coefficients
        yh_out = yh[0]
        ylh = yh_out[:, :, 0, :, :]  # Horizontal details
        yhl = yh_out[:, :, 1, :, :]  # Vertical details
        yhh = yh_out[:, :, 2, :, :]  # Diagonal details

        # Process each wavelet component with CNNs
        approx_features = self.conv_approx(yl)
        horiz_features = self.conv_horiz(ylh)
        vert_features = self.conv_vert(yhl)
        diag_features = self.conv_diag(yhh)

        # Reconstruct each component using the inverse wavelet transform
        ifm = DWTInverse(wave='db3', mode='zero')

        # Reconstruct horizontal component
        rec_horiz = ifm((approx_features, [torch.stack((horiz_features, torch.zeros_like(horiz_features), torch.zeros_like(horiz_features)), dim=2)]))

        # Reconstruct vertical component
        rec_vert = ifm((approx_features, [torch.stack((torch.zeros_like(vert_features), vert_features, torch.zeros_like(vert_features)), dim=2)]))

        # Reconstruct diagonal (spatial) component
        rec_diag = ifm((approx_features, [torch.stack((torch.zeros_like(diag_features), torch.zeros_like(diag_features), diag_features), dim=2)]))

        return rec_horiz, rec_vert, rec_diag

#################################################

def swish(x):
    return x * torch.sigmoid(x)


ACT2FN = {"gelu": torch.nn.functional.gelu, "relu": torch.nn.functional.relu, "swish": swish}


class Attention(nn.Module):
    def __init__(self, vis):
        super(Attention, self).__init__()
        self.vis = vis
        self.num_attention_heads = 12
        self.attention_head_size = int(768 / self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        #self.query = SSL(768)
        self.query = Linear(768, self.all_head_size)
        self.key = Linear(768, self.all_head_size)
        self.value = Linear(768, self.all_head_size)

        self.out = Linear(768, 768)
        self.attn_dropout = Dropout(0.0)
        self.proj_dropout = Dropout(0.1)

        self.softmax = Softmax(dim=-1)

 
    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)
    
    def forward(self, hidden_states):
       
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        
        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = self.softmax(attention_scores)
        weights = attention_probs if self.vis else None
        attention_probs = self.attn_dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        attention_output = self.out(context_layer)
        attention_output = self.proj_dropout(attention_output)
        return attention_output, weights

class Mlp(nn.Module):
    def __init__(self):
        super(Mlp, self).__init__()
        self.fc1 = Linear(768, 3072)
        self.fc2 = Linear(3072, 768)
        self.act_fn = ACT2FN["gelu"]
        self.dropout = Dropout(0.1)

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class Embeddings(nn.Module):
    """Construct the embeddings from patch, position embeddings.
    """
    def __init__(self, img_size, in_channels=3):
        super(Embeddings, self).__init__()
        self.num_layers = (3, 4, 9)
        self.width_factor = 1
        img_size = _pair(img_size)

        grid_size = (16,16)
        patch_size = (img_size[0] // 16 // grid_size[0], img_size[1] // 16 // grid_size[1])
        patch_size_real = (patch_size[0] * 16, patch_size[1] * 16)
        n_patches = (img_size[0] // patch_size_real[0]) * (img_size[1] // patch_size_real[1])  
        
        self.hybrid_model = ResNetV2(block_units=self.num_layers, width_factor=self.width_factor)
        in_channels = self.hybrid_model.width * 16
        
        self.patch_embeddings = Conv2d(in_channels=in_channels,
                                       out_channels=768,
                                       kernel_size=patch_size,
                                       stride=patch_size)
        self.position_embeddings = nn.Parameter(torch.zeros(1, n_patches, 768))

        self.dropout = Dropout(0.1)

        self.wavelet = SSL(3)
        self.conv_layer = nn.Conv2d(in_channels=6, out_channels=3, kernel_size=1, stride=1, padding=0).to(device)
        self.conv_con = nn.Conv2d(in_channels=1024, out_channels=3, kernel_size=1, stride=1, padding=0).to(device)
    def forward(self, x, use_wavelet=True):

        if use_wavelet:
            x_h, x_v, x_sp = self.wavelet(x)
        else:
            # no wavelet contribution
            x_h = torch.zeros_like(x)
            x_v = torch.zeros_like(x)
            x_sp = torch.zeros_like(x)
    
        

        #extract horizontal features by ResNet
        x_h = torch.cat((x, x_h),1)
        x_h = self.conv_layer(x_h)
        x_h, features = self.hybrid_model(x_h)
    
        #extract vertical features by ResNet
        x_v = torch.cat((x, x_v),1)
        x_v = self.conv_layer(x_v)
        x_v, features = self.hybrid_model(x_v)

        #extract diagonal features by ResNet
        x_sp = torch.cat((x, x_sp),1)
        x_sp = self.conv_layer(x_sp)
        x_sp, features = self.hybrid_model(x_sp)

        x = x_h +x_v +x_sp
        
        x = self.patch_embeddings(x)  # (B, hidden. n_patches^(1/2), n_patches^(1/2))
        x = x.flatten(2)
        x = x.transpose(-1, -2)  # (B, n_patches, hidden)

        embeddings = x + self.position_embeddings
        embeddings = self.dropout(embeddings)
        return embeddings, features


class Block(nn.Module):
    def __init__(self, vis):
        super(Block, self).__init__()
        self.hidden_size = 768
        self.attention_norm = LayerNorm(768, eps=1e-6)
        self.ffn_norm = LayerNorm(768, eps=1e-6)
        self.ffn = Mlp()
        self.attn = Attention(vis)

    def forward(self, x):
        h = x
        x = self.attention_norm(x)
        x, weights = self.attn(x)
        x = x + h

        h = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = x + h
        return x, weights

class Encoder(nn.Module):
    def __init__(self, vis):
        super(Encoder, self).__init__()
        self.vis = vis
        self.layer = nn.ModuleList()
        self.encoder_norm = LayerNorm(768, eps=1e-6)
        for _ in range(3):  #number of block of tranformers layer
            layer = Block(vis)
            self.layer.append(copy.deepcopy(layer))

    def forward(self, hidden_states):
        attn_weights = []
        for layer_block in self.layer:
            hidden_states, weights = layer_block(hidden_states)
            if self.vis:
                attn_weights.append(weights)
        encoded = self.encoder_norm(hidden_states)
        return encoded, attn_weights


class Transformer(nn.Module):
    def __init__(self, img_size, vis):
        super(Transformer, self).__init__()
        self.embeddings = Embeddings(img_size=img_size)
        self.encoder = Encoder(vis)

    def forward(self, input_ids):
        embedding_output, features = self.embeddings(input_ids)
        encoded, attn_weights = self.encoder(embedding_output)  # (B, n_patch, hidden)
        return encoded, attn_weights, features


class Conv2dReLU(nn.Sequential):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            padding=0,
            stride=1,
            use_batchnorm=True,
    ):
        conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=not (use_batchnorm),
        )
        relu = nn.ReLU(inplace=True)

        bn = nn.BatchNorm2d(out_channels)

        super(Conv2dReLU, self).__init__(conv, bn, relu)


class DecoderBlock(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            skip_channels=3,
            use_batchnorm=True,
    ):
        super().__init__()
        self.conv1 = Conv2dReLU(
            in_channels + skip_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.conv2 = Conv2dReLU(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)

    def forward(self, x, skip=None):
        x = self.up(x)
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class SegmentationHead(nn.Sequential):

    def __init__(self, in_channels, out_channels, kernel_size=3, upsampling=1):
        conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        upsampling = nn.UpsamplingBilinear2d(scale_factor=upsampling) if upsampling > 1 else nn.Identity()
        super().__init__(conv2d, upsampling)


class DecoderCup(nn.Module):
    def __init__(self):
        super().__init__()
        head_channels = 512
        self.conv_more = Conv2dReLU(
            768,
            head_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=True,
        )

        self.projection_layer = nn.Conv2d(768, 512, kernel_size=1)
        decoder_channels = (256, 128, 64, 16)
        in_channels = [head_channels] + list(decoder_channels[:-1])
        out_channels = decoder_channels
        skip_channels = [512, 256, 64, 16]
        self.n_skip = 3
        for i in range(4-self.n_skip):  # re-select the skip channels according to n_skip
            skip_channels[3-i]=0

        blocks = [
            DecoderBlock(in_ch, out_ch, sk_ch) for in_ch, out_ch, sk_ch in zip(in_channels, out_channels, skip_channels)
        ]
        self.blocks = nn.ModuleList(blocks)

    def forward(self, hidden_states, features=None):
        B, n_patch, hidden = hidden_states.size()  # reshape from (B, n_patch, hidden) to (B, h, w, hidden)
        h, w = int(np.sqrt(n_patch)), int(np.sqrt(n_patch))
        x = hidden_states.permute(0, 2, 1)
        x = x.contiguous().view(B, hidden, h, w)
        
        x = self.conv_more(x)
  
        for i, decoder_block in enumerate(self.blocks):
            if features is not None:
                skip = features[i] if (i < self.n_skip) else None
            else:
                skip = None
            x = decoder_block(x, skip=skip)
        return x

#////////////////////////////////////////////
#Using Transformers as Encoder
#///////////////////////////////////////////
class VisionTransformer(nn.Module):
    def __init__(self, img_size=256, num_classes=21843, zero_head=False, vis=False):
        super(VisionTransformer, self).__init__()
        self.zero_head = zero_head
        self.transformer = Transformer(img_size, vis)
        self.decoder = DecoderCup()
        self.segmentation_head = SegmentationHead(
            in_channels=16,
            out_channels=3,
            kernel_size=3,
        )
        
    def forward(self, x, return_attn=False, use_wavelet=True):
        if x.size()[1] == 1:
            x = x.repeat(1,3,1,1)
       
        #x_input = x
        #x = self.wavelet_transform(x)
        #x = x + x_input
        
        x, attn_weights, features = self.transformer(x)  # (B, n_patch, hidden)
        x = self.decoder(x, features)
        
        x = self.segmentation_head(x)

        if return_attn:
            return x, attn_weights
        return x
'''


def build_wavtrgan(path):
    src = WAVTRGAN_SOURCE
    for old, new in [("DWTForward(J=1, mode='zero', wave='db3').cuda()", "DWTForward(J=1, mode='zero', wave='db3').to(x.device)"),
                     ("ifm = DWTInverse(wave='db3', mode='zero')", "ifm = DWTInverse(wave='db3', mode='zero').to(x.device)")]:
        assert src.count(old) == 1, old
        src = src.replace(old, new)
    module = types.ModuleType("wavtrgan_generator")
    module.device = device   # used by Embeddings.__init__
    exec(compile(src, "wavtrgan_generator", "exec"), module.__dict__)
    net = module.VisionTransformer(img_size=256, num_classes=9)
    net.load_state_dict(load_generator_state(path))   # strict: the checkpoint must match this architecture
    return net.to(device).eval()


if WAVTRGAN_WEIGHTS:
    wavtrgan = build_wavtrgan(WAVTRGAN_WEIGHTS)
    print("WavTRGAN: %.3f s per image" % run_on_inputs(wavtrgan, WAVTRGAN_DIR))
    wavtrgan_scores = score_folder(WAVTRGAN_DIR, REAL_DIR, true_mask_dir=globals().get("TRUE_MASK_DIR"),
                                   input_dir=INPUT_DIR)

# %% [markdown]
# ## IR-SDE
#
# IR-SDE (Luo et al., ICML 2023) inpaints 256x256 CelebA-HQ faces whose holes are filled with white, without being given the mask, i.e. the white setting of this notebook. Download its inpainting generator from the authors' "Weights and Results" Google Drive folder linked at https://github.com/Algolzw/image-restoration-sde (inpainting model), add the file to the notebook as a dataset and set `IRSDE_WEIGHTS`; the code is cloned from GitHub (internet on). The released model was trained on the "thin" masks of RePaint, which cover about 30% of the image on average, and is evaluated here without retraining, so report it as such. Sampling follows the authors' `test.py` and `ir-sde.yml`: 100 reverse SDE steps from the corrupted image plus noise (`max_sigma = 30`, cosine schedule).

# %%
IRSDE_WEIGHTS = ""   # e.g. "/kaggle/input/irsde-inpainting/ir-sde.pth"
IRSDE_REPO = "image-restoration-sde"
IRSDE_DIR = "fake_images_irsde"


def build_irsde(weights, repo=IRSDE_REPO):
    import importlib.util
    import subprocess
    import sys
    if not os.path.isdir(repo):
        subprocess.run(["git", "clone", "--depth", "1", "https://github.com/Algolzw/image-restoration-sde", repo],
                       check=True)

    def load_module(name, path, package=False):   # under its own name: "models" / "utils" are common module names
        spec = importlib.util.spec_from_file_location(
            name, path, submodule_search_locations=[os.path.dirname(path)] if package else None)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        return module

    arch = load_module("irsde_modules", os.path.join(repo, "codes/config/inpainting/models/modules/__init__.py"),
                       package=True)
    sde_utils = load_module("irsde_sde_utils", os.path.join(repo, "codes/utils/sde_utils.py"))
    net = arch.ConditionalUNet(in_nc=3, out_nc=3, nf=64, depth=4)   # network_G of options/test/ir-sde.yml
    state = torch.load(weights, map_location="cpu")
    if any(k.startswith("ema_model.") for k in state):   # a file of the EMA wrapper: keep the averaged weights
        state = {k[len("ema_model."):]: v for k, v in state.items() if k.startswith("ema_model.")}
    net.load_state_dict({k[len("module."):] if k.startswith("module.") else k: v for k, v in state.items()})
    sde = sde_utils.IRSDE(max_sigma=30, T=100, schedule="cosine", eps=0.005, device=device)   # sde of ir-sde.yml
    sde.set_model(net.to(device).eval())
    return sde


def irsde_inpaint(sde, lq):
    """Reverse SDE from the corrupted images lq (B, 3, H, W) in [0, 1]; sde.reverse_sde without the progress bar."""
    sde.set_mu(lq)
    x = sde.noise_state(lq)
    for t in reversed(range(1, sde.T + 1)):
        x = sde.reverse_sde_step(x, sde.score_fn(x, t), t)
    return x


if IRSDE_WEIGHTS:
    torch.manual_seed(opt.seed)
    irsde = build_irsde(IRSDE_WEIGHTS)
    print("IR-SDE: %.3f s per image" % run_on_inputs(lambda x: irsde_inpaint(irsde, x), IRSDE_DIR))
    irsde_scores = score_folder(IRSDE_DIR, REAL_DIR, true_mask_dir=globals().get("TRUE_MASK_DIR"),
                                input_dir=INPUT_DIR)
