!pip install -q pytorch_wavelets      # wavelet layers of WavTRGAN (Kaggle: Settings -> Internet ON)

# =====================================================================================================
# WavTRGAN revision - generate the output images of a saved WavTRGAN checkpoint on the test set.
# Paste this whole cell into a Kaggle notebook (GPU on) and run it. It
#   1. defines the WavTRGAN network (needed so torch.load can rebuild the saved model),
#   2. loads your checkpoint (e.g. netG_model_epoch_30.pth, saved with torch.save(net_g)),
#   3. runs it on every corrupted test image (256 x 256, no crop, no flip) and saves the output under the
#      SAME file name as the input, ready for metrics_kaggle.py.
# Edit the paths in "GENERATION SETTINGS" near the end of the cell.
# =====================================================================================================

# ----------------------------- WavTRGAN network (do not edit) ----------------------------------------
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")   # used inside the network code
# Ablation switches of the training notebook, set to the paper's model. They are only used to build a new
# model: a model loaded with torch.load keeps the settings it was trained with.
USE_WAVELET, N_TRANSFORMER_BLOCKS = True, 12
WAVELET, WAVE_LEVELS, FUSION = "db3", 1, "idwt"

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
    """Wavelet block: one DWT, one convolution per sub-band, then the directional fusion of Eq. (3).

    fusion "idwt" (paper): for each direction, the inverse DWT of the approximation and of that direction's
    detail sub-bands, with the other two directions set to zero.
    fusion "concat" (ablation): the same processed sub-bands, upsampled to the input size and concatenated
    (approximation + that direction's details) instead of the inverse DWT.
    wave / levels: wavelet basis and number of DWT levels. With levels > 1 the detail convolutions are
    shared across levels, so every setting has the same number of parameters.
    """
    def __init__(self, channels, wave="db3", levels=1, fusion="idwt"):
        super(SSL, self).__init__()
        if fusion not in ("idwt", "concat"):
            raise ValueError("fusion must be 'idwt' or 'concat', not %r" % fusion)
        self.wave, self.levels, self.fusion = wave, levels, fusion

        # Convolutional layers for processing wavelet components
        self.conv_approx = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.conv_horiz = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.conv_vert = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.conv_diag = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        with torch.autocast(device_type=x.device.type, enabled=False):   # pytorch_wavelets needs float32 (amp)
            return self._forward(x.float())

    def _forward(self, x):
        # models saved before these options existed have no such attributes: they used db3, 1 level, IDWT
        wave, levels = getattr(self, "wave", "db3"), getattr(self, "levels", 1)
        fusion = getattr(self, "fusion", "idwt")

        # Apply DWT to decompose the input image
        yl, yh = DWTForward(J=levels, mode='zero', wave=wave).to(x.device)(x)

        # Process each wavelet component with CNNs. details[d][j]: direction d at level j,
        # d = 0 horizontal, 1 vertical, 2 diagonal details
        approx_features = self.conv_approx(yl)
        convs = (self.conv_horiz, self.conv_vert, self.conv_diag)
        details = [[conv(h[:, :, d, :, :]) for h in yh] for d, conv in enumerate(convs)]

        if fusion == "concat":
            up = lambda t: F.interpolate(t, size=x.shape[-2:], mode="bilinear", align_corners=False)
            return tuple(torch.cat([up(approx_features)] + [up(t) for t in details[d]], dim=1) for d in range(3))

        # Reconstruct each direction with the inverse wavelet transform, the other two directions set to zero
        ifm = DWTInverse(wave=wave, mode='zero').to(x.device)
        return tuple(
            ifm((approx_features, [torch.stack([t if k == d else torch.zeros_like(t) for k in range(3)], dim=2)
                                   for t in details[d]]))
            for d in range(3))

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

        if USE_WAVELET:
            self.wavelet = SSL(3, WAVELET, WAVE_LEVELS, FUSION)
        # input + one direction image (6 channels); with fusion "concat": input + approximation + one detail per level
        in_ch = 6 if FUSION == "idwt" else 3 * (2 + WAVE_LEVELS)
        self.conv_layer = nn.Conv2d(in_channels=in_ch, out_channels=3, kernel_size=1, stride=1, padding=0).to(device)
        self.conv_con = nn.Conv2d(in_channels=1024, out_channels=3, kernel_size=1, stride=1, padding=0).to(device)
    def forward(self, x):
        if not hasattr(self, 'wavelet'):      # ablation: no wavelet block, one ResNet pass on the input
            x, features = self.hybrid_model(x)
            x = self.patch_embeddings(x)
            embeddings = x.flatten(2).transpose(-1, -2) + self.position_embeddings
            return self.dropout(embeddings), features

        x_h, x_v, x_sp = self.wavelet(x)

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
        for _ in range(N_TRANSFORMER_BLOCKS):  #number of block of tranformers layer
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
        
    def forward(self, x):
        if x.size()[1] == 1:
            x = x.repeat(1,3,1,1)
       
        #x_input = x
        #x = self.wavelet_transform(x)
        #x = x + x_input
        
        x, attn_weights, features = self.transformer(x)  # (B, n_patch, hidden)
        x = self.decoder(x, features)
        
        x = self.segmentation_head(x)
        return x


# =====================================================================================================
# GENERATION SETTINGS - edit these paths
# =====================================================================================================
import os
import numpy as np
from PIL import Image

MODEL_PATH = "/kaggle/input/YOUR-CHECKPOINT/netG_model_epoch_30.pth"   # the reported model (epoch 30)
CORRUPTED_DIR = "/kaggle/input/YOUR-DATASET/test/b"                   # corrupted test images (white strokes)
CLEAN_DIR = "/kaggle/input/YOUR-DATASET/test/a"                       # clean images, same names (range check only)
OUT_DIR = "/kaggle/working/outputs/wavtrgan"                          # the generated images are written here
SIZE = 256
VALUE_RANGE = "auto"     # "auto": tested on a few images; or "[-1,1]" (old training code) / "[0,1]" (revision code)
# -----------------------------------------------------------------------------------------------------

EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp")
for p in (MODEL_PATH, CORRUPTED_DIR):
    if not os.path.exists(p):
        raise SystemExit("not found: %s\n  -> run  !ls /kaggle/input  to see the right path" % p)

model = torch.load(MODEL_PATH, map_location=device, weights_only=False)
if isinstance(model, torch.nn.DataParallel):
    model = model.module
if isinstance(model, dict):
    raise SystemExit("MODEL_PATH holds only weights (a state_dict), not the full model: tell me and I will adapt the cell")
model = model.to(device).eval()
print("model loaded: %.2f M parameters, wavelet block: %s, Transformer blocks: %d" % (
    sum(p.numel() for p in model.parameters()) / 1e6, hasattr(model.transformer.embeddings, "wavelet"),
    len(model.transformer.encoder.layer)))


def load(path):
    return np.asarray(Image.open(path).convert("RGB").resize((SIZE, SIZE), Image.BICUBIC), dtype=np.float32) / 255.0


def run(img01, in_range, out_range):
    """img01: H x W x 3 in [0, 1] -> output H x W x 3 in [0, 1]."""
    x = torch.from_numpy(img01).permute(2, 0, 1)[None].to(device)
    if in_range == "[-1,1]":
        x = x * 2 - 1
    with torch.no_grad():
        y = model(x)[0].permute(1, 2, 0).float().cpu().numpy()
    if out_range == "[-1,1]":
        y = (y + 1) / 2
    return np.clip(y, 0, 1)


names = sorted(f for f in os.listdir(CORRUPTED_DIR) if f.lower().endswith(EXTS))
print("%d corrupted test images in %s" % (len(names), CORRUPTED_DIR))

# value range of the training code: try the four combinations on a few test images, keep the best PSNR
if VALUE_RANGE == "auto":
    clean = {os.path.splitext(f)[0]: os.path.join(CLEAN_DIR, f) for f in os.listdir(CLEAN_DIR)} if os.path.isdir(CLEAN_DIR) else {}
    sample = [n for n in names if os.path.splitext(n)[0] in clean][:16]
    if not sample:
        raise SystemExit("CLEAN_DIR has no image with the same names: set CLEAN_DIR, or VALUE_RANGE by hand")
    best = None
    for in_r in ("[-1,1]", "[0,1]"):
        for out_r in ("[-1,1]", "[0,1]"):
            ps = []
            for n in sample:
                gt = load(clean[os.path.splitext(n)[0]])
                mse = ((run(load(os.path.join(CORRUPTED_DIR, n)), in_r, out_r) - gt) ** 2).mean()
                ps.append(10 * np.log10(1.0 / max(mse, 1e-10)))
            print("  input %-7s output %-7s  PSNR on %d test images: %.2f dB" % (in_r, out_r, len(sample), np.mean(ps)))
            if best is None or np.mean(ps) > best[0]:
                best = (np.mean(ps), in_r, out_r)
    IN_RANGE, OUT_RANGE = best[1], best[2]
else:
    IN_RANGE = OUT_RANGE = VALUE_RANGE
print("using input %s, output %s" % (IN_RANGE, OUT_RANGE))

os.makedirs(OUT_DIR, exist_ok=True)
for i, n in enumerate(names, 1):
    y = run(load(os.path.join(CORRUPTED_DIR, n)), IN_RANGE, OUT_RANGE)
    Image.fromarray((y * 255).round().astype(np.uint8)).save(os.path.join(OUT_DIR, os.path.splitext(n)[0] + ".png"))
    if i % 500 == 0 or i == len(names):
        print("  %d / %d saved" % (i, len(names)))
print("done: outputs in %s  -> use this folder in OUT_DIRS of metrics_kaggle.py" % OUT_DIR)
