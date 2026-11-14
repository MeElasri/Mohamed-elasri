# Kaggle cell: GFLOPs of CMT and T-former with fvcore (the tool now used for WavTRGAN, CAML and Blind-Omni-Wav-Net).
# Paste into one cell of a Kaggle notebook (Internet on) and run. CPU is enough; it takes about a minute.
!pip -q install fvcore thop timm einops
!git clone -q --depth 1 https://github.com/keunsoo-ko/CMT.git
!git clone -q --depth 1 https://github.com/dengyecode/T-former_image_inpainting.git

import io, contextlib, sys, warnings
import torch
from fvcore.nn import FlopCountAnalysis
from thop import profile
warnings.filterwarnings("ignore")

x = torch.randn(1, 3, 256, 256)
mask = (torch.rand(1, 1, 256, 256) > 0.55).float()


def count(build, name):
    net = build().eval()
    with torch.no_grad(), contextlib.redirect_stdout(io.StringIO()):
        f = FlopCountAnalysis(net, (x, mask))
        f.unsupported_ops_warnings(False)
        f.uncalled_modules_warnings(False)
        fv = f.total() / 1e9
        th = profile(build().eval(), inputs=(x, mask), verbose=False)[0] / 1e9
    params = sum(p.numel() for p in net.parameters()) / 1e6
    print("%-9s params %.2f M | fvcore %.2f GFLOPs | thop %.2f" % (name, params, fv, th))


sys.path.insert(0, "CMT")
from network.network_pro import Inpaint               # CMT generator, 256 x 256 (official defaults)
count(Inpaint, "CMT")
sys.path.remove("CMT")
for k in [k for k in sys.modules if k == "network" or k.startswith("network.")]:
    del sys.modules[k]

sys.path.insert(0, "T-former_image_inpainting")
from model.network import Generator                   # T-former generator (ngf=48, as in define_g)
count(lambda: Generator(ngf=48), "T-former")
