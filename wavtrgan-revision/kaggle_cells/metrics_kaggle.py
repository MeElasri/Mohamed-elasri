# =====================================================================================================
# WavTRGAN revision - ALL metrics of Tables 3, 5 and 6 with one code, for every method of one setting.
# Paste this whole cell into a Kaggle notebook (Internet ON for FID) and run it once per setting:
#   CelebA-HQ 40-50%, CelebA-HQ 10-20%, FFHQ 40-50%, FFHQ 10-20%.
#
#   Full image : PSNR, SSIM on the raw output, RGB, 256 x 256           (Tables 3 and 5)
#   Hole only  : PSNR_h, SSIM_h over the corrupted pixels               (Table 6)
#   FID        : outputs vs. the ground-truth test images (pytorch-fid Inception, 2048-d)
#   Check      : hole size, and "Input (no inpainting)" = the corrupted image itself, as a reference row
# =====================================================================================================
import os
import csv
import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity

# ----------------------------- 1. EDIT THESE PATHS ---------------------------------------------------
SETTING = "CelebA-HQ 40-50%"
GT_DIR = "/kaggle/input/YOUR-DATASET/test/a"                    # clean ground-truth test images
INPUT_DIR = "/kaggle/input/YOUR-DATASET/test/b"                 # corrupted test images (white strokes), same names
OUT_DIRS = {                                                    # one line per method: folder of its output images
    "WavTRGAN": "/kaggle/input/YOUR-OUTPUTS/wavtrgan",          # epoch-30 model
    # "CAML": "/kaggle/input/YOUR-OUTPUTS/caml",
}
MASK_DIR = None                                                 # None: the hole = white pixels of the corrupted image
SIZE = (256, 256)                                               # evaluation resolution
FID = True                                                      # needs:  !pip install -q pytorch-fid  (Internet on)
SAVE_DIR = "/kaggle/working/metrics"                            # per-image CSV files are written here
# -----------------------------------------------------------------------------------------------------

EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp")


def index(folder):
    if not os.path.isdir(folder):
        raise SystemExit("folder not found: %s\n  -> run  !ls /kaggle/input  to see the right path" % folder)
    return {os.path.splitext(f)[0]: os.path.join(folder, f)
            for f in sorted(os.listdir(folder)) if f.lower().endswith(EXTS)}


def rgb(path):
    im = Image.open(path).convert("RGB")
    if im.size != SIZE:
        im = im.resize(SIZE, Image.BICUBIC)
    return np.asarray(im, dtype=np.float64)


def hole_of(stem, gt, inp):
    if MASK_DIR:
        im = Image.open(masks[stem]).convert("L")
        m = np.asarray(im) > 127
        if im.size != SIZE:
            shrink = im.size[0] > SIZE[0]
            im = Image.fromarray(m.astype(np.uint8) * 255).resize(SIZE, Image.BICUBIC if shrink else Image.NEAREST)
            m = np.asarray(im) > (5 if shrink else 127)
        return m
    return (np.abs(inp - gt).max(axis=2) > 0.05 * 255) | (inp >= 250).all(axis=2)


def psnr(a, b, m=None):
    mse = ((a - b) ** 2)[m].mean() if m is not None else ((a - b) ** 2).mean()
    return 10 * np.log10(255.0 ** 2 / mse) if mse > 0 else np.nan


# ----------------------------- 2. FID helpers (pytorch-fid weights) ----------------------------------
def fid_model():
    import torch
    from pytorch_fid.inception import InceptionV3
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    return InceptionV3([InceptionV3.BLOCK_INDEX_BY_DIM[2048]]).to(dev).eval(), dev


def fid_stats(images, model, dev, batch=50):
    import torch
    feats = []
    with torch.no_grad():
        for i in range(0, len(images), batch):
            x = torch.from_numpy(np.stack(images[i:i + batch]) / 255.0).permute(0, 3, 1, 2).float().to(dev)
            feats.append(model(x)[0].squeeze(-1).squeeze(-1).cpu().numpy())
    f = np.concatenate(feats).astype(np.float64)
    return f.mean(axis=0), np.cov(f, rowvar=False)


def frechet(mu1, s1, mu2, s2, eps=1e-6):
    from scipy import linalg
    covmean = linalg.sqrtm(s1 @ s2)
    if not np.isfinite(covmean).all():
        off = np.eye(s1.shape[0]) * eps
        covmean = linalg.sqrtm((s1 + off) @ (s2 + off))
    covmean = covmean.real
    d = mu1 - mu2
    return max(0.0, float(d @ d + np.trace(s1) + np.trace(s2) - 2 * np.trace(covmean)))


# ----------------------------- 3. CHECK THE FOLDERS --------------------------------------------------
gts, inputs = index(GT_DIR), index(INPUT_DIR)
masks = index(MASK_DIR) if MASK_DIR else None
methods = {"Input (no inpainting)": INPUT_DIR, **OUT_DIRS}
print("ground truth: %d images, e.g. %s" % (len(gts), list(gts)[:3]))
for name, folder in methods.items():
    outs = index(folder)
    n = sum(s in outs for s in gts)
    print("%-22s %d images | %d match the ground-truth names" % (name, len(outs), n))
    if n == 0:
        raise SystemExit("no file name of %s matches the ground truth (12.png <-> 12.png)" % name)

# ----------------------------- 4. METRICS ------------------------------------------------------------
os.makedirs(SAVE_DIR, exist_ok=True)
if FID:
    try:
        inception, dev = fid_model()
    except ImportError:
        print("pytorch-fid is not installed: run  !pip install -q pytorch-fid  (Internet on), or set FID = False")
        FID = False
stems = [s for s in gts if s in inputs and (masks is None or s in masks)]
gt_cache = {s: rgb(gts[s]) for s in stems}
holes = {s: hole_of(s, gt_cache[s], rgb(inputs[s])) for s in stems}
gt_stats = fid_stats([gt_cache[s] for s in stems], inception, dev) if FID else None

rows = []
for name, folder in methods.items():
    outs = index(folder)
    per, imgs = [], []
    for s in stems:
        if s not in outs:
            continue
        gt, out, m = gt_cache[s], rgb(outs[s]), holes[s]
        ssim_full, smap = structural_similarity(gt, out, data_range=255, channel_axis=2, full=True)
        per.append({"image": s, "hole_%": 100 * m.mean(), "PSNR": psnr(gt, out), "SSIM": ssim_full,
                    "PSNR_h": psnr(gt, out, m) if m.any() else np.nan, "SSIM_h": smap[m].mean() if m.any() else np.nan})
        if FID:
            imgs.append(out)
    with open(os.path.join(SAVE_DIR, "%s_%s.csv" % (SETTING.replace(" ", "_").replace("%", ""),
                                                   name.split(" ")[0])), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(per[0].keys()))
        w.writeheader()
        w.writerows(per)
    col = lambda k: np.array([r[k] for r in per], dtype=float)
    fid = frechet(*gt_stats, *fid_stats(imgs, inception, dev)) if FID else np.nan
    rows.append((name, len(per), col("hole_%"), col("PSNR"), col("SSIM"), col("PSNR_h"), col("SSIM_h"), fid))

# ----------------------------- 5. RESULTS ------------------------------------------------------------
print("\n%s  (%d test images)" % (SETTING, len(stems)))
print("%-22s %6s %7s | %7s %6s | %15s %14s | %6s" % ("method", "images", "hole%", "PSNR", "SSIM",
                                                     "PSNR_h", "SSIM_h", "FID"))
for name, n, hole, p, s, ph, sh, fid in rows:
    print("%-22s %6d %6.1f%% | %7.2f %6.3f | %6.2f +/- %5.2f %6.3f +/- %5.3f | %6.2f" % (
        name, n, np.nanmean(hole), np.nanmean(p), np.nanmean(s), np.nanmean(ph), np.nanstd(ph, ddof=1),
        np.nanmean(sh), np.nanstd(sh, ddof=1), fid))
print("\nper-image values saved in", SAVE_DIR)
