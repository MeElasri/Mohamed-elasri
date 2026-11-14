# =====================================================================================================
# Table 6 - hole-region PSNR_h and SSIM_h. Paste this whole cell into a Kaggle notebook and run it.
# Needs only numpy, Pillow and scikit-image (preinstalled on Kaggle). No model, no GPU.
# Run it once per setting (CelebA-HQ 40-50%, CelebA-HQ 10-20%, FFHQ 40-50%, FFHQ 10-20%).
# =====================================================================================================
import os
import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity

# ----------------------------- 1. EDIT THESE PATHS ---------------------------------------------------
SETTING = "CelebA-HQ 40-50%"                                    # only used in the printout
GT_DIR = "/kaggle/input/YOUR-DATASET/test/a"                    # clean ground-truth test images
OUT_DIRS = {                                                    # one line per method (output images)
    "WavTRGAN": "/kaggle/input/YOUR-OUTPUTS/wavtrgan",          # your generated images (epoch-30 model)
}
MASK_DIR = None                                                 # no mask folder: the hole = white pixels of INPUT_DIR
INPUT_DIR = "/kaggle/input/YOUR-DATASET/test/b"                 # corrupted test images (white strokes), same names
SIZE = (256, 256)                                               # evaluation resolution
# -----------------------------------------------------------------------------------------------------

EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp")


def index(folder):
    """{file name without extension: full path} for the images of a folder."""
    if not os.path.isdir(folder):
        raise SystemExit("folder not found: %s\n  -> run  !ls /kaggle/input  to see the right path" % folder)
    return {os.path.splitext(f)[0]: os.path.join(folder, f)
            for f in sorted(os.listdir(folder)) if f.lower().endswith(EXTS)}


def rgb(path):
    im = Image.open(path).convert("RGB")
    if im.size != SIZE:
        im = im.resize(SIZE, Image.BICUBIC)
    return np.asarray(im, dtype=np.float64)


def hole_mask(path_mask, path_input, gt):
    """True = corrupted pixel."""
    if path_mask is not None:
        im = Image.open(path_mask).convert("L")
        m = np.asarray(im) > 127                                # binary first (JPEG masks have soft edges)
        if im.size != SIZE:
            shrink = im.size[0] > SIZE[0]
            im = Image.fromarray(m.astype(np.uint8) * 255).resize(SIZE, Image.BICUBIC if shrink else Image.NEAREST)
            m = np.asarray(im) > (5 if shrink else 127)
        return m
    inp = rgb(path_input)                                       # no mask: white pixels or pixels that differ from GT
    return (np.abs(inp - gt).max(axis=2) > 0.05 * 255) | (inp >= 250).all(axis=2)


# ----------------------------- 2. CHECK THE FOLDERS --------------------------------------------------
gts = index(GT_DIR)
masks = index(MASK_DIR) if MASK_DIR else None
inputs = None if MASK_DIR else index(INPUT_DIR)
print("ground truth: %d images, e.g. %s" % (len(gts), list(gts)[:3]))
for name, folder in OUT_DIRS.items():
    outs = index(folder)
    common = [s for s in gts if s in outs]
    print("%-20s %d images, e.g. %s | %d match the ground-truth names" % (name, len(outs), list(outs)[:3], len(common)))
    if not common:
        raise SystemExit("no file name of %s matches the ground truth: the names must be the same (12.png <-> 12.png)" % name)
holes = masks if masks is not None else inputs
missing = [s for s in gts if s not in holes]
if missing:
    print("warning: %d ground-truth images have no %s, e.g. %s (they are skipped)"
          % (len(missing), "mask" if masks is not None else "input", missing[:3]))

# ----------------------------- 3. HOLE-REGION SCORES -------------------------------------------------
print("\n%s" % SETTING)
for name, folder in OUT_DIRS.items():
    outs = index(folder)
    psnr_h, ssim_h, psnr_f, ssim_f, ratio = [], [], [], [], []
    for s, gt_path in gts.items():
        if s not in outs or s not in holes:
            continue
        gt, out = rgb(gt_path), rgb(outs[s])
        m = hole_mask(masks[s] if masks is not None else None, inputs[s] if inputs is not None else None, gt)
        if m.sum() == 0:
            continue
        mse = ((gt - out) ** 2)[m].mean()                       # PSNR_h: MSE over the hole pixels (RGB)
        if mse > 0:
            psnr_h.append(10 * np.log10(255.0 ** 2 / mse))
        s_full, smap = structural_similarity(gt, out, data_range=255, channel_axis=2, full=True)
        ssim_h.append(smap[m].mean())                           # SSIM_h: SSIM map averaged over the hole
        mse_f = ((gt - out) ** 2).mean()                         # full image, for the check against Tables 3/5
        psnr_f.append(10 * np.log10(255.0 ** 2 / mse_f) if mse_f > 0 else np.nan)
        ssim_f.append(s_full)
        ratio.append(m.mean())
    print("%-20s %d images | PSNR_h = %.2f +/- %.2f dB | SSIM_h = %.3f +/- %.3f" % (
        name, len(ssim_h), np.mean(psnr_h), np.std(psnr_h, ddof=1), np.mean(ssim_h), np.std(ssim_h, ddof=1)))
    print("%-20s check: hole = %.1f%% of the image (min %.1f%%, max %.1f%%) | full image PSNR = %.2f dB, SSIM = %.3f" % (
        "", 100 * np.mean(ratio), 100 * np.min(ratio), 100 * np.max(ratio), np.nanmean(psnr_f), np.mean(ssim_f)))
