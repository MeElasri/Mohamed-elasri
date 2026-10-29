"""Full-image and hole-region metrics for inpainting.

All images are float arrays in [0, 1], shape HxWx3. Masks are boolean HxW
arrays with True = hole (corrupted pixel).

Hole-region metrics only look at pixels inside the mask, so they are
identical whether or not the output is composited with the uncorrupted
input. That makes them the fairest comparison between blind methods (which
never see the mask) and non-blind methods (which are given it).
"""

import numpy as np
from PIL import Image
from scipy import linalg
from skimage.metrics import structural_similarity

# PSNR is capped so a perfect reconstruction does not turn a mean into inf.
MAX_PSNR = 100.0

# RGB SSIM with scikit-image's default 7x7 uniform window, as in the WavTRGAN notebooks.
SSIM_SETTINGS = {"win_size": 7, "gaussian_weights": False, "use_sample_covariance": True}


def load_rgb(path, size=None):
    img = Image.open(path).convert("RGB")
    if size is not None and img.size != size:
        img = img.resize(size, Image.BICUBIC)
    return np.asarray(img, dtype=np.float64) / 255.0


def load_mask(path, hole_value="white", size=None):
    img = Image.open(path).convert("L")
    if size is not None and img.size != size:
        img = img.resize(size, Image.NEAREST)
    mask = np.asarray(img) > 127
    return mask if hole_value == "white" else ~mask


def composite(pred, source, mask):
    """Keep pred inside the hole and source (the uncorrupted input) outside it."""
    return np.where(mask[..., None], pred, source)


def psnr(pred, gt, mask=None):
    sq_err = (pred - gt) ** 2
    mse = sq_err[mask].mean() if mask is not None else sq_err.mean()
    if mse == 0:
        return MAX_PSNR
    return min(MAX_PSNR, 10.0 * np.log10(1.0 / mse))


def ssim(pred, gt, mask=None):
    """SSIM over the whole image, or the SSIM map averaged over hole pixels."""
    score, ssim_map = structural_similarity(
        pred, gt, channel_axis=2, data_range=1.0, full=True, **SSIM_SETTINGS
    )
    if mask is None:
        return float(score)
    return float(ssim_map[mask].mean())


def l1(pred, gt, mask=None):
    abs_err = np.abs(pred - gt)
    return float(abs_err[mask].mean() if mask is not None else abs_err.mean())


def frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """FID between two Gaussians, as in pytorch-fid.

    Reimplemented because pytorch-fid 0.3.0 calls sqrtm(..., disp=False),
    which SciPy 1.18 no longer accepts.
    """
    covmean = linalg.sqrtm(sigma1 @ sigma2)
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset) @ (sigma2 + offset))
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            raise ValueError(f"imaginary component {np.max(np.abs(covmean.imag))}")
        covmean = covmean.real
    diff = mu1 - mu2
    # FID is non-negative; clamp the rounding error of identical statistics.
    return max(0.0, float(diff @ diff + np.trace(sigma1) + np.trace(sigma2) - 2 * np.trace(covmean)))


def image_metrics(pred, gt, mask, source=None):
    """All per-image metrics for one prediction.

    source is the image used outside the hole when compositing; it defaults
    to gt, which equals the corrupted input outside the hole.
    """
    if source is None:
        source = gt
    comp = composite(pred, source, mask)
    return {
        "mask_ratio": float(mask.mean()),
        "psnr_full": psnr(pred, gt),
        "ssim_full": ssim(pred, gt),
        "l1_full": l1(pred, gt),
        "psnr_full_comp": psnr(comp, gt),
        "ssim_full_comp": ssim(comp, gt),
        "l1_full_comp": l1(comp, gt),
        "psnr_hole": psnr(pred, gt, mask),
        "ssim_hole": ssim(pred, gt, mask),
        "l1_hole": l1(pred, gt, mask),
    }
