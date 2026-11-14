# Kaggle cell: Fig. 4 (ablation of the wavelet block) - picks the representative test image, saves the six panels
# under the file names used in the LaTeX, and prints the four values for the caption.
#
# Before this cell, produce the outputs of both models on the CelebA-HQ 40-50% test set with wavtrgan-evaluation.ipynb
# (configuration cell), once per model:
#   with wavelet    (12 blocks): MODEL_PATH = <WavTRGAN checkpoint>,     OUT_ROOT = "/kaggle/working/eval_with"
#   without wavelet (12 blocks): MODEL_PATH = <no-wavelet checkpoint>,   OUT_ROOT = "/kaggle/working/eval_without"
# Each run writes input/, gt/ and wavtrgan/ (the outputs) under OUT_ROOT, with the same file names.
import os
import shutil
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

GT_DIR = "/kaggle/working/eval_with/gt"
INPUT_DIR = "/kaggle/working/eval_with/input"
OUT_WITHOUT = "/kaggle/working/eval_without/wavtrgan"   # (a) without wavelet
OUT_WITH = "/kaggle/working/eval_with/wavtrgan"         # (b) with wavelet (proposed)
FIG_DIR = "/kaggle/working/fig4"                        # upload its content to Overleaf, keeping the folders


def rgb(folder, stem):
    for f in os.listdir(folder):
        if os.path.splitext(f)[0] == stem:
            return np.asarray(Image.open(os.path.join(folder, f)).convert("RGB").resize((256, 256), Image.BICUBIC),
                              dtype=np.float64)
    raise FileNotFoundError("%s has no file named %s" % (folder, stem))


stems = sorted(os.path.splitext(f)[0] for f in os.listdir(GT_DIR) if not f.startswith("."))
scores = {}                                             # stem -> (PSNR a, SSIM a, PSNR b, SSIM b), full image
for s in stems:
    gt = rgb(GT_DIR, s)
    a, b = rgb(OUT_WITHOUT, s), rgb(OUT_WITH, s)
    scores[s] = (peak_signal_noise_ratio(gt, a, data_range=255), structural_similarity(gt, a, data_range=255, channel_axis=2),
                 peak_signal_noise_ratio(gt, b, data_range=255), structural_similarity(gt, b, data_range=255, channel_axis=2))
v = np.array([scores[s] for s in stems])
v[~np.isfinite(v)] = np.nan
mean_a, mean_b = np.nanmean(v[:, 0]), np.nanmean(v[:, 2])
print("test averages: without wavelet %.2f dB, with wavelet %.2f dB  (paper tables: 31.45 and 33.94 dB)" % (mean_a, mean_b))

# representative image: PSNR closest to the test average of BOTH models
best = min(stems, key=lambda s: abs(scores[s][0] - mean_a) + abs(scores[s][2] - mean_b)
           if np.isfinite(scores[s][0]) and np.isfinite(scores[s][2]) else np.inf)
pa, sa, pb, sb = scores[best]
print("chosen test image:", best)

for sub in ("ablation", "without_res", "with_res"):
    os.makedirs(os.path.join(FIG_DIR, sub), exist_ok=True)
gt, inp = rgb(GT_DIR, best), rgb(INPUT_DIR, best)
a, b = rgb(OUT_WITHOUT, best), rgb(OUT_WITH, best)
save = lambda img, path: Image.fromarray(np.clip(img, 0, 255).round().astype(np.uint8)).save(os.path.join(FIG_DIR, path))
save(inp, "ablation/input.png")
save(gt, "ablation/ground_truth.png")
save(a, "without_res/without_fake_img.png")
save(b, "with_res/with_fake_img.png")
err_a, err_b = np.abs(gt - a).mean(axis=2) / 255, np.abs(gt - b).mean(axis=2) / 255
vmax = max(err_a.max(), err_b.max())                   # same colour scale for both error maps
plt.imsave(os.path.join(FIG_DIR, "without_res/mae_without.png"), err_a, cmap="jet", vmin=0, vmax=vmax)
plt.imsave(os.path.join(FIG_DIR, "with_res/mae_with.png"), err_b, cmap="jet", vmin=0, vmax=vmax)
shutil.make_archive(FIG_DIR, "zip", FIG_DIR)

print("\nCaption values:")
print("(a) Without wavelet: PSNR=%.2f dB, SSIM=%.3f" % (pa, sa))
print("(b) With wavelet:    PSNR=%.2f dB, SSIM=%.3f" % (pb, sb))
print("\nsaved the six panels to %s (and %s.zip)" % (FIG_DIR, FIG_DIR))
