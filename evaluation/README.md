# Inpainting evaluation toolkit

Evaluates any inpainting method (blind or non-blind) from folders of images, so
WavTRGAN and every baseline are scored with exactly the same masks, splits and
metric code.

| Script | What it does | Reviewer comment |
|---|---|---|
| `masks.py` | Deterministic free-form masks per mask-ratio bin, optional blind corruption (noise / constant / donor-image fill), `manifest.json` with the seed and every mask's ratio | 6, 8 |
| `evaluate.py` | PSNR, SSIM, L1 on the full image, on the composited image and on the hole only; per-bin means; optional FID | 2 |
| `aggregate.py` | Mean ± std over seeds, LaTeX/Markdown tables, best (**bold**) and second-best (<u>underline</u>) marked from the printed values | 3, 6 |
| `pick_examples.py` | Qualitative examples where every method scores close to its own test-set average | 6 (Fig. 4) |

## Install

```bash
pip install -r requirements-eval.txt       # numpy, pillow, scipy, scikit-image
pip install torch torchvision pytorch-fid  # only for --fid
```

## Workflow

```bash
# 1. Fixed test masks and blind-corrupted inputs (once; release these with the code)
python -m evaluation.masks --gt data/Test/gt --out data/Test/masks \
    --size 256 --seed 2024 --fill image --corrupted-out data/Test/input

# 2. Run every method on data/Test/input. Non-blind methods also get
#    data/Test/masks; blind methods get the corrupted input only.
#    Save outputs with the same file stems as the ground truth.

# 3. Score each run (one per seed for WavTRGAN)
python -m evaluation.evaluate --method WavTRGAN --gt data/Test/gt \
    --masks data/Test/masks --input data/Test/input \
    --pred results/wavtrgan_seed0 --out eval/wavtrgan_seed0 --fid

# 4. Tables (overall, or one mask-ratio bin with --scope 30-40%)
python -m evaluation.aggregate \
    --method WavTRGAN eval/wavtrgan_seed0/summary.json eval/wavtrgan_seed1/summary.json eval/wavtrgan_seed2/summary.json \
    --method T-former eval/tformer/summary.json \
    --metrics psnr_full ssim_full fid_full psnr_hole ssim_hole --out tables/table2

# 5. Representative qualitative examples
python -m evaluation.pick_examples --csv WavTRGAN=eval/wavtrgan_seed0/per_image.csv \
    --csv T-former=eval/tformer/per_image.csv --metric psnr_full --top 5
```

Run `python -m pytest` from the repository root to run the tests.

## Conventions

- Files are matched by stem across `--gt`, `--pred`, `--masks`, `--input`
  (extensions may differ). Every ground-truth image must have an output and a mask.
- Masks are single-channel PNGs with white (255) = hole; use `--hole-value black`
  for the opposite convention.
- Outputs whose size differs from the ground truth are an error unless
  `--resize-pred` is passed. The number of resized outputs is stored in
  `summary.json`. A resolution mismatch is a common reason a pretrained
  baseline scores far below its published numbers.

## Metric definitions

| Key | Meaning |
|---|---|
| `*_full` | Raw network output against ground truth, over the whole image |
| `*_full_comp` | Output pasted into the uncorrupted input (output inside the hole, input outside), over the whole image |
| `*_hole` | Hole pixels only. Identical with or without compositing |
| `fid_full`, `fid_full_comp` | FID (pytorch-fid, Inception pool3, 2048-d) between ground truth and outputs |

- **PSNR**: images in [0, 1], MSE over all RGB values of the region, capped at 100 dB.
- **SSIM**: RGB, Gaussian window (σ = 1.5), as in Wang et al. (2004). Hole SSIM
  is the full-image SSIM map averaged over hole pixels.
- **FID**: pytorch-fid's Inception weights; the Fréchet distance is computed in
  `metrics.py` because pytorch-fid 0.3.0 breaks on SciPy 1.18. With fewer than
  2048 test images the covariance is singular (SciPy warns) and FID is noisy;
  Paris StreetView, for instance, has only 100 test images.
- **Std over seeds**: sample standard deviation (n − 1).

Hole-only metrics are the fairest way to compare blind and non-blind methods.
Full-image metrics on raw outputs reward methods that leave uncorrupted pixels
alone, and compositing uses the ground-truth mask, which a blind method never
receives.

## Protocol paragraph for the paper (fill in the brackets)

> All methods are evaluated on the same [N] test images at [256×256] resolution,
> with the same [N] free-form masks generated once with a fixed seed ([2024]) and
> distributed evenly over five hole-ratio bins (10–20 % to 50–60 %). Corrupted
> regions are filled with [patches from other test images]. Blind methods,
> including WavTRGAN, receive only the corrupted image; non-blind baselines
> additionally receive the ground-truth mask. We report PSNR and SSIM both on
> the full image and on the corrupted region only. Full-image metrics are
> computed on the raw network output [and, in Table X, after compositing it with
> the uncorrupted input]. Hole-region metrics are unaffected by compositing.
> SSIM uses a Gaussian window (σ = 1.5) on RGB images. FID is computed between
> the [N] outputs and the ground-truth images with pytorch-fid. WavTRGAN results
> are the mean ± standard deviation over three training seeds.
