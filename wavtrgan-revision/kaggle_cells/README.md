# Kaggle cells for the revision

Each file is one notebook cell: paste it into a Kaggle notebook, edit the paths at the top, and run it.

| Cell | What it gives |
|---|---|
| `metrics_kaggle.py` | All metrics of one setting for every method: PSNR, SSIM (full image), PSNR_h, SSIM_h (hole only), FID, hole size, and an "Input (no inpainting)" reference row. Run `!pip install -q pytorch-fid` first. |
| `hole_scores_kaggle.py` | Only PSNR_h and SSIM_h (mean +/- std) for every method of one setting. No extra package. |
| `fig4_kaggle.py` | Fig. 4: picks the test image whose PSNR is closest to the test average of both models (with and without wavelet), saves the six panels and prints the caption values. |
| `count_gflops_cmt_tformer.py` | GFLOPs (fvcore) and parameters of CMT and T-former from their official code. |

Without a mask folder (`MASK_DIR = None`), the hole of each test image is the set of white pixels of its corrupted version
(plus pixels that differ from the ground truth by more than 5%). Outputs are matched to the ground truth by file name.
The metric definitions are the same as in `wavtrgan-evaluation.ipynb` and `evaluation/`.
