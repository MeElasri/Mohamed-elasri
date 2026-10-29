# =============================================================================================
# Corrected evaluation for the WavTRGAN revision (one notebook cell, self-contained)
#
# Part 1  generate_wavtrgan(): runs WavTRGAN on the test pairs WITHOUT random crop / flip and
#         saves input, ground truth and output under the SAME file name (needed for fair,
#         reproducible pairs and for hole-only metrics).
# Part 2  evaluate(): scores every method on the same pairs:
#         - PSNR / SSIM on float images (no uint8 overflow), full image, RGB
#         - PSNR / SSIM on the corrupted region only (hole) and on the rest (known)
#         - LPIPS (AlexNet), FID (torch-fidelity)
#         Files are matched by NAME (sorted), never by glob order.
#
# Folder layout expected for every method: one folder of outputs whose file names are the same
# as in GT_DIR / INPUT_DIR (e.g. 0001.png). Run baselines on INPUT_DIR and save their outputs
# under the same names.
# =============================================================================================
import os, csv
import numpy as np
import torch
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

EXTS = (".png", ".jpg", ".jpeg")
HOLE_THRESHOLD = 0.05          # a pixel is "corrupted" if input and ground truth differ by > 0.05 (any channel)


def load_rgb(path, size=None):
    im = Image.open(path).convert("RGB")          # also drops the alpha channel of plt.imsave PNGs
    if size is not None and im.size != size:
        im = im.resize(size, Image.BICUBIC)
    return np.asarray(im, dtype=np.float64)       # float64 in [0, 255]: no uint8 wrap-around


def names_in(folder):
    return sorted(f for f in os.listdir(folder) if f.lower().endswith(EXTS))


# ---------------------------------------------------------------------------------------------
# Part 1: WavTRGAN outputs on fixed pairs (run in the WavTRGAN notebook, after the model cells)
# ---------------------------------------------------------------------------------------------
@torch.no_grad()
def generate_wavtrgan(model, test_root, out_root, direction="b2a", size=256, device="cuda"):
    """test_root/{a,b}: clean / corrupted pairs with identical names. Writes out_root/{input,gt,wavtrgan}."""
    corrupted_dir, clean_dir = (os.path.join(test_root, "b"), os.path.join(test_root, "a")) if direction == "b2a" \
        else (os.path.join(test_root, "a"), os.path.join(test_root, "b"))
    dirs = {k: os.path.join(out_root, k) for k in ("input", "gt", "wavtrgan")}
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)
    model = model.to(device).eval()
    for name in names_in(corrupted_dir):
        stem = os.path.splitext(name)[0] + ".png"
        inp = load_rgb(os.path.join(corrupted_dir, name), (size, size)) / 255.0
        gt = load_rgb(os.path.join(clean_dir, name), (size, size)) / 255.0
        x = torch.from_numpy(inp).permute(2, 0, 1)[None].float().to(device)   # [0, 1], as in training
        y = model(x).clamp(0, 1)[0].permute(1, 2, 0).cpu().numpy()
        for key, arr in (("input", inp), ("gt", gt), ("wavtrgan", y)):
            Image.fromarray((arr * 255).round().astype(np.uint8)).save(os.path.join(dirs[key], stem))
    print("saved", len(names_in(dirs["input"])), "pairs to", out_root)


# ---------------------------------------------------------------------------------------------
# Part 2: metrics for all methods on the same pairs
# ---------------------------------------------------------------------------------------------
def masked_psnr(gt, pr, m):
    """PSNR over the pixels where m is True (m: H x W bool), images float in [0, 255]."""
    if m.sum() == 0:
        return np.nan
    mse = ((gt - pr) ** 2)[m].mean()
    return np.inf if mse == 0 else 10 * np.log10(255.0 ** 2 / mse)


def masked_ssim(smap, m):
    """Mean of an SSIM map (H x W x C, from structural_similarity(..., full=True)) over the pixels where m is True."""
    if m.sum() == 0:
        return np.nan
    return smap.mean(axis=2)[m].mean()


def evaluate(methods, gt_dir, input_dir, out_csv="metrics.csv", lpips_on=True, fid_on=True, device=None):
    """methods: {"WavTRGAN": folder, "CAML": folder, ...}. Every folder must contain the same file names."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    gt_names = names_in(gt_dir)
    lp = None
    if lpips_on:
        import lpips
        lp = lpips.LPIPS(net="alex", verbose=False).to(device).eval()
    rows = []
    for method, folder in methods.items():
        stems = {os.path.splitext(f)[0]: f for f in names_in(folder)}
        pairs = [(g, stems[os.path.splitext(g)[0]]) for g in gt_names if os.path.splitext(g)[0] in stems]
        missing = len(gt_names) - len(pairs)
        assert pairs, "%s: no file names in common with %s" % (method, gt_dir)
        r = {k: [] for k in ("psnr", "ssim", "psnr_hole", "ssim_hole", "psnr_known", "lpips", "ratio")}
        for g, f in pairs:
            gt = load_rgb(os.path.join(gt_dir, g))
            pr = load_rgb(os.path.join(folder, f), size=(gt.shape[1], gt.shape[0]))
            inp = load_rgb(os.path.join(input_dir, g), size=(gt.shape[1], gt.shape[0]))
            hole = (np.abs(inp - gt).max(axis=2) > HOLE_THRESHOLD * 255)
            r["psnr"].append(peak_signal_noise_ratio(gt, pr, data_range=255))
            ssim_val, smap = structural_similarity(gt, pr, data_range=255, channel_axis=2, full=True)   # one SSIM pass
            r["ssim"].append(ssim_val)
            r["psnr_hole"].append(masked_psnr(gt, pr, hole))
            r["ssim_hole"].append(masked_ssim(smap, hole))
            r["psnr_known"].append(masked_psnr(gt, pr, ~hole))
            r["ratio"].append(hole.mean())
            if lp is not None:
                t = lambda a: torch.from_numpy(a / 127.5 - 1).permute(2, 0, 1)[None].float().to(device)
                with torch.no_grad():
                    r["lpips"].append(lp(t(pr), t(gt)).item())
        finite = lambda v: np.array(v, dtype=float)[np.isfinite(v)]
        row = {"method": method, "images": len(pairs), "missing": missing,
               "PSNR": finite(r["psnr"]).mean(), "SSIM": np.mean(r["ssim"]),
               "PSNR_hole": finite(r["psnr_hole"]).mean(), "SSIM_hole": np.nanmean(r["ssim_hole"]),
               "PSNR_known": finite(r["psnr_known"]).mean(),
               "LPIPS": np.mean(r["lpips"]) if r["lpips"] else np.nan,
               "hole_ratio": np.mean(r["ratio"])}
        if fid_on:
            import torch_fidelity
            row["FID"] = torch_fidelity.calculate_metrics(
                input1=folder, input2=gt_dir, cuda=torch.cuda.is_available(), fid=True, verbose=False
            )["frechet_inception_distance"]
        rows.append(row)
        print("%-22s n=%d%s  PSNR %.2f  SSIM %.4f | hole: PSNR %.2f SSIM %.4f | known PSNR %.2f | LPIPS %.4f%s" % (
            method, row["images"], " (%d missing!)" % missing if missing else "", row["PSNR"], row["SSIM"],
            row["PSNR_hole"], row["SSIM_hole"], row["PSNR_known"], row["LPIPS"],
            "  FID %.2f" % row["FID"] if "FID" in row else ""))
    with open(out_csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("mean corrupted fraction of the test set: %.1f%%  -> %s" % (100 * rows[0]["hole_ratio"], out_csv))
    return rows


# ---------------------------------------------------------------------------------------------
# Sanity check: the old uint8 code vs this code on two unrelated images (should be ~8 dB here)
# ---------------------------------------------------------------------------------------------
def sanity_check():
    rng = np.random.default_rng(0)
    a = rng.integers(0, 256, (64, 64, 3)).astype(np.uint8)
    b = rng.integers(0, 256, (64, 64, 3)).astype(np.uint8)
    old = 20 * np.log10(255 / np.sqrt(np.mean((a - b) ** 2)))           # the old, wrong formula
    new = peak_signal_noise_ratio(a.astype(np.float64), b.astype(np.float64), data_range=255)
    print("unrelated random images: old uint8 PSNR %.2f dB (wrong), corrected %.2f dB" % (old, new))


# ---------------------------------------------------------------------------------------------
# Example use (edit the paths)
# ---------------------------------------------------------------------------------------------
# sanity_check()
# generate_wavtrgan(model, "/kaggle/input/<dataset>/test", "eval_40_50")          # WavTRGAN outputs
# evaluate({"WavTRGAN": "eval_40_50/wavtrgan",
#           "Blind-Omni-Wav-Net": "eval_40_50/omniwavnet",                         # same file names
#           "CAML": "eval_40_50/caml"},
#          gt_dir="eval_40_50/gt", input_dir="eval_40_50/input", out_csv="metrics_40_50.csv")
