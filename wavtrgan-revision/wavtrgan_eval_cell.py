# =============================================================================================
# Corrected evaluation for the WavTRGAN revision (one notebook cell, self-contained)
#
# Part 1  generate_wavtrgan(): runs WavTRGAN on the test pairs WITHOUT random crop / flip and
#         saves input, ground truth and output under the SAME file name (needed for fair,
#         reproducible pairs and for hole-only metrics). If the test set has exact masks
#         (test_root/mask, written by make_corrupted_dataset.py), they are saved too: give them
#         to the non-blind baselines and to evaluate(mask_dir=...).
# Part 2  evaluate(): scores every method on the same pairs:
#         - PSNR / SSIM on float images (no uint8 overflow), full image, RGB
#         - the same after compositing (output inside the hole, input outside)
#         - PSNR / SSIM on the corrupted region only (hole) and PSNR on the rest (known)
#         - LPIPS (AlexNet), FID (torch-fidelity)
#         The hole is the exact mask when mask_dir is given. Without it, the hole is estimated as the
#         pixels where input and ground truth differ by > HOLE_THRESHOLD, which misses corrupted
#         pixels whose fill looks like the original: use exact masks for the paper.
#         Files are matched by NAME (sorted), never by glob order.
#         Besides out_csv (one row per method), it writes per method <out_csv>_<method>_per_image.csv
#         and <out_csv>_<method>.json: the JSON files are the input of evaluation/aggregate.py
#         (mean +/- std over seeds), the per-image CSVs that of evaluation/pick_examples.py.
#
# Folder layout expected for every method: one folder of outputs whose file names are the same
# as in GT_DIR / INPUT_DIR (e.g. 0001.png). Run baselines on INPUT_DIR and save their outputs
# under the same names.
# =============================================================================================
import os, csv, json
import numpy as np
import torch
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

EXTS = (".png", ".jpg", ".jpeg")
HOLE_THRESHOLD = 0.05          # without exact masks: a pixel is "corrupted" if input and ground truth differ by > 0.05


def load_rgb(path, size=None):
    im = Image.open(path).convert("RGB")          # also drops the alpha channel of plt.imsave PNGs
    if size is not None and im.size != size:
        im = im.resize(size, Image.BICUBIC)
    return np.asarray(im, dtype=np.float64)       # float64 in [0, 255]: no uint8 wrap-around


def load_mask(path, size):
    """Binary mask (True = corrupted) from a PNG with white holes, resized with nearest neighbour."""
    im = Image.open(path).convert("L")
    if im.size != size:
        im = im.resize(size, Image.NEAREST)
    return np.asarray(im) > 127


def names_in(folder):
    return sorted(f for f in os.listdir(folder) if f.lower().endswith(EXTS))


# ---------------------------------------------------------------------------------------------
# Part 1: WavTRGAN outputs on fixed pairs (run in the WavTRGAN notebook, after the model cells)
# ---------------------------------------------------------------------------------------------
@torch.no_grad()
def generate_wavtrgan(model, test_root, out_root, direction="b2a", size=256, device="cuda"):
    """test_root/{a,b}: clean / corrupted pairs with identical names (and optionally test_root/mask).
    Writes out_root/{input,gt,wavtrgan} and, when test_root/mask exists, out_root/mask."""
    corrupted_dir, clean_dir = (os.path.join(test_root, "b"), os.path.join(test_root, "a")) if direction == "b2a" \
        else (os.path.join(test_root, "a"), os.path.join(test_root, "b"))
    mask_src = os.path.join(test_root, "mask")
    keys = ("input", "gt", "wavtrgan") + (("mask",) if os.path.isdir(mask_src) else ())
    dirs = {k: os.path.join(out_root, k) for k in keys}
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
        if "mask" in dirs:
            mask = load_mask(os.path.join(mask_src, stem), (size, size))
            Image.fromarray(mask.astype(np.uint8) * 255).save(os.path.join(dirs["mask"], stem))
    print("saved", len(names_in(dirs["input"])), "pairs to", out_root)
    if "mask" not in dirs:
        print("WARNING: %s has no mask/ folder, so hole metrics will use the threshold estimate. "
              "Regenerate the test set with make_corrupted_dataset.py to get exact masks." % test_root)


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


def ratio_bin(ratio):
    """10% bin of the hole ratio, labelled as in evaluation/evaluate.py (e.g. "30-40%")."""
    low = min(int(ratio * 10), 9) * 10
    return "%d-%d%%" % (low, low + 10)


def finite_mean(values):
    """Mean over finite values: a perfect pixel region (inf dB) or an empty hole (nan) is left out."""
    v = np.array(values, dtype=float)
    v = v[np.isfinite(v)]
    return float(v.mean()) if v.size else float("nan")


def evaluate(methods, gt_dir, input_dir, out_csv="metrics.csv", lpips_on=True, fid_on=True, device=None, mask_dir=None):
    """methods: {"WavTRGAN": folder, "CAML": folder, ...}. Every folder must contain the same file names.
    mask_dir: exact masks (white = corrupted) with the same names; None = threshold estimate."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    gt_names = names_in(gt_dir)
    mask_kind = "exact" if mask_dir else "threshold %.2f" % HOLE_THRESHOLD
    if not mask_dir:
        print("WARNING: no mask_dir, the hole is estimated from |input - gt| > %.2f. "
              "Use exact masks for the paper." % HOLE_THRESHOLD)
    lp = None
    if lpips_on:
        import lpips
        lp = lpips.LPIPS(net="alex", verbose=False).to(device).eval()
    base = os.path.splitext(out_csv)[0]
    rows = []
    for method, folder in methods.items():
        stems = {os.path.splitext(f)[0]: f for f in names_in(folder)}
        pairs = [(g, stems[os.path.splitext(g)[0]]) for g in gt_names if os.path.splitext(g)[0] in stems]
        missing = len(gt_names) - len(pairs)
        assert pairs, "%s: no file names in common with %s" % (method, gt_dir)
        per_image, resized = [], 0
        for g, f in pairs:
            gt = load_rgb(os.path.join(gt_dir, g))
            size = (gt.shape[1], gt.shape[0])
            pr = load_rgb(os.path.join(folder, f))
            if pr.shape != gt.shape:              # e.g. a baseline that outputs another resolution
                pr = load_rgb(os.path.join(folder, f), size=size)
                resized += 1
            inp = load_rgb(os.path.join(input_dir, g), size=size)
            if mask_dir:
                hole = load_mask(os.path.join(mask_dir, os.path.splitext(g)[0] + ".png"), size)
            else:
                hole = np.abs(inp - gt).max(axis=2) > HOLE_THRESHOLD * 255
            comp = np.where(hole[..., None], pr, inp)   # output inside the hole, uncorrupted input outside
            ssim_val, smap = structural_similarity(gt, pr, data_range=255, channel_axis=2, full=True)   # one SSIM pass
            r = {"image": os.path.splitext(g)[0], "bin": ratio_bin(hole.mean()), "mask_ratio": float(hole.mean()),
                 "psnr_full": peak_signal_noise_ratio(gt, pr, data_range=255), "ssim_full": ssim_val,
                 "psnr_full_comp": peak_signal_noise_ratio(gt, comp, data_range=255),
                 "ssim_full_comp": structural_similarity(gt, comp, data_range=255, channel_axis=2),
                 "psnr_hole": masked_psnr(gt, pr, hole), "ssim_hole": masked_ssim(smap, hole),
                 "psnr_known": masked_psnr(gt, pr, ~hole)}
            if lp is not None:
                t = lambda a: torch.from_numpy(a / 127.5 - 1).permute(2, 0, 1)[None].float().to(device)
                with torch.no_grad():
                    r["lpips_full"] = lp(t(pr), t(gt)).item()
            per_image.append(r)

        keys = [k for k in per_image[0] if k not in ("image", "bin")]
        overall = {k: finite_mean([r[k] for r in per_image]) for k in keys}
        if fid_on:
            import torch_fidelity
            overall["fid_full"] = torch_fidelity.calculate_metrics(
                input1=folder, input2=gt_dir, cuda=torch.cuda.is_available(), fid=True, verbose=False
            )["frechet_inception_distance"]
        bins = {}
        for label in sorted({r["bin"] for r in per_image}):
            in_bin = [r for r in per_image if r["bin"] == label]
            bins[label] = {"n": len(in_bin), **{k: finite_mean([r[k] for r in in_bin]) for k in keys}}

        tag = base + "_" + "".join(c if c.isalnum() else "_" for c in method)
        with open(tag + "_per_image.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(per_image[0].keys()))
            w.writeheader()
            w.writerows(per_image)
        with open(tag + ".json", "w") as fh:
            json.dump({"method": method, "n_images": len(pairs),
                       "protocol": {"mask": mask_kind, "resized_outputs": resized,
                                    "ssim": "RGB, 7x7 uniform window (scikit-image default)"},
                       "overall": overall, "bins": bins}, fh, indent=2)

        row = {"method": method, "images": len(pairs), "missing": missing, "resized": resized, "mask": mask_kind,
               "PSNR": overall["psnr_full"], "SSIM": overall["ssim_full"],
               "PSNR_comp": overall["psnr_full_comp"], "SSIM_comp": overall["ssim_full_comp"],
               "PSNR_hole": overall["psnr_hole"], "SSIM_hole": overall["ssim_hole"],
               "PSNR_known": overall["psnr_known"], "LPIPS": overall.get("lpips_full", np.nan),
               "hole_ratio": overall["mask_ratio"]}
        if fid_on:
            row["FID"] = overall["fid_full"]
        rows.append(row)
        print("%-22s n=%d%s%s  PSNR %.2f  SSIM %.4f | composited: PSNR %.2f SSIM %.4f | hole: PSNR %.2f SSIM %.4f"
              " | known PSNR %.2f | LPIPS %.4f%s" % (
                  method, row["images"], " (%d missing!)" % missing if missing else "",
                  " (%d resized)" % resized if resized else "", row["PSNR"], row["SSIM"],
                  row["PSNR_comp"], row["SSIM_comp"], row["PSNR_hole"], row["SSIM_hole"], row["PSNR_known"],
                  row["LPIPS"], "  FID %.2f" % row["FID"] if "FID" in row else ""))
    with open(out_csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("mean corrupted fraction of the test set: %.1f%% (%s mask)  -> %s" % (
        100 * rows[0]["hole_ratio"], mask_kind, out_csv))
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
# generate_wavtrgan(model, "/kaggle/input/<dataset>/test", "eval_40_50")          # WavTRGAN outputs (+ mask/)
# evaluate({"WavTRGAN": "eval_40_50/wavtrgan",
#           "Blind-Omni-Wav-Net": "eval_40_50/omniwavnet",                         # same file names
#           "CAML": "eval_40_50/caml"},
#          gt_dir="eval_40_50/gt", input_dir="eval_40_50/input", mask_dir="eval_40_50/mask",
#          out_csv="metrics_40_50.csv")
