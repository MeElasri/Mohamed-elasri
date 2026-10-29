"""Evaluate one method's inpainting outputs against ground truth.

Files are matched across folders by file stem (extensions may differ).
Writes <out>/per_image.csv and <out>/summary.json with overall and
per-mask-ratio-bin means of full-image, composited and hole-only metrics.

Usage:
    python -m evaluation.evaluate --method WavTRGAN --gt data/Test/gt \
        --masks data/Test/masks --input data/Test/input \
        --pred results/wavtrgan_seed0 --out eval/wavtrgan_seed0 --fid
"""

import argparse
import csv
import json
import os
import tempfile

import numpy as np
from PIL import Image

from evaluation.masks import list_images
from evaluation.metrics import (MAX_PSNR, SSIM_SETTINGS, composite, frechet_distance, image_metrics, load_mask,
                                load_rgb)

DEFAULT_EDGES = [i / 10 for i in range(11)]      # 10% bins, as in the WavTRGAN notebooks


def index_by_stem(folder):
    return {os.path.splitext(f)[0]: os.path.join(folder, f) for f in list_images(folder)}


def bin_label(low, high):
    return f"{round(low * 100)}-{round(high * 100)}%"


def assign_bin(ratio, edges):
    for low, high in zip(edges[:-1], edges[1:]):
        if low <= ratio < high or (high == edges[-1] and ratio == high):
            return bin_label(low, high)
    raise ValueError(f"mask ratio {ratio} outside bin edges {edges}")


def mean_metrics(rows, keys):
    return {k: float(np.mean([r[k] for r in rows])) for k in keys}


def to_uint8(img):
    return (np.clip(img, 0, 1) * 255 + 0.5).astype(np.uint8)


def fid_statistics(folders, device, batch_size):
    """Inception pool3 (2048-d) mean and covariance per folder, with pytorch-fid weights."""
    from pytorch_fid.fid_score import compute_statistics_of_path
    from pytorch_fid.inception import InceptionV3

    model = InceptionV3([InceptionV3.BLOCK_INDEX_BY_DIM[2048]]).to(device)
    return {k: compute_statistics_of_path(d, model, batch_size, 2048, device, num_workers=0)
            for k, d in folders.items()}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--method", required=True, help="method name stored in the summary")
    parser.add_argument("--gt", required=True, help="ground-truth images")
    parser.add_argument("--pred", required=True, help="method outputs")
    parser.add_argument("--masks", required=True, help="masks; see --hole-value")
    parser.add_argument("--input", help="corrupted inputs, used outside the hole when compositing (default: gt)")
    parser.add_argument("--out", required=True, help="output folder")
    parser.add_argument("--hole-value", choices=["white", "black"], default="white",
                        help="mask colour that marks the hole (default: white)")
    parser.add_argument("--bin-edges", type=lambda s: [float(x) for x in s.split(",")], default=DEFAULT_EDGES,
                        help="hole-ratio bin edges, e.g. 0,0.1,0.2,...,1")
    parser.add_argument("--resize-pred", action="store_true",
                        help="bicubic-resize outputs whose size differs from gt (otherwise an error)")
    parser.add_argument("--fid", action="store_true", help="also compute FID (needs pytorch-fid)")
    parser.add_argument("--fid-device", default="cuda")
    parser.add_argument("--fid-batch-size", type=int, default=50)
    args = parser.parse_args(argv)

    gt_files = index_by_stem(args.gt)
    if not gt_files:
        raise SystemExit(f"no images found in {args.gt}")
    pred_files = index_by_stem(args.pred)
    mask_files = index_by_stem(args.masks)
    input_files = index_by_stem(args.input) if args.input else None

    missing = [s for s in gt_files if s not in pred_files or s not in mask_files
               or (input_files is not None and s not in input_files)]
    if missing:
        raise SystemExit(f"{len(missing)} images lack an output, mask or input, e.g. {missing[:5]}")

    os.makedirs(args.out, exist_ok=True)
    fid_root = tempfile.TemporaryDirectory() if args.fid else None
    if fid_root:
        fid_dirs = {k: os.path.join(fid_root.name, k) for k in ("gt", "pred", "comp")}
        for d in fid_dirs.values():
            os.makedirs(d)

    rows, resized = [], 0
    for stem in sorted(gt_files):
        gt = load_rgb(gt_files[stem])
        size = (gt.shape[1], gt.shape[0])
        pred = load_rgb(pred_files[stem])
        if pred.shape != gt.shape:
            if not args.resize_pred:
                raise SystemExit(f"{stem}: output size {pred.shape[1::-1]} != gt size {size}; "
                                 "pass --resize-pred to resize outputs")
            pred = load_rgb(pred_files[stem], size)
            resized += 1
        mask = load_mask(mask_files[stem], args.hole_value)
        if mask.shape != gt.shape[:2]:
            raise SystemExit(f"{stem}: mask size {mask.shape[::-1]} != gt size {size}")
        if not mask.any():
            raise SystemExit(f"{stem}: mask has no hole pixels (check --hole-value)")
        source = load_rgb(input_files[stem]) if input_files else gt

        row = {"image": stem, **image_metrics(pred, gt, mask, source)}
        row["bin"] = assign_bin(row["mask_ratio"], args.bin_edges)
        rows.append(row)

        if fid_root:
            Image.fromarray(to_uint8(gt)).save(os.path.join(fid_dirs["gt"], stem + ".png"))
            Image.fromarray(to_uint8(pred)).save(os.path.join(fid_dirs["pred"], stem + ".png"))
            comp = composite(pred, source, mask)
            Image.fromarray(to_uint8(comp)).save(os.path.join(fid_dirs["comp"], stem + ".png"))

    metric_keys = [k for k in rows[0] if k not in ("image", "bin")]
    with open(os.path.join(args.out, "per_image.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["image", "bin", *metric_keys])
        writer.writeheader()
        writer.writerows(rows)

    overall = mean_metrics(rows, metric_keys)
    if fid_root:
        stats = fid_statistics(fid_dirs, args.fid_device, args.fid_batch_size)
        overall["fid_full"] = frechet_distance(*stats["gt"], *stats["pred"])
        overall["fid_full_comp"] = frechet_distance(*stats["gt"], *stats["comp"])
        fid_root.cleanup()

    bins = {}
    for low, high in zip(args.bin_edges[:-1], args.bin_edges[1:]):
        label = bin_label(low, high)
        in_bin = [r for r in rows if r["bin"] == label]
        if in_bin:
            bins[label] = {"n": len(in_bin), **mean_metrics(in_bin, metric_keys)}

    summary = {
        "method": args.method,
        "n_images": len(rows),
        "protocol": {
            "gt": args.gt,
            "pred": args.pred,
            "masks": args.masks,
            "hole_value": args.hole_value,
            "composite_source": "input" if args.input else "gt",
            "resized_outputs": resized,
            "ssim": {"color": "rgb", **SSIM_SETTINGS},
            "psnr_cap_db": MAX_PSNR,
            "hole_ssim": "full-image SSIM map averaged over hole pixels",
        },
        "overall": overall,
        "bins": bins,
    }
    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"{args.method}: {len(rows)} images"
          + (f" ({resized} outputs resized)" if resized else ""))
    for k in ("psnr_full", "psnr_full_comp", "psnr_hole", "ssim_full", "ssim_hole", "fid_full"):
        if k in overall:
            print(f"  {k:16s} {overall[k]:.4f}")


if __name__ == "__main__":
    main()
