"""Deterministic free-form mask generation and blind corruption.

Every mask is drawn from its own RNG seeded with (seed, index), so a given
mask never changes when the test set is reordered or extended. Masks are
spread evenly over the mask-ratio bins (image i goes to bin i % n_bins).

Usage:
    python -m evaluation.masks --gt data/Test/gt --out data/Test/masks \
        --size 256 --seed 2024 --fill image --corrupted-out data/Test/input
"""

import argparse
import json
import math
import os

import numpy as np
from PIL import Image, ImageDraw

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp")
DEFAULT_BINS = [(0.1, 0.2), (0.2, 0.3), (0.3, 0.4), (0.4, 0.5), (0.5, 0.6)]


def list_images(folder):
    return sorted(f for f in os.listdir(folder) if f.lower().endswith(IMAGE_EXTENSIONS))


def draw_stroke(draw, size, rng):
    """One random brush stroke: a polyline of thick segments with round joints."""
    min_width, max_width = max(2, size // 40), max(4, size // 10)
    num_vertices = int(rng.integers(4, 13))
    x, y = rng.uniform(0, size, 2)
    width = int(rng.integers(min_width, max_width + 1))
    for _ in range(num_vertices):
        angle = rng.uniform(0, 2 * math.pi)
        length = rng.uniform(size / 16, size / 4)
        nx = float(np.clip(x + length * math.cos(angle), 0, size - 1))
        ny = float(np.clip(y + length * math.sin(angle), 0, size - 1))
        draw.line([(x, y), (nx, ny)], fill=255, width=width)
        r = width / 2
        draw.ellipse([nx - r, ny - r, nx + r, ny + r], fill=255)
        x, y = nx, ny


def generate_mask(size, ratio_range, rng, max_tries=1000):
    """Return a uint8 HxW mask (255 = hole) whose hole ratio lies in ratio_range."""
    low, high = ratio_range
    for _ in range(max_tries):
        canvas = Image.new("L", (size, size), 0)
        draw = ImageDraw.Draw(canvas)
        ratio = 0.0
        while ratio < low:
            draw_stroke(draw, size, rng)
            ratio = np.count_nonzero(np.asarray(canvas)) / (size * size)
        if ratio <= high:
            return np.asarray(canvas, dtype=np.uint8)
    raise RuntimeError(f"could not hit ratio range {ratio_range} in {max_tries} tries")


def mask_rng(seed, index):
    return np.random.default_rng([seed, index])


def corrupt(gt, mask, fill, rng, donor=None):
    """Blind corruption: replace the hole region of gt (HxWx3 uint8) with fill content.

    fill: "noise" (uniform RGB noise), "constant" (white) or "image" (the
    corresponding region of a donor image, as in VCNet-style protocols).
    """
    hole = (mask > 127)[..., None]
    if fill == "noise":
        content = rng.integers(0, 256, gt.shape, dtype=np.uint8)
    elif fill == "constant":
        content = np.full_like(gt, 255)
    elif fill == "image":
        if donor is None:
            raise ValueError("fill='image' needs a donor image")
        content = donor
    else:
        raise ValueError(f"unknown fill: {fill}")
    return np.where(hole, content, gt)


def load_rgb_uint8(path, size):
    img = Image.open(path).convert("RGB")
    if img.size != (size, size):
        img = img.resize((size, size), Image.BICUBIC)
    return np.asarray(img, dtype=np.uint8)


def parse_bins(text):
    bins = []
    for part in text.split(","):
        low, high = part.split("-")
        bins.append((float(low), float(high)))
    return bins


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", required=True, help="output folder for masks (PNG, 255 = hole)")
    parser.add_argument("--gt", help="ground-truth folder; one mask is made per image, with the same file stem")
    parser.add_argument("--count", type=int, help="number of masks when --gt is not given")
    parser.add_argument("--size", type=int, default=256,
                        help="mask size; --gt images must already be this size for evaluate.py")
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--bins", type=parse_bins, default=DEFAULT_BINS,
                        help="comma-separated hole-ratio bins, e.g. 0.1-0.2,0.2-0.3")
    parser.add_argument("--fill", choices=["noise", "constant", "image"],
                        help="also write blind-corrupted inputs with this fill (needs --gt and --corrupted-out)")
    parser.add_argument("--corrupted-out", help="output folder for corrupted inputs")
    parser.add_argument("--donors", help="folder of donor images for --fill image (default: --gt)")
    args = parser.parse_args(argv)

    if args.gt:
        names = [os.path.splitext(f)[0] for f in list_images(args.gt)]
        gt_files = list_images(args.gt)
    elif args.count:
        names = [f"{i:05d}" for i in range(args.count)]
        gt_files = None
    else:
        parser.error("give --gt or --count")
    if args.fill and not (args.gt and args.corrupted_out):
        parser.error("--fill needs --gt and --corrupted-out")

    os.makedirs(args.out, exist_ok=True)
    if args.corrupted_out:
        os.makedirs(args.corrupted_out, exist_ok=True)
    donor_dir = args.donors or args.gt
    donor_files = list_images(donor_dir) if args.fill == "image" else None

    manifest = {
        "seed": args.seed,
        "size": args.size,
        "bins": args.bins,
        "hole_value": 255,
        "fill": args.fill,
        "masks": [],
    }
    for i, name in enumerate(names):
        rng = mask_rng(args.seed, i)
        ratio_range = args.bins[i % len(args.bins)]
        mask = generate_mask(args.size, ratio_range, rng)
        Image.fromarray(mask).save(os.path.join(args.out, name + ".png"))
        entry = {"name": name, "bin": list(ratio_range),
                 "ratio": round(float(np.count_nonzero(mask)) / mask.size, 4)}

        if args.fill:
            gt = load_rgb_uint8(os.path.join(args.gt, gt_files[i]), args.size)
            donor = None
            if args.fill == "image":
                # Never use the image itself as its own donor.
                choices = [f for f in donor_files if os.path.splitext(f)[0] != name or donor_dir != args.gt]
                donor_file = choices[int(rng.integers(len(choices)))]
                donor = load_rgb_uint8(os.path.join(donor_dir, donor_file), args.size)
                entry["donor"] = donor_file
            corrupted = corrupt(gt, mask, args.fill, rng, donor)
            Image.fromarray(corrupted).save(os.path.join(args.corrupted_out, name + ".png"))
        manifest["masks"].append(entry)

    with open(os.path.join(args.out, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"wrote {len(names)} masks to {args.out}")


if __name__ == "__main__":
    main()
