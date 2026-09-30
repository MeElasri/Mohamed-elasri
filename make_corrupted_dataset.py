"""Create corrupted versions of CelebA-HQ (or any image folder) for blind inpainting experiments.

For every clean image of <src>/<split>/a (or <src>/<split> if there is no "a" folder), it writes
    <dst>/<split>/a/<name>.png      the clean image, resized to --size
    <dst>/<split>/b/<name>.png      the corrupted image (the input of the network)
    <dst>/<split>/mask/<name>.png   the exact corrupted region (white = corrupted)
which is the folder layout of wavtr-flow-matching.ipynb: set opt.root_path = <dst>.

Mask shapes (--mask):   strokes (free-form brush strokes), boxes (random rectangles), center (the classic
                        central square with half the image side), mixed (strokes or boxes), folder (mask images
                        from --mask-dir, e.g. the irregular masks of Liu et al.)
Hole contents (--fill): white, color (one random color), noise (random color + Gaussian noise),
                        patch (the same region of another image of the split), mixed (one of the four)
--ratio sets the range of the corrupted fraction for strokes and boxes; each image draws its target uniformly.
Every image has its own random generator seeded by (--seed, split, index), so the output is reproducible.

Example (Kaggle):
    !python make_corrupted_dataset.py --src /kaggle/input/celeba-hq-img-full-50/CelebA-HQ-img \
        --dst /kaggle/working/celebahq_mixed --mask strokes --fill mixed
"""
import argparse
import math
import os

import numpy as np
from PIL import Image, ImageDraw

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg")


def list_images(folder):
    return sorted(f for f in os.listdir(folder) if f.lower().endswith(IMAGE_EXTENSIONS))


def load_rgb(path, size):
    return np.asarray(Image.open(path).convert("RGB").resize((size, size), Image.BICUBIC), dtype=np.float32) / 255


# ---------------------------------------------------------------- masks (boolean arrays, True = corrupted)
def stroke_mask(rng, size, target):
    """Free-form brush strokes, as in DeepFill v2, until at least `target` of the pixels are covered."""
    canvas = Image.new("L", (size, size), 0)
    draw = ImageDraw.Draw(canvas)
    mask = np.zeros((size, size), bool)
    for _ in range(200):
        if mask.mean() >= target:
            break
        x, y = rng.uniform(0, size, 2)
        width = rng.uniform(0.04, 0.12) * size
        for _ in range(rng.integers(4, 10)):
            angle = rng.uniform(0, 2 * math.pi)
            length = rng.uniform(0.05, 0.25) * size
            x2 = float(np.clip(x + length * math.cos(angle), 0, size - 1))
            y2 = float(np.clip(y + length * math.sin(angle), 0, size - 1))
            draw.line([(x, y), (x2, y2)], fill=255, width=int(width))
            draw.ellipse([x2 - width / 2, y2 - width / 2, x2 + width / 2, y2 + width / 2], fill=255)
            x, y = x2, y2
        mask = np.asarray(canvas) > 127
    return mask


def box_mask(rng, size, target):
    """Random rectangles (sides 10-50% of the image) until at least `target` of the pixels are covered."""
    mask = np.zeros((size, size), bool)
    for _ in range(200):
        if mask.mean() >= target:
            break
        h, w = (rng.uniform(0.1, 0.5, 2) * size).astype(int)
        y, x = rng.integers(0, size - h + 1), rng.integers(0, size - w + 1)
        mask[y:y + h, x:x + w] = True
    return mask


def center_mask(size):
    """Central square with half the image side (128 x 128 for 256 x 256 images): 25% of the pixels."""
    mask = np.zeros((size, size), bool)
    q = size // 4
    mask[q:size - q, q:size - q] = True
    return mask


def folder_mask(rng, size, path, invert, augment):
    mask = np.asarray(Image.open(path).convert("L").resize((size, size), Image.NEAREST)) > 127
    if invert:
        mask = ~mask
    if augment:   # training split: random rotations and flips give more masks per file
        mask = np.rot90(mask, int(rng.integers(4)))
        if rng.random() < 0.5:
            mask = mask[:, ::-1]
    return np.ascontiguousarray(mask)


def make_mask(rng, args, index, mask_files, augment):
    kind = args.mask
    if kind == "mixed":
        kind = str(rng.choice(["strokes", "boxes"]))
    if kind == "center":
        return center_mask(args.size)
    if kind == "folder":
        path = mask_files[int(rng.integers(len(mask_files)))] if augment else mask_files[index % len(mask_files)]
        return folder_mask(rng, args.size, path, args.mask_invert, augment)
    target = rng.uniform(*args.ratio)
    return stroke_mask(rng, args.size, target) if kind == "strokes" else box_mask(rng, args.size, target)


# ---------------------------------------------------------------- hole contents (float arrays H x W x 3 in [0, 1])
def make_fill(rng, kind, size, other_image):
    if kind == "mixed":
        kind = str(rng.choice(["white", "color", "noise", "patch"]))
    if kind == "white":
        return np.ones((size, size, 3), np.float32)
    if kind == "color":
        return np.broadcast_to(rng.uniform(0, 1, 3).astype(np.float32), (size, size, 3))
    if kind == "noise":
        base = rng.uniform(0.2, 0.8, 3)
        return np.clip(base + rng.normal(0, 0.25, (size, size, 3)), 0, 1).astype(np.float32)
    if kind == "patch":
        return other_image()
    raise ValueError("unknown fill " + kind)


# ---------------------------------------------------------------- main
def process_split(args, split, split_id, mask_files):
    src = os.path.join(args.src, split, "a")
    if not os.path.isdir(src):
        src = os.path.join(args.src, split)
    names = list_images(src)[:args.limit or None]
    augment = split == "train"
    out = {k: os.path.join(args.dst, split, k) for k in ("a", "b", "mask")}
    for folder in out.values():
        os.makedirs(folder, exist_ok=True)
    ratios, preview = [], []
    for i, name in enumerate(names):
        rng = np.random.default_rng([args.seed, split_id, i])
        clean = load_rgb(os.path.join(src, name), args.size)
        mask = make_mask(rng, args, i, mask_files, augment)

        def other_image():   # the same region of another image of this split
            j = int(rng.integers(len(names) - 1))
            return load_rgb(os.path.join(src, names[j + (j >= i)]), args.size)

        fill = make_fill(rng, args.fill, args.size, other_image)
        m = mask[..., None].astype(np.float32)
        corrupted = clean * (1 - m) + fill * m
        stem = os.path.splitext(name)[0] + ".png"
        Image.fromarray((clean * 255 + 0.5).astype(np.uint8)).save(os.path.join(out["a"], stem))
        Image.fromarray((corrupted * 255 + 0.5).astype(np.uint8)).save(os.path.join(out["b"], stem))
        Image.fromarray(mask.astype(np.uint8) * 255).save(os.path.join(out["mask"], stem))
        ratios.append(mask.mean())
        if len(preview) < 6:
            preview.append(np.concatenate([corrupted, np.repeat(m, 3, axis=2), clean], axis=1))
        if (i + 1) % 1000 == 0:
            print("  %s: %d / %d" % (split, i + 1, len(names)), flush=True)
    grid = np.concatenate(preview, axis=0)
    Image.fromarray((grid * 255 + 0.5).astype(np.uint8)).save(os.path.join(args.dst, "preview_%s.jpg" % split))
    hist = np.histogram(ratios, bins=np.linspace(0, 1, 11))[0]
    print("%s: %d images, corrupted fraction %.1f%% on average" % (split, len(names), 100 * np.mean(ratios)))
    print("  images per 10%% bin (0-10%%, 10-20%%, ...): %s" % " ".join(str(h) for h in hist))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--src", required=True, help="clean images: <src>/<split>/a or <src>/<split>")
    p.add_argument("--dst", required=True, help="output root, used as opt.root_path in the notebook")
    p.add_argument("--splits", nargs="+", default=["train", "test"])
    p.add_argument("--size", type=int, default=256)
    p.add_argument("--mask", default="strokes", choices=["strokes", "boxes", "center", "mixed", "folder"])
    p.add_argument("--mask-dir", default="", help="--mask folder: mask images, white = hole")
    p.add_argument("--mask-invert", action="store_true", help="--mask folder: the holes are black")
    p.add_argument("--ratio", type=float, nargs=2, default=[0.1, 0.6], help="corrupted fraction range")
    p.add_argument("--fill", default="mixed", choices=["white", "color", "noise", "patch", "mixed"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--limit", type=int, default=0, help="> 0: only the first N images of each split (tests)")
    args = p.parse_args()
    mask_files = []
    if args.mask == "folder":
        mask_files = [os.path.join(args.mask_dir, f) for f in list_images(args.mask_dir)]
        if not mask_files:
            raise SystemExit("no mask images found in --mask-dir " + args.mask_dir)
    for split_id, split in enumerate(args.splits):
        process_split(args, split, split_id, mask_files)
    print("done: set opt.root_path = %r in the notebook" % args.dst)


if __name__ == "__main__":
    main()
