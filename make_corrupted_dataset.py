"""Create corrupted versions of CelebA-HQ (or any image folder) for (blind) inpainting experiments.

For every clean image of <src>/<split>/a (or <src>/<split> if there is no "a" folder), it writes
    <dst>/<split>/a/<name>.png      the clean image, resized to --size
    <dst>/<split>/b/<name>.png      the corrupted image (the input of the network)
    <dst>/<split>/mask/<name>.png   the exact corrupted region (white = corrupted)
which is the folder layout of wavtr-flow-matching.ipynb: set opt.root_path = <dst>.

Mask shapes (--mask)
    strokes  free-form brush strokes, as in DeepFill v2
    boxes    random rectangles
    center   the central square with half the image side (the classic 128 x 128 hole at 256 x 256)
    mixed    strokes or boxes
    folder   mask images from --mask-dir, e.g. the irregular mask test set of Liu et al. (partial convolutions,
             https://nv-adlr.github.io/publication/partialconv-inpainting), the standard masks of inpainting
             papers. Whether the holes are white or black is detected automatically (--mask-invert overrides it).
             With --ratio, only the masks whose holes cover that fraction of the image are used, e.g.
             --ratio 0.3 0.4 for the 30-40% category of the NVIDIA test set.
Hole contents (--fill)
    white, color (one random color), noise (a random color plus Gaussian noise),
    patch (a natural image: a random crop of an image of --patch-dir if given, e.g. Places2 or ImageNet images
           as in VCNet, otherwise the same region of another image of the split), mixed (one of the four)
Blending: --alpha sets the opacity range of the fill (e.g. --alpha 0.6 1 leaves the face partly visible, as in the
contaminated images of VCNet and Phutke et al.), --feather softens the mask border (Gaussian sigma in pixels).
Every image has its own random generator seeded by (--seed, split, index), so the output is reproducible.

Examples (Kaggle):
    # strokes with mixed fills
    !python make_corrupted_dataset.py --src /kaggle/input/celeba-hq-img-full-50/CelebA-HQ-img \
        --dst /kaggle/working/celebahq_mixed --mask strokes --fill mixed
    # NVIDIA irregular test masks, 30-40% category, test images only
    !python make_corrupted_dataset.py --src /kaggle/input/celeba-hq-img-full-50/CelebA-HQ-img \
        --dst /kaggle/working/celebahq_nvidia_30_40 --splits test --mask folder \
        --mask-dir /kaggle/input/<nvidia-masks>/testing_mask_dataset --ratio 0.3 0.4 --fill white
"""
import argparse
import math
import os

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg")
DEFAULT_RATIO = (0.1, 0.6)   # corrupted fraction of strokes and boxes when --ratio is not given


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


def read_mask_file(path):
    return np.asarray(Image.open(path).convert("L")) > 127


def mask_folder(args):
    """Mask files of --mask-dir (holes white or black), optionally restricted to a hole-size range."""
    files = [os.path.join(args.mask_dir, f) for f in list_images(args.mask_dir)]
    if not files:
        raise SystemExit("no mask images found in --mask-dir " + args.mask_dir)
    white = np.array([read_mask_file(f).mean() for f in files])   # hole size measured at the original resolution
    invert = args.mask_invert == "yes" or (args.mask_invert == "auto" and np.median(white) > 0.5)
    holes = 1 - white if invert else white
    print("mask folder: %d masks, holes are %s (%s), median hole %.0f%% of the image" % (
        len(files), "black" if invert else "white", "detected" if args.mask_invert == "auto" else "given",
        100 * np.median(holes)))
    if args.ratio:
        keep = (holes > args.ratio[0]) & (holes <= args.ratio[1])
        files = [f for f, k in zip(files, keep) if k]
        print("  %d masks with holes of %.0f-%.0f%% of the image" % (len(files), 100 * args.ratio[0], 100 * args.ratio[1]))
        if not files:
            raise SystemExit("no mask in this --ratio range")
    return files, invert


def folder_mask(rng, size, path, invert, augment):
    mask = np.asarray(Image.open(path).convert("L").resize((size, size), Image.NEAREST)) > 127
    if invert:
        mask = ~mask
    if augment:   # training split: random rotations and flips give more masks per file
        mask = np.rot90(mask, int(rng.integers(4)))
        if rng.random() < 0.5:
            mask = mask[:, ::-1]
    return np.ascontiguousarray(mask)


def make_mask(rng, args, index, mask_files, invert, augment):
    kind = args.mask
    if kind == "mixed":
        kind = str(rng.choice(["strokes", "boxes"]))
    if kind == "center":
        return center_mask(args.size)
    if kind == "folder":
        path = mask_files[int(rng.integers(len(mask_files)))] if augment else mask_files[index % len(mask_files)]
        return folder_mask(rng, args.size, path, invert, augment)
    target = rng.uniform(*(args.ratio or DEFAULT_RATIO))
    return stroke_mask(rng, args.size, target) if kind == "strokes" else box_mask(rng, args.size, target)


# ---------------------------------------------------------------- hole contents (float arrays H x W x 3 in [0, 1])
def natural_patch(rng, size, path):
    """A random square crop (30-100% of the shorter side) of a natural image, resized to the image size."""
    img = Image.open(path).convert("RGB")
    w, h = img.size
    c = max(1, int(min(w, h) * rng.uniform(0.3, 1.0)))
    x, y = int(rng.integers(0, w - c + 1)), int(rng.integers(0, h - c + 1))
    return np.asarray(img.crop((x, y, x + c, y + c)).resize((size, size), Image.BICUBIC), dtype=np.float32) / 255


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
def process_split(args, split, split_id, mask_files, invert, patch_files):
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
        mask = make_mask(rng, args, i, mask_files, invert, augment)

        def other_image():
            if patch_files:   # a natural image of --patch-dir
                return natural_patch(rng, args.size, patch_files[int(rng.integers(len(patch_files)))])
            j = int(rng.integers(len(names) - 1))   # the same region of another image of this split
            return load_rgb(os.path.join(src, names[j + (j >= i)]), args.size)

        fill = make_fill(rng, args.fill, args.size, other_image)
        weight = mask.astype(np.float32)
        if args.feather > 0:
            blurred = Image.fromarray((weight * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(args.feather))
            weight = np.asarray(blurred, dtype=np.float32) / 255
        alpha = rng.uniform(*args.alpha) if args.alpha[0] < args.alpha[1] else args.alpha[0]
        weight = alpha * weight
        corrupted = clean * (1 - weight[..., None]) + fill * weight[..., None]
        mask = weight > 0.02   # every pixel that the corruption changes noticeably
        stem = os.path.splitext(name)[0] + ".png"
        Image.fromarray((clean * 255 + 0.5).astype(np.uint8)).save(os.path.join(out["a"], stem))
        Image.fromarray((corrupted * 255 + 0.5).astype(np.uint8)).save(os.path.join(out["b"], stem))
        Image.fromarray(mask.astype(np.uint8) * 255).save(os.path.join(out["mask"], stem))
        ratios.append(mask.mean())
        if len(preview) < 6:
            preview.append(np.concatenate([corrupted, np.repeat(mask[..., None], 3, axis=2), clean], axis=1))
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
    p.add_argument("--mask-dir", default="", help="--mask folder: folder of mask images")
    p.add_argument("--mask-invert", default="auto", choices=["auto", "yes", "no"],
                   help="--mask folder: are the holes black? auto: yes if most pixels of most masks are white")
    p.add_argument("--ratio", type=float, nargs=2, default=None,
                   help="strokes/boxes: range of the corrupted fraction (default 0.1 0.6); "
                        "folder: use only masks whose holes cover this fraction")
    p.add_argument("--fill", default="mixed", choices=["white", "color", "noise", "patch", "mixed"])
    p.add_argument("--patch-dir", default="", help="patch fill: folder of natural images to take the patches from")
    p.add_argument("--alpha", type=float, nargs=2, default=[1.0, 1.0], help="opacity range of the fill")
    p.add_argument("--feather", type=float, default=0.0, help="Gaussian blur of the mask border, in pixels")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--limit", type=int, default=0, help="> 0: only the first N images of each split (tests)")
    args = p.parse_args()
    mask_files, invert = mask_folder(args) if args.mask == "folder" else ([], False)
    patch_files = [os.path.join(args.patch_dir, f) for f in list_images(args.patch_dir)] if args.patch_dir else []
    if args.patch_dir and not patch_files:
        raise SystemExit("no images found in --patch-dir " + args.patch_dir)
    for split_id, split in enumerate(args.splits):
        process_split(args, split, split_id, mask_files, invert, patch_files)
    print("done: set opt.root_path = %r in the notebook" % args.dst)


if __name__ == "__main__":
    main()
