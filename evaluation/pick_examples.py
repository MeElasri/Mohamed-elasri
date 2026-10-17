"""Pick qualitative examples that are representative for every method.

For each image, every method's score is turned into a z-score against that
method's own test-set mean and std. Images are ranked by the largest
absolute z-score, so the top picks are images where no method looks much
better or worse than its own average.

Usage:
    python -m evaluation.pick_examples \
        --csv WavTRGAN=eval/wavtrgan_seed0/per_image.csv \
        --csv CAML=eval/caml/per_image.csv --metric psnr_full --bin 30-40%
"""

import argparse
import csv

import numpy as np


def read_scores(path, metric, bin_label=None):
    with open(path, newline="") as f:
        return {r["image"]: float(r[metric]) for r in csv.DictReader(f)
                if bin_label is None or r["bin"] == bin_label}


def rank_images(scores_by_method):
    """Return [(image, max_abs_z, {method: score})] sorted from most representative."""
    common = set.intersection(*(set(s) for s in scores_by_method.values()))
    if not common:
        raise ValueError("no image is shared by all methods")
    stats = {}
    for method, scores in scores_by_method.items():
        values = np.array([scores[i] for i in common])
        stats[method] = (values.mean(), values.std() or 1.0)

    ranked = []
    for image in common:
        per_method = {m: s[image] for m, s in scores_by_method.items()}
        z = max(abs(per_method[m] - stats[m][0]) / stats[m][1] for m in per_method)
        ranked.append((image, float(z), per_method))
    ranked.sort(key=lambda r: (r[1], r[0]))
    return ranked, {m: s[0] for m, s in stats.items()}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--csv", action="append", required=True, metavar="NAME=PATH",
                        help="per_image.csv of one method; repeat per method")
    parser.add_argument("--metric", default="psnr_full")
    parser.add_argument("--bin", help="only consider images in this mask-ratio bin, e.g. 30-40%%")
    parser.add_argument("--top", type=int, default=5)
    args = parser.parse_args(argv)

    scores = {}
    for item in args.csv:
        name, path = item.split("=", 1)
        scores[name] = read_scores(path, args.metric, args.bin)

    ranked, means = rank_images(scores)
    methods = list(scores)
    print("mean      " + "  ".join(f"{m}={means[m]:.3f}" for m in methods))
    for image, z, values in ranked[:args.top]:
        print(f"{image:10s} max|z|={z:.2f}  " + "  ".join(f"{m}={values[m]:.3f}" for m in methods))


if __name__ == "__main__":
    main()
