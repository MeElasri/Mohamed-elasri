"""Build comparison tables (mean +/- std over seeds) from summary.json files.

Best and second-best are marked from the values as printed (after
rounding), so the marks always agree with what the reader sees. Ties share
a mark.

Usage:
    python -m evaluation.aggregate \
        --method WavTRGAN eval/wavtrgan_seed{0,1,2}/summary.json \
        --method T-former eval/tformer/summary.json \
        --metrics psnr_full ssim_full fid_full psnr_hole ssim_hole \
        --out tables/table2
"""

import argparse
import json
import os
import sys

import numpy as np

LOWER_IS_BETTER = ("l1", "fid", "lpips")
DECIMALS = {"psnr": 2, "ssim": 3, "l1": 4, "fid": 2, "lpips": 3}
NAMES = {"psnr": "PSNR", "ssim": "SSIM", "l1": "L1", "fid": "FID", "lpips": "LPIPS"}
SUFFIXES = {"_full_comp": " (comp.)", "_full": "", "_hole": " (hole)"}


def metric_base(metric):
    return metric.split("_")[0]


def higher_is_better(metric):
    return metric_base(metric) not in LOWER_IS_BETTER


def display_name(metric):
    name = NAMES.get(metric_base(metric), metric)
    for suffix, label in SUFFIXES.items():
        if metric.endswith(suffix):
            return name + label
    return name


def load_values(paths, metric, scope):
    values = []
    for path in paths:
        with open(path) as f:
            summary = json.load(f)
        block = summary["overall"] if scope == "overall" else summary["bins"].get(scope, {})
        if metric in block:
            values.append(block[metric])
    if 0 < len(values) < len(paths):
        print(f"warning: {metric} found in only {len(values)} of {len(paths)} runs: {paths}", file=sys.stderr)
    return values


def summarize(values, decimals):
    """Rounded mean and sample std (None for a single run)."""
    if not values:
        return None, None
    mean = round(float(np.mean(values)), decimals)
    std = round(float(np.std(values, ddof=1)), decimals) if len(values) > 1 else None
    return mean, std


def rank_marks(means, higher):
    """Map each method to 'best', 'second' or None from its rounded mean."""
    distinct = sorted({m for m in means.values() if m is not None}, reverse=higher)
    marks = {}
    for method, mean in means.items():
        if mean is not None and distinct and mean == distinct[0]:
            marks[method] = "best"
        elif mean is not None and len(distinct) > 1 and mean == distinct[1]:
            marks[method] = "second"
        else:
            marks[method] = None
    return marks


def build_table(methods, metrics, scope):
    """Return {method: {metric: (mean, std, mark)}}."""
    table = {name: {} for name, _ in methods}
    for metric in metrics:
        decimals = DECIMALS.get(metric_base(metric), 3)
        stats = {name: summarize(load_values(paths, metric, scope), decimals) for name, paths in methods}
        marks = rank_marks({n: s[0] for n, s in stats.items()}, higher_is_better(metric))
        for name, (mean, std) in stats.items():
            table[name][metric] = (mean, std, marks[name])
    return table


def format_cell(mean, std, mark, metric, style):
    if mean is None:
        return "--" if style == "latex" else "–"
    decimals = DECIMALS.get(metric_base(metric), 3)
    text = f"{mean:.{decimals}f}"
    if std is not None:
        text += (r"$\pm$" if style == "latex" else " ± ") + f"{std:.{decimals}f}"
    if mark == "best":
        return rf"\textbf{{{text}}}" if style == "latex" else f"**{text}**"
    if mark == "second":
        return rf"\underline{{{text}}}" if style == "latex" else f"<u>{text}</u>"
    return text


def to_latex(table, metrics):
    arrows = [r"$\uparrow$" if higher_is_better(m) else r"$\downarrow$" for m in metrics]
    lines = [
        r"\begin{tabular}{l" + "c" * len(metrics) + "}",
        r"\toprule",
        "Method & " + " & ".join(f"{display_name(m)}{a}" for m, a in zip(metrics, arrows)) + r" \\",
        r"\midrule",
    ]
    for method, cells in table.items():
        row = [format_cell(*cells[m], m, "latex") for m in metrics]
        lines.append(f"{method} & " + " & ".join(row) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


def to_markdown(table, metrics):
    arrows = ["↑" if higher_is_better(m) else "↓" for m in metrics]
    lines = [
        "| Method | " + " | ".join(f"{display_name(m)} {a}" for m, a in zip(metrics, arrows)) + " |",
        "|---|" + "---|" * len(metrics),
    ]
    for method, cells in table.items():
        lines.append(f"| {method} | " + " | ".join(format_cell(*cells[m], m, "markdown") for m in metrics) + " |")
    return "\n".join(lines) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--method", nargs="+", action="append", required=True, metavar=("NAME", "SUMMARY"),
                        help="method name followed by one summary.json per seed; repeat per method")
    parser.add_argument("--metrics", nargs="+",
                        default=["psnr_full", "ssim_full", "fid_full", "psnr_hole", "ssim_hole"])
    parser.add_argument("--scope", default="overall", help="'overall' or a bin label such as 30-40%%")
    parser.add_argument("--out", help="write <out>.tex and <out>.md (default: print markdown)")
    args = parser.parse_args(argv)

    methods = []
    for entry in args.method:
        if len(entry) < 2:
            parser.error(f"--method {entry[0]} needs at least one summary.json")
        methods.append((entry[0], entry[1:]))

    table = build_table(methods, args.metrics, args.scope)
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out + ".tex", "w") as f:
            f.write(to_latex(table, args.metrics))
        with open(args.out + ".md", "w") as f:
            f.write(to_markdown(table, args.metrics))
        print(f"wrote {args.out}.tex and {args.out}.md")
    else:
        print(to_markdown(table, args.metrics))


if __name__ == "__main__":
    main()
