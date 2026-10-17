import csv
import json
import os

import numpy as np
import pytest
from PIL import Image

from evaluation import aggregate, evaluate, masks, pick_examples
from evaluation.metrics import MAX_PSNR, composite, image_metrics, psnr, ssim

SIZE = 64


def random_image(rng):
    return rng.random((SIZE, SIZE, 3))


def square_mask():
    mask = np.zeros((SIZE, SIZE), dtype=bool)
    mask[16:48, 16:48] = True
    return mask


def save_rgb(path, img):
    Image.fromarray((np.clip(img, 0, 1) * 255 + 0.5).astype(np.uint8)).save(path)


# --- masks -----------------------------------------------------------------

@pytest.mark.parametrize("ratio_range", masks.DEFAULT_BINS)
def test_mask_ratio_in_range(ratio_range):
    mask = masks.generate_mask(SIZE, ratio_range, masks.mask_rng(0, 0))
    ratio = np.count_nonzero(mask) / mask.size
    assert ratio_range[0] <= ratio <= ratio_range[1]


def test_masks_are_deterministic_per_index():
    a = masks.generate_mask(SIZE, (0.2, 0.3), masks.mask_rng(7, 3))
    b = masks.generate_mask(SIZE, (0.2, 0.3), masks.mask_rng(7, 3))
    c = masks.generate_mask(SIZE, (0.2, 0.3), masks.mask_rng(7, 4))
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)


def test_corrupt_only_touches_hole():
    rng = np.random.default_rng(0)
    gt = rng.integers(0, 256, (SIZE, SIZE, 3), dtype=np.uint8)
    donor = rng.integers(0, 256, (SIZE, SIZE, 3), dtype=np.uint8)
    mask = square_mask().astype(np.uint8) * 255
    out = masks.corrupt(gt, mask, "image", rng, donor)
    hole = mask > 0
    assert np.array_equal(out[~hole], gt[~hole])
    assert np.array_equal(out[hole], donor[hole])


# --- metrics ---------------------------------------------------------------

def test_identical_images():
    gt = random_image(np.random.default_rng(0))
    assert psnr(gt, gt) == MAX_PSNR
    assert ssim(gt, gt) == pytest.approx(1.0)


def test_hole_psnr_known_value():
    gt = np.full((SIZE, SIZE, 3), 0.5)
    mask = square_mask()
    pred = gt.copy()
    pred[mask] += 0.1  # MSE inside the hole is 0.01 -> 20 dB
    assert psnr(pred, gt, mask) == pytest.approx(20.0)
    assert psnr(pred, gt) > 20.0  # full image dilutes the error with perfect pixels


def test_hole_metrics_ignore_errors_outside_hole():
    rng = np.random.default_rng(1)
    gt = random_image(rng)
    mask = square_mask()
    pred = np.where(mask[..., None], gt, rng.random(gt.shape))
    m = image_metrics(pred, gt, mask)
    assert m["psnr_hole"] == MAX_PSNR
    assert m["l1_hole"] == 0.0
    assert m["psnr_full"] < 20.0
    # Compositing restores the outside, so the composited image is perfect.
    assert m["psnr_full_comp"] == MAX_PSNR


def test_hole_metrics_are_invariant_to_compositing():
    rng = np.random.default_rng(2)
    gt, pred = random_image(rng), random_image(rng)
    mask = square_mask()
    comp = composite(pred, gt, mask)
    assert psnr(comp, gt, mask) == psnr(pred, gt, mask)


# --- evaluate (end to end) -------------------------------------------------

@pytest.fixture
def dataset(tmp_path):
    rng = np.random.default_rng(3)
    gt_dir = tmp_path / "gt"
    gt_dir.mkdir()
    for i in range(6):
        save_rgb(gt_dir / f"img{i}.png", random_image(rng))
    masks.main(["--gt", str(gt_dir), "--out", str(tmp_path / "masks"), "--size", str(SIZE),
                "--fill", "noise", "--corrupted-out", str(tmp_path / "input")])
    return tmp_path


def run_evaluate(dataset, pred_dir, *extra):
    out = dataset / ("eval_" + os.path.basename(pred_dir))
    evaluate.main(["--method", "m", "--gt", str(dataset / "gt"), "--pred", str(pred_dir),
                   "--masks", str(dataset / "masks"), "--input", str(dataset / "input"),
                   "--out", str(out), *extra])
    with open(out / "summary.json") as f:
        return json.load(f), out


def test_evaluate_perfect_and_identity_outputs(dataset):
    perfect, _ = run_evaluate(dataset, dataset / "gt")
    assert perfect["n_images"] == 6
    assert perfect["overall"]["psnr_full"] == MAX_PSNR
    assert perfect["overall"]["psnr_hole"] == MAX_PSNR

    # Returning the corrupted input unchanged: perfect outside, noise inside.
    identity, out = run_evaluate(dataset, dataset / "input")
    o = identity["overall"]
    assert o["psnr_hole"] < o["psnr_full"] < MAX_PSNR
    assert o["psnr_full"] == o["psnr_full_comp"]
    assert sum(b["n"] for b in identity["bins"].values()) == 6
    with open(out / "per_image.csv") as f:
        assert len(list(csv.DictReader(f))) == 6


def test_evaluate_rejects_size_mismatch_unless_resized(dataset):
    small = dataset / "small"
    small.mkdir()
    for f in os.listdir(dataset / "gt"):
        Image.open(dataset / "gt" / f).resize((SIZE // 2, SIZE // 2)).save(small / f)
    with pytest.raises(SystemExit):
        run_evaluate(dataset, small)
    summary, _ = run_evaluate(dataset, small, "--resize-pred")
    assert summary["protocol"]["resized_outputs"] == 6


def test_evaluate_rejects_empty_mask(dataset):
    Image.new("L", (SIZE, SIZE), 0).save(dataset / "masks" / "img0.png")
    with pytest.raises(SystemExit, match="no hole pixels"):
        run_evaluate(dataset, dataset / "gt")


def test_evaluate_reports_missing_outputs(dataset):
    partial = dataset / "partial"
    partial.mkdir()
    save_rgb(partial / "img0.png", np.zeros((SIZE, SIZE, 3)))
    with pytest.raises(SystemExit, match="lack an output"):
        run_evaluate(dataset, partial)


# --- aggregate -------------------------------------------------------------

def write_summary(path, overall):
    with open(path, "w") as f:
        json.dump({"overall": overall, "bins": {}}, f)
    return str(path)


def test_marks_follow_rounded_values_and_direction(tmp_path):
    # Mirrors the Table 2 issue: T-former must get the SSIM and FID marks.
    ours = [write_summary(tmp_path / f"ours{s}.json", {"ssim_full": v, "fid_full": f})
            for s, (v, f) in enumerate([(0.9128, 5.60), (0.9131, 5.70), (0.9130, 5.65)])]
    tformer = [write_summary(tmp_path / "tformer.json", {"ssim_full": 0.9150, "fid_full": 5.42})]
    other = [write_summary(tmp_path / "other.json", {"ssim_full": 0.9000, "fid_full": 7.00})]
    table = aggregate.build_table([("Ours", ours), ("T-former", tformer), ("Other", other)],
                                  ["ssim_full", "fid_full"], "overall")
    assert table["T-former"]["ssim_full"] == (0.915, None, "best")
    assert table["Ours"]["ssim_full"][2] == "second"
    assert table["T-former"]["fid_full"][2] == "best"
    assert table["Ours"]["fid_full"][2] == "second"
    assert table["Other"]["fid_full"][2] is None
    assert table["Ours"]["fid_full"][0] == 5.65
    assert table["Ours"]["fid_full"][1] == pytest.approx(0.05)


def test_ties_share_the_mark():
    marks = aggregate.rank_marks({"a": 31.2, "b": 31.2, "c": 30.0}, higher=True)
    assert marks == {"a": "best", "b": "best", "c": "second"}


def test_latex_output(tmp_path):
    a = [write_summary(tmp_path / "a.json", {"psnr_full": 31.0}), write_summary(tmp_path / "a2.json", {"psnr_full": 31.2})]
    b = [write_summary(tmp_path / "b.json", {"psnr_full": 30.0})]
    out = str(tmp_path / "table")
    aggregate.main(["--method", "A", *a, "--method", "B", *b, "--metrics", "psnr_full", "fid_full", "--out", out])
    tex = open(out + ".tex").read()
    assert r"\textbf{31.10$\pm$0.14}" in tex
    assert r"\underline{30.00}" in tex
    assert "--" in tex  # missing FID


# --- pick_examples ---------------------------------------------------------

def test_pick_examples_prefers_images_near_every_mean():
    scores = {
        "A": {"x": 30.0, "y": 31.0, "z": 32.0},
        "B": {"x": 20.0, "y": 25.0, "z": 30.0},
    }
    ranked, means = pick_examples.rank_images(scores)
    assert ranked[0][0] == "y"
    assert means["B"] == pytest.approx(25.0)
