"""
Run every method that supports samples/synthetic_2023 -- every Table 2 row
except Xiong -- through eval/run_config.py's own staging and run path (import
and reuse eval.run_config.run_one; no staging logic is duplicated here), score
each with eval/score.py exactly as Table 2 is scored, and compare against
prediction_samples/expected_synthetic.csv.

These are a CONSISTENCY CHECK, not a measure of performance: the set's
irradiance is invented as smooth curves, so scores here say nothing about how
the methods compare on real data and are not the paper's ranking. What this
checks is that every method still runs correctly and reproduces its own past
result on your machine.

Three-level result per row: EXACT (bit-identical to expected_synthetic.csv),
CLOSE (within the row's family tolerance -- different hardware, CUDA/PyTorch/
cuDNN versions and TF32/kernel choices move floating-point results slightly,
and this set's 50 diffusion steps can amplify that) or FAIL. EXACT and CLOSE
both mean the method runs correctly; only FAIL is an error. SkyGPT never
reports EXACT (its sampler is unseeded, see PORTING_NOTES.md) -- CLOSE or FAIL
only. See samples/synthetic_2023/README.md for the tolerances, by family.

Xiong is not supported: its forecast() fits an observation-noise term on the
validation split (see PORTING_NOTES.md), which this set leaves empty by
design.

Usage:
  python eval/check_synthetic.py                       # run + score + compare
  python eval/check_synthetic.py --rows DLinearIclr,SunSet
  python eval/check_synthetic.py --score-only           # score existing outputs/, no re-run
  python eval/check_synthetic.py --write-expected       # (re)write expected_synthetic.csv from
                                                          # this run's own scores, instead of comparing.
                                                          # Combined with --rows, only the named rows are
                                                          # touched (central values + check; an existing
                                                          # tolerance is preserved) -- every other row, and
                                                          # every tolerance column, is copied through
                                                          # unchanged.
"""
import argparse
import csv
import glob
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from main import process
from eval.reproduce_comparison import SHORT_TO_ROW, _find_existing_npz
from eval.run_config import run_one
from eval.score import _reconstruct_spm

CONFIG_DIR = "configs/samples/synthetic"
EXPECTED_CSV = "prediction_samples/expected_synthetic.csv"
CELLS = ("rmse", "fs", "crps", "crpss")
CSV_FIELDS = ["method", "rmse", "fs", "crps", "crpss", "check",
              "rmse_tol", "fs_tol", "crps_tol", "crpss_tol"]

# Method family -> tolerance. Measured by running every method (except SkyGPT, whose tolerance instead
# comes from four independent runs of SkyGPT itself, since its sampler is unseeded) once on the default GPU
# settings and once on GPU with torch.backends.cuda.matmul.allow_tf32/cudnn.allow_tf32 off and
# cudnn.benchmark off/deterministic on -- no source edits, just those four global flags set before calling
# eval.run_config.run_one. CPU could not be measured: loading the released (GPU-trained) checkpoints fails
# on a CPU-only machine (main.py's torch.load calls have no map_location; see PORTING_NOTES.md), so no
# method can complete a CPU run yet. Tolerance = twice the largest deviation seen, rounded up to a clean
# number, with a floor (0.01 W/m² on RMSE/CRPS, 1e-4 on FS/CRPSS) so a family that showed ~zero deviation
# still gets a usable, non-zero tolerance -- then, on top of that, effective_tolerance() below adds a 0.5%
# relative floor so a correct run on genuinely different hardware still passes. "proposed" covers both of
# the proposed model's arms (propFuture, propPast) as one family.
FAMILY_TOLERANCE = {
    "scalar":   {"rmse": 0.01, "fs": 0.0001, "crps": 0.01, "crpss": 0.0001},
    "sunset":   {"rmse": 0.01, "fs": 0.0001, "crps": 0.01, "crpss": 0.0001},
    "video":    {"rmse": 0.1,  "fs": 0.001,  "crps": 0.1,  "crpss": 0.002},
    "proposed": {"rmse": 0.08, "fs": 0.001,  "crps": 0.01, "crpss": 0.0001},
    "skygpt":   {"rmse": 16.0, "fs": 0.15,   "crps": 16.0, "crpss": 0.27},
}
FAMILY = {
    "DLinearIclr": "scalar", "DLinearNoIclr": "scalar",
    "PatchTSTIclr": "scalar", "PatchTSTNoIclr": "scalar",
    "TimeXerIclr": "scalar", "TimeXerNoIclr": "scalar",
    "SunSet": "sunset", "SunSetNoIclr": "sunset",
    "ConvLSTM": "video", "PhyDNet": "video", "PhyDNetGAN": "video",
    "propFuture": "proposed", "propPast": "proposed",
    "SkyGPT": "skygpt",
}
# SPM is a fixed formula (k_origin * clipped Iclr) in plain float64 numpy, not a model -- unaffected by
# GPU numerics, always expected EXACT. The tolerance is only a floor for schema completeness.
SPM_TOLERANCE = {"rmse": 0.01, "crps": 0.01}


def _configs(wanted=None):
    paths = sorted(glob.glob(os.path.join(CONFIG_DIR, "*.yaml")))
    out = []
    for path in paths:
        name = os.path.splitext(os.path.basename(path))[0]
        if wanted is not None and name not in wanted:
            continue
        if name not in SHORT_TO_ROW:
            raise SystemExit("%s does not match any known row (see eval/reproduce_comparison.py "
                              "ROWS); rename it to a known short name." % path)
        out.append((name, path))
    return out


def score_row(row, npz_path):
    scored = row["score"]({"npz": npz_path, **row["kwargs"]})
    # float(): eval/score.py's crpss_convB is a bare numpy float64, unlike its
    # other outputs -- normalise to plain float so every cell here is one.
    return {
        "rmse": scored["rmse"],
        "fs": scored.get("fs"),
        "crps": scored.get("crps"),
        "crpss": (float(scored["crpss_convB"]) if "crpss_convB" in scored else None),
    }, scored


def _fmt(v):
    # repr(), not a fixed-precision %f -- exact rows are compared bit for bit
    # against a re-parsed value, so the CSV must round-trip full float64
    # precision (same convention as prediction_samples/expected_table2.csv).
    return "" if v is None else repr(float(v))


def _row_dict(name, cells, tol, check):
    return {
        "method": name,
        "rmse": _fmt(cells.get("rmse")), "fs": _fmt(cells.get("fs")),
        "crps": _fmt(cells.get("crps")), "crpss": _fmt(cells.get("crpss")),
        "check": check,
        "rmse_tol": _fmt(tol.get("rmse")), "fs_tol": _fmt(tol.get("fs")),
        "crps_tol": _fmt(tol.get("crps")), "crpss_tol": _fmt(tol.get("crpss")),
    }


def write_expected(rows_scored, spm):
    """Write/merge prediction_samples/expected_synthetic.csv. rows_scored is
    this invocation's own [(name, cells)] -- every row, unless --rows narrowed
    it. A row this invocation did not run (only possible with --rows) is
    copied verbatim from the file already on disk, so a --rows-scoped write
    only ever touches the rows it named; everything else -- every other row,
    and every tolerance column, including the named rows' own -- is left
    exactly as it was, unless a named row is brand new, in which case its
    family's FAMILY_TOLERANCE default is used."""
    existing = {}
    if os.path.exists(EXPECTED_CSV):
        with open(EXPECTED_CSV) as f:
            existing = {r["method"]: r for r in csv.DictReader(f)}

    scored_by_name = dict(rows_scored)
    all_names = sorted(set(existing) | set(scored_by_name) | {"SPM"})

    out = []
    for name in all_names:
        if name == "SPM":
            out.append(_row_dict("SPM", spm, SPM_TOLERANCE, "exact"))
        elif name in scored_by_name:
            cells = scored_by_name[name]
            old = existing.get(name)
            check = "tolerance" if name == "SkyGPT" else "exact"
            tol = dict(FAMILY_TOLERANCE[FAMILY[name]])
            if old:
                check = old.get("check") or check
                for cell in CELLS:
                    old_v = old.get(cell + "_tol", "")
                    if old_v not in (None, ""):
                        tol[cell] = float(old_v)
            out.append(_row_dict(name, cells, tol, check))
        else:
            out.append(existing[name])  # not run this invocation -- copied through unchanged

    os.makedirs(os.path.dirname(EXPECTED_CSV), exist_ok=True)
    with open(EXPECTED_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        w.writeheader()
        w.writerows(out)
    print("wrote %s (%d row(s) from this run, %d row(s) carried over unchanged)"
          % (EXPECTED_CSV, len(scored_by_name) + 1, len(out) - len(scored_by_name) - 1))


def load_expected():
    with open(EXPECTED_CSV) as f:
        return {row["method"]: row for row in csv.DictReader(f)}


RELATIVE_FLOOR = 0.005
# fs/crpss are dimensionless skill scores that can sit near zero, where 0.5% of the expected value is not
# a usable tolerance -- an absolute floor under it, same as RMSE/CRPS's family tolerance already gives them.
ABS_FLOOR = {"fs": 0.001, "crpss": 0.001}


def effective_tolerance(method, cell, tol_raw, want):
    """max(the measured/family tolerance already in the CSV, 0.5% of the expected value, and -- for fs/crpss
    only -- 0.001) so a correct run on different hardware still passes even where the measured tolerance
    (from one GPU's own numerics) undershoots. SPM is exempt: a fixed formula (k_origin * clipped Iclr) in
    plain float64 numpy, not a model, unaffected by GPU numerics -- always expected exact, never given a
    tolerance floor."""
    if method == "SPM":
        return 0.0
    measured = float(tol_raw) if tol_raw else 0.0
    return max(measured, RELATIVE_FLOOR * abs(want), ABS_FLOOR.get(cell, 0.0))


def compare(cells, expected_row):
    """(level, details): 'EXACT' if every cell matches bit for bit, 'CLOSE' if
    every cell is exact or within its effective tolerance (effective_tolerance,
    read from the CSV's own *_tol column and central value), 'FAIL' if any cell
    exceeds it. A row whose check=='tolerance' (SkyGPT) never returns EXACT --
    its sampler is unseeded, so one run landing on the recorded value by
    chance is not a guarantee the next one will too."""
    allow_exact = expected_row["check"] != "tolerance"
    any_close = False
    any_fail = False
    details = []
    for cell in CELLS:
        exp_raw = expected_row[cell]
        if exp_raw == "":
            continue
        want = float(exp_raw)
        got = cells[cell]
        tol = effective_tolerance(expected_row["method"], cell, expected_row.get(cell + "_tol", ""), want)
        if allow_exact and got == want:
            continue
        if abs(got - want) <= tol:
            any_close = True
            continue
        any_fail = True
        details.append("%s: got %.6f want %.6f tol %.6g" % (cell, got, want, tol))
    if any_fail:
        return "FAIL", details
    if any_close:
        return "CLOSE", details
    return "EXACT", details


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rows", default=None, help="comma-separated config names to run (default: all)")
    ap.add_argument("--score-only", action="store_true",
                     help="score existing outputs/ without re-running inference")
    ap.add_argument("--write-expected", action="store_true",
                     help="write/merge prediction_samples/expected_synthetic.csv from this run's scores "
                          "instead of comparing against it -- see module docstring for --rows behaviour")
    args = ap.parse_args()

    wanted = set(args.rows.split(",")) if args.rows else None
    configs = _configs(wanted)
    if not configs:
        raise SystemExit("no matching configs under %s" % CONFIG_DIR)

    rows_scored = []
    npz_paths = {}
    for name, path in configs:
        row = SHORT_TO_ROW[name]
        if args.score_only:
            proc = process(config_file=path)
            npz_path = _find_existing_npz(proc, row)
            if npz_path is None:
                raise SystemExit("--score-only: no existing output for %s under outputs/ "
                                  "(run without --score-only first)" % name)
        else:
            proc = run_one(path, name)
            npz_path = os.path.join(proc.config["ckpt_path"], row["npz"])
        npz_paths[name] = npz_path
        cells, _ = score_row(row, npz_path)
        rows_scored.append((name, cells))
        print("%-16s RMSE %9.4f  FS %+9.4f  CRPS %9s  CRPSS %9s"
              % (name, cells["rmse"], cells["fs"],
                 ("%9.4f" % cells["crps"]) if cells["crps"] is not None else "--",
                 ("%+9.4f" % cells["crpss"]) if cells["crpss"] is not None else "--"))

    # SPM: k_origin/iclr_y/GT are dataset-derived (not model output), the same for every method run
    # on this set -- reconstruct from one successful npz's own arrays and confirm identical across all.
    anchor = np.load(npz_paths[configs[0][0]], allow_pickle=True)
    gt = np.asarray(anchor["GT"], dtype=np.float64)
    spm_pred = _reconstruct_spm(anchor)
    spm_se = (spm_pred - gt) ** 2
    spm_ae = np.abs(spm_pred - gt)
    spm = {"rmse": float(np.sqrt(spm_se.mean())), "crps": float(spm_ae.mean())}
    print("%-16s RMSE %9.4f  FS %9s  CRPS %9.4f  CRPSS %9s" % ("SPM", spm["rmse"], "--", spm["crps"], "--"))

    for name, _ in configs:
        d = np.load(npz_paths[name], allow_pickle=True)
        if not (np.array_equal(np.asarray(d["GT"], dtype=np.float64), gt) and
                np.array_equal(np.asarray(d["k_origin"], dtype=np.float64),
                                np.asarray(anchor["k_origin"], dtype=np.float64)) and
                np.array_equal(np.asarray(d["iclr_y"], dtype=np.float64),
                                np.asarray(anchor["iclr_y"], dtype=np.float64))):
            raise SystemExit("REFUSING: %s's GT/k_origin/iclr_y do not match %s's -- the runs did "
                              "not score the same windows." % (name, configs[0][0]))

    print()
    if args.write_expected or not os.path.exists(EXPECTED_CSV):
        write_expected(rows_scored, spm)
        print("Xiong: not supported (needs a validation split)")
        return

    expected = load_expected()
    print("%-16s%-8s%s" % ("Method", "Result", "Detail"))
    print("-" * 70)
    n_exact = n_close = n_fail = 0
    for name, cells in rows_scored:
        if name not in expected:
            raise SystemExit("no expected_synthetic.csv row for %s" % name)
        level, details = compare(cells, expected[name])
        n_exact += level == "EXACT"
        n_close += level == "CLOSE"
        n_fail += level == "FAIL"
        print("%-16s%-8s%s" % (name, level, "; ".join(details)))

    spm_level, spm_details = compare({"rmse": spm["rmse"], "fs": None, "crps": spm["crps"], "crpss": None},
                                      expected["SPM"])
    n_exact += spm_level == "EXACT"
    n_close += spm_level == "CLOSE"
    n_fail += spm_level == "FAIL"
    print("%-16s%-8s%s" % ("SPM", spm_level, "; ".join(spm_details)))

    print("Xiong: not supported (needs a validation split)")
    print("-" * 70)
    n_total = n_exact + n_close + n_fail
    print("%d/%d methods pass (%d exact, %d close)" % (n_exact + n_close, n_total, n_exact, n_close))
    if n_fail:
        sys.exit(1)


if __name__ == "__main__":
    main()
