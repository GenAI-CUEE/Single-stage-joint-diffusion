"""
Runs one config end to end with the released weights -- the entry point
`eval/reproduce_comparison.py --mode regenerate` doesn't otherwise expose for
an arbitrary config file (it only runs the fixed rows in its own ROWS table)
and `main.py` doesn't expose at all (no CLI args, and it does not stage the
checkpoint). Stages the checkpoint exactly the way regenerate mode does --
imports and calls `eval.reproduce_comparison`'s own `_stage`/`_checksums`,
never copies them -- then calls `main.py`'s `process(...).main()`.
Metric-free: this only runs inference and reports where its outputs landed;
it does not score anything (use `eval/reproduce_comparison.py` for that).

Usage:
  python eval/run_config.py --config configs/samples/joint_x0hat_sample2023.yaml --like propFuture

`--like` names the row (see prediction_samples/MANIFEST.tsv's `method` column, or
`eval/reproduce_comparison.py`'s ROWS/`--rows`) whose released checkpoint and
staging recipe (arm, text2image flag) to reuse for `--config` -- the given
config is expected to be that row's config with only data/output fields
changed, same arm, same released weights.

Optional: `--dump-frames case_id:subset_index[,case_id:subset_index...]`
switches on `models/common/frame_export.py`'s per-window `gt_h*.png`/
`pred_h*.png` export (see its module docstring; a no-op unless
`frame_manifest.csv` exists in the working directory). `subset_index` is the
window's 0-based position in THIS run's own subset order (`--config`'s
`subset_indices`, or full-test-set order if unset) -- not the canonical
1998-row position, since a different subset does not preserve it. Writes
`frame_manifest.csv` at the repository root (where `models/common/
frame_export.py` reads it from) for the duration of the run only, and removes
it afterward.
"""
import argparse
import csv
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import process
from eval.reproduce_comparison import WEIGHTS_DIR, SHORT_TO_ROW, _checksums, _stage

FRAME_MANIFEST_PATH = "frame_manifest.csv"


def _parse_dump_frames(spec):
    rows = []
    for entry in spec.split(","):
        case_id, _, idx = entry.partition(":")
        if not case_id or not idx:
            raise SystemExit("--dump-frames entries must be case_id:subset_index, got %r" % entry)
        rows.append({"case_id": case_id, "subset_index": int(idx)})
    return rows


def run_one(config_file, like, dump_frames=None):
    """Stage the checkpoint `like`'s row recipe names and run `config_file`
    end to end, exactly as the CLI below does -- factored out so a driver that
    runs many configs (eval/check_synthetic.py) can call it directly instead
    of duplicating the staging logic."""
    if like not in SHORT_TO_ROW:
        raise SystemExit("Unknown --like %r (known: %s)" % (like, ", ".join(sorted(SHORT_TO_ROW))))
    row = SHORT_TO_ROW[like]

    checksums = _checksums()
    rec = checksums[row["arm"]]

    proc = process(config_file=config_file)

    src = os.path.join(WEIGHTS_DIR, rec["release_filename"])
    if row.get("text2image"):
        net = proc.config["Text2ImageJoint"]["Network"]
        dst = os.path.join(proc.config["Main_logging_path"], net["pretrained_ckpt_dir"],
                            net["pretrained_ckpt_file"])
    else:
        dst = os.path.join(proc.config["ckpt_path"], rec["original_filename"])
    _stage(src, dst)
    print("staged %s -> %s" % (src, dst))

    wrote_manifest = False
    if dump_frames:
        with open(FRAME_MANIFEST_PATH, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["case_id", "subset_index"])
            w.writeheader()
            w.writerows(_parse_dump_frames(dump_frames))
        wrote_manifest = True
        print("wrote %s for this run" % FRAME_MANIFEST_PATH)

    t0 = time.time()
    try:
        proc.main()
    finally:
        if wrote_manifest and os.path.exists(FRAME_MANIFEST_PATH):
            os.remove(FRAME_MANIFEST_PATH)
    t1 = time.time()

    print("run complete in %.1fs -- ckpt_path: %s" % (t1 - t0, proc.config["ckpt_path"]))
    return proc


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True, help="config file to run")
    ap.add_argument("--like", required=True,
                     help="short row name whose staging recipe to reuse (e.g. propFuture)")
    ap.add_argument("--dump-frames", default=None,
                     help="case_id:subset_index[,case_id:subset_index...] -- see module docstring")
    args = ap.parse_args()
    run_one(args.config, args.like, dump_frames=args.dump_frames)


if __name__ == "__main__":
    main()
