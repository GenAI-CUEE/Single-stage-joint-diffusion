"""
Shared cross-model frame-export helper for qualitative frame panels that
compare methods side by side on the same test cases.

The bug this exists to prevent: per-epoch visualisation frames from
evaluation() come from a shuffled, unsubsetted DataLoader, so "sample 0" of a
dumped batch is a different random test window per model -- cross-model
frame panels built from those would silently compare different weather.
forecast() is the aligned path (shuffle=False, subset-filtered via
_maybe_subset, drop_last=False), so row r (0-based) within the returned
[1998, ...] arrays is the SAME real-world sample for every model, since every
model's forecast() loads the identical skygpt_subset_indices.npy in the same
order (verified directly). This module dumps frames keyed by that row number
r, not by batch position.

Call once per batch, from inside each image-generating model's forecast()
loop, with the batch's uint8 [B,T,C,H,W] GT/pred frame tensors and the row
range [row_start, row_start+B) that batch covers. A no-op for any batch that
doesn't overlap frame_manifest.csv, and a no-op entirely unless that manifest
file is present (it is an optional, hand-picked list of cases; none ships
with this release).

**Per-run output dir (2026-08-12).** The original convention wrote every
method's frames to the SAME shared path (`frames/{case_id}/
{method}_{gt|pred}_h{h}.png`), keyed only by method name -- so when the same
method's forecast() ran more than once under different configs (e.g. a sweep
over the number of diffusion sampling steps, all writing
"text2imagejoint_x0hat_pred_h5.png"), each run silently overwrote the last
one's frames. The LAST run to execute wins, with no record of which run that
was -- the frames on disk can silently stop matching the numbers being
reported elsewhere for that method. Pass `out_dir` (typically
`{config['logging_version_path']}/test-images/ramp-images`, i.e. inside that
specific run's own results folder) to write there instead: one location per
run, so different configs of the same method can never collide, and the
frames always live next to the exact checkpoint/npz that produced them. When
`out_dir` is given, filenames drop the method prefix (`gt_h{h}.png` /
`pred_h{h}.png`) since the directory itself is already method+run-specific.
`out_dir=None` (the default) preserves the original shared-path behavior
unchanged, for methods not yet migrated to this convention.
"""
import csv
import os

import numpy as np
from PIL import Image

MANIFEST_PATH = "frame_manifest.csv"
OUT_DIR = "frames"

_MANIFEST_CACHE = None


def _load_manifest(path=MANIFEST_PATH):
    rows = []
    if not os.path.exists(path):
        return rows
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            rows.append({"case_id": row["case_id"], "subset_index": int(row["subset_index"])})
    return rows


def dump_manifest_frames(method_name, row_start, gt_u8, pred_u8, manifest_path=MANIFEST_PATH,
                          out_dir=None):
    """gt_u8 / pred_u8: uint8 tensors (torch or numpy) shaped [B,T,C,H,W],
    already denormalized to [0,255]. row_start: the 0-based row index, within
    the shared 1998-sample subset, of the FIRST item in this batch (running
    counter incremented by each batch's actual size B -- robust to a smaller
    final batch under drop_last=False). out_dir: see module docstring --
    per-run destination; defaults to the shared frames/ path.
    """
    global _MANIFEST_CACHE
    if _MANIFEST_CACHE is None:
        _MANIFEST_CACHE = _load_manifest(manifest_path)
    if not _MANIFEST_CACHE:
        return

    gt_np = gt_u8.detach().cpu().numpy() if hasattr(gt_u8, "detach") else np.asarray(gt_u8)
    pred_np = pred_u8.detach().cpu().numpy() if hasattr(pred_u8, "detach") else np.asarray(pred_u8)
    B, T = gt_np.shape[0], gt_np.shape[1]
    row_end = row_start + B

    base_dir = out_dir if out_dir is not None else OUT_DIR
    prefix = "" if out_dir is not None else ("%s_" % method_name)

    for entry in _MANIFEST_CACHE:
        r = entry["subset_index"]
        if row_start <= r < row_end:
            pos = r - row_start
            case_dir = os.path.join(base_dir, entry["case_id"])
            os.makedirs(case_dir, exist_ok=True)
            for h in range(T):
                _save_frame(gt_np[pos, h], os.path.join(case_dir, "%sgt_h%d.png" % (prefix, h + 1)))
                _save_frame(pred_np[pos, h], os.path.join(case_dir, "%spred_h%d.png" % (prefix, h + 1)))


def _save_frame(frame_u8, path):
    arr = np.asarray(frame_u8).astype(np.uint8)     # [C,H,W]
    arr = np.transpose(arr, (1, 2, 0))               # [H,W,C]
    if arr.shape[2] == 1:
        arr = arr[:, :, 0]
    Image.fromarray(arr).save(path)
