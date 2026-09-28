"""
Reproduce the manuscript's comparison table (Table 2). A driver only: every
number printed comes from eval/score.py's and eval/bootstrap.py's copied
functions -- this file contains no metric arithmetic of its own, only the
orchestration needed to get each method's forecast array and ground truth to
the scorer and resample it for confidence intervals.

Two modes:

  --mode stored (default)  Score the shipped prediction_samples/*.npz against
                            ground truth computed from the user's own local
                            SIRTA data. No model, no checkpoint, no GPU.
                            Reproduces the manuscript's numbers exactly --
                            SkyGPT and Xiong included, since their samplers
                            are not reproducible from the weights alone
                            (see report/R9a_ensemble_provenance.md, comparison
                            repo). Rebuild-aware: SIRTA's February 2025
                            re-issue of the 2023 irradiance files turns 86 of
                            the 1998 canonical test windows into unbuildable
                            (nan-dropped) windows for anyone downloading
                            today. This mode detects, by identity (not
                            position), which of the 1998 canonical test_ids
                            the local data can actually build, and compares
                            against whichever expected table matches: all
                            1998 -> expected_table2.csv; exactly the 1912
                            rebuildable ones -> expected_rebuild.csv; any
                            other sample set -> refuses to compare (no
                            reference table exists for it) and exits non-zero.
  --mode regenerate         The original path: run inference from the
                            released weights. Reproduces the deterministic
                            rows exactly; SkyGPT and Xiong regenerate within
                            their confidence intervals, not to the printed
                            digit.

The SPM (persistence baseline) row is not a trained model: it has no
checkpoint and no shipped npz (SPM is measured-irradiance-derived and is on
this release's "never ship" list) -- it is always computed from the user's
own local data, in both modes.
"""
import argparse
import csv
import glob
import hashlib
import json
import os
import shutil
import sys
import pathlib
import tempfile

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import numpy as np

from main import process
from eval.score import score_point, score_ensemble
from eval.bootstrap import block_bootstrap_ci, picp

WEIGHTS_DIR = "weights"
CHECKSUMS = os.path.join(WEIGHTS_DIR, "CHECKSUMS.tsv")
SUBSET_IDS_CSV = "splits/sirta_2023/test_subset_ids.csv"
PICP_ALPHA = 0.10  # 90% nominal interval, the convention used throughout this codebase

PRED_DIR = "prediction_samples"
PRED_CHECKSUMS = os.path.join(PRED_DIR, "CHECKSUMS.tsv")
PRED_EXPECTED = os.path.join(PRED_DIR, "expected_table2.csv")
# Rebuild-aware stored mode: a user downloading SIRTA's files today gets
# the post-February-2025 re-issue, which silently drops 86 of the 1998
# canonical test windows (see build_ground_truth). rebuild_excluded.csv lists
# exactly those 86 test_ids (from the comparison repo's F38 sample-level
# analysis); expected_rebuild.csv is Table 2 recomputed on the remaining 1912.
PRED_EXPECTED_REBUILD = os.path.join(PRED_DIR, "expected_rebuild.csv")
REBUILD_EXCLUDED_CSV = os.path.join(PRED_DIR, "rebuild_excluded.csv")
# The npz themselves are too large to track in git and live on Dropbox, same
# convention as weights/weights_db -- prediction_samples/predictions_db is a
# symlink to that Dropbox folder (see prediction_samples/CHECKSUMS.tsv for
# per-file hashes).
PRED_NPZ_DIR = os.path.join(PRED_DIR, "predictions_db")
# Any one of the 15 configs' Data/ImageSequence block works: every headline
# config scores against the identical 2023 SIRTA test data (confirmed in
# prediction_samples' own build report, R10a step 2 in the comparison repo).
GT_CONFIG = "configs/xiong.yaml"

# One row per Table 2 method. "arm" keys into weights/CHECKSUMS.tsv; "npz" is
# the filename each model's own forecast() writes into its run's checkpoints/
# directory, used in --mode regenerate; "short" is the stored-predictions file
# stem (prediction_samples/<short>.npz), used in --mode stored and by --rows.
# "score"/"kwargs" call straight through to eval/score.py exactly as the
# paper's own scoring did -- identical in both modes; only where the npz comes
# from differs. "text2image": True marks the two rows whose checkpoint is
# loaded by Text2ImageJoint's own pretrained_ckpt_dir/file mechanism instead
# of the generic checkpoint directory every other method uses (see
# main.py::process._get_previous_training vs. models/Text2ImageJoint.py's
# __init__) -- regenerate mode only.
ROWS = [
    # Not a trained model -- the smart-persistence reference every other row's
    # FS/CRPSS is computed against. Computed fresh from local ground truth in
    # both modes (never shipped: SPM is measured-irradiance-derived).
    {"tag": "SPM (persistence baseline)", "short": "SPM", "spm": True},

    {"tag": "SkyGPT (ens N=10)", "short": "SkyGPT", "config": "configs/skygpt.yaml",
     "arm": "SkyGPT", "npz": "skygpt_forecast_reg_imgstats.npz",
     "score": score_ensemble, "kwargs": {"gt_key": "ghi_target", "members_key": "ens_ghi"}},

    {"tag": "Xiong (direct N=10)", "short": "Xiong", "config": "configs/xiong.yaml",
     "arm": "Xiong", "npz": "xiong_forecast_stats.npz",
     "score": score_ensemble, "kwargs": {"members_key": "S", "point_key": "mu_d"}},

    {"tag": "Joint x0hat (direct N=10)", "short": "propFuture",
     "config": "configs/joint_x0hat_seed1.yaml", "arm": "Joint_x0hat_seed1",
     "npz": "text2imagejoint_forecast_stats.npz",
     "text2image": True, "score": score_ensemble, "kwargs": {"members_key": "S"}},

    {"tag": "Joint clean (direct N=10)", "short": "propPast",
     "config": "configs/joint_clean_seed1.yaml", "arm": "Joint_clean_seed1",
     "npz": "text2imagejoint_forecast_stats.npz",
     "text2image": True, "score": score_ensemble, "kwargs": {"members_key": "S"}},

    {"tag": "Sunset (wIclr)", "short": "SunSet", "config": "configs/sunset_wiclr.yaml",
     "arm": "Sunset_wIclr", "npz": "sunset_forecast_stats.npz",
     "score": score_point, "kwargs": {}},
    {"tag": "DLinear (wIclr)", "short": "DLinearIclr", "config": "configs/dlinear_wiclr.yaml",
     "arm": "DLinear_wIclr", "npz": "dlinear_forecast_stats.npz",
     "score": score_point, "kwargs": {}},
    {"tag": "PatchTST (wIclr)", "short": "PatchTSTIclr", "config": "configs/patchtst_wiclr.yaml",
     "arm": "PatchTST_wIclr", "npz": "patchtst_forecast_stats.npz",
     "score": score_point, "kwargs": {}},
    {"tag": "TimeXer (wIclr)", "short": "TimeXerIclr", "config": "configs/timexer_wiclr.yaml",
     "arm": "TimeXer_wIclr", "npz": "timexer_forecast_stats.npz",
     "score": score_point, "kwargs": {}},

    {"tag": "DLinear (noIclr)", "short": "DLinearNoIclr", "config": "configs/dlinear_noiclr.yaml",
     "arm": "DLinear_noIclr", "npz": "dlinear_forecast_stats.npz",
     "score": score_point, "kwargs": {}},
    {"tag": "PatchTST (noIclr)", "short": "PatchTSTNoIclr",
     "config": "configs/patchtst_noiclr.yaml", "arm": "PatchTST_noIclr",
     "npz": "patchtst_forecast_stats.npz", "score": score_point, "kwargs": {}},
    {"tag": "TimeXer (noIclr)", "short": "TimeXerNoIclr", "config": "configs/timexer_noiclr.yaml",
     "arm": "TimeXer_noIclr", "npz": "timexer_forecast_stats.npz",
     "score": score_point, "kwargs": {}},
    {"tag": "Sunset (noIclr)", "short": "SunSetNoIclr", "config": "configs/sunset_noiclr.yaml",
     "arm": "Sunset_noIclr", "npz": "sunset_forecast_stats.npz",
     "score": score_point, "kwargs": {}},

    {"tag": "ConvLSTM (img-reg)", "short": "ConvLSTM", "config": "configs/convlstm.yaml",
     "arm": "ConvLSTM", "npz": "convlstm_forecast_reg_imgstats.npz",
     "score": score_point, "kwargs": {}},
    {"tag": "PhyDNetGAN (img-reg)", "short": "PhyDNetGAN", "config": "configs/phydnetgan.yaml",
     "arm": "PhyDNetGAN", "npz": "phydnetgan_forecast_reg_imgstats.npz",
     "score": score_point, "kwargs": {}},
    {"tag": "PhyDNet (img-reg)", "short": "PhyDNet", "config": "configs/phydnet.yaml",
     "arm": "PhyDNet", "npz": "phydnet_forecast_reg_imgstats.npz",
     "score": score_point, "kwargs": {}},
]
SHORT_TO_ROW = {r["short"]: r for r in ROWS}


def _checksums():
    with open(CHECKSUMS) as f:
        header = f.readline().rstrip("\n").split("\t")
        return {(rec := dict(zip(header, line.rstrip("\n").split("\t"))))["arm"]: rec
                for line in f}


def _day_labels():
    """One calendar-day label per row of the 1998-sample test subset, in the
    same row order every model's forecast() iterates it -- the block
    bootstrap resamples whole days, not individual rows. regenerate mode
    only -- stored mode gets its (possibly rebuild-reduced) day labels from
    build_ground_truth() instead, since its row set may not be the full 1998."""
    with open(SUBSET_IDS_CSV) as f:
        return np.array([row["timestamp_local"][:10] for row in csv.DictReader(f)])


def _canonical_test_ids():
    """The 1998 canonical matched test_ids and their calendar-day labels, in
    the fixed row order every shipped npz and expected_table2.csv row uses.
    test_id identifies a window by its target (y0) timestamp -- the same
    identity build_ground_truth() uses -- so this list is position-independent:
    it still means the same 1998 windows regardless of how many of them the
    local data can actually build."""
    with open(SUBSET_IDS_CSV) as f:
        rows = list(csv.DictReader(f))
    test_id = [row["timestamp_local"] + ":00" for row in rows]
    day = np.array([row["timestamp_local"][:10] for row in rows])
    return test_id, day


def _rebuild_excluded_ids():
    """The 86 canonical test_ids that SIRTA's February 2025 re-issue makes
    unbuildable (prediction_samples/rebuild_excluded.csv, derived from the
    comparison repo's F38 sample-level analysis)."""
    with open(REBUILD_EXCLUDED_CSV) as f:
        return {row["test_id"] for row in csv.DictReader(f)}


# ==================================================================== #
# --mode regenerate: run inference from the released weights (original path)
# ==================================================================== #

def _stage(src, dst):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if not os.path.exists(dst):
        shutil.copy2(src, dst)


def _find_existing_npz(proc, row):
    """Most configs pin a fixed `version:` (load_ckpt: true), so a prior run's
    output always lands at this same proc.config["ckpt_path"]. The two
    Text2ImageJoint configs set load_ckpt: false, so main.py stamps a *fresh*
    timestamped version folder on every process() call regardless of what
    already ran -- a prior completed npz for those can sit under a different
    version folder than this call's own ckpt_path. Search every version
    folder under this row's logging path (not just this call's own) for a
    matching completed npz before concluding there isn't one."""
    mainpath_dir = os.path.dirname(os.path.dirname(proc.config["ckpt_path"]))
    matches = glob.glob(os.path.join(mainpath_dir, "*", "checkpoints", row["npz"]))
    return max(matches, key=os.path.getmtime) if matches else None


def _fresh_version_config_file(config_file, scratch_dir):
    """--force must never overwrite an existing result (most configs pin a
    fixed `version:`, so a plain re-run would land on the same
    proc.config["ckpt_path"] the earlier run used). Write a temp copy of the
    config with load_ckpt forced off -- main.py then stamps a brand-new
    timestamped version folder (main.py:39-91), exactly the mechanism the two
    Text2ImageJoint configs already use unconditionally. This only changes
    which version folder gets used, not which weights get loaded: the
    checkpoint file is staged into that folder before main() is called (same
    as any other row), and load_ckpt is read nowhere else in this repo
    (confirmed: grep across models/, exp_valid.py, train.py, utils/) except
    main.py's own Skygpt-only nested branch (config["Skygpt"/"VQVAE"]
    ["load_ckpt"]), which is handled here too."""
    import yaml
    with open(config_file) as f:
        config = yaml.safe_load(f)
    if config.get("Network_name") == "Skygpt":
        which = "VQVAE" if config.get("Training", {}).get("train_only_VQVAE") else "Skygpt"
        config[which]["load_ckpt"] = False
    else:
        config["load_ckpt"] = False
    tmp_path = os.path.join(scratch_dir, "force_" + os.path.basename(config_file))
    with open(tmp_path, "w") as f:
        yaml.safe_dump(config, f)
    return tmp_path


def run_row_regenerate(row, checksums, scratch_dir, force=False):
    config_file = _fresh_version_config_file(row["config"], scratch_dir) if force else row["config"]
    proc = process(config_file=config_file)
    rec = checksums[row["arm"]]
    npz_path = os.path.join(proc.config["ckpt_path"], row["npz"])

    existing = None if force else _find_existing_npz(proc, row)
    if existing:
        print(">>>> %s: found existing %s, skipping inference" % (row["tag"], existing))
        npz_path = existing
    else:
        src = os.path.join(WEIGHTS_DIR, rec["release_filename"])
        if row.get("text2image"):
            net = proc.config["Text2ImageJoint"]["Network"]
            dst = os.path.join(proc.config["Main_logging_path"], net["pretrained_ckpt_dir"],
                                net["pretrained_ckpt_file"])
        else:
            dst = os.path.join(proc.config["ckpt_path"], rec["original_filename"])
        _stage(src, dst)
        proc.main()  # runs inference (every headline config has Training.is_train: false)

    return row["score"]({"npz": npz_path, **row["kwargs"]})


# ==================================================================== #
# --mode stored: score the shipped prediction_samples/*.npz
# ==================================================================== #

def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _pred_checksums():
    with open(PRED_CHECKSUMS) as f:
        header = f.readline().strip().split("\t")
        out = {}
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = dict(zip(header, line.split("\t")))
            out[rec["file"]] = rec
        return out


def fmt_ref(ref_row):
    h, m, day, month, year = (int(x) for x in ref_row)
    return "%04d-%02d-%02dT%02d:%02d:00" % (year, month, day, h, m)


def build_ground_truth():
    """CPU-only, no GPU, no model: builds the SIRTA test dataset from the
    user's own local data and extracts, for whichever of the 1998 canonical
    test windows the local data can actually build, everything stored mode
    needs -- GT, k_origin, iclr_y (ground truth only; models/Xiong.py:640-641's
    unpack_batch-derived quantities, never a model output) -- plus
    issue_time/target_time/test_id/test_index for the alignment check and
    day_labels for the block bootstrap.

    Rebuild-aware: matches each of the 1998 canonical test_ids
    (splits/sirta_2023/test_subset_ids.csv) against the local build BY IDENTITY (each
    window's own target/y0 timestamp), not by the fixed row position
    splits/sirta_2023/test_subset_indices.npy used to encode. Position is not
    reusable here -- SIRTA's February 2025 re-issue of the 2023 irradiance
    files turns some previously-provisional minutes into nan, and this
    repo's own windowing (dataloader/sirta_dataset.py::making_data2sequence) slides
    positionally over whatever rows survive the join, so a single missing
    minute shifts every later window's position. Matching by identity instead
    means a canonical test_id the local data can't build is simply absent
    from the result (see 'missing_ids'), and every test_id that IS present
    still gets the correct window regardless of how the row count changed.

    Reproduces the same 1998-row order and values prediction_samples' own
    build used when the local data can build all 1998 (verified: matches
    every value in prediction_samples/expected_table2.csv's SPM row and every
    timestamp field in every shipped npz, bit-for-bit)."""
    import yaml
    import torch
    from dataloader.sirta_dataset import SeqDataset_SIRTA

    with open(GT_CONFIG) as f:
        config = yaml.safe_load(f)

    print(">>>> building ground truth from local SIRTA data (no model, no GPU)...")
    ds = SeqDataset_SIRTA(config, torch.device("cpu"), mode="test",
                          attribute_list=["label", "image", "feature"])

    n_local = len(ds.data["ref_y"])
    local_test_id = [fmt_ref(np.asarray(ds.data["ref_y"][i])[0]).replace("T", " ")
                      for i in range(n_local)]
    id_to_idx = {tid: i for i, tid in enumerate(local_test_id)}

    canonical_id, canonical_day = _canonical_test_ids()
    present = [tid for tid in canonical_id if tid in id_to_idx]
    missing_ids = [tid for tid in canonical_id if tid not in id_to_idx]
    present_mask = np.array([tid in id_to_idx for tid in canonical_id])
    day_labels = canonical_day[present_mask]

    idx_list = [id_to_idx[tid] for tid in present]
    N, H = len(idx_list), 8
    GT = np.empty((N, H), dtype=np.float64)
    ICLR_Y = np.empty((N, H), dtype=np.float64)
    KORG = np.empty(N, dtype=np.float64)
    issue_time = np.empty(N, dtype="<U19")
    target_time = np.empty((N, H), dtype="<U19")
    iclr_floor = 1.0

    for row, idx in enumerate(idx_list):
        label_x = np.asarray(ds.data["label_x"][idx])
        label_y = np.asarray(ds.data["label_y"][idx])
        feat_x = np.asarray(ds.data["feature_x"][idx])
        feat_y = np.asarray(ds.data["feature_y"][idx])
        ref_x = np.asarray(ds.data["ref_x"][idx])
        ref_y = np.asarray(ds.data["ref_y"][idx])

        # unpack_batch() (models/common/frame_utils.py) does .float() on
        # these, and k_origin = ghi_x[:,-1] / iclr_x[:,-1].clamp(...) is torch
        # (float32) arithmetic (models/Xiong.py:590-592) -- replicate both the
        # cast and the float32 division so results match bit-for-bit, not
        # just to float32 precision.
        ghi_x_f32 = label_x[:, 0].astype(np.float32)
        ghi_y_f32 = label_y[:, 0].astype(np.float32)
        iclr_x_f32 = feat_x[:, 3].astype(np.float32)
        iclr_y_f32 = feat_y[:, 3].astype(np.float32)

        GT[row] = ghi_y_f32.astype(np.float64)
        ICLR_Y[row] = iclr_y_f32.astype(np.float64)
        k_f32 = ghi_x_f32[-1] / np.maximum(iclr_x_f32[-1], np.float32(iclr_floor))
        KORG[row] = np.float64(k_f32)

        issue_time[row] = fmt_ref(ref_x[-1])
        for h in range(H):
            target_time[row, h] = fmt_ref(ref_y[h])

    with open(os.path.join(config["Data"]["data_folder"], config["Data"]["test_list"])) as f:
        lines = [ln for ln in f.read().split("\n") if ln]
    line_to_pos = {ln: i for i, ln in enumerate(lines)}
    test_id = np.array([t.replace("T", " ") for t in target_time[:, 0]])
    test_index = np.array([line_to_pos[s] for s in test_id])

    print(">>>> ground truth built: %d/%d canonical samples present"
          % (N, len(canonical_id)))
    return dict(GT=GT, k_origin=KORG, iclr_y=ICLR_Y, issue_time=issue_time,
                target_time=target_time, test_id=test_id, test_index=test_index,
                day_labels=day_labels, missing_ids=missing_ids,
                n_canonical=len(canonical_id))


def _verify_and_load_stored(row, gt, pred_checksums, scratch_dir):
    """Verify checksum + alignment for one row's shipped npz, then build a
    merged temp npz (local ground truth + the shipped forecast array(s) for
    the rows gt actually has, matched by test_id) for eval/score.py to read
    unchanged. Refuses (raises) on any checksum or alignment mismatch -- no
    numbers on a mismatch.

    The shipped npz always carries all 1998 canonical rows (frozen once);
    gt may carry fewer of them (rebuild-aware stored mode) -- rows are
    selected by test_id identity, not position, then checked cell-for-cell
    exactly as before."""
    short = row["short"]
    npz_path = os.path.join(PRED_NPZ_DIR, "%s.npz" % short)
    if not os.path.isdir(PRED_NPZ_DIR):
        raise SystemExit(
            "REFUSING to score %s: %s does not exist.\n"
            "The prediction npz are hosted externally, same as the weights -- download them into\n"
            "%s/ (create the directory if needed), matching the filenames in\n"
            "%s. Verify a download with:\n"
            "    sha256sum -c <(awk -F'\\t' 'NR>1 {print $3\"  \"$1}' %s)"
            % (row["tag"], PRED_NPZ_DIR, PRED_NPZ_DIR, PRED_CHECKSUMS, PRED_CHECKSUMS))
    if not os.path.exists(npz_path):
        raise SystemExit("REFUSING to score %s: %s not found" % (row["tag"], npz_path))

    rec = pred_checksums.get("%s.npz" % short)
    if rec is None:
        raise SystemExit("REFUSING to score %s: no CHECKSUMS.tsv entry for %s.npz"
                          % (row["tag"], short))
    actual_sha = _sha256(npz_path)
    if actual_sha != rec["sha256"]:
        raise SystemExit("REFUSING to score %s: SHA-256 mismatch for %s.npz\n"
                          "  expected %s\n  actual   %s"
                          % (row["tag"], short, rec["sha256"], actual_sha))

    shipped = np.load(npz_path, allow_pickle=True)
    shipped_id_to_row = {tid: i for i, tid in enumerate(shipped["test_id"])}
    try:
        rows = np.array([shipped_id_to_row[tid] for tid in gt["test_id"]])
    except KeyError as e:
        raise SystemExit(
            "REFUSING to score %s: test_id %s (present in the local SIRTA data) is not "
            "one of the shipped npz's 1998 canonical rows." % (row["tag"], e.args[0]))

    for key in ("issue_time", "target_time", "test_id", "test_index"):
        if not np.array_equal(shipped[key][rows], gt[key]):
            raise SystemExit(
                "REFUSING to score %s: %s does not match the local SIRTA data -- "
                "the shipped predictions do not align with this test set." % (row["tag"], key))

    gt_key = row["kwargs"].get("gt_key", "GT")
    merged = {gt_key: gt["GT"], "k_origin": gt["k_origin"], "iclr_y": gt["iclr_y"]}
    for key in shipped.files:
        if key not in ("issue_time", "target_time", "test_index", "test_id", "meta"):
            merged[key] = shipped[key][rows]

    os.makedirs(scratch_dir, exist_ok=True)
    tmp_path = os.path.join(scratch_dir, "merged_%s.npz" % short)
    np.savez(tmp_path, **merged)
    return tmp_path


def run_row_stored(row, gt, pred_checksums, scratch_dir):
    npz_path = _verify_and_load_stored(row, gt, pred_checksums, scratch_dir)
    return row["score"]({"npz": npz_path, **row["kwargs"]})


def spm_scored(gt):
    """SPM: never shipped (measured-irradiance-derived), computed directly
    from local ground truth in both modes."""
    spm = gt["k_origin"][:, None] * np.clip(gt["iclr_y"], 1.0, None)
    se_spm = (spm - gt["GT"]) ** 2
    return {"_se_model": se_spm, "_se_spm": se_spm}


# ==================================================================== #
# Bootstrap / CI / comparison
# ==================================================================== #

def add_ci(scored, day_labels):
    ci = block_bootstrap_ci(
        scored["_se_model"], scored["_se_spm"], day_labels,
        crps_model=scored.get("_crps_model"), ae_spm=scored.get("_ae_spm"))
    scored["_ci"] = ci
    if "_members" in scored:
        covered, _, _ = picp(scored["_members"], scored["_gt"], PICP_ALPHA)
        scored["_picp"] = float(covered.mean())
    return scored


def _fmt_ci(v, key, fmt):
    if key not in v.get("_ci", {}):
        return "--"
    point, lo, hi = v["_ci"][key]
    return ("%s [%s, %s]" % (fmt, fmt, fmt)) % (point, lo, hi)


def print_table(results):
    cols = ("Method", "RMSE [95% CI]", "FS [95% CI]", "CRPS [95% CI]",
            "CRPSS [95% CI]", "PICP@90%")
    widths = (28, 25, 27, 25, 27, 10)
    hdr = "".join("%-*s" % (w, c) for w, c in zip(widths, cols))
    print(hdr)
    print("-" * len(hdr))
    for tag, v in results:
        if "crps" not in v["_ci"]:  # the SPM row -- reference only, no FS/CRPS/CRPSS
            # (block_bootstrap_ci always returns "fs", trivially 0 for SPM's
            # self-comparison against se_spm; "crps" is the reliable marker)
            row = (tag, _fmt_ci(v, "rmse", "%.2f"), "--", "--", "--", "--")
        else:
            crpss_key = "crpss" if "crpss" in v["_ci"] else None
            row = (
                tag,
                _fmt_ci(v, "rmse", "%.2f"),
                _fmt_ci(v, "fs", "%+.3f"),
                _fmt_ci(v, "crps", "%.2f"),
                _fmt_ci(v, "crpss", "%+.3f") if crpss_key else "--",
                "%.3f" % v["_picp"] if "_picp" in v else "--",
            )
        print("".join("%-*s" % (w, c) for w, c in zip(widths, row)))


def _load_expected(path):
    with open(path) as f:
        return {row["method"]: row for row in csv.DictReader(f)}


CELLS = ("rmse", "rmse_lo", "rmse_hi", "fs", "fs_lo", "fs_hi",
         "crps", "crps_lo", "crps_hi", "crpss", "crpss_lo", "crpss_hi")


def _cell_values(v):
    """(cell_name, value_or_None) for all 12 cells, from a scored+CI'd row."""
    ci = v["_ci"]
    out = {}
    for key in ("rmse", "fs", "crps", "crpss"):
        if key in ci:
            point, lo, hi = ci[key]
            out[key], out[key + "_lo"], out[key + "_hi"] = point, lo, hi
        else:
            out[key] = out[key + "_lo"] = out[key + "_hi"] = None
    return out


def compare_stored(results, expected):
    """Exact-match comparison (stored mode): every cell must equal
    expected_table2.csv's value exactly. The driver decides pass/fail, not
    the reader."""
    print()
    print("%-28s%-10s%-14s%s" % ("Method", "Cell", "Computed", "Expected  Result"))
    print("-" * 90)
    n_pass = n_total = 0
    for tag, v in results:
        exp = expected[v["short"]]
        cells = _cell_values(v)
        for cell in CELLS:
            exp_raw = exp[cell]
            if exp_raw == "":
                continue  # SPM has no fs/crps/crpss
            n_total += 1
            got = cells[cell]
            want = float(exp_raw)
            ok = (got == want)
            n_pass += ok
            status = "PASS" if ok else "FAIL"
            if not ok:
                print("%-28s%-10s%-14r%-10r%s" % (tag, cell, got, want, status))
    print("-" * 90)
    print("%d/%d cells PASS" % (n_pass, n_total))
    return n_pass == n_total


def classify_sample_set(gt):
    """Which expected table (if any) matches gt's sample set, given gt's
    missing_ids relative to the 1998 canonical test_ids. Returns
    (branch, expected_path, message):
      "full"    -- all 1998 present -> expected_table2.csv
      "rebuild" -- missing exactly the 86 test_ids SIRTA's February 2025
                   re-issue makes unbuildable -> expected_rebuild.csv (the
                   1912-sample table)
      "other"   -- any other sample set -- no reference table exists for it;
                   expected_path is None and the caller must not compare."""
    missing = set(gt["missing_ids"])
    if not missing:
        return "full", PRED_EXPECTED, (
            ">>>> all %d canonical test samples present; comparing against "
            "expected_table2.csv." % gt["n_canonical"])

    excluded = _rebuild_excluded_ids()
    if missing == excluded:
        return "rebuild", PRED_EXPECTED_REBUILD, (
            ">>>> local data is missing exactly the %d test samples SIRTA's February 2025 "
            "re-issue makes unbuildable -- this matches SIRTA's post-February-2025 files. "
            "Comparing against the %d-sample rebuild table (expected_rebuild.csv)."
            % (len(excluded), gt["n_canonical"] - len(excluded)))

    extra = sorted(missing)
    message = (
        "REFUSING to compare: %d of the %d canonical test_id(s) are missing from the local "
        "dataset, and this is not the known SIRTA post-February-2025 exclusion set (%d ids). "
        "No reference table exists for this sample set.\n"
        "Missing test_id(s) (%d total), first 20:\n  %s"
        % (len(missing), gt["n_canonical"], len(excluded), len(missing),
           "\n  ".join(extra[:20])))
    return "other", None, message


def drop_test_ids(gt, ids_to_drop):
    """A copy of gt with the given test_ids removed -- the same shape
    build_ground_truth() would return if the local dataset had never been
    able to build windows for them. Used by ad hoc rebuild-awareness test scripts
    (not shipped) to simulate a data re-issue without touching real
    data or requiring a shipped 'ok to drop' flag."""
    ids_to_drop = set(ids_to_drop)
    keep = np.array([tid not in ids_to_drop for tid in gt["test_id"]])
    out = dict(gt)
    for key in ("GT", "k_origin", "iclr_y", "issue_time", "target_time",
                "test_id", "test_index", "day_labels"):
        out[key] = gt[key][keep]
    out["missing_ids"] = sorted(set(gt["missing_ids"]) | ids_to_drop)
    return out


def run_stored_mode(gt, rows, scratch_dir):
    """The full stored-mode path from an already-built ground-truth dict:
    score every row, print the table, then compare against whichever
    expected table (if any) matches gt's sample set. Exits non-zero
    (sys.exit) if no reference table applies to gt's sample set, or if any
    cell fails to match the one that does. Shared by main() (the real run)
    and the rebuild-awareness test scripts, which pass a gt modified by
    drop_test_ids() to simulate a rebuild without touching real data."""
    pred_checksums = _pred_checksums()
    results = []
    for row in rows:
        if row.get("spm"):
            scored = spm_scored(gt)
        else:
            scored = run_row_stored(row, gt, pred_checksums, scratch_dir)
        scored["short"] = row["short"]
        results.append((row["tag"], add_ci(scored, gt["day_labels"])))

    print_table(results)

    branch, expected_path, message = classify_sample_set(gt)
    print()
    print(message)
    if branch == "other":
        sys.exit(1)

    expected = _load_expected(expected_path)
    ok = compare_stored(results, expected)
    if not ok:
        sys.exit(1)


def compare_regenerate(results, expected):
    """regenerate mode: not an exact-match gate (SkyGPT/Xiong are genuinely
    stochastic -- R9a). Print each cell's difference from the expected value
    and whether the regenerated point estimate lies inside the expected CI."""
    print()
    print("%-28s%-8s%-16s%-16s%-12s%s" % ("Method", "Metric", "Regenerated", "Expected",
                                            "Diff", "In expected CI?"))
    print("-" * 100)
    n_in_ci = n_total = 0
    for tag, v in results:
        exp = expected[v["short"]]
        ci = v["_ci"]
        for key in ("rmse", "fs", "crps", "crpss"):
            if key not in ci or exp[key] == "":
                continue
            point = ci[key][0]
            want = float(exp[key])
            lo, hi = float(exp[key + "_lo"]), float(exp[key + "_hi"])
            in_ci = lo <= point <= hi
            n_total += 1
            n_in_ci += in_ci
            print("%-28s%-8s%-16.6f%-16.6f%-12.6f%s" %
                  (tag, key, point, want, point - want, "yes" if in_ci else "no"))
    print("-" * 100)
    print("%d/%d point estimates inside the expected 95%% CI" % (n_in_ci, n_total))


# ==================================================================== #
# main
# ==================================================================== #

def main():
    ap = argparse.ArgumentParser(
        description="Reproduce the manuscript's comparison table (Table 2).")
    ap.add_argument("--mode", choices=["stored", "regenerate"], default="stored",
                     help="stored (default): score prediction_samples/ against local ground "
                          "truth, no model/GPU. regenerate: run inference from the released "
                          "weights (original path).")
    ap.add_argument("--rows", default=None,
                     help="comma-separated short row names to run (see prediction_samples/"
                          "MANIFEST.tsv's 'method' column, or SPM); default: all rows.")
    ap.add_argument("--force", action="store_true",
                     help="regenerate mode only: always run inference, even if a completed "
                          "result already exists for a row. Writes to a fresh, run-specific "
                          "output directory -- never overwrites an existing result. Without "
                          "this flag, regenerate mode skips a row whose output already exists "
                          "(crash-resilient resume for long multi-row runs).")
    args = ap.parse_args()

    if args.rows:
        wanted = set(args.rows.split(","))
        unknown = wanted - set(SHORT_TO_ROW)
        if unknown:
            raise SystemExit("Unknown --rows entries: %s (known: %s)"
                              % (", ".join(sorted(unknown)), ", ".join(sorted(SHORT_TO_ROW))))
        rows = [r for r in ROWS if r["short"] in wanted]
    else:
        rows = ROWS

    if args.mode == "stored":
        # Rebuild-aware: build_ground_truth() itself determines which
        # of the 1998 canonical samples the local data can build and supplies
        # its own day_labels for whichever set that turns out to be --
        # run_stored_mode does the scoring, branch selection and comparison.
        gt = build_ground_truth()
        with tempfile.TemporaryDirectory(prefix="reproduce_comparison_") as scratch_dir:
            run_stored_mode(gt, rows, scratch_dir)
        return

    day_labels = _day_labels()
    # regenerate mode only needs local ground truth for the SPM row (every
    # other row gets ground truth from its own freshly generated npz) --
    # skip the (multi-minute) local-data build entirely when a --rows
    # selection excludes SPM.
    needs_gt = any(r.get("spm") for r in rows)
    gt = build_ground_truth() if needs_gt else None

    results = []
    with tempfile.TemporaryDirectory(prefix="reproduce_comparison_") as scratch_dir:
        checksums = _checksums()
        scored_by_tag = {}
        for row in rows:
            if row.get("spm"):
                scored = spm_scored(gt)
            else:
                scored = run_row_regenerate(row, checksums, scratch_dir, force=args.force)
            scored["short"] = row["short"]
            scored_by_tag[row["tag"]] = add_ci(scored, day_labels)
        results = [(row["tag"], scored_by_tag[row["tag"]]) for row in rows]

    print_table(results)
    expected = _load_expected(PRED_EXPECTED)
    compare_regenerate(results, expected)


if __name__ == "__main__":
    main()
