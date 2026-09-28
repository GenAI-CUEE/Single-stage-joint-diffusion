"""
CRPS and skill scores as used for the results in the paper.

Every number is recomputed straight from a model's raw saved forecast arrays,
reusing crps_ensemble_fair (models/common/metrics.py) -- the same function
every model's own forecast() method uses -- so results are guaranteed
consistent with the rest of the repo, not a reimplementation that could
silently drift. The smart-persistence baseline (SPM) is always reconstructed
from each run's own saved k_origin/iclr_y arrays, never taken from a stored
"SPM" field.

TWO ROW "kind"s
---------------
point     A deterministic single-point forecast. CRPS == MAE exactly
          (deterministic forecast, M=1 point mass).
ensemble  A genuine multi-member ensemble. CRPS uses the fair/unbiased
          ensemble estimator (crps_ensemble_fair).

Optional per-row keys (all have sensible defaults -- see score_point /
score_ensemble):
  gt_key            ground-truth array name (default "GT")
  pred_key          point-forecast array name (default: the array's own
                    "pred", else S.mean(1))
  sqerr_key         precomputed per-sample squared-error array -- use INSTEAD
                    of pred_key when the array stores sqerr directly and no
                    raw prediction array exists
  abserr_key        precomputed per-sample absolute-error array, paired with
                    sqerr_key for CRPS/CRPSS -- omit to skip CRPS/CRPSS for a
                    row entirely
  signederr_key     precomputed per-sample signed-error array, used INSTEAD
                    of pred_key/pred to compute Bias when sqerr_key is used
                    and no raw prediction array exists
  members_key       ensemble-members array name (default "ens_ghi"),
                    kind: ensemble only
  point_key         override the ensemble's point forecast (mean of members
                    by default) with a separate array -- e.g. a closed-form
                    mixture mean, while members_key still supplies the raw
                    draws for CRPS
  spm_source        path to a DIFFERENT saved array file to reconstruct SPM
                    (k_origin/iclr_y) from, for arrays that don't carry
                    their own
  omit              list of output keys to force to None even if computable

Both functions also return a few leading-underscore arrays alongside the
scalar summaries above -- _se_model/_se_spm [N,H] and, where CRPS is
computable, _crps_model/_ae_spm [N,H] (score_ensemble also returns _members/
_gt [N,M,H]/[N,H]). These are the exact per-sample arrays the scalars above
are pooled from, not additional arithmetic -- eval/bootstrap.py's block
bootstrap and PICP need the per-sample arrays, not just the pooled scalar.
"""
import numpy as np

from models.common.metrics import crps_ensemble_fair


def _arr(d, key):
    return np.asarray(d[key], dtype=np.float64)


def _reconstruct_spm(d):
    """SPM = k_origin * clear-sky irradiance. Never trust a stored 'SPM'
    field -- that field is not kept in sync with the actual baseline."""
    k = _arr(d, "k_origin")
    iclr = np.clip(_arr(d, "iclr_y"), 1.0, None)
    return k[:, None] * iclr


def _spm_for(row, d):
    if "spm_source" in row:
        spm_d = np.load(row["spm_source"], allow_pickle=True)
        return _reconstruct_spm(spm_d)
    return _reconstruct_spm(d)


def score_point(row):
    """Deterministic single-point direction. Returns a dict with whichever of
    rmse/fs/crps/crpss_convA/crpss_convB/bias are computable from the given
    keys."""
    d = np.load(row["npz"], allow_pickle=True)
    gt = _arr(d, row.get("gt_key", "GT"))
    pred = None

    if "sqerr_key" in row:
        sqerr = _arr(d, row["sqerr_key"])
    else:
        pred_key = row.get("pred_key")
        if pred_key:
            pred = _arr(d, pred_key)
        elif "pred" in d.files:
            pred = _arr(d, "pred")
        else:
            pred = _arr(d, "S").mean(axis=1)
        sqerr = (pred - gt) ** 2

    rmse_pooled = float(np.sqrt(sqerr.mean()))
    spm = _spm_for(row, d)
    spm_sqerr = (spm - gt) ** 2
    spm_rmse_pooled = float(np.sqrt(spm_sqerr.mean()))
    out = {"rmse": rmse_pooled, "fs": 1.0 - rmse_pooled / spm_rmse_pooled,
           "_se_model": sqerr, "_se_spm": spm_sqerr}

    if "abserr_key" in row:
        abserr = _arr(d, row["abserr_key"])
    elif pred is not None:
        abserr = np.abs(pred - gt)
    else:
        abserr = None

    if abserr is not None:
        spm_abserr = np.abs(spm - gt)                      # [N,H]
        mae_h = abserr.mean(axis=0)                        # [H]
        spm_mae_h = spm_abserr.mean(axis=0)                # [H]
        crpss_h = 1.0 - mae_h / spm_mae_h                  # deterministic: CRPS == MAE
        out["crps"] = float(mae_h.mean())
        # convA = mean of the per-horizon ratio (legacy); convB = pooled ratio of pooled
        # means (the manuscript convention) -- these diverge whenever spm_mae_h is
        # horizon-heterogeneous.
        out["crpss_convA"] = float(crpss_h.mean())
        out["crpss_convB"] = 1.0 - mae_h.mean() / spm_mae_h.mean()
        out["_crps_model"] = abserr       # deterministic: per-sample CRPS == abs error
        out["_ae_spm"] = spm_abserr

    if "signederr_key" in row:
        out["bias"] = float(_arr(d, row["signederr_key"]).mean())
    elif pred is not None:
        out["bias"] = float((pred - gt).mean())

    for key in row.get("omit", []):
        out.pop(key, None)
        if key == "crpss":
            out.pop("crpss_convA", None)
            out.pop("crpss_convB", None)
    return out


def score_ensemble(row):
    """Genuine multi-member ensemble direction (fair/unbiased CRPS
    estimator)."""
    d = np.load(row["npz"], allow_pickle=True)
    gt = _arr(d, row.get("gt_key", "GT"))
    members = _arr(d, row.get("members_key", "ens_ghi"))    # [N,M,H]
    point_key = row.get("point_key")
    pred = _arr(d, point_key) if point_key else members.mean(axis=1)

    sqerr = (pred - gt) ** 2
    rmse_pooled = float(np.sqrt(sqerr.mean()))
    spm = _spm_for(row, d)
    spm_sqerr = (spm - gt) ** 2
    spm_rmse_pooled = float(np.sqrt(spm_sqerr.mean()))

    crps_model = crps_ensemble_fair(members, gt)             # [N,H]
    ae_spm = np.abs(spm - gt)                                # [N,H]
    crps_h = crps_model.mean(axis=0)                         # [H]
    spm_mae_h = ae_spm.mean(axis=0)
    crpss_h = 1.0 - crps_h / spm_mae_h

    out = {
        "rmse": rmse_pooled,
        "fs": 1.0 - rmse_pooled / spm_rmse_pooled,
        "crps": float(crps_h.mean()),
        # convA = mean of the per-horizon ratio (legacy); convB = pooled ratio (standard).
        "crpss_convA": float(crpss_h.mean()),
        "crpss_convB": 1.0 - crps_h.mean() / spm_mae_h.mean(),
        "bias": float((pred - gt).mean()),
        "_se_model": sqerr, "_se_spm": spm_sqerr,
        "_crps_model": crps_model, "_ae_spm": ae_spm,
        "_members": members, "_gt": gt,
    }
    for key in row.get("omit", []):
        out.pop(key, None)
        if key == "crpss":
            out.pop("crpss_convA", None)
            out.pop("crpss_convB", None)
    return out
