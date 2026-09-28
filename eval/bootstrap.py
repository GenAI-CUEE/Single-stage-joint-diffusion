"""
Block bootstrap for 95% confidence intervals, and two-sided central-interval
PICP/coverage, as used for the results in the paper.

The bootstrap resamples whole test days with replacement (default B=5000)
rather than individual rows, so it respects the within-day autocorrelation of
the 2-minute forecast windows. PICP coverage is the fraction of ground-truth
values falling inside an ensemble's own [alpha/2, 1-alpha/2] empirical
quantile interval.
"""
import numpy as np


def picp(members, gt, alpha):
    """Two-sided central-interval PICP. members [N,M,H] (ensemble draws),
    gt [N,H] -> (covered [N,H] bool, lo [N,H], hi [N,H]). Coverage at the
    nominal (1-alpha) level is covered.mean()."""
    lo = np.percentile(members, 100 * alpha / 2.0, axis=1)
    hi = np.percentile(members, 100 * (1.0 - alpha / 2.0), axis=1)
    covered = (gt >= lo) & (gt <= hi)
    return covered, lo, hi


def _rmse(se, idx):
    return float(np.sqrt(se[idx, :].mean()))


def point_estimate(se_model, se_spm, crps_model=None, ae_spm=None, idx=None):
    """RMSE and forecast skill (vs. SPM), and -- if crps_model/ae_spm are
    given -- CRPS and CRPSS, pooled over the given row indices (all rows if
    idx is None). RMSE/CRPS are pooled over samples and horizons together
    before any ratio is taken (the "convention B" pooling used throughout
    this codebase's scoring), not averaged per-horizon first."""
    idx = np.arange(se_model.shape[0]) if idx is None else idx
    rmse = _rmse(se_model, idx)
    fs = 1.0 - rmse / _rmse(se_spm, idx)
    out = {"rmse": rmse, "fs": fs}
    if crps_model is not None:
        crps = float(crps_model[idx, :].mean())
        crpss = 1.0 - crps / float(ae_spm[idx, :].mean())
        out["crps"] = crps
        out["crpss"] = crpss
    return out


def block_bootstrap_ci(se_model, se_spm, day_labels, crps_model=None, ae_spm=None,
                        B=5000, seed=20260806):
    """95% CI for RMSE/FS(/CRPS/CRPSS) by resampling whole days with
    replacement, B times. se_model/se_spm/crps_model/ae_spm are [N,H] arrays
    (squared error, CRPS, absolute error against the SPM baseline); day_labels
    is length-N, one label per row, grouping rows into the days that get
    resampled as a block. Returns dict[key] -> (point_estimate, ci_lo, ci_hi).
    """
    days = np.unique(day_labels)
    day_to_rows = {d: np.where(day_labels == d)[0] for d in days}
    n_days = len(days)
    rng = np.random.default_rng(seed)

    keys = ["rmse", "fs"] + (["crps", "crpss"] if crps_model is not None else [])
    boot = {k: np.empty(B) for k in keys}
    for b in range(B):
        drawn = rng.choice(days, size=n_days, replace=True)
        idx = np.concatenate([day_to_rows[d] for d in drawn])
        pe = point_estimate(se_model, se_spm, crps_model, ae_spm, idx)
        for k in keys:
            boot[k][b] = pe[k]

    point = point_estimate(se_model, se_spm, crps_model, ae_spm)
    out = {}
    for k in keys:
        lo, hi = np.percentile(boot[k], [2.5, 97.5])
        out[k] = (point[k], float(lo), float(hi))
    return out
