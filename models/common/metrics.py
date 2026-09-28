"""
xiong_metrics.py

Deterministic and probabilistic scoring for the Xiong et al. baseline, in the
same units and conventions as persistence_baselines.py / analyze_ensemble.py so
the numbers drop straight into the comparison table.

TWO THINGS THAT ARE EASY TO GET WRONG AND ARE HANDLED EXPLICITLY HERE
---------------------------------------------------------------------
1. RMSE IS POOLED. Never average per-batch RMSEs -- on heavy-tailed solar error
   distributions that understates the true value badly. Every function here
   takes full arrays and pools over the sample axis.
2. The forecast is a GAUSSIAN MIXTURE (Eq. 23), not a single Gaussian and not an
   equally-weighted sample cloud. CRPS is computed in closed form for the
   mixture; the moment-matched (mu_d, sigma_d) of Eqs. (24)-(25) are reported
   too, but scoring the moment-matched Gaussian instead of the mixture is a
   different (and generally worse) forecast -- don't confuse them.
"""

import numpy as np
from scipy.stats import norm


# --------------------------------------------------------------------------- #
# mixture aggregation, Eqs. (23)-(25)
# --------------------------------------------------------------------------- #
def aggregate_mixture(mu, sigma):
    """
    mu, sigma : [N, M, H]   per-video-sample predictive mean / std (W/m^2)
    returns   : mu_d [N,H], sigma_d [N,H]   moment-matched mixture (Eqs. 24-25)
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    mu_d = mu.mean(axis=1)
    var_d = (sigma ** 2 + mu ** 2).mean(axis=1) - mu_d ** 2
    return mu_d, np.sqrt(np.clip(var_d, 1e-12, None))


# --------------------------------------------------------------------------- #
# deterministic
# --------------------------------------------------------------------------- #
def pooled_rmse(pred, gt):
    """[N,H],[N,H] -> [H]. Pooled over N, per horizon."""
    return np.sqrt(np.mean((np.asarray(pred, np.float64) - np.asarray(gt, np.float64)) ** 2,
                           axis=0))


def pooled_mae(pred, gt):
    return np.mean(np.abs(np.asarray(pred, np.float64) - np.asarray(gt, np.float64)), axis=0)


def pooled_smape(pred, gt, eps=1.0):
    p, g = np.asarray(pred, np.float64), np.asarray(gt, np.float64)
    return 200.0 * np.mean(np.abs(p - g) / np.clip(np.abs(p) + np.abs(g), eps, None), axis=0)


def forecast_skill(rmse_model, rmse_ref):
    """FS = 1 - RMSE_model / RMSE_reference, per horizon."""
    return 1.0 - np.asarray(rmse_model, np.float64) / np.asarray(rmse_ref, np.float64)


# --------------------------------------------------------------------------- #
# probabilistic
# --------------------------------------------------------------------------- #
def _A(mu, sig):
    """A(mu, sigma^2) = 2 sigma phi(mu/sigma) + mu (2 Phi(mu/sigma) - 1)."""
    sig = np.clip(sig, 1e-8, None)
    z = mu / sig
    return 2.0 * sig * norm.pdf(z) + mu * (2.0 * norm.cdf(z) - 1.0)


def crps_gaussian(mu, sigma, y):
    """Closed-form CRPS of a single Gaussian. Arrays broadcast."""
    sigma = np.clip(sigma, 1e-8, None)
    z = (y - mu) / sigma
    return sigma * (z * (2 * norm.cdf(z) - 1) + 2 * norm.pdf(z) - 1.0 / np.sqrt(np.pi))


def _auto_chunk(M, H, budget=2e7):
    """Rows per chunk so the O(M^2) pairwise term stays inside `budget` floats."""
    return max(1, int(budget / max(M * M * H, 1)))


def crps_mixture(mu, sigma, y, weights=None, chunk=None):
    """
    Exact CRPS of an equally-weighted Gaussian mixture.

      CRPS = sum_i w_i A(y - mu_i, sigma_i)
             - 0.5 sum_i sum_j w_i w_j A(mu_i - mu_j, sqrt(sig_i^2 + sig_j^2))

    mu, sigma : [N, M, H]
    y         : [N, H]
    returns   : [N, H] per-sample CRPS (W/m^2). Chunked over N to bound memory
                (the pairwise term is O(N*M^2*H)).
    """
    mu = np.asarray(mu, np.float64)
    sigma = np.clip(np.asarray(sigma, np.float64), 1e-8, None)
    y = np.asarray(y, np.float64)
    N, M, H = mu.shape
    w = np.full(M, 1.0 / M) if weights is None else np.asarray(weights, np.float64)
    chunk = _auto_chunk(M, H) if chunk is None else chunk

    out = np.empty((N, H))
    for s in range(0, N, chunk):
        e = min(s + chunk, N)
        m, sg, yy = mu[s:e], sigma[s:e], y[s:e]
        t1 = np.einsum("i,nih->nh", w, _A(yy[:, None, :] - m, sg))
        dmu = m[:, :, None, :] - m[:, None, :, :]                 # [n,M,M,H]
        dsg = np.sqrt(sg[:, :, None, :] ** 2 + sg[:, None, :, :] ** 2)
        t2 = np.einsum("i,j,nijh->nh", w, w, _A(dmu, dsg))
        out[s:e] = t1 - 0.5 * t2
    return out


def crps_ensemble_fair(samples, y, chunk=None):
    """
    Fair (bias-corrected) empirical CRPS for a sample cloud -- matches the
    estimator already used in analyze_ensemble.py, for cross-checking.

    samples [N,M,H], y [N,H] -> [N,H]. Chunked over N: the pairwise term is
    O(N*M^2*H) and will happily ask for hundreds of GiB if M is large.
    """
    s = np.asarray(samples, np.float64)
    y = np.asarray(y, np.float64)
    N, M, H = s.shape
    chunk = _auto_chunk(M, H) if chunk is None else chunk
    out = np.empty((N, H))
    for a in range(0, N, chunk):
        b = min(a + chunk, N)
        ss, yy = s[a:b], y[a:b]
        t1 = np.abs(ss - yy[:, None, :]).mean(axis=1)
        t2 = np.abs(ss[:, :, None, :] - ss[:, None, :, :]).sum(axis=(1, 2)) / (2 * M * (M - 1))
        out[a:b] = t1 - t2
    return out


def mixture_quantile(mu, sigma, q, lo=None, hi=None, iters=60):
    """
    Quantile of an equally-weighted Gaussian mixture by bisection on the CDF.

    mu, sigma : [N, M, H]; q scalar -> [N, H]
    """
    mu = np.asarray(mu, np.float64)
    sigma = np.clip(np.asarray(sigma, np.float64), 1e-8, None)
    lo = (mu - 8 * sigma).min(axis=1) if lo is None else lo
    hi = (mu + 8 * sigma).max(axis=1) if hi is None else hi
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        cdf = norm.cdf((mid[:, None, :] - mu) / sigma).mean(axis=1)
        lo = np.where(cdf < q, mid, lo)
        hi = np.where(cdf < q, hi, mid)
    return 0.5 * (lo + hi)


def interval_score(lower, upper, y, alpha):
    """IS_alpha = (u-l) + (2/alpha)(l-y)1{y<l} + (2/alpha)(y-u)1{y>u}."""
    return ((upper - lower)
            + (2.0 / alpha) * np.clip(lower - y, 0, None)
            + (2.0 / alpha) * np.clip(y - upper, 0, None))


def mwis(mu, sigma, y, alphas=(0.02, 0.05, 0.1, 0.2, 0.5)):
    """
    Mean weighted interval score (the paper's MWIS), averaged over horizons.

      WIS = 1/(K + 1/2) [ 1/2 |y - median| + sum_k (alpha_k/2) IS_{alpha_k} ]

    Returns (per-horizon WIS [H], per-horizon coverage dict).
    """
    med = mixture_quantile(mu, sigma, 0.5)
    total = 0.5 * np.abs(y - med)
    coverage = {}
    for a in alphas:
        lo = mixture_quantile(mu, sigma, a / 2.0)
        hi = mixture_quantile(mu, sigma, 1.0 - a / 2.0)
        total = total + (a / 2.0) * interval_score(lo, hi, y, a)
        coverage[1 - a] = float(np.mean((y >= lo) & (y <= hi)))
    wis = total / (len(alphas) + 0.5)
    return wis.mean(axis=0), coverage


def sample_from_mixture(mu, sigma, n_draws, rng=None):
    """
    Draw a sample cloud [N, n_draws, H] from the mixture, so the existing
    analyze_ensemble.py (rank histograms, spread-skill, fair CRPS) can be run
    on this baseline unchanged.
    """
    rng = np.random.default_rng() if rng is None else rng
    mu = np.asarray(mu, np.float64)
    sigma = np.asarray(sigma, np.float64)
    N, M, H = mu.shape
    comp = rng.integers(0, M, size=(N, n_draws, H))
    n_idx = np.arange(N)[:, None, None]
    h_idx = np.arange(H)[None, None, :]
    m = mu[n_idx, comp, h_idx]
    s = sigma[n_idx, comp, h_idx]
    return m + s * rng.standard_normal(size=(N, n_draws, H))


def rank_histogram(samples, y):
    """Verification-rank histogram, pooled over samples and horizons."""
    below = (np.asarray(samples) < np.asarray(y)[:, None, :]).sum(axis=1).reshape(-1)
    M = samples.shape[1]
    h = np.bincount(below, minlength=M + 1).astype(np.float64)
    return h / h.sum()


# --------------------------------------------------------------------------- #
# one-call report
# --------------------------------------------------------------------------- #
def full_report(mu, sigma, gt, spm, name="Xiong et al.", alphas=(0.02, 0.05, 0.1, 0.2, 0.5),
                n_draws=100, seed=0):
    """
    mu, sigma : [N,M,H] in W/m^2   sigma per mixture component
    gt        : [N,H]   W/m^2
    spm       : [N,H]   smart-persistence point forecast, for FS / CRPSS
    """
    rng = np.random.default_rng(seed)
    mu_d, sigma_d = aggregate_mixture(mu, sigma)

    rmse = pooled_rmse(mu_d, gt)
    mae = pooled_mae(mu_d, gt)
    smape = pooled_smape(mu_d, gt)
    spm_rmse = pooled_rmse(spm, gt)
    spm_mae = pooled_mae(spm, gt)
    fs = forecast_skill(rmse, spm_rmse)

    crps = crps_mixture(mu, sigma, gt).mean(axis=0)
    crpss = 1.0 - crps / spm_mae          # deterministic baseline CRPS == MAE
    wis, cov = mwis(mu, sigma, gt, alphas=alphas)

    S = sample_from_mixture(mu, sigma, n_draws, rng=rng)
    rh = rank_histogram(S, gt)

    rep = dict(name=name, rmse=rmse, mae=mae, smape=smape, fs=fs,
               spm_rmse=spm_rmse, spm_mae=spm_mae, crps=crps, crpss=crpss,
               wis=wis, coverage=cov, rank_hist=rh, mu_d=mu_d, sigma_d=sigma_d,
               samples=S)
    return rep


def print_report(rep):
    H = len(rep["rmse"])
    hdr = " ".join("i+%d" % (h + 1) for h in range(H))
    p = lambda tag, v, f="%5.1f": print("  %-12s" % tag, " ".join(f % x for x in v),
                                        " | mean " + (f % np.mean(v)))
    print("\n=== %s ===" % rep["name"])
    print("  horizon     ", hdr)
    p("RMSE", rep["rmse"])
    p("MAE", rep["mae"])
    p("sMAPE %", rep["smape"])
    p("SPM RMSE", rep["spm_rmse"])
    p("FS", rep["fs"], "%+5.2f")
    p("CRPS", rep["crps"])
    p("CRPSS", rep["crpss"], "%+5.2f")
    p("WIS", rep["wis"])
    print("  coverage    ", "  ".join("%.0f%%: %.3f" % (100 * k, v)
                                      for k, v in sorted(rep["coverage"].items())))
    rh = rep["rank_hist"]
    print("  rank hist   extremes %.3f (flat would be %.3f) -> %s"
          % (rh[0] + rh[-1], 2.0 / len(rh),
             "UNDER-dispersed" if (rh[0] + rh[-1]) > 2.5 * (2.0 / len(rh))
             else ("over-dispersed" if (rh[0] + rh[-1]) < 1.2 * (2.0 / len(rh))
                   else "roughly calibrated")))
