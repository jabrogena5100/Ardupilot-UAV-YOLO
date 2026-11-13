#!/usr/bin/env python3
"""Cluster-level statistics for the external (FASDD_UAV, FireMan) evaluation. numpy only; no model, no image.

Frames of one video, and frames of near-identical scenes, are strongly correlated, so a frame (or a video) is not an independent
observation. The unit here is the scene cluster. Per-(family, seed, cluster) scores come from `eval/metrics.py::evaluate` through
`cluster_scores`; this module only aggregates them.

  dhash_clusters       connected components of dHash64 hashes within a Hamming radius (FASDD scene clusters)
  cluster_scores       call eval.metrics.evaluate once per cluster -> {cluster: metric}
  cluster_bootstrap    percentile CI for each family and for the paired difference between two families (same resampled clusters)
  sign_flip_test       exact (n <= 20) or seeded Monte Carlo sign-flip test of the paired cluster differences
  leave_one_cluster_out range of the paired mean difference when each cluster is removed in turn
  effective_n          integrated-autocorrelation effective sample size of a series (documents the within-video correlation)

Conventions (fixed before any external result exists):
  * Scores are arrays shaped (n_seeds, n_clusters); a cluster without the metric (e.g. no smoke box) is NaN and is skipped for that
    family. Within every bootstrap replicate a family's score = mean over its seeds, then macro-average over the resampled clusters.
  * Seed variation is NOT part of the interval; `seed_sd` is reported next to it (three seeds do not support a significance claim).
  * The same resampled cluster indices are used for every family in a replicate, so differences are paired.
"""
from __future__ import annotations

import itertools
from typing import Callable, Dict, Iterable, List, Optional, Sequence

import numpy as np


def _popcount(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.uint64)
    if hasattr(np, "bitwise_count"):
        return np.bitwise_count(x).astype(np.int64)
    out = np.zeros(x.shape, np.int64)
    for s in range(64):
        out += ((x >> np.uint64(s)) & np.uint64(1)).astype(np.int64)
    return out


def dhash_clusters(hashes: Sequence[int], radius: int) -> List[int]:
    """Cluster id (0..k-1, in order of first appearance) for each 64-bit hash: images within `radius` bits are linked, components are clusters."""
    h = np.array(hashes, dtype=np.uint64)
    n = len(h)
    parent = list(range(n))

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for i in range(n):
        d = _popcount(h[i + 1:] ^ h[i])
        for j in np.where(d <= radius)[0]:
            ra, rb = find(i), find(i + 1 + int(j))
            if ra != rb:
                parent[ra] = rb
    ids, out = {}, []
    for i in range(n):
        out.append(ids.setdefault(find(i), len(ids)))
    return out


def cluster_scores(preds: Dict[str, np.ndarray], gts: Dict[str, np.ndarray], cluster_of: Dict[str, str],
                   metric: str = "mAP50", evaluate: Optional[Callable] = None) -> Dict[str, float]:
    """{cluster: metric} using the one metric code path. A cluster with no ground-truth box gets NaN for AP metrics."""
    if evaluate is None:
        from eval.metrics import evaluate as evaluate  # imported lazily so this module has no dependency at import time
    groups: Dict[str, List[str]] = {}
    for name in gts:
        groups.setdefault(cluster_of[name], []).append(name)
    out = {}
    for c, names in sorted(groups.items()):
        sub_g = {n: gts[n] for n in names}
        if not any(len(np.asarray(g).reshape(-1, 5)) for g in sub_g.values()):
            out[c] = float("nan")
            continue
        r = evaluate({n: preds.get(n, np.zeros((0, 6))) for n in names}, sub_g)
        v = r[metric]
        out[c] = float(v) if v is not None else float("nan")
    return out


def _family_array(x) -> np.ndarray:
    a = np.asarray(x, float)
    return a[None, :] if a.ndim == 1 else a


def _score(arr: np.ndarray, idx: np.ndarray) -> float:
    """mean over seeds of the macro-average over the resampled clusters (NaN clusters skipped per seed)."""
    sub = arr[:, idx]
    with np.errstate(all="ignore"):
        per_seed = np.nanmean(sub, axis=1)
    return float(np.nanmean(per_seed))


def cluster_bootstrap(scores: Dict[str, np.ndarray], diff: Optional[Sequence[str]] = None, n_boot: int = 10000,
                      seed: int = 0, alpha: float = 0.05) -> dict:
    """scores[family] = (n_seeds, n_clusters). diff = (a, b) reports b - a with the same resamples. Returns point estimates, percentile
    CIs, seed SD and, for the difference, P(diff > 0) over the replicates."""
    fams = {k: _family_array(v) for k, v in scores.items()}
    n_c = {a.shape[1] for a in fams.values()}
    if len(n_c) != 1:
        raise ValueError("all families need the same clusters in the same order")
    n = n_c.pop()
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, n, size=(n_boot, n))
    full = np.arange(n)
    out = {"n_clusters": n, "n_boot": n_boot, "seed": seed, "alpha": alpha, "families": {}}
    reps = {k: np.array([_score(a, d) for d in draws]) for k, a in fams.items()}
    for k, a in fams.items():
        lo, hi = np.nanpercentile(reps[k], [100 * alpha / 2, 100 * (1 - alpha / 2)])
        with np.errstate(all="ignore"):
            seed_means = np.nanmean(a, axis=1)
        out["families"][k] = {"point": _score(a, full), "ci": [float(lo), float(hi)], "n_seeds": a.shape[0],
                              "seed_sd": float(np.std(seed_means, ddof=1)) if a.shape[0] > 1 else None, "n_clusters_with_value": int(np.isfinite(a).any(axis=0).sum())}
    if diff:
        a, b = diff
        d = reps[b] - reps[a]
        lo, hi = np.nanpercentile(d, [100 * alpha / 2, 100 * (1 - alpha / 2)])
        out["difference"] = {"definition": f"{b} minus {a}", "point": out["families"][b]["point"] - out["families"][a]["point"],
                             "ci": [float(lo), float(hi)], "p_greater_than_zero": float((d > 0).mean())}
    return out


def paired_cluster_differences(a, b) -> np.ndarray:
    """per-cluster b - a using each family's seed-mean; clusters missing in either family are dropped."""
    da = np.nanmean(_family_array(a), axis=0)
    db = np.nanmean(_family_array(b), axis=0)
    d = db - da
    return d[np.isfinite(d)]


def sign_flip_test(diffs: Sequence[float], seed: int = 0, n_mc: int = 200000) -> dict:
    """Two-sided test that the mean paired difference is 0, flipping the sign of each cluster difference. Exact for n <= 20."""
    d = np.asarray(diffs, float)
    d = d[np.isfinite(d)]
    n = len(d)
    if n == 0:
        return {"n": 0, "p": None, "method": "none"}
    obs = abs(d.mean())
    if n <= 20:
        signs = np.array(list(itertools.product((-1, 1), repeat=n)), float)
        stat = np.abs((signs * d).mean(axis=1))
        method = "exact"
    else:
        rng = np.random.default_rng(seed)
        # chunked so memory stays bounded (n_mc x n float64 is ~1 GB for ~700 clusters; the 2 GB laptop ran out of RAM)
        step = max(1, 4_000_000 // n)
        stat = np.concatenate([np.abs((rng.choice((-1.0, 1.0), size=(min(step, n_mc - i), n)) * d).mean(axis=1)) for i in range(0, n_mc, step)])
        method = "monte_carlo"
    p = float((np.sum(stat >= obs - 1e-12) + (0 if method == "exact" else 1)) / (len(stat) + (0 if method == "exact" else 1)))
    out = {"n": n, "mean_diff": float(d.mean()), "p": p, "method": method}
    if method == "monte_carlo":
        out["n_mc"] = len(stat)
        out["p_floor"] = 1.0 / (len(stat) + 1)          # smallest value the estimator can return; a Monte Carlo p is never 0
    return out


def format_p(res: dict, short: bool = False) -> str:
    """Human-readable p: exact tests with 4 significant digits; Monte Carlo as an estimate with its floor (never 'p = 0.0000')."""
    p = res.get("p")
    if p is None:
        return "n/a"
    if res.get("method") == "monte_carlo":
        floor = res["p_floor"]
        if p <= floor * (1 + 1e-9):
            if short:
                return f"p-hat = {p:.1e} (MC floor, n_mc = {res['n_mc']}; not 0)"
            return f"p-hat = {p:.1e} (Monte Carlo floor 1/(n_mc+1), n_mc = {res['n_mc']}; no draw reached the observed statistic, so p is at most of that order, not zero)"
        return f"p-hat = {p:.2g} (Monte Carlo, n_mc = {res['n_mc']})"
    return f"p = {p:.4g} (exact)"


def leave_one_cluster_out(diffs: Sequence[float]) -> dict:
    d = np.asarray(diffs, float)
    d = d[np.isfinite(d)]
    if len(d) < 2:
        return {"n": len(d), "min": None, "max": None, "all_same_sign": None}
    means = np.array([np.delete(d, i).mean() for i in range(len(d))])
    return {"n": len(d), "min": float(means.min()), "max": float(means.max()), "all_same_sign": bool((means > 0).all() or (means < 0).all())}


def effective_n(series: Sequence[float], max_lag: Optional[int] = None) -> dict:
    """N / (1 + 2 sum rho_k) with the sum truncated at the first lag where the autocorrelation drops below 0.05."""
    a = np.asarray(series, float)
    a = a[np.isfinite(a)]
    n = len(a)
    if n < 4 or a.std() < 1e-12:
        return {"n": n, "tau": None, "n_eff": None}
    x = (a - a.mean()) / a.std()
    s = 1.0
    for k in range(1, min(n // 2, max_lag or n // 2)):
        r = float(np.dot(x[:n - k], x[k:]) / n)
        if r < 0.05:
            break
        s += 2 * r
    return {"n": n, "tau": s, "n_eff": n / s}
