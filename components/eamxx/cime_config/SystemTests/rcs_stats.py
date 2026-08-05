#!/usr/bin/env python3

"""
Statistical comparison engine for the RCS (Reproducible Climate Statistics)
system test.

RCS asks a narrow question: given two ensembles of short EAMxx simulations
that differ only by the random seed used to perturb the initial condition,
is there evidence that the two ensembles were produced by *different* models?

What the sample actually is
---------------------------
Each ensemble member is one simulation. A member contributes a vector of
numbers per variable; how that vector is built is controlled by
``analysis_type``:

- ``spatiotemporal`` (default): area-weighted global mean at each output
  time, so a member contributes ``n_time`` values.
- ``temporal``: time mean at each column, so a member contributes ``n_col``
  values.
- ``member``: a single area-weighted, time-averaged value, so a member
  contributes exactly one value.

The crucial point -- and the reason this module tracks members explicitly
instead of dumping everything into one flat array -- is that **the values
contributed by a single member are not independent of each other**. Global
monthly means from one simulation are serially correlated and share a
seasonal cycle; column means from one simulation are spatially correlated.
The *members*, on the other hand, are genuinely independent draws, and under
the null hypothesis (same model, different seed) the members of the two
ensembles are exchangeable.

Calibration: how p-values are computed
--------------------------------------
- ``asymptotic`` (default): scipy's closed-form/asymptotic p-value applied to
  the pooled values. Fast, and the historical RCS behavior, but it treats
  every pooled value as an independent observation. Because the pooled values
  are correlated, the effective sample size is smaller than the nominal one
  and these p-values are **anti-conservative** (too eager to reject). The
  report prints an estimated effective sample size so the size of that
  inflation is visible.
- ``member``: the test statistic is recalibrated by permuting whole members
  between the two ensembles. This is exact under the exchangeability of
  members and makes no independence assumption about values *within* a
  member. It is the statistically defensible mode, but its resolution is
  bounded by the number of members: with ``n`` members per ensemble the
  smallest attainable p-value is about ``2 / C(2n, n)`` (0.029 for n=4,
  0.0079 for n=5, 1.6e-4 for n=8). The report states this bound and warns
  when the configured decision threshold is below it.

Test categories
---------------
1. DISTRIBUTION: ks, ad, cvm, epps, energy
2. LOCATION:     mw, ttest, brunner
3. SCALE:        levene, ansari, mood

Combining the per-variable verdicts
-----------------------------------
``global_test`` decides how many per-variable answers become one answer.

- ``variable_count`` (default): correct the p-values across variables and
  count the failures.
- ``calibrated_count``: judge the *number* of rejections against its own
  member-permutation null distribution. This accounts for correlation between
  variables -- which Bonferroni and Benjamini-Hochberg do not -- and, because
  it spends its resolution on a single p-value instead of one per variable, it
  remains usable at the small member counts RCS is normally run with. It is
  also the more sensitive choice against a diffuse change that nudges many
  variables slightly, which is what a climate-altering bug usually looks like.

Interpreting the outcome
------------------------
A variable that is not rejected has *not* been shown to be equivalent; it has
merely failed to be shown different. Two things help turn that into a
statement with content:

- ``power_analysis`` (on by default) reports, per variable, the smallest
  injected shift this configuration would actually have rejected.
- ``equivalence_margin`` runs a TOST (two one-sided tests) on the member-level
  means, so that a pass means "shown equivalent" rather than "not shown
  different".

Usage
-----
From the command line::

    rcs_stats.py /run/dir /base/dir --test_type ks
    rcs_stats.py /run/dir /base/dir --calibration member --json_output ~/rcs

From Python::

    from rcs_stats import run_stats_comparison
    comments, status = run_stats_comparison(run_dir, base_dir)

References
----------
scipy.stats: https://docs.scipy.org/doc/scipy/reference/stats.html
statsmodels multiple testing:
https://www.statsmodels.org/stable/generated/statsmodels.stats.multitest.multipletests.html
"""

# The module is long because each test and each diagnostic carries the
# explanation a reader needs to judge whether it is the right tool.
# pylint: disable=too-many-lines

import os
import re
import glob
import json
import math
import sys
import logging
import tempfile
import warnings
from abc import ABC, abstractmethod
from itertools import combinations

sys.path.append(os.path.join(os.path.dirname(__file__), "../../scripts"))

try:
    from utils import _ensure_pylib_impl

    _ensure_pylib_impl("xarray")
    _ensure_pylib_impl("dask")
    _ensure_pylib_impl("scipy")
    _ensure_pylib_impl("statsmodels", min_version="0.14.0")

    import numpy as np
    import xarray as xr
    from scipy import stats
    from statsmodels.stats.multitest import multipletests
except ImportError as e:
    raise ImportError(f"Could not ensure Python packages: {e}") from e

logger = logging.getLogger(__name__)


# ==========================================================
# Constants
# ==========================================================

#: Coordinate-like variables that are never candidates for testing.
SKIP_VARS = frozenset(
    {
        "time", "time_bnds", "time_bounds", "date", "datesec", "ndcur",
        "nscur", "nsteph", "lat", "lon", "ncol", "lev", "ilev", "hyam",
        "hybm", "hyai", "hybi", "P0", "area", "dyn_dof", "hyai_dyn",
    }
)

#: Default per-ensemble file glob; ``????`` marks the 4-digit instance number.
DEFAULT_FILE_PATTERN = "*.scream_????.h.AVERAGE.*.nc"

#: Default RNG seed so that permutation-based p-values are reproducible.
DEFAULT_SEED = 20250101

#: Enumerate member permutations exhaustively when there are no more than
#: this many; otherwise draw random ones.
MAX_EXACT_PERMUTATIONS = 20000

#: Denominator floor used when forming relative differences.
_TINY = 1.0e-30

TEST_FULL_NAMES = {
    "ks": "Kolmogorov-Smirnov",
    "ad": "Anderson-Darling",
    "cvm": "Cramer-von Mises",
    "epps": "Epps-Singleton",
    "energy": "Energy distance",
    "mw": "Mann-Whitney U",
    "ttest": "Welch's t-test",
    "brunner": "Brunner-Munzel",
    "levene": "Levene",
    "ansari": "Ansari-Bradley",
    "mood": "Mood",
}

ANALYSIS_FULL_NAMES = {
    "spatiotemporal": "area-weighted global mean per output time",
    "temporal": "time mean per column",
    "member": "one area-weighted, time-averaged value per member",
}

#: Map the user-facing correction names onto statsmodels method names.
CORRECTION_METHODS = {
    "bonferroni": ("bonferroni", "Bonferroni (controls family-wise error rate)"),
    "holm": ("holm", "Holm-Bonferroni (controls FWER, uniformly stronger than Bonferroni)"),
    "fdr": ("fdr_bh", "Benjamini-Hochberg (controls false discovery rate)"),
    "fdr_by": ("fdr_by", "Benjamini-Yekutieli (controls FDR under dependence)"),
    "none": (None, "none (no multiple-testing correction)"),
}


# ==========================================================
# Ensemble discovery
# ==========================================================


def _glob_chunk_to_regex(chunk):
    """Translate the literal part of a glob pattern into a regex fragment."""
    out = []
    for ch in chunk:
        if ch == "*":
            out.append("[^/]*")
        elif ch == "?":
            out.append("[^/]")
        else:
            out.append(re.escape(ch))
    return "".join(out)


def _instance_regex(pattern):
    """
    Build a regex that captures the instance number out of a file pattern.

    ``pattern`` must contain exactly one ``????`` placeholder, which is where
    the 4-digit instance number lives. Everything else is treated as an
    ordinary glob.
    """
    parts = pattern.split("????")
    if len(parts) != 2:
        raise ValueError(
            f"File pattern must contain exactly one '????' placeholder: {pattern}"
        )
    prefix, suffix = parts
    return re.compile(
        "^"
        + _glob_chunk_to_regex(prefix)
        + r"(?P<inst>\d{4})"
        + _glob_chunk_to_regex(suffix)
        + "$"
    )


def discover_instances(directory, pattern):
    """
    Find the ensemble members under ``directory``.

    Returns a dict mapping the 4-digit instance string to the sorted list of
    files belonging to that instance. Raises FileNotFoundError when nothing
    matches, since a silently empty ensemble is far worse than a hard error.
    """
    files = sorted(glob.glob(os.path.join(directory, pattern)))
    if not files:
        raise FileNotFoundError(
            f"No files matching '{pattern}' under '{directory}'"
        )

    rgx = _instance_regex(pattern)
    instances = {}
    for path in files:
        match = rgx.match(os.path.basename(path))
        if match is None:
            continue
        instances.setdefault(match.group("inst"), []).append(path)

    if not instances:
        raise FileNotFoundError(
            f"Found {len(files)} file(s) under '{directory}' matching "
            f"'{pattern}', but none carried a 4-digit instance number where "
            f"the '????' placeholder is. Check --run_file_pattern / "
            f"--base_file_pattern."
        )

    return {inst: sorted(paths) for inst, paths in sorted(instances.items())}


def open_ensemble(directory, pattern):
    """Open every member of an ensemble as a lazily-loaded xarray Dataset."""
    instances = discover_instances(directory, pattern)
    datasets = {}
    for inst, paths in instances.items():
        datasets[inst] = xr.open_mfdataset(
            paths,
            decode_times=False,
            data_vars="all",
            combine="by_coords" if len(paths) > 1 else "nested",
            concat_dim=None if len(paths) > 1 else "time",
        )
    return datasets


def close_ensemble(datasets):
    """
    Close every dataset in an ensemble.

    xarray caches open file handles keyed by path, so a caller that runs
    several comparisons in one process -- a driver script sweeping settings,
    or a test harness -- will otherwise both leak descriptors and risk reading
    a stale handle if a path is rewritten between calls. Closing is cheap and
    removes both problems.
    """
    for dataset in (datasets or {}).values():
        try:
            dataset.close()
        except (OSError, RuntimeError, AttributeError) as error:
            logger.debug("Could not close dataset: %s", error)


# ==========================================================
# Sample construction
# ==========================================================


def _vertical_reduce(var):
    """
    Collapse vertical dimensions with an unweighted mean.

    NOTE: this is a mass-unweighted average, and it can hide a difference that
    changes sign with height. Variables whose signal is confined to a few
    levels are therefore harder to detect than column-integrated ones. This is
    a deliberate simplification to keep one test per variable.
    """
    for dim in ("lev", "ilev"):
        if dim in var.dims:
            var = var.mean(dim=dim, skipna=True)
    return var


def get_area_weights(dataset):
    """Return the ``area`` field as an xarray DataArray, or None."""
    for name in ("area",):
        if name in dataset.variables:
            weights = dataset[name]
            if "ncol" in weights.dims:
                return weights.astype("float64")
    return None


def _weighted_spatial_mean(var, weights):
    """
    Area-weighted mean over ``ncol`` that renormalizes over valid points.

    The naive ``(var * weights).sum()`` is biased low wherever ``var`` is
    masked, because the weight of the masked cells is still in the
    denominator. Dividing by the weight actually used fixes that.
    """
    valid = var.notnull()
    numerator = (var.fillna(0.0) * weights).sum(dim="ncol", skipna=True)
    denominator = (weights * valid).sum(dim="ncol", skipna=True)
    return (numerator / denominator).where(denominator > 0)


def _member_values(dataset, var, analysis_type, weights):
    """Reduce one member's data for one variable down to a 1-D float array."""
    data = _vertical_reduce(dataset[var])

    if analysis_type == "temporal":
        if "time" not in data.dims:
            raise ValueError("no time dimension")
        values = data.mean(dim="time", skipna=True).values
        return np.asarray(values, dtype="float64").ravel()

    # spatiotemporal and member both start from a global mean time series
    if weights is not None and "ncol" in data.dims:
        series = _weighted_spatial_mean(data, weights)
    else:
        spatial_dims = [d for d in data.dims if d != "time"]
        series = data.mean(dim=spatial_dims, skipna=True) if spatial_dims else data

    values = np.asarray(series.values, dtype="float64").ravel()

    if analysis_type == "member":
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", "Mean of empty slice")
            values = np.array([np.nanmean(values)]) if values.size else values

    return values


class Sample:
    """
    One ensemble's data for one variable, with member structure preserved.

    ``rows`` is a list with one 1-D array per member. ``pooled`` is the
    concatenation, which is what the test statistics actually consume.
    """

    __slots__ = ("rows", "pooled", "member_means")

    def __init__(self, rows):
        self.rows = rows
        self.pooled = np.concatenate(rows) if rows else np.empty(0)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", "Mean of empty slice")
            self.member_means = np.array(
                [np.nanmean(r) if r.size else np.nan for r in rows]
            )

    @property
    def n_members(self):
        """Number of ensemble members contributing to this sample."""
        return len(self.rows)

    @property
    def size(self):
        """Number of pooled values."""
        return self.pooled.size


def _apply_common_mask(rows_run, rows_base):
    """
    Drop positions that are not finite in *every* member.

    A per-member ``isnan`` filter followed by truncation (what RCS used to do)
    silently compares position ``i`` of one member against a different
    physical location in another. Masking on the intersection keeps positions
    aligned across members, which is what the paired reduction assumes.
    """
    notes = []
    lengths = {r.size for r in rows_run + rows_base}
    if len(lengths) > 1:
        shortest = min(lengths)
        notes.append(
            f"members had unequal sample lengths {sorted(lengths)}; "
            f"truncated to {shortest}"
        )
        rows_run = [r[:shortest] for r in rows_run]
        rows_base = [r[:shortest] for r in rows_base]

    stacked = np.vstack(rows_run + rows_base)
    mask = np.isfinite(stacked).all(axis=0)
    n_dropped = int(mask.size - mask.sum())
    if n_dropped:
        notes.append(
            f"dropped {n_dropped}/{mask.size} positions that were not finite "
            f"in every member"
        )
    if not mask.any():
        raise ValueError("no position is finite across all members")

    return (
        [r[mask] for r in rows_run],
        [r[mask] for r in rows_base],
        notes,
    )


def build_samples(var, run_ens, base_ens, analysis_type, weights):
    """Build aligned Sample objects for one variable, or raise ValueError."""
    rows_run, rows_base = [], []
    for label, ensemble, rows in (
        ("run", run_ens, rows_run),
        ("baseline", base_ens, rows_base),
    ):
        for inst, dataset in sorted(ensemble.items()):
            if var not in dataset.variables:
                raise ValueError(
                    f"absent from {label} member {inst}"
                )
            rows.append(_member_values(dataset, var, analysis_type, weights))

    if not rows_run or not rows_base:
        raise ValueError("one of the ensembles has no members")

    rows_run, rows_base, notes = _apply_common_mask(rows_run, rows_base)
    return Sample(rows_run), Sample(rows_base), notes


# ==========================================================
# Descriptive statistics, effect sizes and diagnostics
# ==========================================================


def _describe(sample):
    """Summary statistics for one sample."""
    values = sample.pooled
    if values.size == 0:
        return {"n": 0, "n_members": sample.n_members}
    return {
        "n": int(values.size),
        "n_members": sample.n_members,
        "n_per_member": int(values.size // max(sample.n_members, 1)),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "std": float(np.std(values, ddof=1)) if values.size > 1 else float("nan"),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "q25": float(np.percentile(values, 25)),
        "q75": float(np.percentile(values, 75)),
        "member_mean_std": (
            float(np.std(sample.member_means, ddof=1))
            if sample.n_members > 1
            else float("nan")
        ),
    }


def _lag1_autocorrelation(rows):
    """Average lag-1 autocorrelation within members (0.0 if not estimable)."""
    correlations = []
    for row in rows:
        if row.size < 4:
            continue
        centered = row - row.mean()
        denominator = float(np.dot(centered, centered))
        if denominator <= _TINY:
            continue
        correlations.append(float(np.dot(centered[:-1], centered[1:])) / denominator)
    if not correlations:
        return 0.0
    return float(np.clip(np.mean(correlations), -0.99, 0.99))


def _effective_sample_size(sample):
    """
    Estimate how many independent observations the pooled sample is worth.

    Uses the standard AR(1) deflation ``n_eff = n * (1 - r) / (1 + r)`` applied
    within each member. Values contributed by different members *are*
    independent, so no deflation is applied across members. This is a
    diagnostic, not a correction: it exists so that an anti-conservative
    asymptotic p-value is at least visibly labelled as such.
    """
    rho = _lag1_autocorrelation(sample.rows)
    deflation = (1.0 - rho) / (1.0 + rho) if rho > 0 else 1.0
    per_member = [max(1.0, r.size * deflation) for r in sample.rows]
    return float(sum(per_member)), rho


def _effect_sizes(run, base):
    """
    Effect-size measures that do not depend on the sample size.

    ``snr`` is the one climate scientists usually care about: the difference
    in ensemble-mean divided by the baseline ensemble's own member-to-member
    spread. A statistically significant difference with snr << 1 is buried
    inside the ensemble's internal variability.
    """
    a, b = run.pooled, base.pooled
    mean_a, mean_b = float(np.mean(a)), float(np.mean(b))
    std_a = float(np.std(a, ddof=1)) if a.size > 1 else float("nan")
    std_b = float(np.std(b, ddof=1)) if b.size > 1 else float("nan")

    scale = (abs(mean_a) + abs(mean_b)) / 2.0
    pooled_std = float("nan")
    if a.size > 1 and b.size > 1:
        dof = a.size + b.size - 2
        pooled_var = ((a.size - 1) * std_a**2 + (b.size - 1) * std_b**2) / dof
        pooled_std = math.sqrt(pooled_var) if pooled_var > 0 else float("nan")

    spread = (
        float(np.std(base.member_means, ddof=1)) if base.n_members > 1 else float("nan")
    )

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", "ks_2samp: Exact calculation unsuccessful")
        ks_distance = float(stats.ks_2samp(a, b).statistic)

    return {
        "mean_diff": mean_a - mean_b,
        "relative_diff": abs(mean_a - mean_b) / (scale + _TINY),
        "mean_diff_pct": 100.0 * (mean_a - mean_b) / (abs(mean_b) + _TINY),
        "median_diff": float(np.median(a) - np.median(b)),
        "std_ratio": std_a / (std_b + _TINY),
        "cohens_d": (
            (mean_a - mean_b) / pooled_std if pooled_std and pooled_std > 0 else float("nan")
        ),
        "snr": (
            abs(mean_a - mean_b) / spread if spread and spread > 0 else float("nan")
        ),
        "ks_distance": ks_distance,
    }


# ==========================================================
# Statistical tests
# ==========================================================


class StatisticalTest(ABC):
    """
    A two-sample statistic plus, where scipy offers one, an asymptotic p-value.

    Subclasses implement ``_statistic_and_pvalue``. The p-value may be None,
    in which case the caller must use permutation calibration.
    """

    #: Minimum number of pooled values per sample for the test to be defined.
    min_samples = 2

    #: True when large |statistic| means "more different" in both directions,
    #: which is what the permutation calibration needs to know.
    two_sided_statistic = True

    #: False when scipy offers no null distribution, so that an incompatible
    #: calibration can be rejected before any data is read.
    provides_asymptotic_pvalue = True

    def __init__(self, alpha):
        self.alpha = alpha

    @abstractmethod
    def _statistic_and_pvalue(self, a, b):
        """Return ``(statistic, pvalue_or_None)``."""

    def evaluate(self, a, b):
        """Compute the statistic and, if available, the asymptotic p-value."""
        if a.size < self.min_samples or b.size < self.min_samples:
            raise ValueError(
                f"{type(self).__name__} needs at least {self.min_samples} "
                f"values per sample (got {a.size} and {b.size})"
            )
        statistic, pvalue = self._statistic_and_pvalue(a, b)
        return float(statistic), (None if pvalue is None else float(pvalue))

    def statistic(self, a, b):
        """Statistic only; used inside the permutation loop."""
        return self._statistic_and_pvalue(a, b)[0]


# --- Distribution tests ---------------------------------------------------


class KSTest(StatisticalTest):
    """
    Kolmogorov-Smirnov two-sample test.

    Sensitive to the largest vertical gap between the two empirical CDFs, so
    it responds mostly to shifts in the bulk of the distribution and is weak
    in the tails. Reasonable default.
    """

    def _statistic_and_pvalue(self, a, b):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", "ks_2samp: Exact calculation unsuccessful")
            result = stats.ks_2samp(a, b)
        return result.statistic, result.pvalue


class AndersonDarlingTest(StatisticalTest):
    """
    Anderson-Darling k-sample test, weighted towards the distribution tails.

    scipy's ``significance_level`` is already a p-value and is clipped to
    [0.001, 0.25]; that clipping makes it useless against a corrected
    threshold, so a permutation p-value is requested whenever scipy supports
    it and the returned value is honestly flagged as clipped otherwise.
    """

    min_samples = 3

    def _statistic_and_pvalue(self, a, b):
        if np.unique(a).size < 2 and np.unique(b).size < 2:
            raise ValueError("both samples are constant")

        kwargs = {"midrank": True}
        if hasattr(stats, "PermutationMethod"):
            kwargs["method"] = stats.PermutationMethod(
                n_resamples=999,
                rng=np.random.default_rng(DEFAULT_SEED),
            )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            try:
                result = stats.anderson_ksamp([a, b], **kwargs)
            except TypeError:
                result = stats.anderson_ksamp([a, b], midrank=True)
        # significance_level IS the p-value (not a percentage).
        return result.statistic, result.significance_level


class CramerVonMisesTest(StatisticalTest):
    """
    Cramer-von Mises two-sample test.

    Integrates the squared CDF gap rather than taking its maximum, so it uses
    the whole distribution instead of one point. Generally a bit more powerful
    than KS against diffuse differences.
    """

    def _statistic_and_pvalue(self, a, b):
        result = stats.cramervonmises_2samp(a, b)
        return result.statistic, result.pvalue


class EppsSingletonTest(StatisticalTest):
    """
    Epps-Singleton test, comparing empirical characteristic functions.

    Unlike KS/CvM it is valid for discrete data and it picks up location and
    scale differences together. Needs a reasonable number of observations.
    """

    min_samples = 5

    def _statistic_and_pvalue(self, a, b):
        result = stats.epps_singleton_2samp(a, b)
        return result.statistic, result.pvalue


class EnergyDistanceTest(StatisticalTest):
    """
    Energy distance between the two empirical distributions.

    Consistent against *any* difference in distribution, but scipy provides no
    null distribution, so this statistic is only meaningful with permutation
    calibration.
    """

    provides_asymptotic_pvalue = False

    def _statistic_and_pvalue(self, a, b):
        return stats.energy_distance(a, b), None


# --- Location tests -------------------------------------------------------


class MannWhitneyUTest(StatisticalTest):
    """
    Mann-Whitney U. Distribution-free test of stochastic dominance; in
    practice, a rank-based comparison of central tendency.
    """

    two_sided_statistic = False

    def _statistic_and_pvalue(self, a, b):
        result = stats.mannwhitneyu(a, b, alternative="two-sided")
        return result.statistic, result.pvalue

    def statistic(self, a, b):
        # U is bounded by n1*n2; recenter so that "far from the middle in
        # either direction" maps to a large positive number.
        u_statistic = stats.mannwhitneyu(a, b, alternative="two-sided").statistic
        return abs(u_statistic - a.size * b.size / 2.0)


class TTest(StatisticalTest):
    """Welch's unequal-variance t-test on the means."""

    def _statistic_and_pvalue(self, a, b):
        result = stats.ttest_ind(a, b, equal_var=False)
        return result.statistic, result.pvalue


class BrunnerMunzelTest(StatisticalTest):
    """
    Brunner-Munzel test of stochastic equality. Unlike Mann-Whitney it does
    not assume equal shapes, so it stays valid when the variances differ.
    """

    min_samples = 3

    def _statistic_and_pvalue(self, a, b):
        result = stats.brunnermunzel(a, b)
        return result.statistic, result.pvalue


# --- Scale tests ----------------------------------------------------------


class LeveneTest(StatisticalTest):
    """Levene's test (median-centered) for equality of variances."""

    def _statistic_and_pvalue(self, a, b):
        statistic, pvalue = stats.levene(a, b, center="median")
        return statistic, pvalue


class AnsariBradleyTest(StatisticalTest):
    """
    Ansari-Bradley rank test for equal scale. Assumes the two samples share a
    location, so pair it with a location test rather than using it alone.
    """

    def _statistic_and_pvalue(self, a, b):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            result = stats.ansari(a, b)
        return result.statistic, result.pvalue


class MoodTest(StatisticalTest):
    """Mood's rank test for a difference in scale."""

    min_samples = 3

    def _statistic_and_pvalue(self, a, b):
        result = stats.mood(a, b)
        return result.statistic, result.pvalue


TEST_REGISTRY = {
    "ks": KSTest,
    "ad": AndersonDarlingTest,
    "cvm": CramerVonMisesTest,
    "epps": EppsSingletonTest,
    "energy": EnergyDistanceTest,
    "mw": MannWhitneyUTest,
    "ttest": TTest,
    "brunner": BrunnerMunzelTest,
    "levene": LeveneTest,
    "ansari": AnsariBradleyTest,
    "mood": MoodTest,
}

TEST_ALIASES = {
    "kolmogorov-smirnov": "ks",
    "anderson-darling": "ad",
    "cm": "cvm",
    "cramer": "cvm",
    "cramer-von-mises": "cvm",
    "epps-singleton": "epps",
    "energy-distance": "energy",
    "mannwhitney": "mw",
    "mann-whitney": "mw",
    "t-test": "ttest",
    "welch": "ttest",
    "brunnermunzel": "brunner",
    "brunner-munzel": "brunner",
    "ansari-bradley": "ansari",
}


def normalize_test_type(test_type):
    """Resolve aliases and validate a test identifier."""
    key = TEST_ALIASES.get(test_type.lower(), test_type.lower())
    if key not in TEST_REGISTRY:
        raise ValueError(
            f"Unknown test type: '{test_type}'\n"
            f"  Distribution: ks, ad, cvm, epps, energy\n"
            f"  Location:     mw, ttest, brunner\n"
            f"  Scale:        levene, ansari, mood"
        )
    return key


def get_test(test_type, alpha=0.01):
    """Instantiate the requested statistical test."""
    return TEST_REGISTRY[normalize_test_type(test_type)](alpha)


# ==========================================================
# Calibration
# ==========================================================


def _n_member_partitions(n_run, n_base):
    """Number of ways to split the pooled members back into two groups."""
    return math.comb(n_run + n_base, n_run)


def decision_threshold(alpha, correction_method, n_tests):
    """
    The smallest p-value threshold any variable can be judged against.

    Bonferroni and Holm both bottom out at ``alpha / n``; so does
    Benjamini-Hochberg, whose most stringent critical value (the one applied
    to the smallest p-value) is also ``alpha / n``. Without a correction the
    threshold is just ``alpha``.
    """
    if correction_method == "none" or n_tests < 1:
        return alpha
    if correction_method == "fdr_by":
        harmonic = sum(1.0 / (i + 1) for i in range(n_tests))
        return alpha / (n_tests * harmonic)
    return alpha / n_tests


def permutation_resolution(n_run, n_base, n_resamples):
    """
    Smallest p-value a member permutation test can produce.

    Exhaustive enumeration bottoms out at ``2 / C(n1+n2, n1)`` because a
    partition and its complement give the same two-sided statistic; random
    sampling bottoms out at ``1 / (n_resamples + 1)``.
    """
    total = _n_member_partitions(n_run, n_base)
    if total <= MAX_EXACT_PERMUTATIONS:
        return 2.0 / total, int(total), True
    return 1.0 / (n_resamples + 1), int(n_resamples), False


def build_assignments(n_run, n_base, n_resamples, rng):
    """
    Build the set of member re-assignments used to calibrate by permutation.

    The list is built **once** and reused for every variable. That is not just
    an optimization: applying the same member re-assignment to all variables
    simultaneously is what lets the global test in ``calibrated_count_test``
    see the correlation *between* variables. Recomputing an independent
    permutation per variable would destroy exactly the structure that makes
    the global null distribution correct.

    Element 0 is always the identity assignment -- the members as they
    actually are -- so the observed statistic is one of the permuted ones and
    needs no special casing.
    """
    n_total = n_run + n_base
    total = _n_member_partitions(n_run, n_base)
    if total <= MAX_EXACT_PERMUTATIONS:
        # combinations() yields (0, 1, ..., n_run-1) first, which is exactly
        # the identity assignment.
        return [tuple(sel) for sel in combinations(range(n_total), n_run)], True

    assignments = [tuple(range(n_run))]
    assignments.extend(
        tuple(sorted(rng.permutation(n_total)[:n_run].tolist()))
        for _ in range(n_resamples)
    )
    return assignments, False


def permuted_statistics(test, run, base, assignments):
    """
    Evaluate the test statistic under every member assignment.

    Returns an array aligned with ``assignments``; entry 0 is the observed
    statistic. Assignments whose statistic cannot be computed become NaN
    rather than silently shifting the others.
    """
    rows = run.rows + base.rows
    n_total = len(rows)
    values = np.full(len(assignments), np.nan)

    for j, selection in enumerate(assignments):
        chosen = set(selection)
        left = np.concatenate([rows[i] for i in range(n_total) if i in chosen])
        right = np.concatenate([rows[i] for i in range(n_total) if i not in chosen])
        try:
            values[j] = abs(test.statistic(left, right))
        except (ValueError, ZeroDivisionError, FloatingPointError):
            continue

    if not np.isfinite(values).any():
        raise ValueError("no member permutation produced a usable statistic")
    return values


def permutation_pvalues(values):
    """
    Turn permuted statistics into a permutation p-value for each assignment.

    ``p_j`` is the fraction of assignments whose statistic is at least as
    extreme as assignment ``j``'s. Entry 0 is therefore the ordinary
    permutation p-value of the observed data, and the remaining entries are
    the p-values the *same* pipeline would have produced had the members been
    grouped differently -- which is what the global test needs.

    Ranking is used rather than an asymptotic formula so that this works
    identically for every statistic, including those (like energy distance)
    that have no closed-form null distribution.
    """
    finite = np.isfinite(values)
    n_valid = int(finite.sum())
    pvalues = np.ones(values.size)
    # rankdata on the negated statistic with method="max" counts, for each
    # entry, how many entries are >= it (ties included).
    pvalues[finite] = stats.rankdata(-values[finite], method="max") / n_valid
    return pvalues, n_valid


def calibrated_count_test(perm_pvalues, global_alpha):
    """
    Judge the ensemble of per-variable tests as a whole.

    Instead of asking "did any single variable reject after a Bonferroni-style
    correction", this asks "is the *number* of rejecting variables larger than
    member exchangeability can explain". The null distribution of that count is
    obtained by re-grouping the members and re-counting, so it automatically
    accounts for the correlation between variables -- which Bonferroni and
    Benjamini-Hochberg both ignore, and which Benjamini-Yekutieli only handles
    by giving up a lot of power.

    Two properties make this worth having:

    - It is far more sensitive to a diffuse change that nudges many variables
      slightly than a per-variable correction is, and that is the signature of
      a climate-altering bug.
    - It needs only one p-value, so it is judged against ``global_alpha``
      rather than ``global_alpha / n_variables``. With 4+4 members the finest
      attainable p-value is 0.029, which is below 0.05 -- so unlike the
      per-variable permutation test, this one can actually reject at the
      ensemble sizes RCS is usually run with.

    ``perm_pvalues`` maps a variable to its per-assignment p-value array; all
    arrays must come from the same assignment list.
    """
    if not perm_pvalues:
        return None

    matrix = np.vstack(list(perm_pvalues.values()))
    counts = (matrix < global_alpha).sum(axis=0)
    observed = int(counts[0])
    n_perm = counts.size
    pvalue = float((counts >= observed).sum() / n_perm)

    return {
        "n_variables": int(matrix.shape[0]),
        "n_assignments": int(n_perm),
        "per_variable_alpha": global_alpha,
        "observed_rejections": observed,
        "null_rejections_median": float(np.median(counts)),
        "null_rejections_p95": float(np.percentile(counts, 95)),
        "null_rejections_max": int(counts.max()),
        "expected_by_chance": float(matrix.shape[0] * global_alpha),
        "pvalue": pvalue,
        "rejected": bool(pvalue < global_alpha),
    }


# ==========================================================
# Power: what change would this configuration have caught?
# ==========================================================

#: Effect sizes probed by the power analysis, in units of the baseline
#: ensemble's member-to-member standard deviation.
MDE_GRID = (0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0)


def minimum_detectable_effect(test, run, base, threshold, grid=MDE_GRID):
    """
    Smallest injected shift this configuration would have rejected.

    A ``PASS`` is only as meaningful as the change it would have caught. This
    shifts the run sample by increasing multiples of the baseline ensemble's
    member-to-member standard deviation and reports the first multiple whose
    p-value falls below ``threshold``. The answer turns "no difference
    detected" into "no difference detected, and a shift of 2 sigma would have
    been".

    Uses the asymptotic p-value, so it is cheap (one extra test evaluation per
    grid point) but inherits that p-value's optimism -- treat it as a lower
    bound on the shift genuinely needed. Returns None when the ensemble spread
    is degenerate or the test has no asymptotic p-value.
    """
    if not test.provides_asymptotic_pvalue or base.n_members < 2:
        return None

    sigma = float(np.std(base.member_means, ddof=1))
    if not np.isfinite(sigma) or sigma <= _TINY:
        return None

    probed = {}
    for delta in grid:
        try:
            _, pvalue = test.evaluate(run.pooled + delta * sigma, base.pooled)
        except (ValueError, ZeroDivisionError, FloatingPointError):
            continue
        if pvalue is None:
            return None
        probed[delta] = float(pvalue)

    if not probed:
        return None

    # Rank-based statistics are step functions of the injected shift, so the
    # p-value is not strictly monotone: a small shift can land on a flat spot
    # and score worse than an even smaller one. Report the smallest shift from
    # which every *larger* probed shift is also rejected, which is the
    # threshold a reader would actually rely on, rather than the first lucky
    # crossing.
    ordered = sorted(probed)
    detected = None
    for i, delta in enumerate(ordered):
        if all(probed[d] < threshold for d in ordered[i:]):
            detected = delta
            break

    return {
        "sigma": sigma,
        "threshold": threshold,
        "mde_in_sigma": detected,
        "mde_absolute": None if detected is None else detected * sigma,
        # A list, not a dict keyed by float: JSON would stringify those keys
        # and they would no longer sort numerically on the way back in.
        "probed": [
            {"shift_in_sigma": shift, "pvalue": pvalue}
            for shift, pvalue in sorted(probed.items())
        ],
    }


# ==========================================================
# Equivalence testing (TOST)
# ==========================================================


# pylint: disable=too-many-locals
def tost_equivalence(run, base, margin_in_sigma, alpha):
    """
    Two one-sided tests for equivalence of the ensemble means.

    Operates on member-level means, which are the only values here that are
    genuinely independent. The equivalence margin is expressed in units of the
    baseline ensemble's own member-to-member standard deviation, so
    ``margin_in_sigma=1`` reads "the ensemble means agree to within one
    baseline ensemble sigma".

    Returns None when the margin cannot be formed (too few members, or a
    degenerate baseline spread).
    """
    x, y = run.member_means, base.member_means
    if x.size < 2 or y.size < 2:
        return None

    sigma = float(np.std(y, ddof=1))
    if not np.isfinite(sigma) or sigma <= _TINY:
        return None

    margin = margin_in_sigma * sigma
    difference = float(np.mean(x) - np.mean(y))
    standard_error = math.sqrt(
        np.var(x, ddof=1) / x.size + np.var(y, ddof=1) / y.size
    )
    if standard_error <= _TINY:
        return None

    # Welch-Satterthwaite degrees of freedom
    var_x, var_y = np.var(x, ddof=1) / x.size, np.var(y, ddof=1) / y.size
    dof = (var_x + var_y) ** 2 / (
        var_x**2 / (x.size - 1) + var_y**2 / (y.size - 1)
    )

    t_lower = (difference + margin) / standard_error
    t_upper = (difference - margin) / standard_error
    p_lower = float(stats.t.sf(t_lower, dof))
    p_upper = float(stats.t.cdf(t_upper, dof))
    p_tost = max(p_lower, p_upper)

    return {
        "margin_in_sigma": margin_in_sigma,
        "margin": margin,
        "baseline_member_sigma": sigma,
        "mean_difference": difference,
        "dof": float(dof),
        "p_tost": p_tost,
        "equivalent": bool(p_tost < alpha),
    }


# ==========================================================
# Multiple-testing correction
# ==========================================================


def apply_multiple_testing_correction(results, alpha, method="bonferroni"):
    """
    Correct the p-values across variables and refresh the decisions.

    Two properties this deliberately guarantees, and which the previous
    hand-rolled implementation did not:

    1. A correction can only ever *remove* rejections. It never turns a
       variable that passed into one that failed.
    2. Gates that sit downstream of the p-value (the magnitude threshold, the
       equivalence requirement) are re-applied afterwards, so a correction
       cannot resurrect a decision they had already settled.
    """
    statsmodels_method, _ = CORRECTION_METHODS[method]
    if statsmodels_method is None:
        for result in results.values():
            result["correction_method"] = "none"
            result["pvalue_corrected"] = result.get("pvalue")
        return results

    names = [
        name
        for name, result in results.items()
        if result.get("pvalue") is not None and np.isfinite(result["pvalue"])
    ]
    if not names:
        return results

    pvalues = np.array([results[name]["pvalue"] for name in names])
    reject, corrected, _, _ = multipletests(
        pvalues, alpha=alpha, method=statsmodels_method
    )

    for i, name in enumerate(names):
        result = results[name]
        result["correction_method"] = method
        result["n_tests_corrected"] = len(names)
        result["pvalue_corrected"] = float(corrected[i])
        result["rejected_uncorrected"] = bool(result["rejected"])
        # Property 1: intersect, never union.
        result["rejected"] = bool(result["rejected"] and reject[i])

    return results


def _finalize_decision(result, magnitude_threshold, equivalence_margin):
    """Turn a rejection flag plus the downstream gates into PASS/FAIL."""
    reasons = []
    fail = bool(result.get("rejected", False))

    if fail and magnitude_threshold is not None:
        relative = result["effect_size"]["relative_diff"]
        result["magnitude_check"] = {
            "relative_difference": relative,
            "magnitude_threshold": magnitude_threshold,
            "exceeds_threshold": bool(relative > magnitude_threshold),
        }
        if relative <= magnitude_threshold:
            fail = False
            reasons.append(
                f"difference is statistically detectable but below the "
                f"magnitude threshold ({relative:.3e} <= {magnitude_threshold:.3e})"
            )

    equivalence = result.get("equivalence")
    if equivalence_margin is not None:
        if equivalence is None:
            fail = True
            reasons.append("equivalence could not be assessed (too few members)")
        elif not equivalence["equivalent"]:
            fail = True
            reasons.append(
                f"equivalence within +/-{equivalence_margin} sigma not "
                f"demonstrated (p_TOST={equivalence['p_tost']:.3e})"
            )

    result["hypothesis"] = "FAIL" if fail else "PASS"
    result["decision_notes"] = reasons
    return result


# ==========================================================
# Per-variable comparison
# ==========================================================


def load_variable_list(path, variable_set=None):
    """
    Read a curated variable list.

    Accepts a JSON list, a JSON object mapping a set name to a list (the shape
    evv4esm's ``ks_vars.json`` uses, which carries both a ``default`` and a
    ``scream`` set), or a plain text file with one name per line.

    Fixing the variable set matters more than it looks. When the tested set is
    "whatever happened to be in the output stream", the multiple-testing
    penalty -- and therefore the meaning of a PASS -- silently changes every
    time someone edits the output yaml. A curated list keeps the criterion
    comparable across runs and across time.
    """
    with open(path, "r", encoding="utf-8") as handle:
        text = handle.read()

    if os.path.splitext(path)[1].lower() == ".json":
        payload = json.loads(text)
        if isinstance(payload, dict):
            if variable_set is None:
                if len(payload) == 1:
                    variable_set = next(iter(payload))
                elif "default" in payload:
                    variable_set = "default"
                else:
                    raise ValueError(
                        f"{path} holds several named sets "
                        f"({', '.join(sorted(payload))}); pick one with "
                        f"--variable_set"
                    )
            if variable_set not in payload:
                raise ValueError(
                    f"Variable set '{variable_set}' not in {path}; available: "
                    f"{', '.join(sorted(payload))}"
                )
            names = payload[variable_set]
        else:
            names = payload
    else:
        names = [line.strip() for line in text.splitlines()]

    names = [str(name).strip() for name in names if str(name).strip()]
    if not names:
        raise ValueError(f"No variable names found in {path}")
    return names


def _select_variables(run_ens, base_ens, requested=None):
    """
    Choose the variables to test.

    Only variables present in *every* member of *both* ensembles are testable;
    anything else is reported as skipped rather than quietly ignored. The
    all-NaN / constant screening happens later, on the reduced samples, so
    that we never pull a full 4-D field into memory just to decide whether to
    look at it.

    Returns ``(testable, skipped, unmatched)``. A curated list will normally
    name variables that a given output stream does not carry; those come back
    in ``unmatched`` and are reported rather than being fatal, since the same
    list is meant to serve several configurations.
    """
    reference = next(iter(run_ens.values()))
    candidates = [
        str(name)
        for name in reference.data_vars
        if str(name) not in SKIP_VARS
        and "time" in reference[name].dims
        and np.issubdtype(reference[name].dtype, np.floating)
    ]

    unmatched = []
    if requested:
        unmatched = sorted(set(requested) - set(candidates))
        candidates = [name for name in candidates if name in requested]
        if not candidates:
            raise ValueError(
                f"None of the {len(requested)} requested variable(s) are "
                f"testable in the run ensemble. Check the names and the "
                f"output stream."
            )
        if unmatched:
            logger.warning(
                "%d requested variable(s) are not in the output and will not "
                "be tested: %s",
                len(unmatched),
                ", ".join(unmatched[:10]) + (" ..." if len(unmatched) > 10 else ""),
            )

    skipped = {}
    testable = []
    for name in candidates:
        absent = [
            f"{label}:{inst}"
            for label, ensemble in (("run", run_ens), ("base", base_ens))
            for inst, dataset in sorted(ensemble.items())
            if name not in dataset.variables
        ]
        if absent:
            skipped[name] = f"absent from member(s) {', '.join(absent)}"
        else:
            testable.append(name)

    return sorted(testable), skipped, unmatched


# pylint: disable=too-many-arguments, too-many-positional-arguments
# pylint: disable=too-many-locals, too-many-branches
def compare_variable(
    var,
    run_ens,
    base_ens,
    test,
    analysis_type,
    weights,
    calibration,
    assignments,
    equivalence_margin,
    mde_threshold=None,
):
    """
    Run the full comparison for a single variable.

    Returns ``(result, perm_pvalues)``. ``perm_pvalues`` is the p-value the
    variable would have had under each member assignment, or None when the
    variable never reached the test (bit-identical or constant samples). The
    caller collects these to run the global calibrated-count test.
    """
    run, base, notes = build_samples(var, run_ens, base_ens, analysis_type, weights)

    if run.size == 0 or base.size == 0:
        raise ValueError("no finite data")

    result = {
        "variable": var,
        "sample1": _describe(run),
        "sample2": _describe(base),
        "notes": notes,
    }

    run_neff, run_rho = _effective_sample_size(run)
    base_neff, base_rho = _effective_sample_size(base)
    result["independence"] = {
        "lag1_autocorrelation_run": run_rho,
        "lag1_autocorrelation_base": base_rho,
        "effective_n_run": run_neff,
        "effective_n_base": base_neff,
        "nominal_n_run": run.size,
        "nominal_n_base": base.size,
    }

    result["effect_size"] = _effect_sizes(run, base)

    if equivalence_margin is not None:
        result["equivalence"] = tost_equivalence(
            run, base, equivalence_margin, test.alpha
        )

    # Degenerate cases resolve before any test is attempted.
    if run.size == base.size and np.array_equal(run.pooled, base.pooled):
        result.update(
            statistic=0.0,
            pvalue=1.0,
            rejected=False,
            calibration="exact",
            reason="samples are bit-identical",
        )
        return _finalize_decision(result, None, equivalence_margin), None

    constant_run = np.unique(run.pooled).size < 2
    constant_base = np.unique(base.pooled).size < 2
    if constant_run and constant_base:
        identical = math.isclose(float(run.pooled[0]), float(base.pooled[0]))
        result.update(
            statistic=0.0 if identical else float("inf"),
            pvalue=1.0 if identical else 0.0,
            rejected=not identical,
            calibration="exact",
            reason=(
                "both samples are constant and equal"
                if identical
                else "both samples are constant but differ"
            ),
        )
        return _finalize_decision(result, None, equivalence_margin), None

    statistic, asymptotic_p = test.evaluate(run.pooled, base.pooled)
    result["statistic"] = statistic
    result["pvalue_asymptotic"] = asymptotic_p

    perm_pvalues = None
    if assignments is not None:
        values = permuted_statistics(test, run, base, assignments)
        perm_pvalues, n_valid = permutation_pvalues(values)
        result["n_permutations"] = n_valid

    if calibration == "member":
        pvalue = float(perm_pvalues[0])
        result["calibration"] = "member_permutation"
    else:
        if asymptotic_p is None:
            raise ValueError(
                f"test '{type(test).__name__}' has no asymptotic p-value; "
                f"use --calibration member"
            )
        pvalue = asymptotic_p
        result["calibration"] = "asymptotic"

    result["pvalue"] = float(pvalue)
    result["rejected"] = bool(np.isfinite(pvalue) and pvalue < test.alpha)

    if mde_threshold is not None:
        result["power"] = minimum_detectable_effect(test, run, base, mde_threshold)

    return result, perm_pvalues


# ==========================================================
# JSON output location
# ==========================================================


def resolve_json_path(json_output, run_dir, test_type, analysis_type):
    """
    Work out where the JSON report should go.

    Precedence: explicit argument, then ``$RCS_JSON_OUTPUT``, then the run
    directory (the historical behavior). The value may be

    - a path ending in ``.json``: used verbatim, parents created as needed;
    - a directory (existing, or with a trailing separator): the report is
      written inside it under a generated name;
    - ``none`` / ``off`` / ``""``: no JSON report is written.

    Returns None when writing is disabled.
    """
    if json_output is None:
        json_output = os.environ.get("RCS_JSON_OUTPUT")
    if json_output is None:
        json_output = run_dir

    target = str(json_output).strip()
    if target.lower() in ("", "none", "off", "no", "disable", "disabled"):
        return None

    target = os.path.abspath(os.path.expanduser(os.path.expandvars(target)))
    filename = f"rcs_{test_type}_{analysis_type}_results.json"

    if target.endswith(".json"):
        return target
    return os.path.join(target, filename)


def write_json_report(path, payload):
    """
    Write the JSON report, degrading gracefully when the location is not
    writable.

    RUNDIR is frequently on a shared filesystem the caller cannot write to, so
    a failure here falls back to the temporary directory and, if that also
    fails, is reported and swallowed. Losing the report must never turn a
    passing comparison into a failing one.
    """
    if path is None:
        return None, None

    def _attempt(candidate):
        os.makedirs(os.path.dirname(candidate) or ".", exist_ok=True)
        with open(candidate, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, default=str)
        return candidate

    try:
        return _attempt(path), None
    except OSError as first_error:
        fallback = os.path.join(tempfile.gettempdir(), os.path.basename(path))
        logger.warning(
            "Could not write JSON report to %s (%s); trying %s",
            path, first_error, fallback,
        )
        try:
            return _attempt(fallback), (
                f"could not write to {path} ({first_error}); "
                f"wrote to {fallback} instead"
            )
        except OSError as second_error:
            return None, (
                f"could not write JSON report to {path} ({first_error}) "
                f"or {fallback} ({second_error})"
            )


# ==========================================================
# Reporting
# ==========================================================


def _rule(char="="):
    return char * 78


# pylint: disable=too-many-locals, too-many-branches, too-many-statements
def format_report(config, ensemble_info, results, skipped, errors, summary):
    """Render the human-readable report."""
    lines = ["", _rule(), "RCS STATISTICAL COMPARISON", _rule()]

    test_type = config["test_type"]
    lines += [
        f"Statistical test     : {TEST_FULL_NAMES[test_type]} ({test_type})",
        f"Analysis type        : {config['analysis_type']} "
        f"({ANALYSIS_FULL_NAMES[config['analysis_type']]})",
        f"Calibration          : {config['calibration']}",
        f"Significance level   : alpha = {config['alpha']}",
        f"Multiple testing     : {CORRECTION_METHODS[config['correction_method']][1]}",
        f"Global test          : {config['global_test']}"
        + (
            f" (global_alpha = {config['global_alpha']})"
            if config["global_test"] == "calibrated_count"
            else ""
        ),
    ]
    if config["global_test"] == "variable_count":
        lines += [
            f"Max failed variables : {config['max_failed_vars']}",
            f"Max failed fraction  : {config['max_failed_fraction']}",
        ]
    if config["magnitude_threshold"] is not None:
        lines.append(
            f"Magnitude threshold  : {config['magnitude_threshold']} "
            f"(relative mean difference required to count as a failure)"
        )
    if config["equivalence_margin"] is not None:
        lines.append(
            f"Equivalence margin   : {config['equivalence_margin']} baseline sigma "
            f"(TOST; PASS requires demonstrated equivalence)"
        )
    lines += [
        f"RNG seed             : {config['seed']}",
        f"Run file pattern     : {config['run_file_pattern']}",
        f"Baseline pattern     : {config['base_file_pattern']}",
    ]

    lines += ["", _rule("-"), "ENSEMBLES", _rule("-")]
    lines += [
        f"Run      : {ensemble_info['n_run']} member(s) "
        f"[{', '.join(ensemble_info['run_instances'])}]",
        f"Baseline : {ensemble_info['n_base']} member(s) "
        f"[{', '.join(ensemble_info['base_instances'])}]",
        f"Variables tested : {len(results)}",
    ]
    if skipped:
        lines.append(f"Variables skipped: {len(skipped)}")
    if errors:
        lines.append(f"Variables errored: {len(errors)}")

    # --- What this configuration can and cannot resolve -------------------
    lines += ["", _rule("-"), "RESOLUTION AND POWER", _rule("-")]
    n_run, n_base = ensemble_info["n_run"], ensemble_info["n_base"]
    effective_alpha = decision_threshold(
        config["alpha"], config["correction_method"], len(results)
    )
    if config["correction_method"] != "none" and results:
        lines.append(
            f"Strictest per-variable threshold after correction: "
            f"{effective_alpha:.3e} (alpha={config['alpha']} over "
            f"{len(results)} variables)"
        )

    p_min, n_perms, exhaustive = permutation_resolution(
        n_run, n_base, config["n_resamples"]
    )
    lines.append(
        f"Member permutation resolution: smallest attainable p-value is "
        f"{p_min:.3e} with {n_run}+{n_base} members "
        f"({'all ' if exhaustive else ''}{n_perms} "
        f"{'partitions' if exhaustive else 'random resamples'})"
    )

    if config["calibration"] == "member" and p_min > effective_alpha:
        lines += [
            "",
            "*** WARNING: with this many members the permutation test can",
            f"*** never reach the decision threshold {effective_alpha:.3e},",
            "*** so no variable can fail and the comparison is vacuous.",
            "*** Remedies, in order of preference:",
            "***   - add ensemble members (the resolution improves roughly",
            "***     like 2/C(2n,n): 9 per ensemble reaches ~4e-5)",
            "***   - test fewer variables (--variables) to soften the",
            "***     correction; every correction here bottoms out at alpha/n",
            "***   - relax --alpha",
        ]
    if config["calibration"] == "asymptotic":
        neffs = [
            r["independence"]["effective_n_run"] / max(r["independence"]["nominal_n_run"], 1)
            for r in results.values()
            if "independence" in r
        ]
        if neffs:
            lines += [
                "",
                f"Median effective/nominal sample size: {np.median(neffs):.2f}",
                "  Asymptotic p-values assume every pooled value is an",
                "  independent observation. Values below 1.0 mean they are",
                "  anti-conservative; --calibration member removes the",
                "  assumption at the cost of resolution.",
            ]

    # --- Summary ----------------------------------------------------------
    # --- What a PASS would have caught ------------------------------------
    mdes = [
        r["power"]["mde_in_sigma"]
        for r in results.values()
        if r.get("power") and r["power"].get("mde_in_sigma") is not None
    ]
    n_with_power = sum(1 for r in results.values() if r.get("power"))
    if n_with_power:
        undetectable = n_with_power - len(mdes)
        lines += ["", _rule("-"), "MINIMUM DETECTABLE EFFECT", _rule("-")]
        lines.append(
            "  Smallest injected shift, in baseline ensemble sigma, that this"
        )
        lines.append(
            "  configuration would have rejected. This is what a PASS is worth."
        )
        if mdes:
            lines.append(
                f"  Median across {len(mdes)} variable(s): "
                f"{np.median(mdes):.2f} sigma  "
                f"(best {min(mdes):.2f}, worst {max(mdes):.2f})"
            )
        if undetectable:
            lines.append(
                f"  {undetectable} variable(s) would not have been rejected "
                f"even at {MDE_GRID[-1]:.0f} sigma"
            )
        lines.append(
            "  Estimated with asymptotic p-values, so treat it as a lower "
            "bound."
        )

    # --- Global test ------------------------------------------------------
    global_result = summary.get("global_test")
    if global_result:
        lines += ["", _rule("-"), "GLOBAL TEST (calibrated rejection count)", _rule("-")]
        lines += [
            f"  Variables rejecting at p < {global_result['per_variable_alpha']}: "
            f"{global_result['observed_rejections']} of "
            f"{global_result['n_variables']}",
            f"  Expected by chance alone       : "
            f"{global_result['expected_by_chance']:.1f}",
            f"  Permutation null               : median "
            f"{global_result['null_rejections_median']:.1f}, 95th pct "
            f"{global_result['null_rejections_p95']:.1f}, max "
            f"{global_result['null_rejections_max']}",
            f"  Global p-value                 : "
            f"{global_result['pvalue']:.4f} over "
            f"{global_result['n_assignments']} member assignments",
            "",
            "  The null distribution above is an empirical calibration of this",
            "  exact configuration on this exact data: it is how many",
            "  rejections the pipeline produces when the members are regrouped",
            "  under the null. It accounts for correlation between variables,",
            "  which Bonferroni and Benjamini-Hochberg do not.",
        ]

    total = summary["total"]
    lines += ["", _rule("-"), "SUMMARY", _rule("-")]
    if total:
        lines += [
            f"Passed : {summary['passed']:5d} ({100.0 * summary['passed'] / total:5.1f}%)",
            f"Failed : {summary['failed']:5d} ({100.0 * summary['failed'] / total:5.1f}%)",
        ]
        if global_result:
            lines += [
                "",
                "  These per-variable counts are diagnostic only. The overall",
                "  verdict comes from the global test above, so they need not",
                "  agree with it -- and any magnitude or equivalence gate",
                "  shapes this table without reaching that verdict.",
            ]
    else:
        lines.append("No variable could be tested.")

    failed = summary["failed_variables"]
    if failed:
        lines += ["", _rule("-"), "FAILED VARIABLES", _rule("-")]
        ranked = sorted(failed, key=lambda v: results[v].get("pvalue", 1.0))
        for var in ranked[:20]:
            result = results[var]
            effect = result["effect_size"]
            sample1, sample2 = result["sample1"], result["sample2"]
            pvalue = result.get("pvalue", float("nan"))
            corrected = result.get("pvalue_corrected")
            lines.append(f"\n{var}:")
            line = f"  p = {pvalue:.4e}"
            if corrected is not None and corrected != pvalue:
                line += f" (corrected {corrected:.4e})"
            line += f"  alpha = {config['alpha']}  [{result.get('calibration', '?')}]"
            lines.append(line)
            lines.append(
                f"  run      : n={sample1['n']:7d}  mean={sample1.get('mean', float('nan')):13.6e}"
                f"  std={sample1.get('std', float('nan')):13.6e}"
            )
            lines.append(
                f"  baseline : n={sample2['n']:7d}  mean={sample2.get('mean', float('nan')):13.6e}"
                f"  std={sample2.get('std', float('nan')):13.6e}"
            )
            lines.append(
                f"  effect   : dmean={effect['mean_diff']:.6e} "
                f"({effect['mean_diff_pct']:+.3f}%)  "
                f"d={effect['cohens_d']:.3f}  snr={effect['snr']:.3f}  "
                f"KS={effect['ks_distance']:.4f}"
            )
            for note in result.get("decision_notes", []):
                lines.append(f"  note     : {note}")
            for note in result.get("notes", []):
                lines.append(f"  data     : {note}")
        if len(ranked) > 20:
            lines.append(f"\n  ... and {len(ranked) - 20} more failed variable(s)")

    # Variables that pass but show a large effect deserve a look even though
    # they did not trip the threshold.
    watchlist = sorted(
        (
            (results[v]["effect_size"]["snr"], v)
            for v in summary["passed_variables"]
            if np.isfinite(results[v]["effect_size"]["snr"])
            and results[v]["effect_size"]["snr"] > 1.0
        ),
        reverse=True,
    )
    if watchlist:
        lines += ["", _rule("-"), "PASSED BUT NOTABLE (snr > 1)", _rule("-")]
        lines.append("  Not rejected, but the ensemble means differ by more than the")
        lines.append("  baseline ensemble's own member spread.")
        for snr, var in watchlist[:10]:
            effect = results[var]["effect_size"]
            lines.append(
                f"  {var}: snr={snr:.2f}  dmean={effect['mean_diff_pct']:+.3f}%  "
                f"p={results[var].get('pvalue', float('nan')):.3e}"
            )

    # Sample-construction notes are material even when everything passes: a
    # comparison that quietly threw away most of the domain is not the
    # comparison the caller thinks they ran.
    noted = {v: r["notes"] for v, r in results.items() if r.get("notes")}
    if noted:
        lines += ["", _rule("-"), "DATA NOTES", _rule("-")]
        for var, notes in sorted(noted.items())[:20]:
            for note in notes:
                lines.append(f"  {var}: {note}")
        if len(noted) > 20:
            lines.append(f"  ... and {len(noted) - 20} more variable(s) with notes")

    if skipped:
        lines += ["", _rule("-"), "SKIPPED VARIABLES", _rule("-")]
        for var, reason in sorted(skipped.items())[:20]:
            lines.append(f"  {var}: {reason}")
        if len(skipped) > 20:
            lines.append(f"  ... and {len(skipped) - 20} more")

    if errors:
        lines += ["", _rule("-"), "ERRORED VARIABLES", _rule("-")]
        lines.append("  These variables could not be tested and are NOT counted")
        lines.append("  as passing. Investigate before trusting the result.")
        for var, reason in sorted(errors.items())[:20]:
            lines.append(f"  {var}: {reason}")
        if len(errors) > 20:
            lines.append(f"  ... and {len(errors) - 20} more")

    lines += ["", _rule(), f"OVERALL: {summary['test_status']}", _rule()]
    lines.append(f"  {summary['status_reason']}")
    if summary.get("json_path"):
        lines.append(f"  Detailed results: {summary['json_path']}")
    if summary.get("json_note"):
        lines.append(f"  NOTE: {summary['json_note']}")

    return "\n".join(lines)


# ==========================================================
# Main entry point
# ==========================================================


# pylint: disable=too-many-arguments, too-many-positional-arguments
# pylint: disable=too-many-locals, too-many-branches, too-many-statements
def run_stats_comparison(
    run_dir,
    base_dir,
    analysis_type="spatiotemporal",
    test_type="ks",
    alpha=None,
    correction_method="bonferroni",
    calibration="asymptotic",
    n_resamples=2000,
    max_failed_vars=0,
    max_failed_fraction=None,
    magnitude_threshold=None,
    equivalence_margin=None,
    variables=None,
    variables_file=None,
    variable_set=None,
    global_test="variable_count",
    global_alpha=0.05,
    power_analysis=True,
    run_file_pattern=None,
    base_file_pattern=None,
    json_output=None,
    seed=DEFAULT_SEED,
    fail_on_error=False,
    critical_fraction=None,
):
    """
    Compare two ensembles and return ``(report_text, "PASS"|"FAIL")``.

    Args:
        run_dir: Directory holding the run ensemble output.
        base_dir: Directory holding the baseline ensemble output.
        analysis_type: ``spatiotemporal`` (area-weighted global mean per output
            time), ``temporal`` (time mean per column) or ``member`` (one value
            per ensemble member).
        test_type: One of ks, ad, cvm, epps, energy, mw, ttest, brunner,
            levene, ansari, mood.
        alpha: Significance level (default 0.01).
        correction_method: bonferroni, holm, fdr, fdr_by or none. A correction
            can only remove rejections, never add them.
        calibration: ``asymptotic`` for scipy's closed-form p-value on the
            pooled values, or ``member`` to permute whole members. ``member``
            is exact under exchangeability of members but its resolution is
            capped by the member count -- see ``permutation_resolution``.
        n_resamples: Number of random member permutations when exhaustive
            enumeration is impractical.
        max_failed_vars: Overall test fails when more variables than this
            fail. Default 0.
        max_failed_fraction: Overall test also fails when the failing fraction
            exceeds this. None disables the fraction criterion.
        magnitude_threshold: Relative mean difference below which a
            statistically detectable difference is not counted as a failure.
        equivalence_margin: When set, additionally require a TOST at this many
            baseline-ensemble sigma to demonstrate equivalence; PASS then means
            "shown equivalent" rather than "not shown different".
        variables: Optional explicit list of variables to test.
        variables_file: Path to a curated variable list -- a JSON list, a JSON
            object of named lists (evv4esm's ``ks_vars.json`` shape), or a
            plain text file with one name per line.
        variable_set: Which named list to take from a JSON object.
        global_test: How the per-variable verdicts become one verdict.
            ``variable_count`` applies a multiple-testing correction and
            counts failures; ``calibrated_count`` compares the number of
            rejections against its member-permutation null distribution, which
            accounts for correlation between variables and stays usable at
            small member counts.
        global_alpha: Significance level for ``calibrated_count``, used both to
            screen each variable and to judge the resulting count.
        power_analysis: Estimate, per variable, the smallest injected shift
            this configuration would have rejected.
        run_file_pattern / base_file_pattern: Globs with a single ``????``
            placeholder marking the 4-digit instance number.
        json_output: Where to write the JSON report -- a ``.json`` path, a
            directory, or ``none`` to disable. Falls back to
            ``$RCS_JSON_OUTPUT`` and then to ``run_dir``.
        seed: RNG seed for permutation calibration.
        fail_on_error: Treat a variable that could not be tested as a failure.
        critical_fraction: Deprecated alias for ``max_failed_fraction``. The
            original parameter documented a per-variable, per-column threshold
            that was never implemented; it now controls the overall failing
            fraction.
    """
    alpha = 0.01 if alpha is None else float(alpha)
    test_type = normalize_test_type(test_type)
    correction_method = correction_method.lower()
    if correction_method not in CORRECTION_METHODS:
        raise ValueError(
            f"Unknown correction_method '{correction_method}'; expected one of "
            f"{', '.join(sorted(CORRECTION_METHODS))}"
        )
    if analysis_type not in ANALYSIS_FULL_NAMES:
        raise ValueError(
            f"Unknown analysis_type '{analysis_type}'; expected one of "
            f"{', '.join(sorted(ANALYSIS_FULL_NAMES))}"
        )
    if calibration not in ("asymptotic", "member"):
        raise ValueError(
            f"Unknown calibration '{calibration}'; expected asymptotic or member"
        )
    if calibration == "asymptotic" and not TEST_REGISTRY[
        test_type
    ].provides_asymptotic_pvalue:
        raise ValueError(
            f"Test '{test_type}' has no asymptotic null distribution in scipy; "
            f"it can only be used with calibration='member'."
        )
    if global_test not in ("variable_count", "calibrated_count"):
        raise ValueError(
            f"Unknown global_test '{global_test}'; expected variable_count "
            f"or calibrated_count"
        )
    if global_test == "calibrated_count":
        # The calibrated count is built from p-values, so gates that sit
        # downstream of the p-value do not reach it. They still shape the
        # per-variable table, which is why they are allowed rather than
        # rejected -- but the user must not assume they bound the verdict.
        ignored = [
            name
            for name, value in (
                ("magnitude_threshold", magnitude_threshold),
                ("equivalence_margin", equivalence_margin),
                ("max_failed_vars", max_failed_vars if max_failed_vars else None),
                ("max_failed_fraction", max_failed_fraction),
            )
            if value is not None
        ]
        if ignored:
            logger.warning(
                "global_test='calibrated_count' decides the overall verdict "
                "from the rejection count, so %s affect the per-variable "
                "table but not the PASS/FAIL result.",
                " and ".join(ignored),
            )

    if critical_fraction is not None and max_failed_fraction is None:
        logger.warning(
            "critical_fraction is deprecated; it is now interpreted as "
            "max_failed_fraction (overall fraction of failing variables)."
        )
        max_failed_fraction = critical_fraction

    run_file_pattern = run_file_pattern or DEFAULT_FILE_PATTERN
    base_file_pattern = base_file_pattern or DEFAULT_FILE_PATTERN

    config = {
        "test_type": test_type,
        "analysis_type": analysis_type,
        "calibration": calibration,
        "alpha": alpha,
        "correction_method": correction_method,
        "n_resamples": n_resamples,
        "max_failed_vars": max_failed_vars,
        "max_failed_fraction": max_failed_fraction,
        "magnitude_threshold": magnitude_threshold,
        "equivalence_margin": equivalence_margin,
        "global_test": global_test,
        "global_alpha": global_alpha,
        "power_analysis": power_analysis,
        "seed": seed,
        "fail_on_error": fail_on_error,
        "run_file_pattern": run_file_pattern,
        "base_file_pattern": base_file_pattern,
        "run_dir": str(run_dir),
        "base_dir": str(base_dir),
    }

    run_ens = open_ensemble(str(run_dir), run_file_pattern)
    base_ens = open_ensemble(str(base_dir), base_file_pattern)

    try:

        ensemble_info = {
            "n_run": len(run_ens),
            "n_base": len(base_ens),
            "run_instances": sorted(run_ens),
            "base_instances": sorted(base_ens),
        }
        if len(run_ens) < 2 or len(base_ens) < 2:
            logger.warning(
                "RCS is comparing ensembles of %d and %d member(s); with fewer "
                "than 2 members per side the ensemble spread is unknown and the "
                "comparison has essentially no diagnostic value.",
                len(run_ens), len(base_ens),
            )

        requested = list(variables) if variables else None
        if variables_file:
            from_file = load_variable_list(variables_file, variable_set)
            requested = sorted(set(requested or []) | set(from_file))

        testable, skipped, unmatched = _select_variables(run_ens, base_ens, requested)
        weights = (
            get_area_weights(next(iter(run_ens.values())))
            if analysis_type in ("spatiotemporal", "member")
            else None
        )
        if analysis_type in ("spatiotemporal", "member") and weights is None:
            logger.warning(
                "No 'area' field found; falling back to an unweighted spatial "
                "mean, which over-weights small grid cells."
            )

        test = get_test(test_type, alpha=alpha)
        rng = np.random.default_rng(seed)

        # One shared assignment list for every variable, so the global test can
        # see the correlation between variables.
        assignments, exhaustive = (None, None)
        if calibration == "member" or global_test == "calibrated_count":
            assignments, exhaustive = build_assignments(
                len(run_ens), len(base_ens), n_resamples, rng
            )
            logger.info(
                "Calibrating with %d member assignments (%s)",
                len(assignments),
                "exhaustive" if exhaustive else "randomly sampled",
            )

        mde_threshold = (
            decision_threshold(alpha, correction_method, max(len(testable), 1))
            if power_analysis
            else None
        )

        results, errors, perm_pvalues = {}, {}, {}
        for var in testable:
            try:
                result, per_assignment = compare_variable(
                    var, run_ens, base_ens, test, analysis_type, weights,
                    calibration, assignments, equivalence_margin, mde_threshold,
                )
                results[var] = result
                if per_assignment is not None:
                    perm_pvalues[var] = per_assignment
                elif assignments is not None:
                    # Bit-identical or constant: never rejects, under any grouping.
                    perm_pvalues[var] = np.ones(len(assignments))
            except (ValueError, KeyError, OSError, IndexError, TypeError,
                    ZeroDivisionError, FloatingPointError) as error:
                logger.warning("Could not test %s: %s", var, error)
                errors[var] = str(error)

        if results:
            results = apply_multiple_testing_correction(results, alpha, correction_method)
        for result in results.values():
            _finalize_decision(result, magnitude_threshold, equivalence_margin)

        global_result = None
        if global_test == "calibrated_count":
            global_result = calibrated_count_test(perm_pvalues, global_alpha)

        failed = sorted(v for v, r in results.items() if r["hypothesis"] == "FAIL")
        passed = sorted(v for v, r in results.items() if r["hypothesis"] == "PASS")
        total = len(results)

        # --- Overall verdict --------------------------------------------------
        status_reasons = []
        test_status = "PASS"
        if total == 0:
            test_status = "FAIL"
            status_reasons.append(
                "no variable could be tested; the comparison is vacuous"
            )
        elif global_test == "calibrated_count":
            # The count of rejections, judged against its own permutation null.
            if global_result["rejected"]:
                test_status = "FAIL"
                status_reasons.append(
                    f"{global_result['observed_rejections']} of "
                    f"{global_result['n_variables']} variables rejected at "
                    f"p<{global_alpha}, more than member exchangeability explains "
                    f"(global p={global_result['pvalue']:.4f}; null median "
                    f"{global_result['null_rejections_median']:.1f}, 95th "
                    f"percentile {global_result['null_rejections_p95']:.1f})"
                )
            else:
                status_reasons.append(
                    f"{global_result['observed_rejections']} of "
                    f"{global_result['n_variables']} variables rejected at "
                    f"p<{global_alpha}, consistent with the permutation null "
                    f"(global p={global_result['pvalue']:.4f})"
                )
        else:
            if len(failed) > max_failed_vars:
                test_status = "FAIL"
                status_reasons.append(
                    f"{len(failed)} variable(s) failed, more than the allowed "
                    f"{max_failed_vars}"
                )
            if max_failed_fraction is not None:
                fraction = len(failed) / total
                if fraction > max_failed_fraction:
                    test_status = "FAIL"
                    status_reasons.append(
                        f"failing fraction {fraction:.4f} exceeds "
                        f"{max_failed_fraction}"
                    )

        # A permutation test whose finest attainable p-value sits above the
        # decision threshold cannot reject anything, no matter what the model
        # does. Reporting PASS in that situation is worse than useless, so treat
        # it as the configuration error it is. The global test escapes this,
        # because it spends its resolution on a single p-value rather than on one
        # per variable.
        if calibration == "member" and total and global_test != "calibrated_count":
            threshold = decision_threshold(alpha, correction_method, total)
            p_min, _, _ = permutation_resolution(
                len(run_ens), len(base_ens), n_resamples
            )
            if p_min > threshold:
                test_status = "FAIL"
                status_reasons.append(
                    f"configuration cannot resolve a difference: member "
                    f"permutation bottoms out at p={p_min:.3e} but the decision "
                    f"threshold is {threshold:.3e}, so no variable could ever "
                    f"fail. Add ensemble members, relax --alpha, switch "
                    f"--correction_method, or use --global_test calibrated_count"
                )
        if global_test == "calibrated_count" and total:
            # Same resolution floor as any permutation test: 2/C(2n,n) when the
            # assignments are enumerated, because a partition and its complement
            # give identical counts. Comparing against 1/n_assignments here would
            # miss configurations that fall between the two.
            p_min, n_perms, _ = permutation_resolution(
                len(run_ens), len(base_ens), n_resamples
            )
            if p_min > global_alpha:
                test_status = "FAIL"
                status_reasons.append(
                    f"configuration cannot resolve a difference: the global test "
                    f"bottoms out at p={p_min:.3e} over {n_perms} member "
                    f"assignments, above global_alpha={global_alpha}, so it could "
                    f"never reject. Add members or raise --global_alpha"
                )

        if unmatched:
            status_reasons.append(
                f"{len(unmatched)} requested variable(s) are not present in the "
                f"output and were not tested"
            )

        if errors and fail_on_error:
            test_status = "FAIL"
            status_reasons.append(f"{len(errors)} variable(s) could not be tested")
        elif errors:
            status_reasons.append(
                f"{len(errors)} variable(s) could not be tested (not counted; "
                f"use fail_on_error to make this fatal)"
            )
        if not status_reasons:
            status_reasons.append(
                f"all {total} tested variable(s) are consistent with the baseline "
                f"ensemble at alpha={alpha}"
            )

        summary = {
            "global_test": global_result,
            "passed": len(passed),
            "failed": len(failed),
            "total": total,
            "unmatched": unmatched,
            "skipped": len(skipped),
            "errored": len(errors),
            "test_status": test_status,
            "status_reason": "; ".join(status_reasons),
            "failed_variables": failed,
            "passed_variables": passed,
        }

        json_path = resolve_json_path(json_output, str(run_dir), test_type, analysis_type)
        written_path, json_note = write_json_report(
            json_path,
            {
                "configuration": config,
                "ensembles": ensemble_info,
                "summary": summary,
                "skipped_variables": skipped,
                "errored_variables": errors,
                "details": results,
            },
        )
        summary["json_path"] = written_path
        summary["json_note"] = json_note

        report = format_report(config, ensemble_info, results, skipped, errors, summary)
        return report, test_status
    finally:
        close_ensemble(run_ens)
        close_ensemble(base_ens)


# ==========================================================
# Command line interface
# ==========================================================


###############################################################################
def parse_command_line(args, description):
###############################################################################
    """Build and run the argument parser."""
    # pylint: disable=import-outside-toplevel
    import argparse
    from pathlib import Path

    program = Path(args[0]).name
    parser = argparse.ArgumentParser(
        usage=f"""\n{program} run_dir base_dir [options]
OR
{program} --help

\033[1mEXAMPLES:\033[0m
    \033[1;32m# Default: KS on area-weighted global means, Bonferroni corrected\033[0m
    > {program} /path/to/run /path/to/baseline

    \033[1;32m# Exact member-level permutation calibration (no iid assumption)\033[0m
    > {program} /path/to/run /path/to/baseline --calibration member

    \033[1;32m# Global test on the rejection count; works with few members\033[0m
    > {program} /path/to/run /path/to/baseline \\
          --analysis_type member --global_test calibrated_count

    \033[1;32m# Reuse evv4esm's curated EAMxx variable list\033[0m
    > {program} /path/to/run /path/to/baseline \\
          --variables_file ks_vars.json --variable_set scream

    \033[1;32m# Write the JSON report somewhere writable\033[0m
    > {program} /path/to/run /path/to/baseline --json_output ~/rcs_reports

    \033[1;32m# Positively demonstrate equivalence within one ensemble sigma\033[0m
    > {program} /path/to/run /path/to/baseline --equivalence_margin 1.0

    \033[1;32m# Custom file patterns\033[0m
    > {program} /path/to/run /path/to/baseline \\
          --run_file_pattern "*.eam_????.h0.*.nc"
""",
        description=description,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
\033[1mAVAILABLE STATISTICAL TESTS:\033[0m

  \033[1mDISTRIBUTION\033[0m (compare the whole distribution):
    ks          Kolmogorov-Smirnov (default; largest CDF gap)
    ad          Anderson-Darling (tail-weighted)
    cvm         Cramer-von Mises (integrated CDF gap)
    epps        Epps-Singleton (characteristic functions; valid for discrete)
    energy      Energy distance (requires --calibration member)

  \033[1mLOCATION\033[0m (compare means/medians):
    mw          Mann-Whitney U
    ttest       Welch's t-test
    brunner     Brunner-Munzel

  \033[1mSCALE\033[0m (compare spread):
    levene      Levene's test
    ansari      Ansari-Bradley
    mood        Mood's test

\033[1mA NOTE ON RIGOR:\033[0m
  The default asymptotic calibration treats every pooled value as an
  independent observation. It is not: global means from one simulation are
  serially correlated. Use --calibration member for a p-value that is exact
  under exchangeability of ensemble members, and read the RESOLUTION AND
  POWER section of the report to see what your member count can resolve.

  Pooling timesteps can also cost more power than it adds, because the
  seasonal cycle ends up in the sample variance. --analysis_type member uses
  one value per ensemble member instead, which is the sampling choice
  evv4esm's MVK test makes.

  Always read MINIMUM DETECTABLE EFFECT before trusting a PASS.
""",
    )

    parser.add_argument("run_dir", help="Directory of the new run ensemble")
    parser.add_argument("base_dir", help="Directory of the baseline ensemble")
    parser.add_argument(
        "--analysis_type",
        default="spatiotemporal",
        choices=sorted(ANALYSIS_FULL_NAMES),
        help="How each member is reduced to a sample (default: spatiotemporal)",
    )
    parser.add_argument(
        "--test_type",
        default="ks",
        choices=sorted(TEST_REGISTRY),
        help="Statistical test identifier (default: ks)",
    )
    parser.add_argument(
        "--calibration",
        default="asymptotic",
        choices=["asymptotic", "member"],
        help="How p-values are computed: scipy's asymptotic formula on the "
        "pooled values, or exact permutation of whole members "
        "(default: asymptotic)",
    )
    parser.add_argument(
        "--n_resamples",
        type=int,
        default=2000,
        help="Random member permutations to draw when exhaustive enumeration "
        "is impractical (default: 2000)",
    )
    parser.add_argument(
        "--alpha", type=float, default=None,
        help="Significance level (default: 0.01)",
    )
    parser.add_argument(
        "--correction_method",
        default="bonferroni",
        choices=sorted(CORRECTION_METHODS),
        help="Multiple-testing correction across variables (default: bonferroni)",
    )
    parser.add_argument(
        "--max_failed_vars", type=int, default=0,
        help="Number of failing variables tolerated (default: 0)",
    )
    parser.add_argument(
        "--max_failed_fraction", type=float, default=None,
        help="Fraction of failing variables tolerated (default: unset)",
    )
    parser.add_argument(
        "--critical_fraction", type=float, default=None,
        help="Deprecated alias for --max_failed_fraction. The original "
        "parameter documented a per-variable, per-column threshold that was "
        "never implemented; it now controls the overall failing fraction.",
    )
    parser.add_argument(
        "--magnitude_threshold", type=float, default=None,
        help="Relative mean difference required before a detectable "
        "difference counts as a failure (default: unset)",
    )
    parser.add_argument(
        "--equivalence_margin", type=float, default=None,
        help="Run a TOST equivalence test with this margin, in units of the "
        "baseline ensemble's member-to-member sigma. PASS then requires "
        "demonstrated equivalence (default: unset)",
    )
    parser.add_argument(
        "--variables", nargs="+", default=None,
        help="Restrict the comparison to these variables (default: all)",
    )
    parser.add_argument(
        "--variables_file", default=None,
        help="Path to a curated variable list: a JSON list, a JSON object of "
        "named lists (evv4esm's ks_vars.json shape), or one name per line. "
        "Fixing the variable set keeps the multiple-testing penalty, and so "
        "the meaning of a PASS, stable across runs.",
    )
    parser.add_argument(
        "--variable_set", default=None,
        help="Which named list to take from a JSON object of sets "
        "(e.g. 'scream' in evv4esm's ks_vars.json)",
    )
    parser.add_argument(
        "--global_test",
        default="variable_count",
        choices=["variable_count", "calibrated_count"],
        help="How per-variable verdicts become one verdict. variable_count "
        "corrects the p-values and counts failures. calibrated_count judges "
        "the NUMBER of rejections against its member-permutation null, which "
        "accounts for correlation between variables and, because it spends "
        "its resolution on a single p-value, still works with few members "
        "(default: variable_count)",
    )
    parser.add_argument(
        "--global_alpha", type=float, default=0.05,
        help="Significance level for --global_test calibrated_count, used "
        "both to screen each variable and to judge the count (default: 0.05)",
    )
    parser.add_argument(
        "--no_power_analysis", dest="power_analysis", action="store_false",
        default=True,
        help="Skip estimating the smallest injected shift this configuration "
        "would have rejected",
    )
    parser.add_argument(
        "--run_file_pattern", default=DEFAULT_FILE_PATTERN,
        help=f"Run ensemble glob, '????' marks the instance number "
        f"(default: {DEFAULT_FILE_PATTERN})",
    )
    parser.add_argument(
        "--base_file_pattern", default=DEFAULT_FILE_PATTERN,
        help=f"Baseline ensemble glob (default: {DEFAULT_FILE_PATTERN})",
    )
    parser.add_argument(
        "--json_output", default=None,
        help="Where to write the JSON report: a .json path, a directory, or "
        "'none' to disable. Falls back to $RCS_JSON_OUTPUT and then to "
        "run_dir. Use this when run_dir is not writable.",
    )
    parser.add_argument(
        "--seed", type=int, default=DEFAULT_SEED,
        help=f"RNG seed for permutation calibration (default: {DEFAULT_SEED})",
    )
    parser.add_argument(
        "--fail_on_error", action="store_true",
        help="Treat a variable that could not be tested as a failure",
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Emit debug-level logging",
    )

    parsed = parser.parse_args(args[1:])
    logging.basicConfig(
        level=logging.DEBUG if parsed.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )
    del parsed.verbose
    return parsed


###############################################################################
def _main_func(description):
###############################################################################
    comments, status = run_stats_comparison(
        **vars(parse_command_line(sys.argv, description))
    )
    print(comments)
    return 0 if status == "PASS" else 1


###############################################################################

if __name__ == "__main__":
    sys.exit(_main_func(__doc__))
