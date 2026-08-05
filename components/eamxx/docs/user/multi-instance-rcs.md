# Multi-instance and NBFB testing capabilities

For NBFB testing, we utilize a "System Test" called `RCS`
which stands for Reproducible Climate Statistics. The premise
of RCS is comparing the statistics of two (potentially differing)
populations to determine if they are statistically identical.
To enable RCS, we utilize CIME's multi-instance capability.

## Multi-instance

A given case can be made to house a number of instances
by modifying the `NINST` variable. For a given component `CMP`,
active or not, `NINST_CMP` can be set to a value greater than 1.
After CIME's setup operation (`case.setup`), `CMP` will have
`NINST_CMP` instances of itself available, with their own runtime
options that can be changed.

For the `scream` component, the support is incomplete because
of the departure from the convention of `user_nl_scream` to
utilize a more readable YAML-based configuration. As a result,
the configuration of `NINST_ATM` when `ATM` is `scream`, the user
must replicate `NINST_ATM` copies of `data/scream_input.yaml` with
`data/scream_input.yaml_0001` where the last digits represent the
number of instances (with added leading zeros). With the new input
files, the users can change each individual file's content so that
ensemble member `_0001` reflects a user-specified configuration.

Beyond `NINST`, the user can also choose to utilize the multi-driver
capability. If `MULTI_DRIVER` is `FALSE` (default), then all instances
will be launched from the same coupler instance, which can be problematic
and can result in out-of-memory errors. Instead, the user can set
`MULTI_DRIVER` to `TRUE` which will result in a coupler instance
for each `NINST`.

## RCS testing

The Reproducible Climate Statistics (RCS) test compares two ensembles of
short simulations that differ only in the random seed used to perturb the
initial condition, and asks whether there is evidence that they were
produced by *different* models.

The statistical machinery lives in
`components/eamxx/cime_config/SystemTests/rcs_stats.py`, which can be driven
by the CIME system test or run standalone on any two directories of ensemble
output.

### What the sample actually is

This is the single most important thing to understand about RCS, because it
determines what the p-values mean.

Each ensemble member is one simulation. For a given variable, a member
contributes a *vector* of numbers; how that vector is built is set by
`--analysis_type`. The values within one member's vector are **not**
independent of one another: global monthly means from a single simulation are
serially correlated and share a seasonal cycle, and column means from a single
simulation are spatially correlated.

The *members*, by contrast, are genuinely independent draws, and under the null
hypothesis (same model, different seed) the members of the two ensembles are
exchangeable. `rcs_stats.py` therefore keeps track of which values came from
which member instead of flattening everything into one anonymous array, and it
offers a calibration mode that exploits exactly that structure.

#### `spatiotemporal` (default)

Area-weighted global mean at each output time, so a member contributes
`n_time` values.

- Vertical dimensions (`lev`, `ilev`) are collapsed with an unweighted mean
- The area weighting is renormalized over the valid (non-masked) cells at each
  time, so a partial land/ocean mask does not bias the global mean low
- Sensitive to global systematic biases
- **Recommended default**

#### `temporal`

Time mean at each column, so a member contributes `n_col` values.

- Detects spatially localized differences that a global mean would average away
- Columns are strongly spatially correlated, so the nominal sample size badly
  overstates the information content; pair this with `--calibration member`
- A column is used only if it is finite in *every* member of *both* ensembles,
  which keeps positions aligned across members

#### `member` (analysis type)

A single area-weighted, time-averaged value per member, so a member
contributes exactly one value.

- The cleanest sample: one independent observation per member, no
  within-member correlation to worry about
- Removes the seasonal cycle from the noise, which can make it *more*
  powerful against a steady bias than `spatiotemporal` despite the far
  smaller sample
- With only a handful of members it has little power against subtle changes

### Calibration: how p-values are computed

`--calibration` selects how the test statistic is turned into a p-value.

#### `asymptotic` (default)

SciPy's closed-form or asymptotic p-value, applied to the pooled values.

Fast, and the historical RCS behavior, but it treats every pooled value as an
independent observation. Because the pooled values are correlated, the
effective sample size is smaller than the nominal one and these p-values are
**anti-conservative** — they reject more readily than their nominal level
suggests. The report prints an estimated effective sample size (from the
within-member lag-1 autocorrelation) so the size of that inflation is visible
rather than hidden.

#### `member` (calibration)

The test statistic is recalibrated by permuting *whole members* between the two
ensembles.

This is exact under exchangeability of members and makes no independence
assumption about the values within a member. It is the statistically
defensible mode. Its cost is resolution: with `n` members per ensemble the
smallest attainable p-value is about `2 / C(2n, n)`, because a partition and
its complement give the same two-sided statistic.

| Members per ensemble | Partitions `C(2n,n)` | Smallest attainable p |
| -------------------- | -------------------- | --------------------- |
| 2                    | 6                    | 0.33                  |
| 3                    | 20                   | 0.10                  |
| 4                    | 70                   | 0.029                 |
| 5                    | 252                  | 0.0079                |
| 6                    | 924                  | 0.0022                |
| 8                    | 12870                | 1.6e-4                |
| 9                    | 48620                | 4.1e-5                |

Up to 8 members per ensemble the enumeration is exhaustive. Beyond that,
`rcs_stats.py` draws `--n_resamples` random permutations instead and the
resolution becomes `1 / (n_resamples + 1)`.

This matters in practice. A Bonferroni-corrected `alpha = 0.01` over ~100
variables puts the decision threshold at `1e-4`, which needs roughly **9
members per ensemble** to be reachable at all. A default `RCS_P4_C4` test has
4, where the finest attainable p-value is 0.029 — well above the threshold, so
no variable could ever fail.

Rather than silently reporting `PASS` in that situation, `rcs_stats.py` treats
it as a configuration error and fails the comparison with an explanation. A
test that cannot fail provides no assurance, and reporting success for it is
worse than reporting nothing.

Remedies, in order of preference: add ensemble members; test fewer variables
with `--variables`; relax `--alpha`. Note that changing `--correction_method`
does not help much, since every correction offered here bottoms out at
`alpha / n` for the smallest p-value.

```shell
# exact calibration, drawing 5000 random member permutations
rcs_stats.py /run/dir /base/dir --calibration member --n_resamples 5000
```

### Available statistical tests

RCS provides 11 two-sample tests in three categories, all built on
[SciPy stats](https://docs.scipy.org/doc/scipy/reference/stats.html).

#### Distribution tests

Compare the entire distribution.

| id | Test | Notes |
| ---- | ------ | ------- |
| `ks` | Kolmogorov-Smirnov | Largest gap between the empirical CDFs. Responds mostly to shifts in the bulk; weak in the tails. **Default.** |
| `ad` | Anderson-Darling | Tail-weighted. See the caveat below. |
| `cvm` | Cramér-von Mises | Integrates the squared CDF gap rather than taking its maximum, so it uses the whole distribution. Usually a little more powerful than KS against diffuse differences. |
| `epps` | Epps-Singleton | Compares empirical characteristic functions. Unlike KS/CvM it is valid for discrete data, and it picks up location and scale together. Needs at least 5 values per sample. |
| `energy` | Energy distance | Consistent against *any* distributional difference, but SciPy provides no null distribution, so it **requires `--calibration member`**. Requesting it with asymptotic calibration is rejected up front. |

!!! note "Anderson-Darling p-values are clipped"
    SciPy's `anderson_ksamp` reports a `significance_level` that is already a
    p-value (not a percentage) and that is clipped to the range
    `[0.001, 0.25]`. That clipping makes it useless against a corrected
    threshold below 0.001, so `rcs_stats.py` asks SciPy for a permutation
    p-value where the installed version supports it. Earlier versions of this
    module divided `significance_level` by 100, treating it as a percentage;
    that was wrong, and it made the `ad` path report `FAIL` for two samples
    drawn from *identical* distributions.

#### Location tests

Compare central tendency.

| id | Test | Notes |
| ---- | ------ | ------- |
| `mw` | Mann-Whitney U | Distribution-free; in practice a rank-based comparison of central tendency. |
| `ttest` | Welch's t-test | Unequal-variance t-test on the means. |
| `brunner` | Brunner-Munzel | Tests stochastic equality. Unlike Mann-Whitney it does not assume equal shapes, so it stays valid when the variances differ. |

#### Scale tests

Compare spread.

| id | Test | Notes |
| ---- | ------ | ------- |
| `levene` | Levene (median-centered) | Robust to non-normality. |
| `ansari` | Ansari-Bradley | Assumes the two samples share a location, so pair it with a location test rather than using it alone. |
| `mood` | Mood | Rank test for a difference in scale. |

### Configuration parameters

All parameters are available as command-line arguments to `rcs_stats.py` and
as keyword arguments to `run_stats_comparison()`.

#### Core

##### `--test_type` (default: `ks`)

Statistical test identifier, from the tables above.

##### `--analysis_type` (default: `spatiotemporal`)

How each member is reduced to a sample: `spatiotemporal`, `temporal` or
`member`.

##### `--calibration` (default: `asymptotic`)

`asymptotic` for SciPy's formula on the pooled values, or `member` for exact
permutation of whole members.

##### `--alpha` (default: `0.01`)

Significance level. Lower values reject less readily.

##### `--n_resamples` (default: `2000`)

Number of random member permutations to draw when exhaustive enumeration is
impractical (more than 8 members per ensemble).

##### `--seed` (default: `20250101`)

RNG seed for permutation calibration. Fixed by default so that repeated runs
on the same data give the same answer; the previous implementation used an
unseeded global RNG, which made the `energy` test non-reproducible.

##### `--variables` (default: all)

Restrict the comparison to an explicit list of variables. Useful both for
debugging and for reducing the multiple-testing penalty.

##### `--variables_file` and `--variable_set`

Read a curated variable list from a file: a JSON list, a JSON object mapping
set names to lists, or a plain text file with one name per line.

Fixing the variable set matters more than it looks. When the tested set is
"whatever happened to be in the output stream", the multiple-testing penalty —
and therefore the meaning of a PASS — silently changes every time someone
edits the output YAML. A curated list keeps the criterion comparable across
runs and across time. It is the approach
[evv4esm](https://github.com/LIVVkit/evv4esm) takes for its MVK test, and its
`ks_vars.json` ships both a `default` set and a 63-variable `scream` set that
can be used directly:

```shell
rcs_stats.py /run/dir /base/dir \
    --variables_file /path/to/evv4esm/extensions/ks_vars.json \
    --variable_set scream
```

Names in the list that the output stream does not carry are reported and
skipped rather than treated as an error, since one list is meant to serve
several configurations. If *nothing* in the list matches, that is an error.

##### `--no_power_analysis`

Skip the minimum-detectable-effect estimate described under
[Output](#output). The estimate is on by default and costs one extra test
evaluation per variable per grid point.

##### `--verbose` (default: off)

Emit debug-level logging while the comparison runs.

#### Combining the per-variable verdicts

`--global_test` (default: `variable_count`) chooses how the per-variable
results become a single verdict.

##### `variable_count` (default)

Correct the p-values across variables, then fail if more than
`--max_failed_vars` (or more than `--max_failed_fraction`) fail. This is the
familiar behavior.

##### `calibrated_count`

Judge the *number* of rejections against its own permutation null
distribution. Members are regrouped between the two ensembles, the whole
pipeline is re-run for every regrouping, and the resulting counts form an
empirical null for "how many variables reject when nothing is wrong".

This is worth having for two reasons.

First, **it accounts for correlation between variables**. Bonferroni and
Benjamini-Hochberg assume the tests are independent or positively dependent in
a specific way; climate variables are neither. Benjamini-Yekutieli is valid
under arbitrary dependence but pays for it with a large loss of power. The
permutation null measures the dependence actually present in your output, so
nothing has to be assumed.

Second, **it stays usable with few members**. It spends its resolution on a
single p-value rather than one per variable, so it is judged against
`--global_alpha` (default 0.05) rather than `alpha / n_variables`. With 4+4
members the finest attainable p-value is 0.029, which is below 0.05 — so
unlike the per-variable permutation test, this one can actually reject at the
ensemble sizes RCS is usually run at.

It is also markedly more sensitive to a *diffuse* change that nudges many
variables a little, which is the characteristic signature of a
climate-altering bug; a per-variable correction is tuned to find a single
large outlier instead.

```shell
rcs_stats.py /run/dir /base/dir \
    --analysis_type member \
    --global_test calibrated_count
```

The report prints the observed count, the number expected by chance, and the
median, 95th percentile and maximum of the permutation null. That null
distribution is an empirical calibration of your exact configuration on your
exact data — if the observed count sits comfortably inside it, the pipeline is
behaving as advertised.

##### `--global_alpha` (default: 0.05)

Used both to screen each variable and to judge the resulting count. If the
number of available member assignments cannot produce a p-value this small,
the comparison fails as a configuration error rather than reporting a PASS it
could not have avoided.

!!! warning "Downstream gates do not reach this verdict"
    `calibrated_count` builds its verdict from p-values, so anything that acts
    *after* the p-value — `--magnitude_threshold`, `--equivalence_margin`,
    `--max_failed_vars`, `--max_failed_fraction` — still shapes the
    per-variable table but does not bound the overall PASS/FAIL. A warning is
    emitted when they are combined. Use `--global_test variable_count` if you
    need those gates to be decisive.

#### Multiple-testing correction

Testing hundreds of variables at `alpha = 0.01` will produce failures by
chance alone. `--correction_method` (default: `bonferroni`) adjusts for that.
It applies to `--global_test variable_count`; `calibrated_count` handles
multiplicity through its permutation null instead.

| value | Method | Controls |
| ------- | -------- | ---------- |
| `bonferroni` | Bonferroni | Family-wise error rate. Conservative. |
| `holm` | Holm-Bonferroni | Family-wise error rate, uniformly more powerful than Bonferroni at the same guarantee. Prefer this over `bonferroni`. |
| `fdr` | Benjamini-Hochberg | False discovery rate. Less conservative; better power. |
| `fdr_by` | Benjamini-Yekutieli | False discovery rate, valid under arbitrary dependence between tests. Appropriate when variables are strongly correlated. |
| `none` | — | No correction. High false-positive rate with many variables. |

The correction is applied through
[statsmodels' `multipletests`](https://www.statsmodels.org/stable/generated/statsmodels.stats.multitest.multipletests.html)
and is guaranteed to only ever *remove* rejections. It can never turn a
variable that passed into one that failed. (The previous hand-rolled
implementation could do exactly that, by re-deciding each variable from its
p-value alone and discarding the magnitude threshold that had already settled
the question.)

#### Failure thresholds

##### `--max_failed_vars` (default: `0`)

The overall test fails when more variables than this fail. Use `0` for strict
NBFB validation.

##### `--max_failed_fraction` (default: unset)

The overall test also fails when the *fraction* of failing variables exceeds
this. Unset by default, which disables the criterion.

!!! warning "`--critical_fraction` has changed meaning"
    `critical_fraction` is still accepted as a deprecated alias for
    `--max_failed_fraction`, but it now controls the overall failing fraction.
    It previously documented a per-variable, per-column threshold that was
    never actually implemented — the parameter was accepted, printed in the
    report, and then ignored. Existing callers should move to
    `--max_failed_fraction`.

##### `--fail_on_error` (default: off)

Treat a variable that could not be tested as a failure. Errored variables are
always listed in the report and are never counted as passing; this flag makes
them fatal. If *no* variable could be tested, the comparison fails regardless,
since a vacuous comparison must not report success.

#### Effect size filtering

##### `--magnitude_threshold` (default: unset)

Minimum relative mean difference, computed as
`|mean1 - mean2| / ((|mean1| + |mean2|) / 2)`, required before a statistically
detectable difference is counted as a failure. Use it to ignore differences
that are real but physically negligible.

#### Equivalence testing

##### `--equivalence_margin` (default: unset)

A variable that is not rejected has **not** been shown to be equivalent; it has
merely failed to be shown different. With few members, almost nothing gets
rejected, and a `PASS` mostly reflects low power rather than agreement.

Setting `--equivalence_margin` adds a TOST (two one-sided tests) on the
member-level means — the only values in the analysis that are genuinely
independent. The margin is expressed in units of the baseline ensemble's own
member-to-member standard deviation, so `--equivalence_margin 1.0` reads "the
ensemble means agree to within one baseline ensemble sigma".

When it is set, `PASS` means *shown equivalent* rather than *not shown
different*, which is the logic a reproducibility claim actually needs. It is
also demanding: demonstrating equivalence takes more members than failing to
detect a difference does.

### Variable selection

A variable is testable when it

- has a `time` dimension,
- has a floating-point type,
- is not a coordinate or grid-description variable (`time`, `lat`, `lon`,
  `lev`, `ilev`, `area`, the hybrid coefficients, and so on), and
- is present in **every** member of **both** ensembles.

Variables that fail the last condition are reported in a `SKIPPED VARIABLES`
section rather than being quietly ignored. The all-NaN and constant screening
happens on the reduced sample, so a full 4-D field is never pulled into memory
just to decide whether to look at it.

### Missing-value handling

- Area-weighted means renormalize over the valid cells at each time, so a
  partial mask does not bias the result
- For `temporal`, a position is used only if it is finite in every member of
  both ensembles, which keeps positions aligned across members. The previous
  implementation dropped NaNs per member and then truncated to the shortest
  array, which silently compared one member's column *i* against a different
  physical location in another member
- Every position dropped this way is reported in a `DATA NOTES` section, so a
  comparison that threw away most of the domain is visible even when it passes

### Output

#### Console and test log

The report is appended to the CIME test log and printed by the standalone
script. It contains:

- the full configuration, including the RNG seed and file patterns
- the member counts and instance numbers of both ensembles
- a `RESOLUTION AND POWER` section: the strictest per-variable threshold after
  correction, the smallest attainable permutation p-value for the member count
  at hand, and (for asymptotic calibration) the median ratio of effective to
  nominal sample size
- a pass/fail summary
- per-variable detail for the worst failures, with sample statistics, p-values
  before and after correction, and effect sizes
- a `MINIMUM DETECTABLE EFFECT` section: the smallest injected shift, in
  baseline ensemble sigma, that this configuration would actually have
  rejected. This is what turns a bare `PASS` into a statement with content —
  "no difference detected, and a 2 sigma shift would have been". It is
  computed by shifting the run sample by increasing multiples of the baseline
  ensemble's member spread and re-running the test, and it reports the
  smallest shift from which *every* larger probed shift is also rejected
  (rank statistics are step functions of the shift, so the first crossing
  alone would be unreliable). Because it uses asymptotic p-values it is a
  lower bound on the shift genuinely needed
- a `GLOBAL TEST` section when `--global_test calibrated_count` is in use
- a `PASSED BUT NOTABLE` section listing variables that were not rejected but
  whose ensemble means differ by more than the baseline ensemble's own member
  spread (`snr > 1`) — these are the near-misses worth a human look
- `DATA NOTES`, `SKIPPED VARIABLES` and `ERRORED VARIABLES` sections
- the overall verdict with the reason it was reached

Effect sizes are reported for every variable, not just the failing ones,
because they answer a different question than the p-value does:

- `mean_diff`, `mean_diff_pct`, `median_diff`, `std_ratio`
- `cohens_d` — mean difference in pooled standard deviations
- `snr` — mean difference divided by the baseline ensemble's member-to-member
  spread. This is usually the number a climate scientist wants: a
  statistically significant difference with `snr << 1` is buried inside the
  ensemble's own internal variability
- `ks_distance` — the KS statistic, as a scale-free summary of distributional
  distance, regardless of which test was selected

#### JSON report

A structured report is written to `rcs_<test>_<analysis>_results.json`
containing `configuration`, `ensembles`, `summary`, `skipped_variables`,
`errored_variables` and per-variable `details`.

##### Choosing where the JSON report is written

`RUNDIR` frequently lives on a shared filesystem that the person running the
comparison cannot write to. `--json_output` controls the destination:

| value | Behavior |
| ------- | ---------- |
| a path ending in `.json` | Used verbatim; parent directories are created as needed |
| a directory | The report is written inside it under the generated name |
| `none` (or `off`, `no`, `disabled`) | No JSON report is written |
| unset | Falls back to `$RCS_JSON_OUTPUT`, then to `run_dir` |

```shell
# explicit file
rcs_stats.py /run/dir /base/dir --json_output ~/reports/rcs_run42.json

# directory
rcs_stats.py /run/dir /base/dir --json_output ~/reports

# skip it entirely
rcs_stats.py /run/dir /base/dir --json_output none

# or set it once for a whole session
export RCS_JSON_OUTPUT=$HOME/rcs_reports
```

If the chosen location turns out to be unwritable, the report falls back to the
system temporary directory and the substitution is noted in the console output.
If that also fails, the failure is reported and the comparison continues:
losing the report never turns a passing comparison into a failing one.

Within the CIME system test, the report defaults to the **case directory**
rather than `RUNDIR`, since the user necessarily owns the case directory.
Setting `RCS_JSON_OUTPUT` in the environment overrides that.

### Running RCS tests

#### Within the CIME system test framework

To run RCS as a CIME system test you must request multiple instances, either
by adjusting the runtime settings discussed above, or by appending `_N#` (same
driver) or `_C#` (multiple drivers) to the test name.

You must also enable a perturbation across instances. Adding a perturbed field
to the `scream` configuration (e.g.
`initial_conditions::perturbed_fields="T_mid"`) suffices. The RCS test then
gives each instance a different seed so that each follows a different
trajectory. RCS is designed to return identical seeds, and therefore identical
results, unless code or configuration changes introduce numerical or climate
differences.

A testmod exists that enables the perturbation and sets up the monthly average
output stream that RCS copies across instances:

```shell
./cime/scripts/create_test RCS_P4_C4.$RES.$COMPSET.$MACH.eamxx-perturb
```

`RCS_P4_C4` gives 4 multi-driver instances each using a pelayout of 4, with the
`eamxx-perturb` testmod applied during setup. `$RES`, `$COMPSET` and `$MACH`
are the usual create_test arguments.

!!! tip "Four members is a small ensemble"
    `_C4` is cheap and catches gross errors, but see the resolution table
    above: with 4 members per ensemble the permutation calibration cannot
    reach a corrected threshold, and even the asymptotic calibration has
    limited power against subtle changes. For a result you intend to rely on,
    prefer 8 or more members per ensemble.

#### Standalone command-line usage

The comparison runs independently of CIME on any two directories of ensemble
output, which is useful for re-analyzing archived results, trying different
statistical settings without re-running simulations, and debugging.

```text
# Default: KS on area-weighted global means, Bonferroni corrected
rcs_stats.py /run/dir /base/dir
```

```text
# Exact member-level permutation calibration
rcs_stats.py /run/dir /base/dir --calibration member
```

```text
# Per-member scalars, which removes the seasonal cycle from the noise
rcs_stats.py /run/dir /base/dir --analysis_type member --calibration member
```

```text
# Positively demonstrate equivalence within one ensemble sigma
rcs_stats.py /run/dir /base/dir \
    --analysis_type member \
    --equivalence_margin 1.0
```

```text
# Holm correction, report to a writable location
rcs_stats.py /run/dir /base/dir \
    --correction_method holm \
    --json_output ~/rcs_reports
```

```text
# Require practical as well as statistical significance
rcs_stats.py /run/dir /base/dir \
    --magnitude_threshold 0.01
```

```text
# Focus on a few variables to soften the multiple-testing penalty
rcs_stats.py /run/dir /base/dir \
    --variables T_mid ps surf_flux \
    --calibration member
```

```text
# Global test on the rejection count, which works at small member counts
rcs_stats.py /run/dir /base/dir \
    --analysis_type member \
    --global_test calibrated_count
```

```text
# Reuse evv4esm's curated EAMxx variable set
rcs_stats.py /run/dir /base/dir \
    --variables_file /path/to/evv4esm/extensions/ks_vars.json \
    --variable_set scream
```

```text
# Compare ensembles written with different filename conventions
rcs_stats.py /run/dir /base/dir \
    --run_file_pattern "*.eam_????.h0.*.nc" \
    --base_file_pattern "*.scream_????.h.AVERAGE.*.nc"
```

The exit status is 0 for `PASS` and 1 for `FAIL`, so the script can be used
directly in a shell pipeline.

##### File pattern customization

`--run_file_pattern` and `--base_file_pattern` (both defaulting to
`*.scream_????.h.AVERAGE.*.nc`) accept an ordinary glob containing exactly one
`????` placeholder marking the 4-digit instance number. Files that match the
glob but do not carry a 4-digit number in that position are ignored, and a
pattern with no placeholder is rejected with a clear error.

##### Getting help

```text
rcs_stats.py --help
```

### What RCS does not test

RCS compares two *climates*: it asks whether the long-run statistics of the
two ensembles differ. That is the question that matters scientifically, but it
is an expensive and blunt instrument — every verdict costs a year of
simulation per member, and a change has to survive being averaged over a year
and a globe before RCS can see it.

Two complementary approaches in
[evv4esm](https://github.com/LIVVkit/evv4esm) test the *model operator*
instead of the attractor it produces, and they are far cheaper and far more
sensitive. Neither is implemented here; they need different run
configurations, not different post-processing, so they belong in separate
system tests.

#### Perturbation growth (PGN)

Perturb the initial condition at the level of machine epsilon and compare how
that perturbation grows through each physics parameterization within a
*single* time step, against the spread the unmodified code produces. Because
it looks at one step rather than one year, essentially any change to the
equations shows up, and the failure is localized to the parameterization that
caused it. Cost: one time step.

#### Time step convergence (TSC)

Run a truth ensemble at a 1-second time step and reference and test ensembles
at 2 seconds for 10 simulated minutes, compute each ensemble's RMSD against
truth, and test whether the mean difference of those RMSDs is zero. This
checks that the *discretization error behaves the same way*, which catches
changes RCS would need many ensemble members to resolve. Cost: 10 simulated
minutes.

A complete NBFB verification story wants all three: PGN and TSC to catch
changes to the model operator quickly and diagnostically, and RCS to answer
the question they cannot — whether the climate itself moved.

### Known limitations

These are properties of the method, not bugs, and they bound what a `PASS`
means.

- **Vertical averaging is mass-unweighted.** `lev` and `ilev` are collapsed
  with a plain mean, so a difference that changes sign with height can cancel
  itself out. Variables whose signal is confined to a few levels are harder to
  detect than column-integrated ones.
- **Asymptotic p-values are anti-conservative.** They assume independence that
  the pooled values do not have. The reported effective sample size shows how
  far off they are; `--calibration member` removes the assumption at the cost
  of resolution.
- **Power is set by the member count, not the number of timesteps.** Adding
  output frequency inflates the nominal sample size without adding much
  information. Adding members is what actually helps. For reference,
  evv4esm's MVK test defaults to **30 members per ensemble**, against RCS's
  usual 4.
- **Pooling timesteps can destroy the very power it appears to add.** The
  default `spatiotemporal` mode puts the seasonal cycle into the sample
  variance, so a steady offset has to compete against a spread it does not
  belong to. On synthetic ensembles where `--analysis_type member` detects a
  4 sigma shift, `spatiotemporal` can fail to detect 16 sigma in the same
  variable. If in doubt, read the `MINIMUM DETECTABLE EFFECT` section of the
  report — and consider `--analysis_type member`, which is the sampling
  choice MVK makes.
- **A `PASS` is not a proof of equivalence** unless `--equivalence_margin` is
  set. Without it, `PASS` means only that the configured test did not detect a
  difference.
