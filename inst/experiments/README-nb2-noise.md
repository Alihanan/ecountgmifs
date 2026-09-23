# NB2 observation-noise study

This is an univariate, unpenalized simulation for the fractional-response
reviewer comment. It measures distortion introduced by observation noise,
clamping, and the gamma-extended NB2 objective. It does not establish that the
gamma extension is a normalized continuous likelihood, or address sparsity,
predictor correlation, prior quality, or high-dimensional network recovery.

## Install and start

The changed-files ZIP is relative to the package root. Extract it **inside a
copy of the supplied `ecountgmifs` source directory**, replacing matching files.
It is a patch to `ecountgmifs(20260923-112418).zip`, not a standalone package.
Reinstall the package so the new C++ registration is available:

```sh
R CMD INSTALL --preclean /path/to/ecountgmifs
```

```r
library(ecountgmifs)
source(system.file("experiments", "nb2_noise_study.R", package = "ecountgmifs"))

# Small pilot covering every scenario:
pilot <- run.nb2.noise.study(repetitions = 5)

# The planned study. The output directory must not already exist.
study <- run.nb2.noise.study(
  repetitions = 500,
  output.directory = "nb2_noise_results"
)
```

Sourcing the script does not run an experiment. No new runtime dependencies
are introduced. The package uses its existing Rcpp, RcppArmadillo, and nloptr
dependencies. The generated Rcpp wrappers and Rd help files are supplied;
regenerating them is not required to install the patch.

## Generator interface

```r
noise <- noise.gaussian(sd = 1)
set.seed(12)
data <- generate.nb2(n = 100, mu = 5, dispersion = 0.5, noise = noise)
nb2.noise.moments(mu = 5, dispersion = 0.5, noise = noise)
nb2.loglik(data$y, mu = 5, dispersion = 0.5)
nb2.loglik(data$floored, mu = 5, dispersion = 0.5)
nb2.loglik(data$rounded, mu = 5, dispersion = 0.5)

# A new noise family needs only a draw callback. Population moments are optional.
custom <- noise.generator(
  name = "small uniform",
  draw = function(n) runif(n, -0.1, 0.1),
  parameters = list(min = -0.1, max = 0.1)
)
```

`noise.generator()` returns a readable list with name, draw callback, optional
moment callback, and fixed parameters. The built-ins are `noise.none()`,
`noise.uniform.positive()`, `noise.uniform.centered()`, and `noise.gaussian(sd)`.
See `?noise.generator`, `?generate.nb2`, `?nb2.noise.moments`, and `?nb2.loglik`.
Callbacks draw noise independently of the counts. Every negative noisy response
is clamped to zero, including custom-noise responses. Samples retain latent
counts, noise draws, values before clamping, observed values, floored/rounded
values, and a clamping indicator.

## Parameter and likelihood conventions

The latent NB2 variance is `mu + dispersion * mu^2`; R's size is
`1 / dispersion`. Dispersion is unrelated to the elastic-net mixing parameter.
Zero dispersion generates Poisson counts. Positive dispersion remains NB in
the generator, even when extremely small.

`nb2.loglik()` delegates to the existing `NB2Family::negloglik()` in
`src/example.h` and reverses its sign. It retains all terms, including
`lgamma(y + 1)`, and uses the existing mean caps `[1e-12, 1e12]` and Poisson
fallback for dispersion `<= 1e-8`. The likelihood formulas, fitting optimizer,
gradients, and model selection criteria were not changed. The new
`prepare_response()` method shares cache preparation with ordinary fitting.

The distinction between an exact NB generator and the likelihood's numerical
Poisson approximation matters near zero dispersion, but not at the generating
dispersions in this study. At zero requested mean the likelihood still uses
its positive mean cap, as the current fitting implementation does.

## Design

| Parameter | Values |
|---|---|
| Latent mean | 0.5, 5, 50 |
| Latent NB2 dispersion | 0.1, 0.5, 2 |
| Training sample size | 100, 1000 |
| Independent integer test sample size | 1000 |
| Replications per setting | 500 |
| Noise arms | None; U(0,1); U(-0.5,0.5); N(0,1/12); N(0,1) |

There are 18 generating settings, 90 setting/arm combinations, and four
response methods: latent oracle, untransformed noisy response, floor, round.
Each replication shares the latent training counts and integer test sample
across all arms and methods. Repeated oracle rows in different arms are the
same fit; they are not additional independent replications.

Flooring U(0,1) noise and rounding centered-uniform noise recover the counts
almost surely after clamping. These are the primary integer comparators.
Both floor and round are nevertheless recorded for all arms as requested.
Gaussian conversion is approximate, so its converted samples are not claimed
to follow the original NB law. R uses ties-to-even rounding; exact half ties
have probability zero in the proposed continuous distributions.

`fit.nb2.intercept()` fixes the mean at the sample mean, capped as in C++, then
profiles dispersion. It checks zero and the endpoints of `[1e-6, 100]`, scans
41 log-spaced positive values, and refines grid-local maxima with `optimize()`.
The positive interval `(0, 1e-6)` is excluded to limit cancellation in the
existing gamma expression near zero. Search-boundary hits are explicitly
reported; they require sensitivity analysis before interpreting fitted bias.
`dispersion.bounds` and `search.points` are configurable. All-zero samples
have unidentified dispersion and retain `NA` rather than an invented estimate.

## Outputs and interpretation

`study$results` contains one row per setting, replication, arm, and method:

- Sample mean, unbiased sample variance, moment dispersion, zero proportion,
  clamping proportion, and fraction differing from latent counts.
- C++ log objective at the generating parameters and at the fitted parameters,
  both total and per observation.
- Fitted mean and dispersion, errors against latent parameters, and paired
  differences from the oracle estimates.
- Mean log score on the common independent integer test counts and its paired
  difference from the oracle score. Higher scores indicate better prediction
  of the **latent integer counts**, not the noisy observations.
- Fit status (`ok`, `poisson_boundary`, `positive_lower_bound`, `upper_bound`,
  `all_zero`, or `failed`), mean-cap indicator, and error message.

`study$summary` reports latent-parameter bias, RMSE, paired differences, average
diagnostics, boundary/failure rates, and Monte Carlo standard errors. It reports
the number of finite replications for each metric. Missing estimates are
excluded from the corresponding metric only; failure counts remain visible.
RMSE MCSE uses a delta-method approximation and is undefined at zero RMSE.
Boundary fits remain in the bias/RMSE summaries, with their frequencies shown.

`study$theoretical.moments` describes the **untransformed, clamped** population
in each arm. Its moment dispersion is `(variance - mean) / mean^2`, which may
be negative. It is neither the latent generating dispersion nor necessarily
the pseudo-true gamma-objective dispersion. Custom noise without a moments
callback returns `NA` in this table.

Do not rank methods using raw objective values across `y`, `floor(y)`, and
`round(y)`: these use different responses, and the fractional objective is not
a normalized continuous density. Use parameter errors and the common integer
test score for performance comparisons. The raw objectives are retained to
inspect numerical discrepancies and within-response objective shape.

`study$representative.data` retains the first replicate of every cell and arm.
For example, obtain a centered objective surface for one such response:

```r
data <- pilot$representative.data$cell_1_gaussian_sd1
surface <- nb2.objective.surface(
  data$y,
  mu.values = seq(0.1, 1.5, length.out = 31),
  dispersion.values = c(0, exp(seq(log(0.01), log(3), length.out = 30)))
)
z <- matrix(surface$relative.loglik.per.observation, nrow = 31)
contour(unique(surface$mu), unique(surface$dispersion), z,
        xlab = "Mean", ylab = "NB2 dispersion")
points(0.5, 0.1, pch = 4)  # latent truth for cell 1
fit <- fit.nb2.intercept(data$y)
points(fit$mu, fit$dispersion, pch = 19)
```

Each surface is centered at its own grid maximum and divided by sample size.
Such contours are not calibrated confidence regions. Extend grids if their
maxima fall at an edge.

The optional output directory contains per-cell CSV checkpoints, summary and
theory CSV files, `study.rds`, and `sessionInfo.txt`. Settings retain the seed,
RNG kind, package/R versions, grid, noise objects, and numerical settings.
The caller's RNG state is restored. Reordering arms or changing the grid changes
subsequent draws; repeat the same settings for exact reproducibility.

## Mathematics and provenance

`inst/paper/nb2_noise_moments.tex` is a standalone short LaTeX appendix with its
bibliography. It separates standard independence identities and this study's
centered-uniform calculation from the published censored-normal formulas of
Foi et al. (2008). Ouimet (2023) provides precedent for uniform NB jittering,
not a justification of the gamma kernel as a continuous density.

Gaussian moments average the **clipping corrections** for counts up to
`ceiling(10 * sd)` and add the exact unclipped moments. This avoids truncating
the NB tail. `nb2.noise.moments()` returns absolute bounds on omitted Gaussian
corrections; these do not include floating-point roundoff. For custom very
large Gaussian SDs, the moment function refuses sums over a million terms.

## Validation

Run the included package tests with:

```r
library(ecountgmifs)
testthat::test_local("/path/to/ecountgmifs")
```

The tests cover agreement with integer `dnbinom()`, the existing fractional R
reference, mean caps and fallback, response-cache refresh, shared counts,
zero clamping, uniform recovery, moments against numerical integration,
reproducible grid execution, Monte Carlo summaries, and intercept profiling
against an independent two-parameter integer NB optimization.

For this patch, compilation and installation succeeded with R 4.3.3,
Rcpp 1.0.12, RcppArmadillo 0.12.8.1.0, and nloptr 2.0.3 on Linux. The included
tests passed. A separate three-replication pilot covered all 90 setting/arm
combinations (1,080 method-level rows), with no failed fits; the output-file
path and a fractional two-parameter fit were also checked. This pilot is a
software check, not the final 500-replication scientific experiment.

## Files in the patch

Five existing files change: `NAMESPACE`, `R/RcppExports.R`,
`src/RcppExports.cpp`, `src/ecountgmifs.cpp`, and `src/example.h`.
The remaining files are additions: two R implementation files, four Rd help
files, this guide and the grid script, the LaTeX appendix, and two test files.
The archive excludes compiled objects and shared libraries. `--preclean`
ensures that any binaries already present in the original source are rebuilt.
