# ecountgmifs

`ecountgmifs` is an R package for fitting extended generalized monotone incremental forward stagewise models for high-dimensional count data.

The package is designed for sparse regression problems where the response is count-valued and the number of predictors may be large relative to the number of samples. It supports negative-binomial and Poisson model families, elastic-net-style stagewise updates, prior-weighted penalties, unpenalized covariates, offsets, and information-criterion-based model selection along the solution path.

## Installation

Install the package from GitHub:

```r
# install.packages("remotes")
remotes::install_github("Alihanan/ecountgmifs")
```

For substantive analyses, increase this value or omit the explicit example control and use the package default.

## Basic usage

```r
library(ecountgmifs)

set.seed(1)
X <- matrix(rpois(100 * 20, lambda = 5), nrow = 100, ncol = 20)
colnames(X) <- paste0("X", seq_len(ncol(X)))
y <- rpois(100, lambda = 10)

fit <- ecountgmifs(
  X = X,
  y = y,
  family = "negative.binomial",
  enet.alpha = 0.75
)

print(fit)
summary(fit)
coef(fit, state = "terminal")
plot(fit)
```

## Packaged RNA-seq example dataset

The package includes `mrna97_rnaseq`, an RNA-seq example dataset derived from the regulatory-network setting described by Anuarbekov and Kléma (2025). The original test set used in the paper is not included in this package dataset.

```r
data(mrna97_rnaseq)

names(mrna97_rnaseq)
dim(mrna97_rnaseq$Y)
dim(mrna97_rnaseq$X)
dim(mrna97_rnaseq$prior)
dim(mrna97_rnaseq$truth)
```

For the packaged object used here, `Y` has 434 samples and 97 target mRNAs, `X` has 434 samples and 2636 candidate predictors, and the prior/reference matrices use **predictor-by-target orientation**. Thus, for target `j`, use `prior[, j]` and `truth[, j]`.

A single-target prior-weighted fit is therefore:

```r
library(ecountgmifs)
data(mrna97_rnaseq)

j <- 1L

Y <- mrna97_rnaseq$Y
X <- mrna97_rnaseq$X
prior <- mrna97_rnaseq$prior

prior_j <- prior[, j]
weights <- ifelse(prior_j, 0.1, 1)

stopifnot(length(weights) == ncol(X))

fit <- ecountgmifs(
  X = X,
  y = Y[, j],
  weight.vec = weights,
  family = "negative.binomial",
  enet.alpha = 0.75
)
```

## Packaged T-100 airport passenger-flow dataset

The package also includes `airport_t100`, a monthly U.S. airport passenger-flow dataset derived from the Bureau of Transportation Statistics T-100 Domestic Segment benchmark. The data cover January 1990 through December 2025, giving 432 monthly observations.

The object contains an airport-level monthly count matrix together with directed route-reference matrices at several temporal-persistence thresholds:

* `GT1`: route observed in at least 1 month;
* `GT12`: route observed in at least 12 months;
* `GT120`: route observed in at least 120 months;
* `GT216`: route observed in at least 216 months;
* `GT432`: route observed in all 432 months.

The reference matrices use destination-by-origin orientation. For destination airport `i`, the response is the corresponding airport count series and the remaining airport series are candidate predictors.

```r
library(ecountgmifs)
data(airport_t100)

i <- 1L

X <- airport_t100$X_out
y <- airport_t100$X_in[, i]
truth_i <- airport_t100$GT12[i, ]

stopifnot(
  ncol(X) == length(truth_i),
  identical(colnames(X), airport_t100$airport)
)

fit_airport <- ecountgmifs(
  X = X,
  y = y,
  family = "negative.binomial",
  enet.alpha = 0.75,
  control = ecountgmifs.control(
    stagewise.iteration.max = 500L
  )
)
```

## Preprocessing utilities

For RNA-seq-style count matrices, sample-level normalization can be applied before predictor transformation:

```r
library(ecountgmifs)
data(mrna97_rnaseq)

X <- mrna97_rnaseq$X

X_cpm <- normalize_counts(
  X,
  method = "cpm"
)

X_pre <- transform_predictors(
  X_cpm,
  method = "standardize_log1p"
)
```

Available count normalizations are:

* `"none"`: keep the original scale;
* `"cpm"`: counts per million scaling;
* `"tmm"`: TMM normalization through `edgeR`, if `edgeR` is installed.

Available RNA-seq predictor transformations are:

* `"none"`
* `"log1p"`
* `"asinh"`
* `"standardize"`
* `"standardize_log1p"`
* `"standardize_asinh"`

For negative-binomial models, the response should usually remain on the original count scale. When a sample-level offset is needed, request the normalization factor and pass it to `ecountgmifs()`:

```r
library(ecountgmifs)
data(mrna97_rnaseq)

X_raw <- mrna97_rnaseq$X
y <- mrna97_rnaseq$Y[, 1L]

norm <- normalize_counts(
  X_raw,
  method = "cpm",
  return.offset = TRUE
)

X <- norm$x
offset <- norm$offset

fit_offset <- ecountgmifs(
  X = X,
  y = y,
  offset = offset,
  family = "negative.binomial"
)
```

For monthly airport passenger-flow matrices, temporal transformations can reduce seasonal and system-wide temporal effects:

```r
library(ecountgmifs)
data(airport_t100)

X <- airport_t100$X
month <- airport_t100$sample

X_air <- transform_time_series(
  X,
  method = "log1p_month_demean",
  month = month
)
```

Available temporal transformations are:

* `"none"`
* `"log1p_month_demean"`
* `"month_z"`
* `"log1p_month_z"`
* `"log1p_gaussian_smooth"`
* `"log1p_gaussian_smooth_month_demean"`
* `"log1p_diff1"`

## Prior-weighted fitting

Prior information can be supplied through `weight.vec`. Smaller weights reduce the penalty for selected predictors. This example defines its own data and weights and can be run independently:

```r
library(ecountgmifs)

set.seed(2)
X <- matrix(rpois(100 * 20, lambda = 5), nrow = 100, ncol = 20)
colnames(X) <- paste0("X", seq_len(ncol(X)))
y <- rpois(100, lambda = 10)

weights <- rep(1, ncol(X))
weights[1:5] <- 0.1

fit_prior <- ecountgmifs(
  X = X,
  y = y,
  weight.vec = weights,
  family = "negative.binomial",
  enet.alpha = 0.75
)
```

For the packaged RNA-seq dataset, use the **column** corresponding to target `j` because the stored prior matrix is predictor by target:

```r
library(ecountgmifs)
data(mrna97_rnaseq)

j <- 1L

X <- mrna97_rnaseq$X
y <- mrna97_rnaseq$Y[, j]
prior_j <- mrna97_rnaseq$prior[, j]
weights <- ifelse(prior_j, 0.1, 1)

stopifnot(length(weights) == ncol(X))

fit_prior <- ecountgmifs(
  X = X,
  y = y,
  weight.vec = weights,
  family = "negative.binomial",
  enet.alpha = 0.75
)
```

## Modular families, links, and criteria

Families, link functions, fused family-link implementations, and information criteria are interchangeable modules. Each example below defines its own data.

Built-in modules avoid runtime compilation and are the simplest choice for ordinary fits:

```r
library(ecountgmifs)

set.seed(3)
X <- matrix(rpois(100 * 20, lambda = 5), nrow = 100, ncol = 20)
colnames(X) <- paste0("X", seq_len(ncol(X)))
y <- rpois(100, lambda = 10)

fit_modular <- ecountgmifs(
  X = X,
  y = y,
  family = plugin.family.nb2.builtin(),
  link = plugin.link.log.builtin(),
  criteria = list(
    AIC = criterion("AIC", type = "full"),
    BIC = criterion("BIC", type = "nnz")
  ),
  enet.alpha = 0.75
)
```

The same fitting interface can mix implementations. For example, the family can remain built in while the link is an R closure and one criterion is live-compiled C++:

```r
library(ecountgmifs)

set.seed(4)
X <- matrix(rpois(100 * 20, lambda = 5), nrow = 100, ncol = 20)
colnames(X) <- paste0("X", seq_len(ncol(X)))
y <- rpois(100, lambda = 10)

fit_mixed <- ecountgmifs(
  X = X,
  y = y,
  family = plugin.family.nb2.builtin(),
  link = plugin.link.capped.log.r.closure(
    mu.min.cap = 1e-12,
    mu.max.cap = 1e12
  ),
  criteria = list(
    EBIC = criterion(
      "EBIC",
      type = "nnz",
      gamma = 0.5
    ),
    AIC_live = criterion(
      "AIC",
      type = "full",
      implementation = "live",
      cache = TRUE
    )
  ),
  enet.alpha = 0.75
)
```

`gamma` is specific to EBIC. `cache` is specific to `implementation = "live"`. The available criterion implementations are `"builtin"`, `"live"`, `"r.environment"`, and `"r.closure"`.

A fused family-link module can be supplied instead of separate family and link modules:

```r
library(ecountgmifs)

set.seed(5)
X <- matrix(rpois(100 * 20, lambda = 5), nrow = 100, ncol = 20)
colnames(X) <- paste0("X", seq_len(ncol(X)))
y <- rpois(100, lambda = 10)

fit_fused <- ecountgmifs(
  X = X,
  y = y,
  family.link = plugin.family.link.nb2.log.builtin(),
  criteria = list(
    AIC = criterion("AIC", type = "full"),
    BIC = criterion("BIC", type = "full")
  ),
  enet.alpha = 0.75
)
```

## Extracting coefficients

The following complete example creates a fit and then extracts several points from its path:

```r
library(ecountgmifs)

set.seed(6)
X <- matrix(rpois(100 * 20, lambda = 5), nrow = 100, ncol = 20)
colnames(X) <- paste0("X", seq_len(ncol(X)))
y <- rpois(100, lambda = 10)

fit_modular <- ecountgmifs(
  X = X,
  y = y,
  family = plugin.family.nb2.builtin(),
  link = plugin.link.log.builtin(),
  criteria = list(
    AIC = criterion("AIC", type = "full"),
    BIC = criterion("BIC", type = "nnz")
  ),
  enet.alpha = 0.75
)

beta_path <- coef(fit_modular)
beta_terminal <- coef(fit_modular, state = "terminal")

beta_by_criterion <- coef(
  fit_modular,
  criterion = c(
    "AIC.builtin",
    "BIC.nnz.builtin"
  )
)

beta_by_iteration <- coef(
  fit_modular,
  iteration = c(50, 100),
  include.theta = TRUE
)
```

For a multi-alpha object, select alpha directly:

```r
library(ecountgmifs)

set.seed(7)
X <- matrix(rpois(100 * 20, lambda = 5), nrow = 100, ncol = 20)
colnames(X) <- paste0("X", seq_len(ncol(X)))
y <- rpois(100, lambda = 10)

fit_multi <- ecountgmifs(
  X = X,
  y = y,
  family = "negative.binomial",
  enet.alpha = c(0.50, 0.75, 1.00),
  criteria = list(
    AIC = criterion("AIC", type = "full"),
    BIC = criterion("BIC", type = "full")
  )
)

beta_alpha <- coef(
  fit_multi,
  alpha = 0.75,
  criterion = "BIC.builtin"
)
```

For a prior-strength grid, select both alpha and eta:

```r
library(ecountgmifs)

set.seed(8)
X <- matrix(rpois(100 * 20, lambda = 5), nrow = 100, ncol = 20)
colnames(X) <- paste0("X", seq_len(ncol(X)))
y <- rpois(100, lambda = 10)

weights <- rep(1, ncol(X))
weights[1:5] <- 0.1

prior_binary <- setNames(
  weights < 1,
  colnames(X)
)

weight_prior <- ecountgmifs.weight.prior(
  prior = prior_binary,
  eta = c(1, 10, 100),
  label = "Example prior"
)

fit_grid <- ecountgmifs(
  X = X,
  y = y,
  weight.vec = weight_prior,
  enet.alpha = c(0.50, 0.75, 1.00),
  family = "negative.binomial",
  criteria = list(
    AIC = criterion("AIC", type = "full"),
    BIC = criterion("BIC", type = "full")
  )
)

beta_grid <- coef(
  fit_grid,
  alpha = 0.75,
  eta = 10,
  criterion = "BIC.builtin"
)
```

## Extracting all fitted parameters

`parameters()` returns every optimized parameter group: penalized coefficients (`beta`), unpenalized coefficients (`theta`), family parameters, and link parameters. `params()` is the shorter alias.

This example defines and fits its own data before calling the accessors:

```r
library(ecountgmifs)

set.seed(9)
X <- matrix(rpois(100 * 20, lambda = 5), nrow = 100, ncol = 20)
colnames(X) <- paste0("X", seq_len(ncol(X)))
y <- rpois(100, lambda = 10)

fit_modular <- ecountgmifs(
  X = X,
  y = y,
  family = plugin.family.nb2.builtin(),
  link = plugin.link.log.builtin(),
  criteria = list(
    AIC = criterion("AIC", type = "full"),
    BIC = criterion("BIC", type = "nnz")
  ),
  enet.alpha = 0.75
)

all_paths <- parameters(fit_modular)
names(all_paths)

terminal_parameters <- params(
  fit_modular,
  state = "terminal"
)

bic_parameters <- params(
  fit_modular,
  criterion = "BIC.nnz.builtin"
)

bic_parameters$beta
bic_parameters$theta
bic_parameters$family
bic_parameters$link

beta_at_bic <- bparams(
  fit_modular,
  criterion = "BIC.nnz.builtin"
)

theta_at_terminal <- thparams(
  fit_modular,
  state = "terminal"
)

family_at_bic <- fparams(
  fit_modular,
  criterion = "BIC.nnz.builtin"
)

link_at_terminal <- lparams(
  fit_modular,
  state = "terminal"
)
```

The preferred compact aliases are `bparams()`, `thparams()`, `fparams()`, and `lparams()`. Descriptive aliases such as `beta.parameters()`, `pen.parameters()`, `theta.parameters()`, `nonpen.parameters()`, `family.parameters()`, and `link.parameters()` are also exported.

Parameter-free families or links return zero-length vectors for one state and zero-column matrices for paths.

## Inspecting the fitted model modules

The following example is self-contained and shows `family()`, `link()`, and `criteria()` metadata:

```r
library(ecountgmifs)

set.seed(10)
X <- matrix(rpois(100 * 20, lambda = 5), nrow = 100, ncol = 20)
colnames(X) <- paste0("X", seq_len(ncol(X)))
y <- rpois(100, lambda = 10)

fit_modular <- ecountgmifs(
  X = X,
  y = y,
  family = plugin.family.nb2.builtin(),
  link = plugin.link.log.builtin(),
  criteria = list(
    AIC = criterion("AIC", type = "full"),
    BIC = criterion("BIC", type = "nnz")
  ),
  enet.alpha = 0.75
)

family_info <- family(fit_modular)
family_info$name
family_info$parameter_count
family_info$function_name
family_info$constructor

link_info <- link(fit_modular)
link_info$name
link_info$call

criterion_info <- criteria(fit_modular)
names(criterion_info$criteria)
criterion_info$criteria[[1L]]$function_name
criterion_info$selected
```

## Model-selection criteria

The compact `criterion()` constructor covers the standard criterion and degrees-of-freedom combinations:

```r
library(ecountgmifs)

criteria <- list(
  AIC = criterion("AIC", type = "full"),
  BIC_nnz = criterion("BIC", type = "nnz"),
  SABIC_hedf = criterion("SABIC", type = "hedf"),
  EBIC = criterion("EBIC", type = "nnz", gamma = 0.5)
)
```

The older constructors such as `AIC_nnz()` and `BIC_hedf()` remain available for compatibility.

## Main features

* Stagewise sparse regression for high-dimensional count outcomes
* Negative-binomial and Poisson model families
* Elastic-net mixing through `enet.alpha`
* Optional prior-weighted penalty rescaling
* Support for offsets and unpenalized covariates through `offset` and `w`
* Information-criterion-based solution-path selection
* Runtime-compilable custom C++ selection criteria with read-only context access
* Public RNA-seq and T-100 airport example datasets
* R interface with C++ implementation through Rcpp

## Related paper

This package accompanies the manuscript:

**Extended generalized monotone incremental forward stagewise regression for penalized negative binomial path-following modeling of high-dimensional count data**  
Alikhan Anuarbekov and Jiří Kléma

The packaged `mrna97_rnaseq` dataset is adapted from the RNA-seq and prior-knowledge setup described in:

Anuarbekov, A. and Kléma, J. (2025). Utilizing RNA-seq data in monotone iterative generalized linear model to elevate prior knowledge quality of the circRNA-miRNA-mRNA regulatory axis. *BMC Bioinformatics*, 26, 139. https://doi.org/10.1186/s12859-025-06161-w

The packaged `airport_t100` dataset is adapted from the U.S. Bureau of Transportation Statistics T-100 Domestic Segment passenger-flow benchmark used in the accompanying eGMIFS study.

Please cite the associated paper if you use this package or dataset in academic work.

## License

This package is licensed under the MIT License. See the `LICENSE` file for details.
