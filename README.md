# ecountgmifs

`ecountgmifs` is an R package for fitting extended generalized monotone incremental forward stagewise models for high-dimensional count data.

The package is designed for sparse regression problems where the response is count-valued and the number of predictors may be large relative to the number of samples. It supports negative-binomial and Poisson model families, elastic-net-style stagewise updates, prior-weighted penalties, unpenalized covariates, and information-criterion-based model selection along the solution path.

## Installation

Install the package from GitHub:

```r
# install.packages("remotes")
remotes::install_github("Alihanan/ecountgmifs")
```

## Basic usage

```r
library(ecountgmifs)

# Example data
set.seed(1)
X <- matrix(rpois(100 * 20, lambda = 5), nrow = 100, ncol = 20)
y <- rpois(100, lambda = 10)

# Fit model
fit <- ecountgmifs(
  X = X,
  y = y,
  family = "negative.binomial",
  enet.alpha = 0.75
)

# Inspect results
print(fit)
summary(fit)
coef(fit)
plot(fit)
```

## Packaged RNA-seq example dataset

The package includes `mrna97_rnaseq`, an RNA-seq example dataset derived from the regulatory-network setting described by Anuarbekov and Kléma (2025). The original test set used in the paper is not included in this package dataset.

```r
data(mrna97_rnaseq)

names(mrna97_rnaseq)
#> [1] "Y"      "X"      "prior"  "truth"  "sample"

dim(mrna97_rnaseq$Y)
#> 434 97

dim(mrna97_rnaseq$X)
#> 434 2636

dim(mrna97_rnaseq$prior)
#> 97 2636

dim(mrna97_rnaseq$truth)
#> 97 2636

head(mrna97_rnaseq$sample)
```

The dataset is stored as a list with five elements:

* `Y`: response matrix with 434 samples and 97 mRNA target variables.
* `X`: predictor matrix with 434 samples and 2636 candidate predictor variables.
* `prior`: binary prior-knowledge matrix with 97 rows and 2636 columns, using target-by-predictor orientation.
* `truth`: binary ground-truth matrix with the same orientation as `prior`.
* `sample`: sample-level metadata with fine-grained and higher-level tissue annotations.

A basic single-target fit can be run as follows:

```r
data(mrna97_rnaseq)

j <- 1

Y <- mrna97_rnaseq$Y
X <- mrna97_rnaseq$X
prior <- mrna97_rnaseq$prior

fit <- ecountgmifs(
  X = X,
  y = Y[, j],
  weight.vec = ifelse(prior[j, ], 0.1, 1),
  family = "negative.binomial",
  enet.alpha = 0.75
)
```


## Preprocessing utilities

The package includes preprocessing helpers matching the normalization and transformation strategies used in the RNA-seq and airport experiments.

For RNA-seq-style count matrices, sample-level normalization can be applied before predictor transformation:

```r
data(mrna97_rnaseq)

X_cpm <- normalize_counts(
  mrna97_rnaseq$X,
  method = "cpm"
)

X_pre <- transform_predictors(
  X_cpm,
  method = "standardize_log1p"
)
```

Available count normalizations are:

* `"none"`: keep the original scale.
* `"cpm"`: counts per million scaling.
* `"tmm"`: TMM normalization through `edgeR`, if `edgeR` is installed.

Available RNA-seq predictor transformations are:

* `"none"`
* `"log1p"`
* `"asinh"`
* `"standardize"`
* `"standardize_log1p"`
* `"standardize_asinh"`

For negative-binomial models, the response should usually remain on the original count scale. When an offset is needed, request the normalization factor:

```r
norm <- normalize_counts(
  mrna97_rnaseq$X,
  method = "cpm",
  return.offset = TRUE
)

X_cpm <- norm$x
offset <- norm$offset
```

For monthly airport passenger-flow matrices, temporal transformations can be applied to reduce seasonal and system-wide temporal effects:

```r
data(airport_t100)

X_air <- transform_time_series(
  airport_t100$X,
  method = "log1p_month_demean",
  month = airport_t100$sample$month
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

Prior information can be supplied through `weight.vec`. Smaller weights reduce the penalty for selected predictors.

```r
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

For the packaged RNA-seq dataset, the prior matrix can be used to construct target-specific penalty weights:

```r
data(mrna97_rnaseq)

j <- 1

weights <- ifelse(mrna97_rnaseq$prior[j, ], 0.1, 1)

fit_prior <- ecountgmifs(
  X = mrna97_rnaseq$X,
  y = mrna97_rnaseq$Y[, j],
  weight.vec = weights,
  family = "negative.binomial",
  enet.alpha = 0.75
)
```

## Model-selection criteria

The package includes example utilities for compiling information-criterion functions used to select points along the fitted solution path.

```r
bic_nnz <- BIC_nnz()
bic_hedf <- BIC_hedf()

fit <- ecountgmifs(
  X = X,
  y = y,
  criteria = list(
    BIC_nnz = bic_nnz,
    BIC_hedf = bic_hedf
  )
)

The example criteria are compiled C++ functions. Repeated calls with the same source code are cached within the R session.

## Main features

* Stagewise sparse regression for high-dimensional count outcomes
* Negative-binomial and Poisson model families
* Elastic-net mixing through `enet.alpha`
* Optional prior-weighted penalty rescaling
* Support for unpenalized covariates through `w`
* Information-criterion-based solution-path selection
* Runtime-compilable custom C++ selection criteria with read-only context access
* Public RNA-seq example dataset with prior and ground-truth interaction matrices
* R interface with C++ implementation through Rcpp

## Related paper

This package accompanies the manuscript:

**ecountGMIFS: Extended generalized monotone incremental forward stagewise regression for penalized negative binomial path-following modeling of high-dimensional count data**  
Alikhan Anuarbekov and Jiří Kléma

The packaged `mrna97_rnaseq` dataset is adapted from the RNA-seq and prior-knowledge setup described in:

Anuarbekov, A. and Kléma, J. (2025). Utilizing RNA-seq data in monotone iterative generalized linear model to elevate prior knowledge quality of the circRNA-miRNA-mRNA regulatory axis. *BMC Bioinformatics*, 26, 139. https://doi.org/10.1186/s12859-025-06161-w

Please cite the associated paper if you use this package or dataset in academic work.

## License

This package is licensed under the MIT License. See the `LICENSE` file for details.
