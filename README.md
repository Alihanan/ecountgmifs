# ecountgmifs

`ecountgmifs` is an R package for fitting extended generalized monotone incremental forward stagewise models for high-dimensional count data.

The package is designed for sparse regression problems where the response is count-valued and the number of predictors may be large relative to the number of samples. It supports negative-binomial and Poisson model families, elastic-net-style stagewise updates, prior-weighted penalties, and information-criterion-based model selection along the solution path.

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

## Model-selection criteria

The package includes information-criterion utilities for selecting points along the fitted solution path:

```r
fit <- ecountgmifs(
  X = X,
  y = y,
  criteria = list(
    AIC_nnz = AIC_nnz,
    BIC_nnz = BIC_nnz,
    SABIC_nnz = SABIC_nnz
  )
)
```

## Main features

* Stagewise sparse regression for high-dimensional count outcomes
* Negative-binomial and Poisson model families
* Elastic-net mixing through `enet.alpha`
* Optional prior-weighted penalty rescaling
* Support for unpenalized covariates through `w`
* Information-criterion-based solution-path selection
* R interface with C++ implementation through Rcpp

## Related paper

This package accompanies the manuscript:

**ecountGMIFS: Extended generalized monotone incremental forward stagewise regression for penalized negative binomial path-following modeling of high-dimensional count data**
Alikhan Anuarbekov and Jiří Kléma

Please cite the associated paper if you use this package in academic work.

## License

This package is licensed under GPL-3 or later. See the `LICENSE` file for details.
