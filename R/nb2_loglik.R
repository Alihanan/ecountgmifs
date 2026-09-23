#' Evaluate the package's C++ NB2 log-likelihood objective
#'
#' @param y Non-empty finite vector of non-negative responses. Fractional
#'   responses are accepted using the existing gamma-function extension.
#' @param mu A finite non-negative scalar mean, or a vector matching `y`.
#' @param dispersion Non-negative scalar NB2 dispersion, with variance
#'   `mu + dispersion * mu^2` for integer NB counts.
#' @param sum Logical; return the sum (default) or one term per observation.
#' @param mu.min.cap,mu.max.cap Positive finite mean caps, identical to the
#'   built-in NB2 family's defaults. Means are clamped inside C++.
#' @param poisson.fallback.eps Non-negative dispersion threshold at or below
#'   which the C++ family evaluates its Poisson-limit objective.
#'
#' @details
#' Calls `NB2Family::negloglik()` in `src/example.h`, after preparing its
#' response cache, and reverses its sign. No independent R likelihood is used.
#' All response-dependent terms, including `lgamma(y + 1)`, are retained.
#' A fresh family instance is used for each call, so the prepared cache cannot
#' be stale. The per-observation option favors clarity over speed.
#'
#' On integer responses this is the ordinary NB log likelihood, subject to
#' the family's mean caps and Poisson fallback. On fractional responses it
#' is a working objective, not a normalized continuous probability density.
#' Raw values computed on `y`, `floor(y)`, and `round(y)` are descriptive
#' diagnostics and must not be ranked as evidence for different models.
#'
#' @return A numeric scalar, or a numeric vector when `sum = FALSE`.
#'   These are log-likelihood values, not negative log-likelihood values.
#' @examples
#' y <- c(0, 1, 4, 10)
#' nb2.loglik(y, mu = 3, dispersion = 0.5)
#' base::sum(stats::dnbinom(y, mu = 3, size = 2, log = TRUE))
#' nb2.loglik(c(0, 1.2, 4.7), mu = 3, dispersion = 0.5)
#' @export
nb2.loglik <- function(y, mu, dispersion, sum = TRUE,
                       mu.min.cap = 1e-12, mu.max.cap = 1e12,
                       poisson.fallback.eps = 1e-8) {
  if (length(y) == 0L) stop("`y` must not be empty.", call. = FALSE)
  .nb2.check.vector(y, "y", length(y), nonnegative = TRUE)
  if (length(mu) == 1L) {
    .nb2.check.scalar(mu, "mu", lower = 0)
    mu <- rep(mu, length(y))
  }
  .nb2.check.vector(mu, "mu", length(y), nonnegative = TRUE)
  .nb2.check.scalar(dispersion, "dispersion", lower = 0)
  .nb2.check.scalar(poisson.fallback.eps, "poisson.fallback.eps", lower = 0)
  .r.validate.caps(mu.min.cap, mu.max.cap)
  if (!is.logical(sum) || length(sum) != 1L || is.na(sum)) {
    stop("`sum` must be TRUE or FALSE.", call. = FALSE)
  }

  evaluate <- function(response, mean) {
    nb2_loglik_cpp(response, mean, dispersion, mu.min.cap, mu.max.cap,
                   poisson.fallback.eps)
  }
  if (sum) return(evaluate(y, mu))
  vapply(seq_along(y), function(i) evaluate(y[i], mu[i]), numeric(1))
}
