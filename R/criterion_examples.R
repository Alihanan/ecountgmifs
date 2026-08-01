# Auxiliary-data and mutable-state criterion examples ---------------------

.r.prepare.test.criterion.data <- function(
    Xtest,
    ytest,
    wtest,
    offsettest,
    mu.min.cap,
    mu.max.cap,
    poisson.fallback.eps
) {
  Xtest <- as.matrix(Xtest)
  ytest <- as.numeric(ytest)
  wtest <- as.matrix(wtest)
  offsettest <- as.numeric(offsettest)

  if (!is.numeric(Xtest) || !is.numeric(wtest)) {
    stop("`Xtest` and `wtest` must be numeric matrices.", call. = FALSE)
  }
  if (nrow(Xtest) != length(ytest) || nrow(wtest) != length(ytest)) {
    stop("Test matrices and `ytest` must have equal row counts.", call. = FALSE)
  }
  if (length(offsettest) != length(ytest)) {
    stop("`offsettest` must have length `length(ytest)`.", call. = FALSE)
  }
  if (
      anyNA(Xtest) || any(!is.finite(Xtest)) ||
      anyNA(wtest) || any(!is.finite(wtest)) ||
      anyNA(ytest) || any(!is.finite(ytest)) || any(ytest < 0) ||
      anyNA(offsettest) || any(!is.finite(offsettest))
  ) {
    stop("Test data must contain finite values and non-negative counts.", call. = FALSE)
  }

  .r.validate.caps(mu.min.cap, mu.max.cap)
  if (
      length(poisson.fallback.eps) != 1L ||
      !is.finite(poisson.fallback.eps) ||
      poisson.fallback.eps < 0
  ) {
    stop("`poisson.fallback.eps` must be finite and non-negative.", call. = FALSE)
  }

  list(
    Xtest = Xtest,
    ytest = ytest,
    log.factorial = lgamma(ytest + 1),
    wtest = wtest,
    offsettest = offsettest,
    mu.min.cap = as.numeric(mu.min.cap),
    mu.max.cap = as.numeric(mu.max.cap),
    poisson.fallback.eps = as.numeric(poisson.fallback.eps)
  )
}

.r.test.criterion.dispersion <- function(state) {
  if (length(state$family_parameters) != 1L) {
    stop("The test-data criterion requires one NB2 dispersion parameter.", call. = FALSE)
  }
  state$family_parameters[[1L]]
}

#' @rdname custom.plugins.r.environment
#' @param Xtest Numeric test predictor matrix with one column per penalized
#'   coefficient.
#' @param ytest Numeric test response vector.
#' @param wtest Numeric test design matrix with one column per unpenalized
#'   coefficient.
#' @param offsettest Numeric test offset vector.
#' @param mu.min.cap,mu.max.cap Positive finite lower and upper caps for `mu`.
#' @param poisson.fallback.eps Positive threshold below which NB2 is evaluated
#'   with its Poisson limit.
#' @return An external pointer to an R-backed criterion.
#' @export
example.r.test.criterion.environment <- function(
    Xtest,
    ytest,
    wtest,
    offsettest = rep(0, length(ytest)),
    mu.min.cap = 1e-12,
    mu.max.cap = 1e12,
    poisson.fallback.eps = 1e-8
) {
  workspace <- list2env(
    .r.prepare.test.criterion.data(
      Xtest,
      ytest,
      wtest,
      offsettest,
      mu.min.cap,
      mu.max.cap,
      poisson.fallback.eps
    ),
    envir = r.environment()
  )

  r.criterion(
    name = "TestNegloglikEnvironment",
    environment = workspace,
    evaluate = function(input, control, state, environment) {
      eta <- environment$offsettest +
        drop(environment$Xtest %*% state$beta) +
        drop(environment$wtest %*% state$theta)

      .r.nb2.negloglik(
        environment$ytest,
        .r.capped.exp(
          eta,
          environment$mu.min.cap,
          environment$mu.max.cap
        )$mu,
        .r.test.criterion.dispersion(state),
        environment$mu.min.cap,
        environment$mu.max.cap,
        environment$poisson.fallback.eps,
        log.factorial = environment$log.factorial
      )
    }
  )
}

#' @rdname custom.plugins.r.closure
#' @param Xtest Numeric test predictor matrix with one column per penalized
#'   coefficient.
#' @param ytest Numeric test response vector.
#' @param wtest Numeric test design matrix with one column per unpenalized
#'   coefficient.
#' @param offsettest Numeric test offset vector.
#' @param mu.min.cap,mu.max.cap Positive finite lower and upper caps for `mu`.
#' @param poisson.fallback.eps Positive threshold below which NB2 is evaluated
#'   with its Poisson limit.
#' @return An external pointer to an R-backed criterion.
#' @export
example.r.test.criterion.closure <- function(
    Xtest,
    ytest,
    wtest,
    offsettest = rep(0, length(ytest)),
    mu.min.cap = 1e-12,
    mu.max.cap = 1e12,
    poisson.fallback.eps = 1e-8
) {
  data <- .r.prepare.test.criterion.data(
    Xtest,
    ytest,
    wtest,
    offsettest,
    mu.min.cap,
    mu.max.cap,
    poisson.fallback.eps
  )
  Xtest <- data$Xtest
  ytest <- data$ytest
  log.factorial <- data$log.factorial
  wtest <- data$wtest
  offsettest <- data$offsettest
  mu.min.cap <- data$mu.min.cap
  mu.max.cap <- data$mu.max.cap
  poisson.fallback.eps <- data$poisson.fallback.eps

  r.criterion(
    name = "TestNegloglikClosure",
    evaluate = function(input, control, state, environment) {
      eta <- offsettest +
        drop(Xtest %*% state$beta) +
        drop(wtest %*% state$theta)

      .r.nb2.negloglik(
        ytest,
        .r.capped.exp(eta, mu.min.cap, mu.max.cap)$mu,
        .r.test.criterion.dispersion(state),
        mu.min.cap,
        mu.max.cap,
        poisson.fallback.eps,
        log.factorial = log.factorial
      )
    }
  )
}

#' @rdname custom.plugins.r.environment
#' @export
example.r.counted.aic.criterion.environment <- function() {
  workspace <- r.environment(evaluation.count = 0L)

  r.criterion(
    name = "CountedAICEnvironment",
    environment = workspace,
    evaluate = function(input, control, state, environment) {
      environment$evaluation.count <- environment$evaluation.count + 1L
      2 * state$negloglik + 2 * .r.effective.parameter.count(state)
    }
  )
}

#' @rdname custom.plugins.r.closure
#' @export
example.r.counted.aic.criterion.closure <- function() {
  evaluation.count <- 0L

  r.criterion(
    name = "CountedAICClosure",
    evaluate = function(input, control, state, environment) {
      evaluation.count <<- evaluation.count + 1L
      2 * state$negloglik + 2 * .r.effective.parameter.count(state)
    }
  )
}
