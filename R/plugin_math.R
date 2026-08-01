.r.log1pexp <- function(x) {
  x <- as.numeric(x)
  result <- numeric(length(x))
  positive <- x > 0
  result[positive] <- x[positive] + log1p(exp(-x[positive]))
  result[!positive] <- log1p(exp(x[!positive]))
  result
}

.r.logistic <- function(x) {
  x <- as.numeric(x)
  result <- numeric(length(x))
  nonnegative <- x >= 0
  negative.exp <- exp(-x[nonnegative])
  result[nonnegative] <- 1 / (1 + negative.exp)
  positive.exp <- exp(x[!nonnegative])
  result[!nonnegative] <- positive.exp / (1 + positive.exp)
  result
}

.r.clamp <- function(x, lower, upper) {
  pmin(pmax(x, lower), upper)
}

.r.capped.exp <- function(eta, mu.min.cap, mu.max.cap) {
  eta.lower <- log(mu.min.cap)
  eta.upper <- log(mu.max.cap)
  eta.safe <- pmin(pmax(eta, eta.lower), eta.upper)
  mu <- exp(eta.safe)

  list(
    mu = mu,
    d.mu.d.eta = mu * (eta > eta.lower & eta < eta.upper)
  )
}

.r.validate.caps <- function(mu.min.cap, mu.max.cap) {
  if (
      length(mu.min.cap) != 1L || !is.finite(mu.min.cap) ||
      length(mu.max.cap) != 1L || !is.finite(mu.max.cap) ||
      mu.min.cap <= 0 || mu.min.cap >= mu.max.cap
  ) {
    stop("Invalid mean clamping bounds.", call. = FALSE)
  }
}

.r.validate.nb2 <- function(
    mu.min.cap,
    mu.max.cap,
    poisson.fallback.eps,
    dispersion.initial,
    dispersion.lower.bound,
    dispersion.upper.bound
) {
  .r.validate.caps(mu.min.cap, mu.max.cap)

  if (
      length(poisson.fallback.eps) != 1L ||
      !is.finite(poisson.fallback.eps) ||
      poisson.fallback.eps < 0
  ) {
    stop("`poisson.fallback.eps` must be finite and non-negative.", call. = FALSE)
  }

  if (
      length(dispersion.initial) != 1L || !is.finite(dispersion.initial) ||
      length(dispersion.lower.bound) != 1L || is.na(dispersion.lower.bound) ||
      length(dispersion.upper.bound) != 1L || is.na(dispersion.upper.bound) ||
      dispersion.lower.bound < 0 ||
      dispersion.lower.bound >= dispersion.upper.bound ||
      dispersion.initial < dispersion.lower.bound ||
      dispersion.initial > dispersion.upper.bound
  ) {
    stop("Invalid NB2 dispersion initial value or bounds.", call. = FALSE)
  }
}

.r.validate.log.factorial <- function(y, log.factorial) {
  if (
      !is.numeric(log.factorial) ||
      length(log.factorial) != length(y) ||
      anyNA(log.factorial) ||
      any(!is.finite(log.factorial))
  ) {
    stop(
      "Prepared `log.factorial` must be a finite numeric vector matching `y`.",
      call. = FALSE
    )
  }

  log.factorial
}

.r.poisson.negloglik <- function(
    y,
    mu,
    mu.min.cap,
    mu.max.cap,
    log.factorial = lgamma(y + 1)
) {
  log.factorial <- .r.validate.log.factorial(y, log.factorial)
  mu.safe <- .r.clamp(mu, mu.min.cap, mu.max.cap)
  sum(mu.safe - y * log(mu.safe) + log.factorial)
}

.r.poisson.grad <- function(y, mu) {
  list(
    d_negloglik_d_mu = 1 - y / mu,
    d_negloglik_d_family_parameters = numeric()
  )
}

.r.nb2.negloglik <- function(
    y,
    mu,
    dispersion,
    mu.min.cap,
    mu.max.cap,
    poisson.fallback.eps,
    log.factorial
) {
  log.factorial <- .r.validate.log.factorial(y, log.factorial)

  if (dispersion <= poisson.fallback.eps) {
    return(
      .r.poisson.negloglik(
        y,
        mu,
        mu.min.cap,
        mu.max.cap,
        log.factorial = log.factorial
      )
    )
  }
  if (dispersion <= 0) {
    return(Inf)
  }

  mu.safe <- .r.clamp(mu, mu.min.cap, mu.max.cap)
  r <- 1 / dispersion
  lgamma.r <- lgamma(r)
  dispersion.mu <- dispersion * mu.safe
  log.one.plus <- log1p(dispersion.mu)

  -sum(
    y * (log(dispersion.mu) - log.one.plus) -
      r * log.one.plus +
      lgamma(y + r) -
      log.factorial -
      lgamma.r
  )
}

.r.nb2.grad <- function(y, mu, dispersion, poisson.fallback.eps) {
  if (dispersion <= poisson.fallback.eps) {
    return(list(
      d_negloglik_d_mu = 1 - y / mu,
      d_negloglik_d_family_parameters = 0
    ))
  }

  r <- 1 / dispersion
  d.negloglik.d.mu <-
    -y / mu + (1 + dispersion * y) / (1 + dispersion * mu)

  r.plus.mu <- r + mu
  d.negloglik.d.r <- sum(
    digamma(r) -
      digamma(y + r) -
      log(r / r.plus.mu) -
      1 +
      (r + y) / r.plus.mu
  )

  list(
    d_negloglik_d_mu = d.negloglik.d.mu,
    d_negloglik_d_family_parameters = -d.negloglik.d.r / dispersion^2
  )
}

.r.effective.parameter.count <- function(state) {
  sum(state$beta != 0) +
    length(state$theta) +
    length(state$family_parameters) +
    length(state$link_parameters)
}
