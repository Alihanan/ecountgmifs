#' Generate a sparse negative-binomial regression dataset
#'
#' @param n Number of observations.
#' @param p Number of slopes, excluding the intercept. Zero gives the old
#'   intercept-only experiment.
#' @param mu Positive marginal mean of the latent counts, averaged over random X.
#' @param dispersion Non-negative NB2 dispersion: Var(Y|X) = m + dispersion*m^2.
#' @param active Number of nonzero slopes; default min(3, p).
#' @param signal.sd Standard deviation of X beta. Ignored when p is zero.
#' @param rho AR(1) predictor correlation, strictly between -1 and 1.
#' @param coefficients Optional named or unnamed vector of length p+1,
#'   intercept first. Overrides the generated coefficient pattern and mu.
#' @details Rows of X are independent normal vectors with covariance
#'   Sigma[j,k] = rho^abs(j-k). The first active slopes alternate signs and
#'   are scaled to beta' Sigma beta = signal.sd^2. The intercept is
#'   log(mu) - signal.sd^2/2, giving E_X[exp(intercept + X beta)] = mu.
#'   Covariates are not restandardized using the sample. No noise is added here;
#'   apply a noise generator to the returned count vector.
#' @return List containing X, integer counts, conditional means and variances,
#'   intercept and slopes, true support, zero offsets and generator settings.
#' @export
generate.nb2.glm <- function(n, p = 5, mu = 5, dispersion = 0.5,
                              active = min(3, p), signal.sd = 0.75, rho = 0.3,
                              coefficients = NULL) {
  for (name in c("n", "p", "active")) {
    value <- get(name)
    .nb2.check.scalar(value, name, lower = if (name == "n") 1 else 0)
    if (value != floor(value)) stop(name, " must be an integer.")
  }
  .nb2.check.parameters(mu, dispersion)
  if (mu == 0 || active > p) stop("Require mu > 0 and active <= p.")
  .nb2.check.scalar(signal.sd, "signal.sd", lower = 0)
  .nb2.check.scalar(rho, "rho")
  if (abs(rho) >= 1) stop("Require abs(rho) < 1.")

  X <- matrix(stats::rnorm(n * p), nrow = n, ncol = p)
  colnames(X) <- if (p > 0) paste0("x", seq_len(p)) else character()
  if (p > 1) {
    for (j in 2:p) X[, j] <- rho * X[, j - 1] + sqrt(1 - rho^2) * X[, j]
  }
  if (is.null(coefficients)) {
    slopes <- numeric(p)
    predictor.variance <- 0
    if (active > 0 && signal.sd > 0) {
      signs <- rep(c(1, -1), length.out = active)
      covariance <- rho^abs(outer(seq_len(active), seq_len(active), "-"))
      scale <- signal.sd / sqrt(drop(crossprod(signs, covariance %*% signs)))
      slopes[seq_len(active)] <- signs * scale
      predictor.variance <- signal.sd^2
    }
    coefficients <- c(log(mu) - predictor.variance / 2, slopes)
  }
  .nb2.check.vector(coefficients, "coefficients", p + 1)
  names(coefficients) <- c("(Intercept)", colnames(X))
  eta <- drop(coefficients[1] + X %*% coefficients[-1])
  conditional.mean <- exp(eta)
  if (any(!is.finite(conditional.mean)) || any(conditional.mean <= 0)) {
    stop("Conditional means exceed numerical range.")
  }
  count <- if (dispersion == 0) stats::rpois(n, conditional.mean) else
    stats::rnbinom(n, mu = conditional.mean, size = 1 / dispersion)
  list(X = X, count = count, mu = conditional.mean,
       variance = conditional.mean + dispersion * conditional.mean^2,
       eta = eta, offset = rep(0, n), coefficients = coefficients,
       support = which(coefficients[-1] != 0), dispersion = dispersion,
       settings = list(n = n, p = p, marginal.mean = mu, dispersion = dispersion,
                       active = active, signal.sd = signal.sd, rho = rho))
}

#' Fit only the existing C++ unpenalized NB2 initialization
#'
#' @param X Numeric n-by-p matrix of slopes. An intercept is added internally;
#'   a matrix with zero columns is supported.
#' @param y Non-negative response vector; fractional values are permitted.
#' @param dispersion Supply a non-negative number to hold dispersion fixed.
#'   NULL estimates it by the existing alternating initial-fit routine.
#' @param offset Optional known log-mean offset.
#' @param start Optional initial coefficients, intercept first. The default
#'   uses log(mean(y)) for the intercept and zero slopes.
#' @param control Named list overriding outer.maxit (200), inner.maxeval (1000),
#'   tolerance (1e-9), coefficient.algorithm (11), dispersion.algorithm (28),
#'   dispersion.initial (0.5), dispersion.lower (1e-8), dispersion.upper (1e4).
#'   Algorithm numbers are NLopt enum values, as in ecountgmifs.control().
#' @details This bridge uses the package's exact fit_null_model routine, with
#'   all supplied predictors unpenalized. It runs neither a saturated fit nor
#'   stagewise fitting. Optimization uses the mean objective (total divided
#'   by n) and correspondingly scaled gradients for stable gradient steps;
#'   returned likelihoods use the original total. The default coefficient
#'   optimizer is NLopt L-BFGS (11); the family uses SBPLX (28). These settings
#'   are explicit experimental controls, not changes to ecountgmifs defaults.
#'   Fixed dispersion uses a parameter-free family that
#'   delegates objective and gradients to the existing NB2 implementation.
#'   Its bounds are not approximated by a narrow interval.
#'
#'   Require n > p+1 and full design rank for this coefficient-recovery study.
#'   The returned convergence flag describes the original outer stopping rule;
#'   inspect optimizer codes and coefficient score too. No inference or sparse
#'   selection is supplied. Fractional responses use an extended objective,
#'   not a proved normalized continuous probability density.
#' @return List containing coefficients, dispersion, fitted values, loglik,
#'   convergence diagnostics, normalized coefficient score, settings and start.
#' @export
nb2.glm.fit <- function(X, y, dispersion = NULL, offset = NULL,
                        start = NULL, control = list()) {
  if (!is.matrix(X) || !is.numeric(X) || any(!is.finite(X))) stop("X must be a finite numeric matrix.")
  n <- nrow(X)
  .nb2.check.vector(y, "y", n, nonnegative = TRUE)
  if (n <= ncol(X) + 1) stop("Unpenalized recovery requires n > p + 1.")
  if (sum(y) == 0) stop("All-zero response has no finite intercept MLE.")
  design <- cbind("(Intercept)" = 1, X)
  if (qr(design)$rank < ncol(design)) stop("The design is rank deficient.")
  if (is.null(colnames(X))) colnames(design)[-1] <- paste0("x", seq_len(ncol(X)))
  if (is.null(offset)) offset <- rep(0, n)
  .nb2.check.vector(offset, "offset", n)
  if (is.null(start)) start <- c(log(mean(y)) - mean(offset), rep(0, ncol(X)))
  .nb2.check.vector(start, "start", ncol(design))
  if (!is.null(dispersion)) .nb2.check.scalar(dispersion, "dispersion", lower = 0)
  defaults <- list(outer.maxit = 200L, inner.maxeval = 1000L, tolerance = 1e-9,
                   coefficient.algorithm = 11L, dispersion.algorithm = 28L,
                   dispersion.initial = 0.5, dispersion.lower = 1e-8,
                   dispersion.upper = 1e4)
  if (!is.list(control) || (length(control) && (is.null(names(control)) ||
      any(!names(control) %in% names(defaults)) || anyDuplicated(names(control))))) {
    stop("control must be a named list of documented settings.")
  }
  settings <- utils::modifyList(defaults, control)
  for (name in names(settings)) .nb2.check.scalar(settings[[name]], name, lower = 0)
  for (name in c("outer.maxit", "inner.maxeval", "coefficient.algorithm", "dispersion.algorithm")) {
    if (settings[[name]] != floor(settings[[name]])) stop(name, " must be an integer.")
  }
  if (settings$outer.maxit < 1 || settings$inner.maxeval < 1 || settings$tolerance <= 0 ||
      settings$dispersion.lower >= settings$dispersion.upper ||
      settings$dispersion.initial < settings$dispersion.lower ||
      settings$dispersion.initial > settings$dispersion.upper) stop("Invalid fitting controls.")
  fit <- nb2_glm_fit_cpp(design, y, offset, start, !is.null(dispersion),
                        if (is.null(dispersion)) 0 else dispersion,
                        settings$dispersion.initial, settings$dispersion.lower,
                        settings$dispersion.upper, settings$outer.maxit,
                        settings$inner.maxeval, settings$tolerance,
                        settings$coefficient.algorithm, settings$dispersion.algorithm)
  fit$coefficients <- setNames(as.numeric(fit$coefficients), colnames(design))
  fit$dispersion <- if (is.null(dispersion)) as.numeric(fit$family_parameters) else dispersion
  fit$linear.predictors <- drop(design %*% fit$coefficients + offset)
  fit$fitted.values <- exp(fit$linear.predictors)
  fit$score.max.per.observation <- max(abs(crossprod(
    design, (y - fit$fitted.values) / (1 + fit$dispersion * fit$fitted.values)))) / n
  fit$dispersion.at.bound <- is.null(dispersion) &&
    (fit$dispersion <= settings$dispersion.lower * 1.01 ||
     fit$dispersion >= settings$dispersion.upper * 0.99)
  fit$control <- settings
  fit$start <- start
  fit$offset <- offset
  fit
}
