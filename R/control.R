#' Auxiliary control for extended count GMIFS fitting
#'
#' @description
#' Creates a control object for [ecountgmifs()]. This function is normally used
#' through the `control` argument of [ecountgmifs()].
#'
#' @details
#' The returned values control the forward-stagewise path, adaptive step size,
#' convergence rules, elastic-net calculations, internal NLopt optimization,
#' initialization of unpenalized coefficients, and path-state storage.
#'
#' The state-tracking strategy controls how much information is retained along
#' the fitted path. The `"active.set.change"` strategy stores a compressed path,
#' while `"all.iteration"` stores every iteration.
#'
#' Module-specific controls do not belong in this object. For example,
#' negative-binomial dispersion initialization, dispersion bounds, mean caps,
#' and the Poisson fallback threshold are supplied when constructing the
#' negative-binomial family object.
#'
#' @param iteration.max Positive integer. Maximum number of stagewise
#'   iterations.
#' @param epsilon.max Positive numeric value. Maximum permitted stagewise step
#'   size.
#' @param epsilon.start Positive numeric value. Initial stagewise step size.
#' @param epsilon.min Non-negative numeric value. Minimum permitted stagewise
#'   step size.
#' @param tol Positive numeric value. General numerical convergence tolerance.
#' @param loglik.reltol.cutoff Non-negative relative log-likelihood cutoff used
#'   by the stopping rule corresponding to the pseudo-\eqn{R^2} threshold.
#' @param enet.abs.tol Positive absolute tolerance used by the weighted
#'   elastic-net solver.
#' @param enet.rel.tol Non-negative relative tolerance used by the weighted
#'   elastic-net solver.
#' @param enet.max.iter Positive integer. Maximum number of elastic-net solver
#'   iterations.
#' @param state.track.strategy Character value specifying the path-state
#'   tracking strategy. One of:
#'   \describe{
#'     \item{\code{"active.set.change"}}{
#'       Store states when the active set changes.
#'     }
#'     \item{\code{"all.iteration"}}{
#'       Store the state at every stagewise iteration.
#'     }
#'     \item{\code{"every.k.iteration"}}{
#'       Store the state every \code{state.track.freq} iterations.
#'     }
#'     \item{\code{"none"}}{
#'       Do not store stagewise path states.
#'     }
#'   }
#' @param state.track.freq Positive integer. Tracking frequency used for
#'   `state.track.strategy = "every.k.iteration"`.
#' @param verbose Logical value. Whether fitting progress should be printed.
#' @param include.data Logical value. Whether input data should be included in
#'   the returned object.
#' @param theta.initial Numeric vector containing initial values of the
#'   unpenalized coefficients. A scalar may be expanded by [ecountgmifs()] to
#'   the number of columns of the unpenalized design matrix.
#' @param theta.lower.bounds Numeric vector containing lower bounds for
#'   `theta`. Infinite values are allowed. A scalar may be expanded by
#'   [ecountgmifs()].
#' @param theta.upper.bounds Numeric vector containing upper bounds for
#'   `theta`. Infinite values are allowed. A scalar may be expanded by
#'   [ecountgmifs()].
#' @param nlopt.algorithm Integer NLopt algorithm identifier. The default,
#'   `28L`, corresponds to `NLOPT_LN_NELDERMEAD`.
#' @param nlopt.xtol.rel Non-negative relative parameter tolerance for NLopt.
#'   A value of zero disables this stopping condition.
#' @param nlopt.ftol.rel Non-negative relative objective tolerance for NLopt.
#'   A value of zero disables this stopping condition.
#' @param nlopt.maxeval Positive integer. Maximum number of NLopt objective
#'   evaluations.
#'
#' @return A list of control parameters.
#'
#' @export
ecountgmifs.control <- function(
    iteration.max = 10000L,
    epsilon.max = 0.01,
    epsilon.start = 1e-6,
    epsilon.min = .Machine$double.eps,
    tol = 1e-8,
    loglik.reltol.cutoff = 0.25,
    enet.abs.tol = 1e-10,
    enet.rel.tol = 1e-6,
    enet.max.iter = 99L,
    state.track.strategy = c(
      "active.set.change",
      "all.iteration",
      "every.k.iteration",
      "none"
    ),
    state.track.freq = 10L,
    verbose = FALSE,
    include.data = FALSE,
    theta.initial = 0,
    theta.lower.bounds = -Inf,
    theta.upper.bounds = Inf,
    nlopt.algorithm = 28L,
    nlopt.xtol.rel = tol,
    nlopt.ftol.rel = tol,
    nlopt.maxeval = 100L
) {
  check_scalar_numeric <- function(
    value,
    name,
    lower = -Inf,
    lower_inclusive = TRUE
  ) {
    valid_lower <- if (lower_inclusive) {
      value >= lower
    } else {
      value > lower
    }

    if (
      !is.numeric(value) ||
      length(value) != 1L ||
      is.na(value) ||
      !is.finite(value) ||
      !valid_lower
    ) {
      comparison <- if (lower_inclusive) ">=" else ">"

      stop(
        sprintf(
          "value of '%s' must be a finite number %s %s",
          name,
          comparison,
          format(lower)
        ),
        call. = FALSE
      )
    }
  }

  check_positive_integer <- function(value, name) {
    if (
      !is.numeric(value) ||
      length(value) != 1L ||
      is.na(value) ||
      !is.finite(value) ||
      value <= 0 ||
      value != floor(value) ||
      value > .Machine$integer.max
    ) {
      stop(
        sprintf(
          "value of '%s' must be a positive integer",
          name
        ),
        call. = FALSE
      )
    }
  }

  check_logical_scalar <- function(value, name) {
    if (
      !is.logical(value) ||
      length(value) != 1L ||
      is.na(value)
    ) {
      stop(
        sprintf(
          "value of '%s' must be TRUE or FALSE",
          name
        ),
        call. = FALSE
      )
    }
  }

  check_theta_vector <- function(
    value,
    name,
    finite = FALSE
  ) {
    if (
      !is.numeric(value) ||
      length(value) == 0L ||
      anyNA(value) ||
      any(is.nan(value)) ||
      (finite && any(!is.finite(value)))
    ) {
      requirement <- if (finite) {
        "finite numeric values"
      } else {
        "numeric values without NA or NaN"
      }

      stop(
        sprintf(
          "value of '%s' must contain %s",
          name,
          requirement
        ),
        call. = FALSE
      )
    }

    as.numeric(value)
  }

  state.track.strategy <-
    match.arg(state.track.strategy)

  check_positive_integer(
    iteration.max,
    "iteration.max"
  )

  check_scalar_numeric(
    epsilon.max,
    "epsilon.max",
    lower = 0,
    lower_inclusive = FALSE
  )

  check_scalar_numeric(
    epsilon.start,
    "epsilon.start",
    lower = 0,
    lower_inclusive = FALSE
  )

  check_scalar_numeric(
    epsilon.min,
    "epsilon.min",
    lower = 0,
    lower_inclusive = TRUE
  )

  if (epsilon.min > epsilon.start) {
    stop(
      "value of 'epsilon.min' must be <= 'epsilon.start'",
      call. = FALSE
    )
  }

  if (epsilon.start > epsilon.max) {
    stop(
      "value of 'epsilon.start' must be <= 'epsilon.max'",
      call. = FALSE
    )
  }

  check_scalar_numeric(
    tol,
    "tol",
    lower = 0,
    lower_inclusive = FALSE
  )

  check_scalar_numeric(
    loglik.reltol.cutoff,
    "loglik.reltol.cutoff",
    lower = 0,
    lower_inclusive = TRUE
  )

  check_scalar_numeric(
    enet.abs.tol,
    "enet.abs.tol",
    lower = 0,
    lower_inclusive = FALSE
  )

  check_scalar_numeric(
    enet.rel.tol,
    "enet.rel.tol",
    lower = 0,
    lower_inclusive = TRUE
  )

  check_positive_integer(
    enet.max.iter,
    "enet.max.iter"
  )

  check_positive_integer(
    state.track.freq,
    "state.track.freq"
  )

  check_logical_scalar(
    verbose,
    "verbose"
  )

  check_logical_scalar(
    include.data,
    "include.data"
  )

  theta.initial <- check_theta_vector(
    theta.initial,
    "theta.initial",
    finite = TRUE
  )

  theta.lower.bounds <- check_theta_vector(
    theta.lower.bounds,
    "theta.lower.bounds"
  )

  theta.upper.bounds <- check_theta_vector(
    theta.upper.bounds,
    "theta.upper.bounds"
  )

  theta.length <- max(
    length(theta.initial),
    length(theta.lower.bounds),
    length(theta.upper.bounds)
  )

  theta.lengths <- c(
    length(theta.initial),
    length(theta.lower.bounds),
    length(theta.upper.bounds)
  )

  if (any(theta.lengths != 1L & theta.lengths != theta.length)) {
    stop(
      paste0(
        "'theta.initial', 'theta.lower.bounds', and ",
        "'theta.upper.bounds' must have equal lengths or length one"
      ),
      call. = FALSE
    )
  }

  theta.initial <- rep(
    theta.initial,
    length.out = theta.length
  )

  theta.lower.bounds <- rep(
    theta.lower.bounds,
    length.out = theta.length
  )

  theta.upper.bounds <- rep(
    theta.upper.bounds,
    length.out = theta.length
  )

  if (any(theta.lower.bounds > theta.upper.bounds)) {
    stop(
      paste0(
        "each value of 'theta.lower.bounds' must be <= ",
        "the corresponding value of 'theta.upper.bounds'"
      ),
      call. = FALSE
    )
  }

  if (
    any(theta.initial < theta.lower.bounds) ||
    any(theta.initial > theta.upper.bounds)
  ) {
    stop(
      paste0(
        "each value of 'theta.initial' must lie within ",
        "its corresponding bounds"
      ),
      call. = FALSE
    )
  }

  if (
    !is.numeric(nlopt.algorithm) ||
    length(nlopt.algorithm) != 1L ||
    is.na(nlopt.algorithm) ||
    !is.finite(nlopt.algorithm) ||
    nlopt.algorithm < 0 ||
    nlopt.algorithm != floor(nlopt.algorithm)
  ) {
    stop(
      "value of 'nlopt.algorithm' must be a non-negative integer",
      call. = FALSE
    )
  }

  check_scalar_numeric(
    nlopt.xtol.rel,
    "nlopt.xtol.rel",
    lower = 0,
    lower_inclusive = TRUE
  )

  check_scalar_numeric(
    nlopt.ftol.rel,
    "nlopt.ftol.rel",
    lower = 0,
    lower_inclusive = TRUE
  )

  check_positive_integer(
    nlopt.maxeval,
    "nlopt.maxeval"
  )

  list(
    iteration.max = as.integer(iteration.max),
    epsilon.max = epsilon.max,
    epsilon.start = epsilon.start,
    epsilon.min = epsilon.min,
    tol = tol,
    loglik.reltol.cutoff = loglik.reltol.cutoff,
    enet.abs.tol = enet.abs.tol,
    enet.rel.tol = enet.rel.tol,
    enet.max.iter = as.integer(enet.max.iter),
    state.track.strategy = state.track.strategy,
    state.track.freq = as.integer(state.track.freq),
    verbose = verbose,
    include.data = include.data,
    theta.initial = theta.initial,
    theta.lower.bounds = theta.lower.bounds,
    theta.upper.bounds = theta.upper.bounds,
    nlopt.algorithm = as.integer(nlopt.algorithm),
    nlopt.xtol.rel = nlopt.xtol.rel,
    nlopt.ftol.rel = nlopt.ftol.rel,
    nlopt.maxeval = as.integer(nlopt.maxeval)
  )
}
