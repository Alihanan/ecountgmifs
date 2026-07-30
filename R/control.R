#' Auxiliary control for extended count GMIFS fitting
#'
#' @description
#' Creates a control object for [ecountgmifs()]. Null-model and stagewise
#' iteration limits are independent. NLopt settings are shared by parameter
#' block: nonpenalized, family, and link parameters.
#'
#' @param null.iteration.max Positive integer. Maximum number of outer
#'   theta/link/family alternations used to fit the null model.
#' @param stagewise.iteration.max Positive integer. Maximum number of completed
#'   forward-stagewise iterations.
#' @param null.family.parameter.abs.tol Positive numeric value. Absolute
#'   tolerance used when comparing family parameters between consecutive
#'   null-model outer iterations.
#' @param stagewise.objective.rel.tol Non-negative relative
#'   negative-log-likelihood tolerance for stagewise stopping.
#' @param stagewise.beta.step.norm.tol Non-negative Euclidean beta-step norm
#'   threshold for stagewise stopping.
#' @param epsilon.max Positive maximum stagewise step.
#' @param epsilon.start Positive initial stagewise step.
#' @param epsilon.min Non-negative minimum stagewise step.
#' @param loglik.reltol.cutoff Non-negative relative cutoff for the pseudo-R2
#'   stopping rule.
#' @param enet.abs.tol Positive absolute tolerance for the weighted elastic-net
#'   solver.
#' @param enet.rel.tol Non-negative relative tolerance for the weighted
#'   elastic-net solver.
#' @param enet.max.iter Positive maximum elastic-net bisection iterations.
#' @param state.track.strategy Character path-state tracking strategy.
#' @param state.track.freq Positive integer used by
#'   `state.track.strategy = "every.k.iteration"`.
#' @param verbose Logical. Print fitting progress.
#' @param include.data Logical. Include input data in the returned object.
#' @param theta.initial Initial unpenalized coefficients.
#' @param theta.lower.bounds Lower bounds for unpenalized coefficients.
#' @param theta.upper.bounds Upper bounds for unpenalized coefficients.
#' @param nonpen.nlopt.algorithm,family.nlopt.algorithm,link.nlopt.algorithm
#'   Non-negative NLopt algorithm identifiers. Each setting is shared across
#'   all fitting phases for the corresponding parameter block.
#' @param nonpen.nlopt.xtol.rel,family.nlopt.xtol.rel,link.nlopt.xtol.rel
#'   Non-negative relative parameter tolerances.
#' @param nonpen.nlopt.ftol.rel,family.nlopt.ftol.rel,link.nlopt.ftol.rel
#'   Non-negative relative objective tolerances.
#' @param nonpen.nlopt.maxeval,family.nlopt.maxeval,link.nlopt.maxeval
#'   Positive maximum NLopt evaluation counts.
#'
#' @return A list of control parameters.
#' @export
ecountgmifs.control <- function(
    null.iteration.max = 1000L,
    stagewise.iteration.max = 10000L,
    null.family.parameter.abs.tol = 1e-8,
    stagewise.objective.rel.tol = 1e-8,
    stagewise.beta.step.norm.tol = .Machine$double.eps,
    epsilon.max = 0.01,
    epsilon.start = 1e-6,
    epsilon.min = .Machine$double.eps,
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

    nonpen.nlopt.algorithm = 28L,
    nonpen.nlopt.xtol.rel = 1e-8,
    nonpen.nlopt.ftol.rel = 1e-8,
    nonpen.nlopt.maxeval = 100L,

    family.nlopt.algorithm = 28L,
    family.nlopt.xtol.rel = 1e-8,
    family.nlopt.ftol.rel = 1e-8,
    family.nlopt.maxeval = 100L,

    link.nlopt.algorithm = 28L,
    link.nlopt.xtol.rel = 1e-8,
    link.nlopt.ftol.rel = 1e-8,
    link.nlopt.maxeval = 100L
) {
  check_scalar_numeric <- function(
      value,
      name,
      lower = -Inf,
      lower.inclusive = TRUE
  ) {
    valid.lower <- if (lower.inclusive) {
      value >= lower
    } else {
      value > lower
    }

    if (
      !is.numeric(value) ||
      length(value) != 1L ||
      is.na(value) ||
      !is.finite(value) ||
      !valid.lower
    ) {
      comparison <- if (lower.inclusive) ">=" else ">"

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

  check_algorithm <- function(value, name) {
    if (
      !is.numeric(value) ||
      length(value) != 1L ||
      is.na(value) ||
      !is.finite(value) ||
      value < 0 ||
      value != floor(value) ||
      value > .Machine$integer.max
    ) {
      stop(
        sprintf(
          "value of '%s' must be a non-negative integer",
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
    null.iteration.max,
    "null.iteration.max"
  )

  check_positive_integer(
    stagewise.iteration.max,
    "stagewise.iteration.max"
  )

  check_scalar_numeric(
    null.family.parameter.abs.tol,
    "null.family.parameter.abs.tol",
    lower = 0,
    lower.inclusive = FALSE
  )

  check_scalar_numeric(
    stagewise.objective.rel.tol,
    "stagewise.objective.rel.tol",
    lower = 0
  )

  check_scalar_numeric(
    stagewise.beta.step.norm.tol,
    "stagewise.beta.step.norm.tol",
    lower = 0
  )

  check_scalar_numeric(
    epsilon.max,
    "epsilon.max",
    lower = 0,
    lower.inclusive = FALSE
  )

  check_scalar_numeric(
    epsilon.start,
    "epsilon.start",
    lower = 0,
    lower.inclusive = FALSE
  )

  check_scalar_numeric(
    epsilon.min,
    "epsilon.min",
    lower = 0
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
    loglik.reltol.cutoff,
    "loglik.reltol.cutoff",
    lower = 0
  )

  check_scalar_numeric(
    enet.abs.tol,
    "enet.abs.tol",
    lower = 0,
    lower.inclusive = FALSE
  )

  check_scalar_numeric(
    enet.rel.tol,
    "enet.rel.tol",
    lower = 0
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

  nlopt.prefixes <- c(
    "nonpen",
    "family",
    "link"
  )

  nlopt.algorithms <- c(
    nonpen.nlopt.algorithm,
    family.nlopt.algorithm,
    link.nlopt.algorithm
  )

  nlopt.xtol.rels <- c(
    nonpen.nlopt.xtol.rel,
    family.nlopt.xtol.rel,
    link.nlopt.xtol.rel
  )

  nlopt.ftol.rels <- c(
    nonpen.nlopt.ftol.rel,
    family.nlopt.ftol.rel,
    link.nlopt.ftol.rel
  )

  nlopt.maxevals <- c(
    nonpen.nlopt.maxeval,
    family.nlopt.maxeval,
    link.nlopt.maxeval
  )

  for (i in seq_along(nlopt.prefixes)) {
    check_algorithm(
      nlopt.algorithms[[i]],
      paste0(
        nlopt.prefixes[[i]],
        ".nlopt.algorithm"
      )
    )

    check_scalar_numeric(
      nlopt.xtol.rels[[i]],
      paste0(
        nlopt.prefixes[[i]],
        ".nlopt.xtol.rel"
      ),
      lower = 0
    )

    check_scalar_numeric(
      nlopt.ftol.rels[[i]],
      paste0(
        nlopt.prefixes[[i]],
        ".nlopt.ftol.rel"
      ),
      lower = 0
    )

    check_positive_integer(
      nlopt.maxevals[[i]],
      paste0(
        nlopt.prefixes[[i]],
        ".nlopt.maxeval"
      )
    )
  }

  list(
    null.iteration.max =
      as.integer(null.iteration.max),

    stagewise.iteration.max =
      as.integer(stagewise.iteration.max),

    null.family.parameter.abs.tol =
      null.family.parameter.abs.tol,

    stagewise.objective.rel.tol =
      stagewise.objective.rel.tol,

    stagewise.beta.step.norm.tol =
      stagewise.beta.step.norm.tol,

    epsilon.max = epsilon.max,
    epsilon.start = epsilon.start,
    epsilon.min = epsilon.min,

    loglik.reltol.cutoff =
      loglik.reltol.cutoff,

    enet.abs.tol = enet.abs.tol,
    enet.rel.tol = enet.rel.tol,
    enet.max.iter =
      as.integer(enet.max.iter),

    state.track.strategy =
      state.track.strategy,

    state.track.freq =
      as.integer(state.track.freq),

    verbose = verbose,
    include.data = include.data,

    theta.initial = theta.initial,
    theta.lower.bounds =
      theta.lower.bounds,
    theta.upper.bounds =
      theta.upper.bounds,

    nonpen.nlopt.algorithm =
      as.integer(
        nonpen.nlopt.algorithm
      ),
    nonpen.nlopt.xtol.rel =
      nonpen.nlopt.xtol.rel,
    nonpen.nlopt.ftol.rel =
      nonpen.nlopt.ftol.rel,
    nonpen.nlopt.maxeval =
      as.integer(
        nonpen.nlopt.maxeval
      ),

    family.nlopt.algorithm =
      as.integer(
        family.nlopt.algorithm
      ),
    family.nlopt.xtol.rel =
      family.nlopt.xtol.rel,
    family.nlopt.ftol.rel =
      family.nlopt.ftol.rel,
    family.nlopt.maxeval =
      as.integer(
        family.nlopt.maxeval
      ),

    link.nlopt.algorithm =
      as.integer(
        link.nlopt.algorithm
      ),
    link.nlopt.xtol.rel =
      link.nlopt.xtol.rel,
    link.nlopt.ftol.rel =
      link.nlopt.ftol.rel,
    link.nlopt.maxeval =
      as.integer(
        link.nlopt.maxeval
      )
  )
}
