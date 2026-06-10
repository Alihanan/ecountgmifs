#' Auxiliary control for extended count GMIFS fitting
#'
#' @description
#' Auxiliary function for [ecountgmifs()] fitting. Typically only used when
#' calling [ecountgmifs()] through its `control` argument.
#'
#' @details
#' The \code{control} argument of [ecountgmifs()] is passed to the internal
#' fitting routine, which uses its elements to control the stagewise path,
#' adaptive step-size procedure, internal numerical optimization, stopping
#' rules, and coefficient-path storage. The function
#' \code{ecountgmifs.control()} provides default values and performs basic
#' sanity checking before fitting begins.
#'
#' The state-tracking strategy given by \code{state.track.strategy} controls
#' the amount of path information stored
#' in the returned [ecountgmifs()] object. Tracked quantities include the
#' coefficient vector, active set, step size, dispersion parameter, elastic-net
#' norm, log-likelihood, and other diagnostics. Tracking every iteration gives
#' the most detailed representation of the algorithm path, whereas
#' \code{"active.set.change"} stores a compressed path indexed by changes in
#' the active set.
#'
#' @param iteration.max maximum number of path-following iteration steps
#' @param epsilon.max maximum step size, e.g. upper bound for step doubling
#' @param epsilon.min minimum step size, e.g. everything smaller considered a zero step and considered converged
#' @param epsilon.start initial step size before halving procedure
#' @param tol numerical convergence tolerance for main C++ code
#' @param nlopt.optim.reltol relative tolerance for internal C++ numerical optimization
#'   performed using the \emph{NLopt} library through \link[nloptr]{nloptr}.
#' @param state.track.strategy character. Strategy used to track the algorithm
#'   state along the stagewise path. The tracked state may include quantities
#'   such as the current coefficient vector, active set, step size, dispersion
#'   parameter, elastic-net norm, log-likelihood, and other path diagnostics.
#'   One of:
#'   \describe{
#'     \item{\code{"active.set.change"}}{
#'       Track the current state only when the active set changes. Consecutive
#'       iterations with the same active set overwrite the previous stored
#'       state, so the returned path contains the last iteration associated
#'       with each distinct active set. This is the default and usually the
#'       most memory-efficient informative option.
#'     }
#'     \item{\code{"all.iteration"}}{
#'       Track the state at every path-following iteration. This gives the most
#'       detailed representation of the path, but can require substantial
#'       memory for long paths or high-dimensional designs.
#'     }
#'     \item{\code{"every.k.iteration"}}{
#'       Track the state every \code{state.track.freq} iterations. This gives a
#'       regular subsampling of the path and is useful when full iteration-level
#'       tracking is too large.
#'     }
#'     \item{\code{"none"}}{
#'       Do not track the path state. Only final or selected model summaries are
#'       returned.
#'     }
#'   }
#' @param state.track.freq integer. Tracking frequency used when
#'   \code{state.track.strategy = "every.k.iteration"}. For example,
#'   \code{state.track.freq = 10} stores the algorithm state every 10
#'   iterations. Ignored by the other tracking strategies.
#' @param loglik.reltol.cutoff numeric. Non-negative relative
#'   log-likelihood cutoff used by stopping or model-selection rules comparing
#'   the fitted path to a reference log-likelihood. This quantity corresponds
#'   to the pseudo-\eqn{R^2} threshold used in the accompanying research paper.
#'
#'
#' @return A list of control parameters.
#' @export
ecountgmifs.control <- function(
    iteration.max = 1000000,
    epsilon.max = 0.01,
    epsilon.start = 1e-6,
    epsilon.min = .Machine$double.eps,
    tol = 1e-8,
    nlopt.optim.reltol = tol,
    loglik.reltol.cutoff = 0.25,
    state.track.strategy = c("active.set.change",
                                  "all.iteration",
                                  "every.k.iteration",
                                  "none"),
    state.track.freq = 10

) {

  state.track.strategy <- match.arg(state.track.strategy)
  if (!is.numeric(state.track.freq) ||
      length(state.track.freq) != 1L ||
      is.na(state.track.freq) ||
      state.track.freq <= 0 ||
      state.track.freq != as.integer(state.track.freq)) {
    stop("value of 'state.track.freq' must be a positive integer")
  }

  if (!is.numeric(iteration.max) ||
      length(iteration.max) != 1L ||
      is.na(iteration.max) ||
      iteration.max <= 0 ||
      iteration.max != as.integer(iteration.max)) {
    stop("value of 'iteration.max' must be a positive integer")
  }

  if (!is.numeric(epsilon.max) ||
      length(epsilon.max) != 1L ||
      is.na(epsilon.max) ||
      epsilon.max <= 0) {
    stop("value of 'epsilon.max' must be > 0")
  }

  if (!is.numeric(epsilon.start) ||
      length(epsilon.start) != 1L ||
      is.na(epsilon.start) ||
      epsilon.start <= 0) {
    stop("value of 'epsilon.start' must be > 0")
  }

  if (!is.numeric(epsilon.min) ||
      length(epsilon.min) != 1L ||
      is.na(epsilon.min) ||
      epsilon.min <= 0) {
    stop("value of 'epsilon.min' must be > 0")
  }

  if (epsilon.min > epsilon.start) {
    stop("value of 'epsilon.min' must be <= 'epsilon.start'")
  }

  if (epsilon.start > epsilon.max) {
    stop("value of 'epsilon.start' must be <= 'epsilon.max'")
  }

  if (!is.numeric(tol) ||
      length(tol) != 1L ||
      is.na(tol) ||
      tol <= 0) {
    stop("value of 'tol' must be > 0")
  }

  if (!is.numeric(nlopt.optim.reltol) ||
      length(nlopt.optim.reltol) != 1L ||
      is.na(nlopt.optim.reltol) ||
      nlopt.optim.reltol <= 0) {
    stop("value of 'nlopt.optim.reltol' must be > 0")
  }

  if (!is.numeric(loglik.reltol.cutoff) ||
      length(loglik.reltol.cutoff) != 1L ||
      is.na(loglik.reltol.cutoff) ||
      loglik.reltol.cutoff < 0) {
    stop("value of 'loglik.reltol.cutoff' must be >= 0")
  }

  list(
    state.track.strategy = state.track.strategy,
    state.track.freq = as.integer(state.track.freq),
    iteration.max = as.integer(iteration.max),
    epsilon.max = epsilon.max,
    epsilon.start = epsilon.start,
    epsilon.min = epsilon.min,
    tol = tol,
    nlopt.optim.reltol = nlopt.optim.reltol,
    loglik.reltol.cutoff = loglik.reltol.cutoff
  )
}
