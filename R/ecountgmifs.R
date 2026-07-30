#' Fit an extended count GMIFS model
#'
#' Fits an extended generalized monotone incremental forward-stagewise model
#' for high-dimensional count outcomes.
#'
#' @param X Numeric penalized predictor matrix.
#' @param y Non-negative count response.
#' @param w Numeric unpenalized predictor matrix or `NULL`.
#' @param intercept Logical. Add an intercept column to `w`.
#' @param offset Numeric offset vector or `NULL`.
#' @param weight.vec Positive elastic-net prior weights or `NULL`.
#' @param enet.alpha Elastic-net mixing parameter in `[0, 1]`.
#' @param family Either `"negative.binomial"` or `"poisson"`.
#' @param savefolder Retained for compatibility; currently unused.
#' @param link Either `"log"` or `"softplus"`.
#' @param criteria A criterion external pointer, a list of criterion external
#'   pointers, `NULL` for the example AIC/BIC/SABIC set, or an empty list to
#'   disable criteria.
#' @param verbose Logical. Print fitting progress.
#' @param fixed.dispersion Logical. Fixed NB2 dispersion is not yet supported by
#'   the current plugin interface.
#' @param fixed.dispersion.value Retained for compatibility.
#' @param include.data Logical. Include input data in the result.
#' @param control An object returned by [ecountgmifs.control()].
#'
#' @return An object of class `"ecountgmifs"`.
#' @export
ecountgmifs <- function(
    X,
    y,
    w = NULL,
    intercept = TRUE,
    offset = NULL,
    weight.vec = NULL,
    enet.alpha = 1,
    family = c("negative.binomial", "poisson"),
    savefolder = NULL,
    link = c("log", "softplus"),
    criteria = NULL,
    verbose = FALSE,
    fixed.dispersion = FALSE,
    fixed.dispersion.value = 0,
    include.data = FALSE,
    control = ecountgmifs.control()
) {
  X <- as.matrix(X)
  y <- as.numeric(y)

  if (!is.numeric(X) || nrow(X) == 0L || ncol(X) == 0L) {
    stop("value of 'X' must be a non-empty numeric matrix", call. = FALSE)
  }

  if (!is.numeric(y) || length(y) != nrow(X)) {
    stop("length of numeric 'y' must equal 'nrow(X)'", call. = FALSE)
  }

  if (anyNA(X) || any(!is.finite(X))) {
    stop("value of 'X' must contain finite values", call. = FALSE)
  }

  if (anyNA(y) || any(!is.finite(y)) || any(y < 0)) {
    stop("value of 'y' must contain finite non-negative counts", call. = FALSE)
  }

  if (!is.logical(intercept) || length(intercept) != 1L || is.na(intercept)) {
    stop("value of 'intercept' must be TRUE or FALSE", call. = FALSE)
  }

  if (
    !is.numeric(enet.alpha) ||
    length(enet.alpha) != 1L ||
    is.na(enet.alpha) ||
    !is.finite(enet.alpha) ||
    enet.alpha < 0 ||
    enet.alpha > 1
  ) {
    stop("value of 'enet.alpha' must be in [0, 1]", call. = FALSE)
  }

  if (is.null(w)) {
    if (!intercept) {
      stop("value of 'w' must be supplied when 'intercept' is FALSE", call. = FALSE)
    }

    w <- matrix(1, nrow = nrow(X), ncol = 1L)
    colnames(w) <- "(Intercept)"
  } else {
    w <- as.matrix(w)

    if (!is.numeric(w) || nrow(w) != nrow(X)) {
      stop("numeric 'w' must have the same row count as 'X'", call. = FALSE)
    }

    if (anyNA(w) || any(!is.finite(w))) {
      stop("value of 'w' must contain finite values", call. = FALSE)
    }

    if (intercept) {
      w <- cbind("(Intercept)" = 1, w)
    } else if (ncol(w) == 0L) {
      stop("value of 'w' must contain at least one column", call. = FALSE)
    }
  }

  if (is.null(offset)) {
    offset <- rep(0, nrow(X))
  } else {
    offset <- as.numeric(offset)

    if (
      length(offset) != nrow(X) ||
      anyNA(offset) ||
      any(!is.finite(offset))
    ) {
      stop("value of 'offset' must be a finite vector of length 'nrow(X)'", call. = FALSE)
    }
  }

  if (is.null(weight.vec)) {
    weight.vec <- rep(1, ncol(X))
  } else {
    weight.vec <- as.numeric(weight.vec)

    if (
      length(weight.vec) != ncol(X) ||
      anyNA(weight.vec) ||
      any(!is.finite(weight.vec)) ||
      any(weight.vec <= 0)
    ) {
      stop("value of 'weight.vec' must contain positive finite values", call. = FALSE)
    }
  }

  weight.vec <-
    weight.vec * ncol(X) / sum(weight.vec)

  family <- match.arg(family)
  link <- match.arg(link)

  if (isTRUE(fixed.dispersion) && family == "negative.binomial") {
    stop(
      "fixed NB2 dispersion is not supported by the current family plugin interface",
      call. = FALSE
    )
  }

  family.pointer <- NULL
  link.pointer <- NULL
  family.link.pointer <- NULL

  if (family == "negative.binomial" && link == "log") {
    family.link.pointer <-
      example_create_nb2_log_family_link()
  } else {
    family.pointer <- switch(
      family,
      "negative.binomial" = example_create_nb2_family(),
      "poisson" = example_create_poisson_family()
    )

    link.pointer <- switch(
      link,
      "log" = example_create_log_link(),
      "softplus" = example_create_softplus_link()
    )
  }

  if (is.null(criteria)) {
    criteria <- list(
      AIC = example_create_aic_criterion(),
      BIC = example_create_bic_criterion(),
      SABIC = example_create_sabic_criterion()
    )
  } else if (typeof(criteria) == "externalptr") {
    criteria <- list(criteria)
  }

  if (
    !is.list(criteria) ||
    !all(vapply(criteria, typeof, character(1L)) == "externalptr")
  ) {
    stop(
      "value of 'criteria' must be an external pointer or a list of external pointers",
      call. = FALSE
    )
  }

  strategy <- switch(
    control$state.track.strategy,
    "active.set.change" = 0L,
    "all.iteration" = 1L,
    "every.k.iteration" = 2L,
    "none" = 3L,
    stop("unknown state tracking strategy", call. = FALSE)
  )

  q <- ncol(w)

  expand_theta <- function(value, name) {
    if (length(value) == 1L) {
      return(rep(value, q))
    }

    if (length(value) != q) {
      stop(
        sprintf("length of '%s' must be one or 'ncol(w)'", name),
        call. = FALSE
      )
    }

    value
  }

  theta.initial <-
    expand_theta(control$theta.initial, "theta.initial")

  theta.lower.bounds <-
    expand_theta(control$theta.lower.bounds, "theta.lower.bounds")

  theta.upper.bounds <-
    expand_theta(control$theta.upper.bounds, "theta.upper.bounds")

  out <- ecountgmifs_cpp(
    X = X,
    y = y,
    w = w,
    offset = offset,
    weight_vec = weight.vec,
    enet_alpha = enet.alpha,

    epsilon_start = control$epsilon.start,
    epsilon_max = control$epsilon.max,
    epsilon_min = control$epsilon.min,

    null_iteration_max = control$null.iteration.max,
    stagewise_iteration_max = control$stagewise.iteration.max,
    null_family_parameter_abs_tol =
      control$null.family.parameter.abs.tol,
    stagewise_objective_rel_tol =
      control$stagewise.objective.rel.tol,
    stagewise_beta_step_norm_tol =
      control$stagewise.beta.step.norm.tol,

    family = family.pointer,
    link_func = link.pointer,
    criteria = criteria,

    loglik_reltol_cutoff = control$loglik.reltol.cutoff,
    enet_abs_tol = control$enet.abs.tol,
    enet_rel_tol = control$enet.rel.tol,
    enet_max_iter = control$enet.max.iter,

    verbose = isTRUE(verbose) || isTRUE(control$verbose),
    include_data = isTRUE(include.data) || isTRUE(control$include.data),
    state_track_strategy = strategy,
    state_track_freq = control$state.track.freq,

    theta_initial = theta.initial,
    theta_lower_bounds = theta.lower.bounds,
    theta_upper_bounds = theta.upper.bounds,

    # RcppExports is intentionally unchanged. Its existing phase-specific
    # boundary arguments receive the same parameter-block control values.
    null_nonpen_nlopt_algorithm =
      control$nonpen.nlopt.algorithm,
    null_nonpen_nlopt_xtol_rel =
      control$nonpen.nlopt.xtol.rel,
    null_nonpen_nlopt_ftol_rel =
      control$nonpen.nlopt.ftol.rel,
    null_nonpen_nlopt_maxeval =
      control$nonpen.nlopt.maxeval,

    null_family_nlopt_algorithm =
      control$family.nlopt.algorithm,
    null_family_nlopt_xtol_rel =
      control$family.nlopt.xtol.rel,
    null_family_nlopt_ftol_rel =
      control$family.nlopt.ftol.rel,
    null_family_nlopt_maxeval =
      control$family.nlopt.maxeval,

    null_link_nlopt_algorithm =
      control$link.nlopt.algorithm,
    null_link_nlopt_xtol_rel =
      control$link.nlopt.xtol.rel,
    null_link_nlopt_ftol_rel =
      control$link.nlopt.ftol.rel,
    null_link_nlopt_maxeval =
      control$link.nlopt.maxeval,

    saturated_family_nlopt_algorithm =
      control$family.nlopt.algorithm,
    saturated_family_nlopt_xtol_rel =
      control$family.nlopt.xtol.rel,
    saturated_family_nlopt_ftol_rel =
      control$family.nlopt.ftol.rel,
    saturated_family_nlopt_maxeval =
      control$family.nlopt.maxeval,

    stagewise_nonpen_nlopt_algorithm =
      control$nonpen.nlopt.algorithm,
    stagewise_nonpen_nlopt_xtol_rel =
      control$nonpen.nlopt.xtol.rel,
    stagewise_nonpen_nlopt_ftol_rel =
      control$nonpen.nlopt.ftol.rel,
    stagewise_nonpen_nlopt_maxeval =
      control$nonpen.nlopt.maxeval,

    stagewise_family_nlopt_algorithm =
      control$family.nlopt.algorithm,
    stagewise_family_nlopt_xtol_rel =
      control$family.nlopt.xtol.rel,
    stagewise_family_nlopt_ftol_rel =
      control$family.nlopt.ftol.rel,
    stagewise_family_nlopt_maxeval =
      control$family.nlopt.maxeval,

    stagewise_link_nlopt_algorithm =
      control$link.nlopt.algorithm,
    stagewise_link_nlopt_xtol_rel =
      control$link.nlopt.xtol.rel,
    stagewise_link_nlopt_ftol_rel =
      control$link.nlopt.ftol.rel,
    stagewise_link_nlopt_maxeval =
      control$link.nlopt.maxeval,

    family_link = family.link.pointer
  )

  out$call <- match.call()
  class(out) <- c("ecountgmifs", class(out))
  out
}
