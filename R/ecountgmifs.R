.ecountgmifs.resolve.plugin <- function(plugin, type, envir = parent.frame()) {
  value <- plugin

  if (is.character(value) && length(value) == 1L) {
    value <- get(value, envir = envir, mode = "function", inherits = TRUE)
  }

  if (is.function(value)) {
    value <- value()
  }

  if (!identical(typeof(value), "externalptr")) {
    stop(
      "`", type, "` must be an external pointer or a zero-argument ",
      "constructor returning one.",
      call. = FALSE
    )
  }

  declared.type <- attr(value, "plugin.type", exact = TRUE)
  if (is.null(declared.type)) {
    declared.type <- attr(value, "r.plugin.type", exact = TRUE)
  }
  if (!is.null(declared.type) && !identical(declared.type, type)) {
    stop(
      "`", type, "` received a plugin declared as `", declared.type, "`.",
      call. = FALSE
    )
  }

  validator <- switch(
    type,
    link = inspect_link_plugin,
    family = inspect_family_plugin,
    family.link = inspect_family_link_plugin,
    criterion = inspect_criterion_plugin,
    stop("Unknown plugin type.", call. = FALSE)
  )
  validator(value)

  value
}

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
#' @param weight.vec Positive elastic-net prior weights, an object returned by
#'   [ecountgmifs.weight.prior()], or `NULL`.
#' @param enet.alpha One or more elastic-net mixing parameters in `[0, 1]`.
#'   A vector fits one complete path per alpha and returns an
#'   `ecountgmifs_multi_alpha` object.
#' @param family Either `"negative.binomial"`, `"poisson"`, a family external
#'   pointer, a zero-argument constructor returning one, or its character name.
#' @param savefolder Retained for compatibility; currently unused.
#' @param link Either `"log"`, `"softplus"`, a link external pointer, a
#'   zero-argument constructor returning one, or its character name.
#' @param criteria A criterion pointer, constructor, character constructor name,
#'   list of such specifications, `NULL` for the built-in AIC/BIC/SABIC set, or
#'   an empty list to disable criteria.
#' @param verbose Logical. Print fitting progress.
#' @param fixed.dispersion Logical. Fixed NB2 dispersion is not yet supported by
#'   the current plugin interface.
#' @param fixed.dispersion.value Retained for compatibility.
#' @param include.data Logical. Include input data in the result.
#' @param control An object returned by [ecountgmifs.control()].
#' @param family.link Optional fused family-link pointer, zero-argument
#'   constructor, or its character name. When supplied, it takes precedence over
#'   `family` and `link`.
#'
#' @return An object of class `"ecountgmifs"` for one fit,
#'   `"ecountgmifs_multi_alpha"` for multiple alpha values with a fixed
#'   numeric weight vector, or `"ecountgmifs_tuning_grid"` when a prior-weight
#'   specification supplies one or more prior-strength values.
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
    control = ecountgmifs.control(),
    family.link = NULL
) {
  fit_call <- match.call()

  if (
      !is.numeric(enet.alpha) ||
      length(enet.alpha) == 0L ||
      anyNA(enet.alpha) ||
      any(!is.finite(enet.alpha)) ||
      any(enet.alpha < 0) ||
      any(enet.alpha > 1)
  ) {
    stop("value of 'enet.alpha' must contain finite values in [0, 1]", call. = FALSE)
  }

  enet.alpha <- as.numeric(enet.alpha)

  if (anyDuplicated(enet.alpha)) {
    warning(
      "duplicated values in 'enet.alpha' were removed",
      call. = FALSE
    )
    enet.alpha <- unique(enet.alpha)
  }

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

  predictor_names <- if (is.null(colnames(X))) {
    paste0("X", seq_len(ncol(X)))
  } else {
    make.unique(as.character(colnames(X)))
  }
  colnames(X) <- predictor_names

  weight_prior <- NULL

  if (inherits(weight.vec, "ecountgmifs_weight_prior")) {
    weight_prior <- .ecountgmifs.prepare.weight.prior(
      object = weight.vec,
      predictor_names = predictor_names,
      predictor_count = ncol(X)
    )
  } else if (is.null(weight.vec)) {
    weight.vec <- rep(1, ncol(X))
    names(weight.vec) <- predictor_names
  } else {
    original_weight_names <- names(weight.vec)
    weight.vec <- as.numeric(weight.vec)

    if (!is.null(original_weight_names)) {
      names(weight.vec) <- original_weight_names

      missing_weight <- setdiff(
        predictor_names,
        names(weight.vec)
      )

      if (length(missing_weight) > 0L) {
        stop(
          "Named `weight.vec` is missing predictors: ",
          paste(utils::head(missing_weight, 20L), collapse = ", "),
          if (length(missing_weight) > 20L) " ..." else "",
          call. = FALSE
        )
      }

      weight.vec <- weight.vec[predictor_names]
    }

    if (
      length(weight.vec) != ncol(X) ||
      anyNA(weight.vec) ||
      any(!is.finite(weight.vec)) ||
      any(weight.vec <= 0)
    ) {
      stop("value of 'weight.vec' must contain positive finite values", call. = FALSE)
    }
  }

  if (
      !is.null(weight_prior) &&
      (
        length(weight_prior$eta) > 1L ||
        length(enet.alpha) > 1L
      )
  ) {
    return(
      .ecountgmifs.fit.tuning.grid(
        fit_call = fit_call,
        alpha = enet.alpha,
        weight_prior = weight_prior,
        envir = parent.frame()
      )
    )
  }

  if (!is.null(weight_prior)) {
    weight.vec <- weight_prior$weights[[1L]]
  }

  if (length(enet.alpha) > 1L) {
    return(
      .ecountgmifs.fit.multi.alpha(
        fit_call = fit_call,
        alpha = enet.alpha,
        envir = parent.frame()
      )
    )
  }

  weight.vec <-
    weight.vec * ncol(X) / sum(weight.vec)
  names(weight.vec) <- predictor_names

  builtin.family <- NULL
  builtin.link <- NULL

  family.choices <- c("negative.binomial", "poisson")
  link.choices <- c("log", "softplus")

  if (
      is.character(family) &&
      (length(family) > 1L || family %in% family.choices)
  ) {
    builtin.family <- match.arg(family, family.choices)
  }
  if (
      is.character(link) &&
      (length(link) > 1L || link %in% link.choices)
  ) {
    builtin.link <- match.arg(link, link.choices)
  }

  if (
      isTRUE(fixed.dispersion) &&
      identical(builtin.family, "negative.binomial")
  ) {
    stop(
      "fixed NB2 dispersion is not supported by the current family plugin interface",
      call. = FALSE
    )
  }

  family.pointer <- NULL
  link.pointer <- NULL
  family.link.pointer <- NULL

  if (!is.null(family.link)) {
    family.link.pointer <- .ecountgmifs.resolve.plugin(
      family.link,
      "family.link",
      envir = parent.frame()
    )
  } else if (
      identical(builtin.family, "negative.binomial") &&
      identical(builtin.link, "log")
  ) {
    family.link.pointer <- example_create_nb2_log_family_link()
  } else {
    family.pointer <- if (!is.null(builtin.family)) {
      switch(
        builtin.family,
        "negative.binomial" = example_create_nb2_family(),
        "poisson" = example_create_poisson_family()
      )
    } else {
      .ecountgmifs.resolve.plugin(family, "family", envir = parent.frame())
    }

    link.pointer <- if (!is.null(builtin.link)) {
      switch(
        builtin.link,
        "log" = example_create_log_link(),
        "softplus" = example_create_softplus_link()
      )
    } else {
      .ecountgmifs.resolve.plugin(link, "link", envir = parent.frame())
    }
  }

  if (is.null(criteria)) {
    criteria <- list(
      AIC = plugin.criterion.AIC.builtin(),
      BIC = plugin.criterion.BIC.builtin(),
      SABIC = plugin.criterion.SABIC.builtin()
    )
  } else {
    criteria <- .ecountgmifs.normalize.criteria(
      criteria,
      envir = parent.frame()
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

  out$call <- fit_call

  # Lightweight plotting and printing metadata are retained even when
  # `include.data = FALSE`. The prior-weight vector is needed for the optional
  # ground-truth baseline and is only length p.
  unpenalized_names <- if (is.null(colnames(w))) {
    paste0("theta", seq_len(ncol(w)))
  } else {
    make.unique(as.character(colnames(w)))
  }

  out$input$weight_vec <- weight.vec

  if (!is.null(weight_prior)) {
    out$input$prior_eta <- weight_prior$eta[[1L]]
    out$input$prior_eta_label <- weight_prior$eta_labels[[1L]]
    out$input$prior_selected <- weight_prior$score > 0
    names(out$input$prior_selected) <- predictor_names
    out$weight_prior <- list(
      label = weight_prior$label,
      eta = weight_prior$eta[[1L]],
      eta_label = weight_prior$eta_labels[[1L]],
      score = weight_prior$score
    )
  }

  out <- .ecountgmifs.assign.parameter.names(
    object = out,
    predictor.names = predictor_names,
    unpenalized.names = unpenalized_names
  )

  out$input$response_name <- deparse(substitute(y))

  # Retain compact data summaries needed by diagnostic plots even when the
  # complete design matrices are intentionally omitted from the returned fit.
  out$diagnostics <- list(
    response = y,
    response_name = out$input$response_name,
    response_mean = mean(y),
    response_variance = stats::var(y),
    predictor_mean = colMeans(X),
    predictor_variance = apply(X, 2L, stats::var),
    predictor_nonnegative = apply(X, 2L, function(value) all(value >= 0))
  )
  names(out$diagnostics$predictor_mean) <- predictor_names
  names(out$diagnostics$predictor_variance) <- predictor_names
  names(out$diagnostics$predictor_nonnegative) <- predictor_names

  class(out) <- c("ecountgmifs", class(out))
  out
}
