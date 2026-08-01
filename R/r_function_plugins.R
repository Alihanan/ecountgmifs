.r.scalar.string <- function(x, name) {
  if (!is.character(x) || length(x) != 1L || is.na(x) || !nzchar(x)) {
    stop("`", name, "` must be one non-empty character value.", call. = FALSE)
  }

  x
}

.r.callback <- function(x, name) {
  if (!is.function(x)) {
    stop("`", name, "` must be a function.", call. = FALSE)
  }

  x
}

.r.prepare.callback <- function(x) {
  if (is.null(x)) {
    return(function(input, control, environment) invisible(NULL))
  }

  .r.callback(x, "prepare")
}

.r.as.environment <- function(x) {
  if (is.null(x)) {
    return(new.env(parent = emptyenv()))
  }

  if (is.environment(x)) {
    return(x)
  }

  if (is.list(x)) {
    return(list2env(x, envir = new.env(parent = emptyenv())))
  }

  stop(
    "`environment` must be NULL, an environment, or a named list.",
    call. = FALSE
  )
}

.r.parameter.metadata <- function(
    initial.parameters,
    lower.bounds,
    upper.bounds,
    prefix
) {
  initial.parameters <- as.numeric(initial.parameters)

  if (is.null(lower.bounds)) {
    lower.bounds <- rep(-Inf, length(initial.parameters))
  }

  if (is.null(upper.bounds)) {
    upper.bounds <- rep(Inf, length(initial.parameters))
  }

  lower.bounds <- as.numeric(lower.bounds)
  upper.bounds <- as.numeric(upper.bounds)

  if (
      length(initial.parameters) != length(lower.bounds) ||
      length(initial.parameters) != length(upper.bounds)
  ) {
    stop(
      "Initial values and bounds for ", prefix,
      " parameters must have equal lengths.",
      call. = FALSE
    )
  }

  if (anyNA(initial.parameters) || any(!is.finite(initial.parameters))) {
    stop("Initial ", prefix, " parameters must be finite.", call. = FALSE)
  }

  if (anyNA(lower.bounds) || anyNA(upper.bounds)) {
    stop(prefix, " parameter bounds must not contain NA or NaN.", call. = FALSE)
  }

  if (any(lower.bounds > upper.bounds)) {
    stop(prefix, " lower bounds must not exceed upper bounds.", call. = FALSE)
  }

  if (any(initial.parameters < lower.bounds | initial.parameters > upper.bounds)) {
    stop(
      "Initial ", prefix, " parameters must lie within their bounds.",
      call. = FALSE
    )
  }

  list(
    initial = initial.parameters,
    lower = lower.bounds,
    upper = upper.bounds
  )
}

.r.decorate.plugin <- function(pointer, type, name, environment, callbacks) {
  attr(pointer, "r.plugin.type") <- type
  attr(pointer, "r.plugin.name") <- name
  attr(pointer, "r.plugin.environment") <- environment
  attr(pointer, "r.plugin.callbacks") <- callbacks
  class(pointer) <- unique(c(paste0("r.", type), "r.plugin", class(pointer)))
  pointer
}

#' Create an auxiliary environment for R callback plugins
#'
#' @description
#' Creates an environment that can be retained by an R-backed link, family,
#' family-link, or criterion. It is useful for fixed configuration, large
#' auxiliary objects, shared data, counters, and caches. A callback receives
#' this exact environment as its final argument on every call.
#'
#' A closure is an equally valid alternative when auxiliary values are fixed;
#' see [custom.plugins] and the paired examples documented with the plugin
#' constructors.
#'
#' @param ... Named objects to place in the environment.
#' @param .parent Parent environment. The default is `emptyenv()` because this
#'   object is intended as explicit data storage, not as the callback's lexical
#'   evaluation environment.
#'
#' @return An environment.
#' @export
#' @family R callback plugins
r.environment <- function(..., .parent = emptyenv()) {
  if (!is.environment(.parent)) {
    stop("`.parent` must be an environment.", call. = FALSE)
  }

  values <- list(...)

  if (length(values) > 0L && (is.null(names(values)) || any(!nzchar(names(values))))) {
    stop("All values supplied through `...` must be named.", call. = FALSE)
  }

  list2env(values, envir = new.env(parent = .parent))
}

#' Inspect the environment retained by an R callback plugin
#'
#' @param plugin An object returned by [r.link()], [r.family()],
#'   [r.family.link()], or [r.criterion()].
#'
#' @return The retained environment.
#' @export
#' @family R callback plugins
r.plugin.environment <- function(plugin) {
  environment <- attr(plugin, "r.plugin.environment", exact = TRUE)

  if (!is.environment(environment)) {
    stop("`plugin` is not an R callback plugin.", call. = FALSE)
  }

  environment
}

#' Construct an R-backed link
#'
#' @description
#' Wraps R functions in the built-in C++ child of `IEcountgmifsLinkFunc`.
#' The returned object is a normal external pointer and can be supplied anywhere
#' a native link external pointer is accepted.
#'
#' The optional preparation callback has signature
#' `prepare(input, control, environment)` and is invoked once before fitting.
#' The inverse callback has signature
#' `function(eta, link.parameters, environment)` and returns `mu`.
#' The gradient callback has the same arguments and returns a list containing
#' `d_mu_d_eta` and `d_mu_d_link_parameters`.
#'
#' @param name Non-empty link name.
#' @param prepare Optional one-time preparation callback.
#' @param inverse R inverse-link callback.
#' @param grad R gradient callback.
#' @param initial.parameters Numeric vector of initial link parameters.
#' @param lower.bounds,upper.bounds Numeric parameter bounds. `NULL` creates
#'   infinite bounds.
#' @param environment `NULL`, an environment, or a named list retained by the
#'   C++ adapter and passed to every callback. Fixed values may instead be
#'   captured lexically by closures.
#'
#' @return An external pointer to an `IEcountgmifsLinkFunc` implementation.
#' @export
#' @family R callback plugins
#' @seealso [custom.plugins], [r.environment()], [r.family()], [r.family.link()]
r.link <- function(
    name,
    inverse,
    grad,
    initial.parameters = numeric(),
    lower.bounds = NULL,
    upper.bounds = NULL,
    environment = NULL,
    prepare = NULL
) {
  name <- .r.scalar.string(name, "name")
  prepare <- .r.prepare.callback(prepare)
  inverse <- .r.callback(inverse, "inverse")
  grad <- .r.callback(grad, "grad")
  metadata <- .r.parameter.metadata(
    initial.parameters,
    lower.bounds,
    upper.bounds,
    "link"
  )
  environment <- .r.as.environment(environment)

  pointer <- create_r_link(
    name = name,
    prepare = prepare,
    inverse = inverse,
    grad = grad,
    initial_parameters = metadata$initial,
    lower_bounds = metadata$lower,
    upper_bounds = metadata$upper,
    environment = environment
  )

  .r.decorate.plugin(
    pointer,
    type = "link",
    name = name,
    environment = environment,
    callbacks = list(prepare = prepare, inverse = inverse, grad = grad)
  )
}

#' Construct an R-backed family
#'
#' @description
#' Wraps R functions in the built-in C++ child of `IEcountgmifsFamily`.
#' `prepare(input, control, environment)` is invoked once before fitting.
#' `negloglik(y, mu, family.parameters, environment)` returns one negative
#' log-likelihood value. `grad()` returns a list containing
#' `d_negloglik_d_mu` and `d_negloglik_d_family_parameters`.
#'
#' @param name Non-empty family name.
#' @param prepare Optional one-time preparation callback.
#' @param negloglik R negative-log-likelihood callback.
#' @param grad R gradient callback.
#' @param initial.parameters Numeric vector of initial family parameters.
#' @param lower.bounds,upper.bounds Numeric parameter bounds. `NULL` creates
#'   infinite bounds.
#' @param environment `NULL`, an environment, or a named list retained by the
#'   adapter and passed to every callback.
#'
#' @return An external pointer to an `IEcountgmifsFamily` implementation.
#' @export
#' @family R callback plugins
#' @seealso [custom.plugins], [r.link()], [r.family.link()]
r.family <- function(
    name,
    negloglik,
    grad,
    initial.parameters = numeric(),
    lower.bounds = NULL,
    upper.bounds = NULL,
    environment = NULL,
    prepare = NULL
) {
  name <- .r.scalar.string(name, "name")
  prepare <- .r.prepare.callback(prepare)
  negloglik <- .r.callback(negloglik, "negloglik")
  grad <- .r.callback(grad, "grad")
  metadata <- .r.parameter.metadata(
    initial.parameters,
    lower.bounds,
    upper.bounds,
    "family"
  )
  environment <- .r.as.environment(environment)

  pointer <- create_r_family(
    name = name,
    prepare = prepare,
    negloglik = negloglik,
    grad = grad,
    initial_parameters = metadata$initial,
    lower_bounds = metadata$lower,
    upper_bounds = metadata$upper,
    environment = environment
  )

  .r.decorate.plugin(
    pointer,
    type = "family",
    name = name,
    environment = environment,
    callbacks = list(prepare = prepare, negloglik = negloglik, grad = grad)
  )
}

#' Construct an R-backed fused family-link
#'
#' @description
#' Wraps three R callbacks in the built-in C++ child of
#' `IEcountgmifsFamilyLink`. A fused implementation may provide
#' `d_negloglik_d_eta` directly and can therefore avoid the ordinary separated
#' family/link chain rule.
#'
#' `prepare(input, control, environment)` is invoked once before fitting.
#' `inverse(eta, link.parameters, environment)` returns `mu`.
#' `negloglik(y, mu, family.parameters, environment)` returns one value.
#' `grad(y, eta, mu, family.parameters, link.parameters, environment)` returns
#' all six derivative objects required by `IEcountgmifsFamilyLink`.
#'
#' @param family.name,link.name Non-empty names.
#' @param prepare Optional one-time preparation callback.
#' @param inverse,negloglik,grad R callbacks.
#' @param family.initial.parameters,link.initial.parameters Initial parameters.
#' @param family.lower.bounds,family.upper.bounds Family-parameter bounds.
#' @param link.lower.bounds,link.upper.bounds Link-parameter bounds.
#' @param environment Retained explicit auxiliary environment or named list.
#'
#' @return An external pointer to an `IEcountgmifsFamilyLink` implementation.
#' @export
#' @family R callback plugins
#' @seealso [custom.plugins], [r.family()], [r.link()]
r.family.link <- function(
    family.name,
    link.name,
    inverse,
    negloglik,
    grad,
    family.initial.parameters = numeric(),
    family.lower.bounds = NULL,
    family.upper.bounds = NULL,
    link.initial.parameters = numeric(),
    link.lower.bounds = NULL,
    link.upper.bounds = NULL,
    environment = NULL,
    prepare = NULL
) {
  family.name <- .r.scalar.string(family.name, "family.name")
  link.name <- .r.scalar.string(link.name, "link.name")
  prepare <- .r.prepare.callback(prepare)
  inverse <- .r.callback(inverse, "inverse")
  negloglik <- .r.callback(negloglik, "negloglik")
  grad <- .r.callback(grad, "grad")

  family.metadata <- .r.parameter.metadata(
    family.initial.parameters,
    family.lower.bounds,
    family.upper.bounds,
    "family"
  )
  link.metadata <- .r.parameter.metadata(
    link.initial.parameters,
    link.lower.bounds,
    link.upper.bounds,
    "link"
  )
  environment <- .r.as.environment(environment)

  pointer <- create_r_family_link(
    family_name = family.name,
    link_name = link.name,
    prepare = prepare,
    inverse = inverse,
    negloglik = negloglik,
    grad = grad,
    family_initial_parameters = family.metadata$initial,
    family_lower_bounds = family.metadata$lower,
    family_upper_bounds = family.metadata$upper,
    link_initial_parameters = link.metadata$initial,
    link_lower_bounds = link.metadata$lower,
    link_upper_bounds = link.metadata$upper,
    environment = environment
  )

  .r.decorate.plugin(
    pointer,
    type = "family.link",
    name = paste(family.name, link.name, sep = "/"),
    environment = environment,
    callbacks = list(
      prepare = prepare,
      inverse = inverse,
      negloglik = negloglik,
      grad = grad
    )
  )
}

#' Construct an R-backed model-selection criterion
#'
#' @description
#' Wraps `evaluate(input, control, state, environment)` in the built-in C++
#' child of `IEcountgmifsCriterion`. An optional
#' `prepare(input, control, environment)` callback is invoked once before
#' fitting. The first three evaluation arguments are named R lists
#' representing simplified read-only views of the corresponding public C++ API
#' objects. The environment is the same object retained at construction.
#'
#' @section Criterion callback views:
#' `input` contains `X`, `y`, `w`, `offset`, `weight_vec`, `n`, `p`, `q`,
#' `has_prior`, `enet_alpha`, `family_name`, `link_name`,
#' `family_parameter_count`, and `link_parameter_count`.
#'
#' `control` contains `null_iteration_max`, `stagewise_iteration_max`,
#' `null_family_parameter_abs_tol`, `stagewise_objective_rel_tol`,
#' `stagewise_beta_step_norm_tol`, `epsilon_max`, `epsilon_start`,
#' `epsilon_min`, `loglik_reltol_cutoff`, `enet_abs_tol`, `enet_rel_tol`,
#' `enet_max_iter`, `state_track_strategy`, `state_track_freq`, `verbose`,
#' `include_data`, `theta_initial`, `theta_lower_bounds`, and
#' `theta_upper_bounds`. It also contains `nonpen_nlopt`, `family_nlopt`, and
#' `link_nlopt`; each nested list has `algorithm`, `xtol_rel`, `ftol_rel`, and
#' `maxeval`.
#'
#' `state` contains `beta`, `theta`, `family_parameters`, `link_parameters`,
#' `xbeta`, `wtheta`, `eta`, `mu`, `active_set`, `negloglik`, `criteria`,
#' `iteration`, `pseudo_r2`, and `elapsed_time`.
#'
#' @param name Non-empty criterion name.
#' @param prepare Optional one-time preparation callback.
#' @param evaluate R criterion callback returning one numeric value.
#' @param environment Retained explicit auxiliary environment or named list.
#'
#' @return An external pointer to an `IEcountgmifsCriterion` implementation.
#' @export
#' @family R callback plugins
#' @seealso [custom.plugins], [information.criteria]
r.criterion <- function(name, evaluate, environment = NULL, prepare = NULL) {
  name <- .r.scalar.string(name, "name")
  prepare <- .r.prepare.callback(prepare)
  evaluate <- .r.callback(evaluate, "evaluate")
  environment <- .r.as.environment(environment)

  pointer <- create_r_criterion(
    name = name,
    prepare = prepare,
    evaluate = evaluate,
    environment = environment
  )

  .r.decorate.plugin(
    pointer,
    type = "criterion",
    name = name,
    environment = environment,
    callbacks = list(prepare = prepare, evaluate = evaluate)
  )
}

#' @export
print.r.plugin <- function(x, ...) {
  cat(
    "<R-backed ",
    attr(x, "r.plugin.type", exact = TRUE),
    ": ",
    attr(x, "r.plugin.name", exact = TRUE),
    ">\n",
    sep = ""
  )
  invisible(x)
}
