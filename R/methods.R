.ecountgmifs.path.states <- function(object, required = TRUE) {
  states <- NULL

  if (!is.null(object$path$states)) {
    states <- object$path$states
  } else if (!is.null(object$states)) {
    # Compatibility with early development objects.
    states <- object$states
  }

  if (
      isTRUE(required) &&
      (
        is.null(states) ||
          is.null(states$beta) ||
          length(states$beta) == 0L
      )
  ) {
    stop(
      paste0(
        "No saved coefficient path is available. Fit with a state-tracking ",
        "strategy other than 'none'."
      ),
      call. = FALSE
    )
  }

  states
}


.ecountgmifs.terminal.state <- function(object) {
  if (!is.null(object$terminal_state)) {
    return(object$terminal_state)
  }

  states <- .ecountgmifs.path.states(object)
  index <- length(states$beta)

  list(
    predictors = list(
      parameters = list(
        beta = states$beta[[index]],
        theta = states$theta[[index]],
        family_parameters = states$family_parameters[[index]],
        link_parameters = states$link_parameters[[index]]
      ),
      xbeta = states$xbeta[[index]],
      wtheta = states$wtheta[[index]],
      eta = states$eta[[index]],
      mu = states$mu[[index]],
      active_set = states$active_set[[index]]
    ),
    negloglik = states$negloglik[[index]],
    criteria = vapply(
      states$criteria,
      function(value) value[[index]],
      numeric(1L)
    ),
    iteration = states$iteration[[index]],
    pseudo_r2 = states$pseudo_r2[[index]],
    elapsed_time = states$elapsed_time[[index]]
  )
}


.ecountgmifs.plugin.label <- function(value, fallback = "Unknown") {
  if (is.null(value)) {
    return(fallback)
  }

  if (is.list(value) && length(value) > 0L) {
    value_names <- names(value)

    if (
        !is.null(value_names) &&
        length(value_names) > 0L &&
        nzchar(value_names[[1L]])
    ) {
      return(value_names[[1L]])
    }
  }

  if (
      is.character(value) &&
      length(value) >= 1L &&
      !is.na(value[[1L]])
  ) {
    return(value[[1L]])
  }

  # Compatibility with the original enum-based result format.
  if (is.numeric(value) && length(value) == 1L && !is.na(value)) {
    return(as.character(value))
  }

  fallback
}


.ecountgmifs.input.view <- function(object) {
  if (
      is.list(object) &&
      !is.null(object$input) &&
      is.list(object$input)
  ) {
    return(object$input)
  }

  object
}


.ecountgmifs.family.label <- function(object) {
  input <- .ecountgmifs.input.view(object)

  value <- if (is.list(input) && !is.null(input$family)) {
    input$family
  } else {
    input
  }

  # Compatibility with the original enum-based output.
  if (is.numeric(value) && length(value) == 1L && !is.na(value)) {
    return(
      switch(
        as.character(as.integer(value)),
        "0" = "NEGATIVE_BINOMIAL",
        "1" = "POISSON",
        as.character(value)
      )
    )
  }

  .ecountgmifs.plugin.label(
    value,
    fallback = "Unknown family"
  )
}


.ecountgmifs.link.label <- function(object) {
  input <- .ecountgmifs.input.view(object)

  value <- if (is.list(input) && !is.null(input$link_func)) {
    input$link_func
  } else {
    input
  }

  # Compatibility with the original enum-based output.
  if (is.numeric(value) && length(value) == 1L && !is.na(value)) {
    return(
      switch(
        as.character(as.integer(value)),
        "0" = "LOG_LINK",
        "1" = "SOFTPLUS_LINK",
        as.character(value)
      )
    )
  }

  .ecountgmifs.plugin.label(
    value,
    fallback = "Unknown link"
  )
}


.ecountgmifs.interface.label <- function(object) {
  input <- .ecountgmifs.input.view(object)

  if (
      is.list(input) &&
      !is.null(input$family_link_supplied)
  ) {
    return(
      if (isTRUE(input$family_link_supplied)) {
        "combined (fused family-link)"
      } else {
        "separate family and link"
      }
    )
  }

  "unknown"
}


.ecountgmifs.predictor.names <- function(object, p = NULL) {
  if (is.null(p)) {
    p <- object$input$p
  }

  candidate <- object$input$predictor_names

  if (is.null(candidate) && !is.null(object$input$X)) {
    candidate <- colnames(object$input$X)
  }

  if (is.null(candidate)) {
    terminal <- tryCatch(
      .ecountgmifs.terminal.state(object),
      error = function(error) NULL
    )

    if (!is.null(terminal)) {
      candidate <- names(
        terminal$predictors$parameters$beta
      )
    }
  }

  if (
      is.null(candidate) ||
      length(candidate) != p ||
      anyNA(candidate) ||
      any(!nzchar(candidate))
  ) {
    candidate <- paste0("X", seq_len(p))
  }

  make.unique(as.character(candidate))
}


.ecountgmifs.unpenalized.names <- function(object, q = NULL) {
  if (is.null(q)) {
    q <- object$input$q
  }

  candidate <- object$input$unpenalized_names

  if (is.null(candidate) && !is.null(object$input$w)) {
    candidate <- colnames(object$input$w)
  }

  if (is.null(candidate)) {
    terminal <- tryCatch(
      .ecountgmifs.terminal.state(object),
      error = function(error) NULL
    )

    if (!is.null(terminal)) {
      candidate <- names(
        terminal$predictors$parameters$theta
      )
    }
  }

  if (
      is.null(candidate) ||
      length(candidate) != q ||
      anyNA(candidate) ||
      any(!nzchar(candidate))
  ) {
    candidate <- paste0("theta", seq_len(q))
  }

  make.unique(as.character(candidate))
}



.ecountgmifs.name.vector <- function(value, labels) {
  if (
      !is.null(value) &&
      length(value) == length(labels)
  ) {
    names(value) <- labels
  }

  value
}


.ecountgmifs.name.state <- function(
    state,
    predictor.names,
    unpenalized.names
) {
  if (
      is.null(state) ||
      is.null(state$predictors) ||
      is.null(state$predictors$parameters)
  ) {
    return(state)
  }

  state$predictors$parameters$beta <-
    .ecountgmifs.name.vector(
      state$predictors$parameters$beta,
      predictor.names
    )

  state$predictors$parameters$theta <-
    .ecountgmifs.name.vector(
      state$predictors$parameters$theta,
      unpenalized.names
    )

  if (!is.null(state$predictors$active_set)) {
    state$predictors$active_set <-
      .ecountgmifs.name.vector(
        state$predictors$active_set,
        predictor.names
      )
  }

  state
}


.ecountgmifs.assign.parameter.names <- function(
    object,
    predictor.names = NULL,
    unpenalized.names = NULL
) {
  if (!is.list(object) || is.null(object$input)) {
    stop(
      "`object` is not in the expected ecountgmifs result format.",
      call. = FALSE
    )
  }

  terminal <- .ecountgmifs.terminal.state(object)
  terminal_parameters <- terminal$predictors$parameters

  p <- if (!is.null(object$input$p)) {
    as.integer(object$input$p)
  } else {
    length(terminal_parameters$beta)
  }

  q <- if (!is.null(object$input$q)) {
    as.integer(object$input$q)
  } else {
    length(terminal_parameters$theta)
  }

  if (is.null(predictor.names)) {
    predictor.names <- object$input$predictor_names
  }

  if (
      is.null(predictor.names) &&
      !is.null(object$input$X)
  ) {
    predictor.names <- colnames(object$input$X)
  }

  if (is.null(predictor.names)) {
    predictor.names <- names(terminal_parameters$beta)
  }

  if (
      is.null(predictor.names) ||
      length(predictor.names) != p ||
      anyNA(predictor.names) ||
      any(!nzchar(as.character(predictor.names)))
  ) {
    predictor.names <- paste0("X", seq_len(p))
  }

  predictor.names <- make.unique(
    as.character(predictor.names)
  )

  if (is.null(unpenalized.names)) {
    unpenalized.names <- object$input$unpenalized_names
  }

  if (
      is.null(unpenalized.names) &&
      !is.null(object$input$w)
  ) {
    unpenalized.names <- colnames(object$input$w)
  }

  if (is.null(unpenalized.names)) {
    unpenalized.names <- names(terminal_parameters$theta)
  }

  if (
      is.null(unpenalized.names) ||
      length(unpenalized.names) != q ||
      anyNA(unpenalized.names) ||
      any(!nzchar(as.character(unpenalized.names)))
  ) {
    unpenalized.names <- paste0("theta", seq_len(q))
  }

  unpenalized.names <- make.unique(
    as.character(unpenalized.names)
  )

  object$input$predictor_names <- predictor.names
  object$input$unpenalized_names <- unpenalized.names

  if (
      !is.null(object$input$X) &&
      is.matrix(object$input$X) &&
      ncol(object$input$X) == length(predictor.names)
  ) {
    colnames(object$input$X) <- predictor.names
  }

  if (
      !is.null(object$input$w) &&
      is.matrix(object$input$w) &&
      ncol(object$input$w) == length(unpenalized.names)
  ) {
    colnames(object$input$w) <- unpenalized.names
  }

  if (!is.null(object$input$weight_vec)) {
    object$input$weight_vec <-
      .ecountgmifs.name.vector(
        object$input$weight_vec,
        predictor.names
      )
  }

  if (!is.null(object$terminal_state)) {
    object$terminal_state <- .ecountgmifs.name.state(
      object$terminal_state,
      predictor.names,
      unpenalized.names
    )
  }

  if (!is.null(object$path$null_theta)) {
    object$path$null_theta <-
      .ecountgmifs.name.vector(
        object$path$null_theta,
        unpenalized.names
      )
  }

  if (!is.null(object$path$states)) {
    states <- object$path$states

    if (!is.null(states$beta)) {
      states$beta <- lapply(
        states$beta,
        .ecountgmifs.name.vector,
        labels = predictor.names
      )
    }

    if (!is.null(states$theta)) {
      states$theta <- lapply(
        states$theta,
        .ecountgmifs.name.vector,
        labels = unpenalized.names
      )
    }

    if (!is.null(states$active_set)) {
      states$active_set <- lapply(
        states$active_set,
        .ecountgmifs.name.vector,
        labels = predictor.names
      )
    }

    object$path$states <- states
  }

  if (!is.null(object$states)) {
    if (!is.null(object$states$beta)) {
      object$states$beta <- lapply(
        object$states$beta,
        .ecountgmifs.name.vector,
        labels = predictor.names
      )
    }

    if (!is.null(object$states$theta)) {
      object$states$theta <- lapply(
        object$states$theta,
        .ecountgmifs.name.vector,
        labels = unpenalized.names
      )
    }

    if (!is.null(object$states$active_set)) {
      object$states$active_set <- lapply(
        object$states$active_set,
        .ecountgmifs.name.vector,
        labels = predictor.names
      )
    }
  }

  if (!is.null(object$path$best_criteria)) {
    for (criterion_name in names(object$path$best_criteria)) {
      object$path$best_criteria[[criterion_name]]$state <-
        .ecountgmifs.name.state(
          object$path$best_criteria[[criterion_name]]$state,
          predictor.names,
          unpenalized.names
        )
    }
  }

  if (!is.null(object$path$last_saved_active_set)) {
    object$path$last_saved_active_set <-
      .ecountgmifs.name.vector(
        object$path$last_saved_active_set,
        predictor.names
      )
  }

  if (!is.null(object$stagewise)) {
    for (
        field in c(
          "beta_start",
          "beta_trial",
          "delta_beta"
        )
    ) {
      if (!is.null(object$stagewise[[field]])) {
        object$stagewise[[field]] <-
          .ecountgmifs.name.vector(
            object$stagewise[[field]],
            predictor.names
          )
      }
    }
  }

  object
}


.ecountgmifs.beta.matrix <- function(object) {
  states <- .ecountgmifs.path.states(object)

  beta <- do.call(
    rbind,
    lapply(
      states$beta,
      function(value) as.numeric(value)
    )
  )

  if (is.null(dim(beta))) {
    beta <- matrix(beta, nrow = 1L)
  }

  colnames(beta) <- .ecountgmifs.predictor.names(
    object,
    ncol(beta)
  )

  if (!is.null(states$iteration)) {
    rownames(beta) <- paste0(
      "iter_",
      as.integer(states$iteration)
    )
  }

  beta
}


.ecountgmifs.theta.matrix <- function(object) {
  states <- .ecountgmifs.path.states(object)

  theta <- do.call(
    rbind,
    lapply(
      states$theta,
      function(value) as.numeric(value)
    )
  )

  if (is.null(dim(theta))) {
    theta <- matrix(theta, nrow = 1L)
  }

  colnames(theta) <- .ecountgmifs.unpenalized.names(
    object,
    ncol(theta)
  )

  theta
}


.ecountgmifs.criteria.path <- function(object) {
  states <- .ecountgmifs.path.states(object)

  criteria <- states$criteria

  if (is.null(criteria) || length(criteria) == 0L) {
    return(
      matrix(
        numeric(),
        nrow = length(states$iteration),
        ncol = 0L
      )
    )
  }

  result <- do.call(
    cbind,
    lapply(criteria, as.numeric)
  )

  if (is.null(dim(result))) {
    result <- matrix(result, ncol = 1L)
  }

  colnames(result) <- names(criteria)
  rownames(result) <- paste0(
    "iter_",
    as.integer(states$iteration)
  )

  result
}


.ecountgmifs.best.criteria <- function(object) {
  best <- object$path$best_criteria

  if (is.null(best)) {
    best <- list()
  }

  best
}


.ecountgmifs.match.criteria <- function(object, criteria = NULL) {
  available <- names(.ecountgmifs.best.criteria(object))

  if (is.null(criteria)) {
    return(available)
  }

  criteria <- as.character(criteria)
  unknown <- setdiff(criteria, available)

  if (length(unknown) > 0L) {
    stop(
      "Unknown criterion name(s): ",
      paste(unknown, collapse = ", "),
      ". Available criteria: ",
      paste(available, collapse = ", "),
      call. = FALSE
    )
  }

  criteria
}


.ecountgmifs.state.parameters <- function(state) {
  parameters <- state$predictors$parameters

  list(
    beta = as.numeric(parameters$beta),
    theta = as.numeric(parameters$theta),
    family_parameters = as.numeric(parameters$family_parameters),
    link_parameters = as.numeric(parameters$link_parameters)
  )
}


.ecountgmifs.resolve.state <- function(
    object,
    criterion = NULL,
    iteration = NULL
) {
  if (!is.null(criterion) && !is.null(iteration)) {
    stop(
      "Supply either `criterion` or `iteration`, not both.",
      call. = FALSE
    )
  }

  if (!is.null(criterion)) {
    criterion <- .ecountgmifs.match.criteria(
      object,
      criterion
    )

    if (length(criterion) != 1L) {
      stop(
        "`criterion` must identify exactly one criterion.",
        call. = FALSE
      )
    }

    return(
      list(
        state = object$path$best_criteria[[criterion]]$state,
        label = paste0("criterion ", criterion),
        criterion = criterion
      )
    )
  }

  if (!is.null(iteration)) {
    if (
        !is.numeric(iteration) ||
        length(iteration) != 1L ||
        is.na(iteration) ||
        !is.finite(iteration)
    ) {
      stop(
        "`iteration` must be one finite numeric value.",
        call. = FALSE
      )
    }

    states <- .ecountgmifs.path.states(object)
    iteration_values <- as.numeric(states$iteration)
    index <- which.min(abs(iteration_values - iteration))

    state <- list(
      predictors = list(
        parameters = list(
          beta = states$beta[[index]],
          theta = states$theta[[index]],
          family_parameters = states$family_parameters[[index]],
          link_parameters = states$link_parameters[[index]]
        ),
        xbeta = states$xbeta[[index]],
        wtheta = states$wtheta[[index]],
        eta = states$eta[[index]],
        mu = states$mu[[index]],
        active_set = states$active_set[[index]]
      ),
      negloglik = states$negloglik[[index]],
      criteria = vapply(
        states$criteria,
        function(value) value[[index]],
        numeric(1L)
      ),
      iteration = states$iteration[[index]],
      pseudo_r2 = states$pseudo_r2[[index]],
      elapsed_time = states$elapsed_time[[index]]
    )

    return(
      list(
        state = state,
        label = paste0(
          "saved iteration ",
          as.integer(states$iteration[[index]])
        ),
        criterion = NULL
      )
    )
  }

  list(
    state = .ecountgmifs.terminal.state(object),
    label = "terminal state",
    criterion = NULL
  )
}


.ecountgmifs.criterion.table <- function(object) {
  best <- .ecountgmifs.best.criteria(object)

  if (length(best) == 0L) {
    return(
      data.frame(
        criterion = character(),
        value = numeric(),
        iteration = integer(),
        nonzero = integer(),
        negloglik = numeric(),
        pseudo_r2 = numeric(),
        stringsAsFactors = FALSE
      )
    )
  }

  do.call(
    rbind,
    lapply(
      names(best),
      function(name) {
        selected <- best[[name]]
        beta <- selected$state$predictors$parameters$beta

        data.frame(
          criterion = name,
          value = as.numeric(selected$value),
          iteration = as.integer(selected$state$iteration),
          nonzero = sum(beta != 0),
          negloglik = as.numeric(selected$state$negloglik),
          pseudo_r2 = as.numeric(selected$state$pseudo_r2),
          stringsAsFactors = FALSE
        )
      }
    )
  )
}


#' Extract an ecountgmifs coefficient path or selected coefficient vector
#'
#' Without `criterion` or `iteration`, this returns the saved coefficient path
#' with saved states in rows and penalized predictors in columns. Supplying one
#' selector returns the coefficient vector at the corresponding state.
#'
#' @param object An `ecountgmifs` fit.
#' @param criterion Optional criterion name selecting its best stored state.
#' @param iteration Optional iteration; the nearest saved state is used.
#' @param include.theta Logical. Append unpenalized coefficients when a single
#'   state is selected.
#' @param ... Unused.
#'
#' @return A numeric matrix for the full path or a named numeric vector for one
#'   selected state.
#' @export
coef.ecountgmifs <- function(
    object,
    criterion = NULL,
    iteration = NULL,
    include.theta = FALSE,
    ...
) {
  if (is.null(criterion) && is.null(iteration)) {
    return(.ecountgmifs.beta.matrix(object))
  }

  selected <- .ecountgmifs.resolve.state(
    object,
    criterion = criterion,
    iteration = iteration
  )

  parameters <- .ecountgmifs.state.parameters(
    selected$state
  )

  names(parameters$beta) <- .ecountgmifs.predictor.names(
    object,
    length(parameters$beta)
  )

  if (!isTRUE(include.theta)) {
    return(parameters$beta)
  }

  names(parameters$theta) <- .ecountgmifs.unpenalized.names(
    object,
    length(parameters$theta)
  )

  c(
    parameters$theta,
    parameters$beta
  )
}


#' Print an ecountgmifs fit
#'
#' @param x An `ecountgmifs` fit.
#' @param digits Number of significant digits.
#' @param max.criteria Maximum number of criterion selections to print.
#' @param ... Unused.
#'
#' @export
print.ecountgmifs <- function(
    x,
    digits = max(3L, getOption("digits") - 3L),
    max.criteria = 20L,
    ...
) {
  terminal <- .ecountgmifs.terminal.state(x)
  parameters <- .ecountgmifs.state.parameters(terminal)
  criterion_table <- .ecountgmifs.criterion.table(x)

  cat("ecountgmifs path fit\n")

  if (!is.null(x$call)) {
    cat("\nCall:\n")
    print(x$call)
  }

  cat("\n")
  cat("Family:            ", .ecountgmifs.family.label(x), "\n", sep = "")
  cat("Link:              ", .ecountgmifs.link.label(x), "\n", sep = "")
  cat("Family/link mode:   ", .ecountgmifs.interface.label(x), "\n", sep = "")
  cat("Observations:      ", x$input$n, "\n", sep = "")
  cat("Penalized terms:   ", x$input$p, "\n", sep = "")
  cat("Unpenalized terms: ", x$input$q, "\n", sep = "")
  cat("Elastic-net alpha: ", format(x$input$enet_alpha, digits = digits), "\n", sep = "")
  cat("Prior weights:     ", if (isTRUE(x$input$has_prior)) "yes" else "no", "\n", sep = "")

  if (
      !is.null(x$path$null_negloglik) ||
      !is.null(x$path$saturated_negloglik)
  ) {
    cat("\nReference models:\n")

    if (!is.null(x$path$null_negloglik)) {
      cat(
        "  Null negative log-likelihood:      ",
        format(x$path$null_negloglik, digits = digits),
        "\n",
        sep = ""
      )
    }

    if (!is.null(x$path$saturated_negloglik)) {
      cat(
        "  Saturated negative log-likelihood: ",
        format(x$path$saturated_negloglik, digits = digits),
        "\n",
        sep = ""
      )
    }
  }

  cat("\n")
  cat("Terminal iteration: ", terminal$iteration, "\n", sep = "")
  cat("Nonzero beta:       ", sum(abs(parameters$beta) > 0), "\n", sep = "")
  cat("Negative log-likelihood: ", format(terminal$negloglik, digits = digits), "\n", sep = "")
  cat("Pseudo-R2:           ", format(terminal$pseudo_r2, digits = digits), "\n", sep = "")

  if (!is.null(terminal$criteria) && length(terminal$criteria) > 0L) {
    cat("\nTerminal criterion values:\n")
    print(terminal$criteria, digits = digits)
  }

  if (!is.null(x$path$total_time)) {
    cat("Total time (sec):    ", format(x$path$total_time, digits = digits), "\n", sep = "")
  }

  if (!is.null(x$path$message)) {
    cat("Termination:         ", x$path$message, "\n", sep = "")
  }

  if (nrow(criterion_table) > 0L) {
    cat("\nCriterion-selected states:\n")

    shown <- utils::head(
      criterion_table,
      max.criteria
    )

    print(
      shown,
      row.names = FALSE,
      digits = digits
    )

    if (nrow(criterion_table) > nrow(shown)) {
      cat(
        "... ",
        nrow(criterion_table) - nrow(shown),
        " additional criteria not shown\n",
        sep = ""
      )
    }
  }

  invisible(x)
}


#' Summarize an ecountgmifs fit
#'
#' @param object An `ecountgmifs` fit.
#' @param criterion Optional criterion selecting the state summarized.
#' @param iteration Optional iteration selecting the nearest saved state.
#' @param ground.truth Optional logical or binary vector used to calculate
#'   selection metrics.
#' @param zero.tol Coefficients with absolute value no larger than this are
#'   treated as zero.
#' @param max.coef Maximum number of nonzero coefficients printed later.
#' @param prior.selected Optional logical prior-selected vector. See
#'   [ecountgmifs.metrics()].
#' @param prior.cutoff Cutoff used with `weight_vec` when `prior.selected` is
#'   omitted.
#' @param prior.direction Whether smaller or larger weights indicate prior
#'   selection.
#' @param ... Additional arguments passed to [ecountgmifs.metrics()].
#'
#' @return An object of class `summary.ecountgmifs`.
#' @export
summary.ecountgmifs <- function(
    object,
    criterion = NULL,
    iteration = NULL,
    ground.truth = NULL,
    zero.tol = sqrt(.Machine$double.eps),
    max.coef = 20L,
    prior.selected = NULL,
    prior.cutoff = 1,
    prior.direction = c("lower", "higher"),
    ...
) {
  selected <- .ecountgmifs.resolve.state(
    object,
    criterion = criterion,
    iteration = iteration
  )

  state <- selected$state
  parameters <- .ecountgmifs.state.parameters(state)

  predictor_names <- .ecountgmifs.predictor.names(
    object,
    length(parameters$beta)
  )

  names(parameters$beta) <- predictor_names
  names(parameters$theta) <- .ecountgmifs.unpenalized.names(
    object,
    length(parameters$theta)
  )

  nonzero <- abs(parameters$beta) > zero.tol

  coefficient_table <- data.frame(
    term = predictor_names[nonzero],
    estimate = parameters$beta[nonzero],
    absolute_estimate = abs(parameters$beta[nonzero]),
    stringsAsFactors = FALSE
  )

  if (nrow(coefficient_table) > 0L) {
    coefficient_table <- coefficient_table[
      order(
        coefficient_table$absolute_estimate,
        decreasing = TRUE
      ),
      ,
      drop = FALSE
    ]
  }

  metric_summary <- NULL

  if (!is.null(ground.truth)) {
    prior.direction <- match.arg(prior.direction)

    metric_object <- ecountgmifs.metrics(
      object = object,
      ground.truth = ground.truth,
      zero.tol = zero.tol,
      prior.selected = prior.selected,
      prior.cutoff = prior.cutoff,
      prior.direction = prior.direction,
      ...
    )

    metric_summary <- list(
      selected = .ecountgmifs.metric.for.state(
        object = object,
        state = state,
        ground.truth = metric_object$ground_truth,
        zero.tol = zero.tol,
        zero.division = metric_object$settings$zero_division
      ),
      criteria = metric_object$criteria,
      prior = metric_object$prior
    )
  }

  out <- list(
    call = object$call,
    family = .ecountgmifs.family.label(object),
    link = .ecountgmifs.link.label(object),
    plugin_structure = .ecountgmifs.interface.label(object),
    n = object$input$n,
    p = object$input$p,
    q = object$input$q,
    enet_alpha = object$input$enet_alpha,
    has_prior = isTRUE(object$input$has_prior),
    null_negloglik = if (is.null(object$path$null_negloglik)) {
      NA_real_
    } else {
      as.numeric(object$path$null_negloglik)
    },
    saturated_negloglik = if (
        is.null(object$path$saturated_negloglik)
    ) {
      NA_real_
    } else {
      as.numeric(object$path$saturated_negloglik)
    },
    selected_label = selected$label,
    selected_criterion = selected$criterion,
    iteration = as.integer(state$iteration),
    negloglik = as.numeric(state$negloglik),
    loglik = -as.numeric(state$negloglik),
    pseudo_r2 = as.numeric(state$pseudo_r2),
    nonzero = sum(nonzero),
    theta = parameters$theta,
    family_parameters = parameters$family_parameters,
    link_parameters = parameters$link_parameters,
    coefficients = coefficient_table,
    max_coef = as.integer(max.coef),
    selected_criteria = if (
        is.null(state$criteria) ||
        length(state$criteria) == 0L
    ) {
      data.frame(
        criterion = character(),
        value = numeric(),
        stringsAsFactors = FALSE
      )
    } else {
      data.frame(
        criterion = names(state$criteria),
        value = as.numeric(state$criteria),
        stringsAsFactors = FALSE
      )
    },
    criteria = .ecountgmifs.criterion.table(object),
    metrics = metric_summary,
    timing = c(
      null = object$path$null_time,
      saturated = object$path$saturated_time,
      total = object$path$total_time
    ),
    termination = object$path$message
  )

  class(out) <- "summary.ecountgmifs"
  out
}


#' @export
print.summary.ecountgmifs <- function(
    x,
    digits = max(3L, getOption("digits") - 3L),
    ...
) {
  if (!is.null(x$call)) {
    cat("\nCall:\n")
    print(x$call)
  }

  cat("\nModel:\n")
  cat("  Family:            ", x$family, "\n", sep = "")
  cat("  Link:              ", x$link, "\n", sep = "")
  cat("  Family/link mode:   ", x$plugin_structure, "\n", sep = "")
  cat("  Observations:      ", x$n, "\n", sep = "")
  cat("  Penalized terms:   ", x$p, "\n", sep = "")
  cat("  Unpenalized terms: ", x$q, "\n", sep = "")
  cat("  Elastic-net alpha: ", format(x$enet_alpha, digits = digits), "\n", sep = "")
  cat("  Prior weights:     ", if (x$has_prior) "yes" else "no", "\n", sep = "")

  if (
      is.finite(x$null_negloglik) ||
      is.finite(x$saturated_negloglik)
  ) {
    cat("\nReference models:\n")

    if (is.finite(x$null_negloglik)) {
      cat(
        "  Null negative log-likelihood:      ",
        format(x$null_negloglik, digits = digits),
        "\n",
        sep = ""
      )
    }

    if (is.finite(x$saturated_negloglik)) {
      cat(
        "  Saturated negative log-likelihood: ",
        format(x$saturated_negloglik, digits = digits),
        "\n",
        sep = ""
      )
    }
  }

  cat("\nSelected state: ", x$selected_label, "\n", sep = "")
  cat("  Iteration:       ", x$iteration, "\n", sep = "")
  cat("  Negative log-likelihood: ", format(x$negloglik, digits = digits), "\n", sep = "")
  cat("  Log-likelihood:          ", format(x$loglik, digits = digits), "\n", sep = "")
  cat("  Pseudo-R2:       ", format(x$pseudo_r2, digits = digits), "\n", sep = "")
  cat("  Nonzero beta:    ", x$nonzero, "\n", sep = "")

  if (length(x$theta) > 0L) {
    cat("\nUnpenalized coefficients:\n")
    print(x$theta, digits = digits)
  }

  if (length(x$family_parameters) > 0L) {
    cat("\nFamily parameters:\n")
    print(x$family_parameters, digits = digits)
  }

  if (length(x$link_parameters) > 0L) {
    cat("\nLink parameters:\n")
    print(x$link_parameters, digits = digits)
  }

  cat("\nPenalized coefficients:\n")

  if (nrow(x$coefficients) == 0L) {
    cat("  No nonzero penalized coefficients.\n")
  } else {
    shown <- utils::head(
      x$coefficients[, c("term", "estimate"), drop = FALSE],
      x$max_coef
    )

    print(
      shown,
      row.names = FALSE,
      digits = digits
    )

    if (nrow(x$coefficients) > nrow(shown)) {
      cat(
        "... ",
        nrow(x$coefficients) - nrow(shown),
        " additional nonzero coefficients not shown\n",
        sep = ""
      )
    }
  }

  if (nrow(x$selected_criteria) > 0L) {
    cat("\nCriterion values at summarized state:\n")
    print(
      x$selected_criteria,
      row.names = FALSE,
      digits = digits
    )
  }

  if (nrow(x$criteria) > 0L) {
    cat("\nCriterion-selected states:\n")
    print(
      x$criteria,
      row.names = FALSE,
      digits = digits
    )
  }

  if (!is.null(x$metrics)) {
    cat("\nSelection metrics for summarized state:\n")
    print(
      x$metrics$selected,
      row.names = FALSE,
      digits = digits
    )

    if (!is.null(x$metrics$criteria) && nrow(x$metrics$criteria) > 0L) {
      cat("\nGround-truth metrics at criterion-selected states:\n")
      print(
        x$metrics$criteria[, c(
          "criterion",
          "iteration",
          "selected",
          "precision",
          "recall",
          "F1"
        ), drop = FALSE],
        row.names = FALSE,
        digits = digits
      )
    }

    if (!is.null(x$metrics$prior)) {
      cat("\nPrior-weight baseline:\n")
      print(
        x$metrics$prior,
        row.names = FALSE,
        digits = digits
      )
    }
  }

  if (length(x$timing) > 0L && any(is.finite(x$timing))) {
    cat("\nTiming (seconds):\n")
    print(x$timing, digits = digits)
  }

  if (!is.null(x$termination)) {
    cat("\nTermination:\n  ", x$termination, "\n", sep = "")
  }

  invisible(x)
}


#' Plot an ecountgmifs fit
#'
#' @param x An `ecountgmifs` fit.
#' @param type Plot type: coefficient paths, criterion paths, or ground-truth
#'   metrics.
#' @param ground.truth Required for `type = "metrics"`.
#' @param zero.tol,zero.division,prior.selected,prior.weight.vec,prior.cutoff,prior.direction
#'   Arguments used to construct an [ecountgmifs.metrics()] object.
#' @param ... Arguments passed to the corresponding plotting helper.
#'
#' @return The input fit invisibly, except for `type = "metrics"`, which returns
#'   the generated `ecountgmifs.metrics` object invisibly.
#' @export
plot.ecountgmifs <- function(
    x,
    type = c("coefficients", "criteria", "metrics"),
    ground.truth = NULL,
    zero.tol = sqrt(.Machine$double.eps),
    zero.division = c("zero", "one", "NA"),
    prior.selected = NULL,
    prior.weight.vec = NULL,
    prior.cutoff = 1,
    prior.direction = c("lower", "higher"),
    ...
) {
  type <- match.arg(type)

  switch(
    type,
    coefficients = {
      .ecountgmifs.plot.coefficients(
        x,
        zero.tol = zero.tol,
        ...
      )
      invisible(x)
    },
    criteria = {
      .ecountgmifs.plot.criteria(x, ...)
      invisible(x)
    },
    metrics = {
      if (is.null(ground.truth)) {
        stop(
          "`ground.truth` is required for `type = \"metrics\"`.",
          call. = FALSE
        )
      }

      metric_object <- ecountgmifs.metrics(
        object = x,
        ground.truth = ground.truth,
        zero.tol = zero.tol,
        zero.division = match.arg(zero.division),
        prior.selected = prior.selected,
        prior.weight.vec = prior.weight.vec,
        prior.cutoff = prior.cutoff,
        prior.direction = match.arg(prior.direction)
      )

      plot(metric_object, ...)
      invisible(metric_object)
    }
  )
}
