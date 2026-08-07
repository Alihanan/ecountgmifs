#' Extract all fitted model parameters
#'
#' `parameters()` extracts penalized coefficients, unpenalized coefficients,
#' family parameters, and link parameters from saved path states. `params()` is
#' a shorter alias with identical behavior.
#'
#' With no selector, the saved parameter path is returned. Use
#' `state = "terminal"`, one or more criterion names, or one or more iterations
#' to select particular fitted states.
#'
#' @param object An ecountgmifs fit, multi-alpha fit, or alpha by prior-strength
#'   tuning grid.
#' @param ... Arguments passed to the corresponding class method.
#'
#' @return A named list with components `beta`, `theta`, `family`, and `link`.
#'   Each component is a matrix for a path or multiple states, and a named
#'   numeric vector for one selected state when `drop = TRUE`.
#' @export
parameters <- function(object, ...) {
  UseMethod("parameters")
}


#' @rdname parameters
#' @export
params <- function(object, ...) {
  parameters(object, ...)
}


.ecountgmifs.parameter.group.names <- c(
  "beta",
  "theta",
  "family",
  "link"
)


.ecountgmifs.is.parameter.bundle <- function(value) {
  is.list(value) &&
    identical(names(value), .ecountgmifs.parameter.group.names)
}


.ecountgmifs.extract.parameter.group <- function(value, group) {
  group <- match.arg(group, .ecountgmifs.parameter.group.names)

  if (.ecountgmifs.is.parameter.bundle(value)) {
    return(value[[group]])
  }

  if (!is.list(value)) {
    stop(
      "Unexpected parameter-extraction result.",
      call. = FALSE
    )
  }

  out <- lapply(
    value,
    .ecountgmifs.extract.parameter.group,
    group = group
  )
  names(out) <- names(value)
  out
}


#' Extract penalized coefficients
#'
#' These accessors extract the penalized `beta` coefficient group while
#' preserving the same selectors and return structure as [parameters()].
#' `bparams()` is the preferred compact alias. `beta.parameters()`,
#' `beta.params()`, `pen.parameters()`, `pen.params()`, and `penparams()` are
#' equivalent descriptive aliases.
#'
#' @param object An ecountgmifs fit, multi-alpha fit, or tuning grid.
#' @param ... Arguments passed to [parameters()].
#'
#' @return A numeric vector, matrix, or named list, depending on the selected
#'   object and states.
#' @export beta.parameters
beta.parameters <- function(object, ...) {
  .ecountgmifs.extract.parameter.group(
    parameters(object, ...),
    "beta"
  )
}


#' @rdname beta.parameters
#' @export beta.params
beta.params <- function(object, ...) {
  beta.parameters(object, ...)
}


#' @rdname beta.parameters
#' @export
bparams <- function(object, ...) {
  beta.parameters(object, ...)
}


#' @rdname beta.parameters
#' @export pen.parameters
pen.parameters <- function(object, ...) {
  beta.parameters(object, ...)
}


#' @rdname beta.parameters
#' @export pen.params
pen.params <- function(object, ...) {
  beta.parameters(object, ...)
}


#' @rdname beta.parameters
#' @export
penparams <- function(object, ...) {
  beta.parameters(object, ...)
}


#' Extract unpenalized coefficients
#'
#' `theta()` extracts the unpenalized coefficient group while preserving the
#' same state, criterion, iteration, alpha, eta, and `drop` behavior as
#' [parameters()]. `thparams()` is a compact explicit alias.
#'
#' @param object An ecountgmifs fit, multi-alpha fit, or tuning grid.
#' @param ... Arguments passed to [parameters()].
#'
#' @return A numeric vector, matrix, or named list, depending on the selected
#'   object and states.
#' @export
theta <- function(object, ...) {
  .ecountgmifs.extract.parameter.group(
    parameters(object, ...),
    "theta"
  )
}


#' @rdname theta
#' @export
thparams <- function(object, ...) {
  theta(object, ...)
}


#' @rdname theta
#' @export theta.parameters
theta.parameters <- function(object, ...) {
  theta(object, ...)
}


#' @rdname theta
#' @export theta.params
theta.params <- function(object, ...) {
  theta(object, ...)
}


#' @rdname theta
#' @export
thetaparams <- function(object, ...) {
  theta(object, ...)
}


#' @rdname theta
#' @export nonpen.parameters
nonpen.parameters <- function(object, ...) {
  theta(object, ...)
}


#' @rdname theta
#' @export nonpen.params
nonpen.params <- function(object, ...) {
  theta(object, ...)
}


#' @rdname theta
#' @export
nonpenparams <- function(object, ...) {
  theta(object, ...)
}


#' @rdname theta
#' @export
npparams <- function(object, ...) {
  theta(object, ...)
}


#' Extract family parameters
#'
#' `family.parameters()` extracts fitted family parameters. `fparams()` is the
#' preferred compact alias, while `family.params()` is retained as a descriptive
#' alias. All three preserve the selectors and return structure of
#' [parameters()].
#'
#' @inheritParams theta
#'
#' @return A numeric vector, matrix, or named list, depending on the selected
#'   object and states. Parameter-free families return zero-length vectors or
#'   zero-column matrices.
#' @export family.parameters
family.parameters <- function(object, ...) {
  .ecountgmifs.extract.parameter.group(
    parameters(object, ...),
    "family"
  )
}


#' @rdname family.parameters
#' @export
fparams <- function(object, ...) {
  family.parameters(object, ...)
}


#' @rdname family.parameters
#' @export family.params
family.params <- function(object, ...) {
  family.parameters(object, ...)
}


#' Extract link parameters
#'
#' `link.parameters()` extracts fitted link parameters. `lparams()` is the
#' preferred compact alias, while `link.params()` is retained as a descriptive
#' alias. All three preserve the selectors and return structure of
#' [parameters()].
#'
#' @inheritParams theta
#'
#' @return A numeric vector, matrix, or named list, depending on the selected
#'   object and states. Parameter-free links return zero-length vectors or
#'   zero-column matrices.
#' @export link.parameters
link.parameters <- function(object, ...) {
  .ecountgmifs.extract.parameter.group(
    parameters(object, ...),
    "link"
  )
}


#' @rdname link.parameters
#' @export
lparams <- function(object, ...) {
  link.parameters(object, ...)
}


#' @rdname link.parameters
#' @export link.params
link.params <- function(object, ...) {
  link.parameters(object, ...)
}


.ecountgmifs.auxiliary.parameter.names <- function(value, prefix) {
  value.length <- length(value)

  if (value.length == 0L) {
    return(character())
  }

  out <- names(value)

  if (
      is.null(out) ||
      length(out) != value.length ||
      anyNA(out) ||
      any(!nzchar(out))
  ) {
    out <- paste0(prefix, seq_len(value.length))
  }

  make.unique(as.character(out))
}


.ecountgmifs.all.parameters <- function(object, state) {
  raw <- state$predictors$parameters

  if (is.null(raw)) {
    stop(
      "The selected state does not contain fitted parameters.",
      call. = FALSE
    )
  }

  beta <- as.numeric(raw$beta)
  names(beta) <- .ecountgmifs.predictor.names(object, length(beta))

  theta <- as.numeric(raw$theta)
  names(theta) <- .ecountgmifs.unpenalized.names(object, length(theta))

  family <- as.numeric(raw$family_parameters)
  names(family) <- .ecountgmifs.auxiliary.parameter.names(
    raw$family_parameters,
    "family"
  )

  link <- as.numeric(raw$link_parameters)
  names(link) <- .ecountgmifs.auxiliary.parameter.names(
    raw$link_parameters,
    "link"
  )

  list(
    beta = beta,
    theta = theta,
    family = family,
    link = link
  )
}


.ecountgmifs.parameter.matrix <- function(values, row.names, column.names) {
  widths <- vapply(values, length, integer(1L))

  if (length(unique(widths)) > 1L) {
    stop(
      "A parameter group changes dimension between selected states.",
      call. = FALSE
    )
  }

  width <- if (length(widths) == 0L) 0L else widths[[1L]]

  if (width == 0L) {
    out <- matrix(numeric(), nrow = length(values), ncol = 0L)
  } else {
    out <- do.call(rbind, lapply(values, as.numeric))

    if (is.null(dim(out))) {
      out <- matrix(out, nrow = length(values))
    }
  }

  rownames(out) <- row.names
  colnames(out) <- column.names
  out
}


.ecountgmifs.parameter.path <- function(object) {
  states <- .ecountgmifs.path.states(object)
  terminal <- .ecountgmifs.terminal.state(object)
  terminal.parameters <- .ecountgmifs.all.parameters(object, terminal)

  groups <- list(
    beta = states$beta,
    theta = states$theta,
    family = states$family_parameters,
    link = states$link_parameters
  )

  state.count <- length(states$beta)
  groups <- lapply(
    groups,
    function(value) {
      if (is.null(value)) rep(list(numeric()), state.count) else value
    }
  )

  if (any(vapply(groups, length, integer(1L)) != state.count)) {
    stop(
      "Stored parameter paths have different lengths.",
      call. = FALSE
    )
  }

  row.names <- if (is.null(states$iteration)) {
    paste0("state_", seq_len(state.count))
  } else {
    paste0("iter_", as.integer(states$iteration))
  }

  out <- lapply(
    names(groups),
    function(group) {
      .ecountgmifs.parameter.matrix(
        groups[[group]],
        row.names,
        names(terminal.parameters[[group]])
      )
    }
  )
  names(out) <- names(groups)
  out
}


.ecountgmifs.select.parameter.states <- function(
    object,
    criterion,
    iteration,
    state
) {
  if (!is.null(state)) {
    if (!is.character(state) || length(state) != 1L || is.na(state)) {
      stop(
        "`state` must be NULL, \"path\", or \"terminal\".",
        call. = FALSE
      )
    }

    state <- match.arg(state, c("path", "terminal"))

    if (!is.null(criterion) || !is.null(iteration)) {
      stop(
        "`state` cannot be combined with `criterion` or `iteration`.",
        call. = FALSE
      )
    }
  }

  if (!is.null(criterion) && !is.null(iteration)) {
    stop(
      "Supply either `criterion` or `iteration`, not both.",
      call. = FALSE
    )
  }

  path.requested <-
    identical(state, "path") ||
    (is.null(state) && is.null(criterion) && is.null(iteration))

  if (path.requested) {
    return(list(path = TRUE, states = NULL, labels = NULL))
  }

  if (identical(state, "terminal")) {
    return(
      list(
        path = FALSE,
        states = list(.ecountgmifs.terminal.state(object)),
        labels = "terminal"
      )
    )
  }

  if (!is.null(criterion)) {
    criterion <- .ecountgmifs.match.criteria(object, criterion)

    if (length(criterion) == 0L) {
      stop("No criterion names were supplied.", call. = FALSE)
    }

    states <- lapply(
      criterion,
      function(criterion.name) {
        .ecountgmifs.resolve.state(
          object,
          criterion = criterion.name
        )$state
      }
    )

    return(list(path = FALSE, states = states, labels = criterion))
  }

  if (
      !is.numeric(iteration) ||
      length(iteration) == 0L ||
      anyNA(iteration) ||
      any(!is.finite(iteration))
  ) {
    stop(
      "`iteration` must contain one or more finite numeric values.",
      call. = FALSE
    )
  }

  resolved <- lapply(
    as.numeric(iteration),
    function(iteration.value) {
      .ecountgmifs.resolve.state(
        object,
        iteration = iteration.value
      )
    }
  )

  states <- lapply(
    resolved,
    function(value) value$state
  )
  labels <- vapply(
    states,
    function(value) paste0("iter_", as.integer(value$iteration)),
    character(1L)
  )

  list(
    path = FALSE,
    states = states,
    labels = make.unique(labels)
  )
}


#' @rdname parameters
#' @param criterion Optional criterion name or names selecting their best stored
#'   states.
#' @param iteration Optional numeric iteration or iterations. The nearest saved
#'   state is used for each value.
#' @param state Optional explicit state selector: `"path"` or `"terminal"`.
#'   It cannot be combined with `criterion` or `iteration`.
#' @param drop Logical. For one selected state, return vectors instead of
#'   one-row matrices.
#' @export
parameters.ecountgmifs <- function(
    object,
    criterion = NULL,
    iteration = NULL,
    state = NULL,
    drop = TRUE,
    ...
) {
  selected <- .ecountgmifs.select.parameter.states(
    object,
    criterion,
    iteration,
    state
  )

  if (isTRUE(selected$path)) {
    return(.ecountgmifs.parameter.path(object))
  }

  out <- lapply(
    selected$states,
    function(selected.state) {
      .ecountgmifs.all.parameters(object, selected.state)
    }
  )

  if (length(out) == 1L && isTRUE(drop)) {
    return(out[[1L]])
  }

  groups <- c("beta", "theta", "family", "link")
  result <- lapply(
    groups,
    function(group) {
      values <- lapply(out, function(value) value[[group]])
      .ecountgmifs.parameter.matrix(
        values,
        selected$labels,
        names(values[[1L]])
      )
    }
  )
  names(result) <- groups
  result
}


#' @rdname parameters
#' @param alpha Optional stored alpha value or values, or their labels.
#' @export
parameters.ecountgmifs_multi_alpha <- function(
    object,
    criterion = NULL,
    iteration = NULL,
    alpha = NULL,
    state = NULL,
    drop = TRUE,
    ...
) {
  indices <- .ecountgmifs.multi.alpha.select(object, alpha = alpha)
  out <- lapply(
    indices,
    function(index) {
      parameters(
        object$fits[[index]],
        criterion = criterion,
        iteration = iteration,
        state = state,
        drop = drop,
        ...
      )
    }
  )

  names(out) <- paste0("alpha=", object$alpha_labels[indices])

  if (!is.null(alpha) && length(out) == 1L && isTRUE(drop)) {
    return(out[[1L]])
  }

  out
}


#' @rdname parameters
#' @param eta Optional stored prior-strength value or values, or their labels.
#' @export
parameters.ecountgmifs_tuning_grid <- function(
    object,
    criterion = NULL,
    iteration = NULL,
    alpha = NULL,
    eta = NULL,
    state = NULL,
    drop = TRUE,
    ...
) {
  indices <- .ecountgmifs.tuning.grid.select(
    object,
    alpha = alpha,
    eta = eta
  )
  out <- lapply(
    indices,
    function(index) {
      parameters(
        object$fits[[index]],
        criterion = criterion,
        iteration = iteration,
        state = state,
        drop = drop,
        ...
      )
    }
  )

  names(out) <- names(object$fits)[indices]

  if (length(out) == 1L && isTRUE(drop)) {
    return(out[[1L]])
  }

  out
}
