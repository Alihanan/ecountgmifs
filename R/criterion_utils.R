.is.criterion.pointer <- function(x) {
  identical(typeof(x), "externalptr")
}

.extract.criterion.pointer <- function(x) {
  if (.is.criterion.pointer(x)) {
    return(x)
  }

  if (inherits(x, "ecountgmifs.criterion")) {
    candidates <- c("pointer", "ptr", "external.pointer")

    for (name in candidates) {
      if (!is.null(x[[name]]) && .is.criterion.pointer(x[[name]])) {
        return(x[[name]])
      }
    }
  }

  NULL
}

#' Coerce one object to a criterion external pointer
#'
#' @description
#' Internal helper used by [ecountgmifs()] to resolve a supplied criterion.
#' Accepted inputs are an external pointer, a zero-argument constructor, a
#' character name resolving to such a constructor, or a compatible wrapper
#' object containing an external pointer.
#'
#' @param criterion Criterion specification.
#' @param name Optional list name used for output naming.
#' @param envir Environment used to resolve character function names.
#'
#' @return An external pointer.
#' @keywords internal
.ecountgmifs.as.criterion <- function(
    criterion,
    name = NULL,
    envir = parent.frame()
) {
  value <- criterion

  if (is.character(value) && length(value) == 1L) {
    value <- get(value, envir = envir, mode = "function", inherits = TRUE)
  }

  if (is.function(value)) {
    value <- value()
  }

  pointer <- .extract.criterion.pointer(value)

  if (is.null(pointer)) {
    stop(
      "Each criterion must be an external pointer, a constructor returning ",
      "one, a character constructor name, or a compatible criterion wrapper.",
      call. = FALSE
    )
  }

  tryCatch(
    inspect_criterion_plugin(pointer),
    error = function(error) {
      stop(
        "Invalid criterion external pointer: ",
        conditionMessage(error),
        call. = FALSE
      )
    }
  )

  if (!is.null(name) && nzchar(name)) {
    attr(pointer, "criterion.list.name") <- name
  }

  pointer
}

#' Normalize model-selection criteria
#'
#' @description
#' Internal helper that resolves criterion constructors and character names to
#' a list of external pointers accepted by the C++ input layer.
#'
#' @param criteria Criterion pointer, function, character name, list, or `NULL`.
#' @param envir Environment used to resolve character names.
#'
#' @return A list of criterion external pointers.
#' @keywords internal
.ecountgmifs.normalize.criteria <- function(
    criteria,
    envir = parent.frame()
) {
  if (missing(criteria) || is.null(criteria)) {
    return(list())
  }

  if (
      .is.criterion.pointer(criteria) ||
      inherits(criteria, "ecountgmifs.criterion") ||
      is.function(criteria) ||
      (is.character(criteria) && length(criteria) == 1L)
  ) {
    criteria <- list(criteria)
  }

  if (!is.list(criteria)) {
    stop(
      "`criteria` must be a criterion, constructor, character name, or list.",
      call. = FALSE
    )
  }

  input.names <- names(criteria)
  output <- vector("list", length(criteria))
  output.names <- character(length(criteria))

  for (i in seq_along(criteria)) {
    user.name <- NULL

    if (
        !is.null(input.names) &&
        length(input.names) == length(criteria) &&
        !is.na(input.names[[i]]) &&
        nzchar(input.names[[i]])
    ) {
      user.name <- input.names[[i]]
    }

    output[[i]] <- .ecountgmifs.as.criterion(
      criteria[[i]],
      name = user.name,
      envir = envir
    )

    output.names[[i]] <- if (!is.null(user.name)) {
      user.name
    } else {
      plugin.name <- attr(output[[i]], "r.plugin.name", exact = TRUE)

      if (!is.character(plugin.name) || length(plugin.name) != 1L) {
        plugin.name <- attr(output[[i]], "plugin.name", exact = TRUE)
      }

      if (!is.character(plugin.name) || length(plugin.name) != 1L) {
        plugin.name <- attr(output[[i]], "criterion.name", exact = TRUE)
      }

      if (is.character(plugin.name) && length(plugin.name) == 1L &&
          nzchar(plugin.name)) {
        plugin.name
      } else {
        paste0("criterion_", i)
      }
    }
  }

  names(output) <- output.names
  output
}
