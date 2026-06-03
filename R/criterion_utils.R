#' Coerce one object to an ecountgmifs criterion
#'
#' @description
#' Internal helper used by [ecountgmifs()] to coerce a user-supplied
#' model-selection criterion into an object of class
#' \code{"ecountgmifs.criterion"}.
#'
#' Accepted inputs are:
#' \describe{
#'   \item{compiled criterion object}{
#'     An object of class \code{"ecountgmifs.criterion"}, usually returned by
#'     [ecountgmifs.compile.criterion()] or one of the example criterion
#'     constructors such as [BIC_nnz()].
#'   }
#'   \item{criterion constructor function}{
#'     A zero-argument function returning an object of class
#'     \code{"ecountgmifs.criterion"}, such as [BIC_nnz].
#'   }
#'   \item{criterion constructor name}{
#'     A single character string naming a criterion constructor function, such
#'     as \code{"BIC_nnz"}. The function is searched first in \code{envir} and
#'     then in the \pkg{ecountgmifs} namespace.
#'   }
#' }
#'
#' If \code{name} is supplied, it overrides the criterion object's internal
#' \code{name} field. This is how named elements of the \code{criteria} list in
#' [ecountgmifs()] become output names. If \code{name} is not supplied, the
#' criterion object's own \code{name} field is preserved. If that field is empty,
#' the compiled symbol name is used instead.
#'
#' @param x criterion object, function, or character string. Object to coerce.
#' @param name character or \code{NULL}. Optional output name to assign to the
#'   criterion.
#' @param envir environment. Environment in which character criterion names are
#'   resolved.
#'
#' @return An object of class \code{"ecountgmifs.criterion"}.
#'
#' @keywords internal
.ecountgmifs.as.criterion <- function(x, name = NULL, envir = parent.frame()) {
  if (inherits(x, "ecountgmifs.criterion")) {
    out <- x

  } else if (is.function(x)) {
    out <- x()

    if (!inherits(out, "ecountgmifs.criterion")) {
      stop(
        "criterion function did not return an object of class 'ecountgmifs.criterion'",
        call. = FALSE
      )
    }

  } else if (is.character(x) && length(x) == 1L && !is.na(x)) {
    fun.name <- x

    if (exists(fun.name, envir = envir, mode = "function", inherits = TRUE)) {
      fun <- get(fun.name, envir = envir, mode = "function", inherits = TRUE)
    } else if (exists(fun.name,
                      envir = asNamespace("ecountgmifs"),
                      mode = "function",
                      inherits = FALSE)) {
      fun <- get(fun.name,
                 envir = asNamespace("ecountgmifs"),
                 mode = "function",
                 inherits = FALSE)
    } else {
      stop(
        "could not find criterion function named '", fun.name, "'",
        call. = FALSE
      )
    }

    out <- fun()

    if (!inherits(out, "ecountgmifs.criterion")) {
      stop(
        "criterion function named '", fun.name,
        "' did not return an object of class 'ecountgmifs.criterion'",
        call. = FALSE
      )
    }

  } else {
    stop(
      "each criterion must be an 'ecountgmifs.criterion' object, ",
      "a function returning one, or a character string naming such a function",
      call. = FALSE
    )
  }

  if (!is.null(name) && nzchar(name)) {
    out$name <- name
  } else if (is.null(out$name) || !nzchar(out$name)) {
    out$name <- out$symbol
  }

  out
}


#' Normalize model-selection criteria
#'
#' @description
#' Internal helper used by [ecountgmifs()] to normalize the \code{criteria}
#' argument into a named list of compiled criterion objects.
#'
#' The user may supply:
#' \enumerate{
#'   \item a single object of class \code{"ecountgmifs.criterion"};
#'   \item a zero-argument function returning such an object;
#'   \item a single character string naming such a function; or
#'   \item a list containing any mixture of the above.
#' }
#'
#' Character entries are resolved to functions and called. This allows both
#' package-provided criterion constructors, such as \code{"BIC_nnz"}, and
#' user-defined criterion constructors to be passed by name.
#'
#' If \code{criteria} is a named list, the list names override the internal
#' criterion names stored in the compiled criterion objects. If list elements
#' are unnamed, the \code{name} field returned by
#' [ecountgmifs.compile.criterion()] is used.
#'
#' @param criteria criterion object, function, character string, list, or
#'   \code{NULL}. User-supplied model-selection criteria.
#' @param envir environment. Environment in which character criterion names are
#'   resolved.
#'
#' @return A named list of objects of class \code{"ecountgmifs.criterion"}.
#'
#' @keywords internal
.ecountgmifs.normalize.criteria <- function(criteria, envir = parent.frame()) {
  if (missing(criteria) || is.null(criteria)) {
    return(list())
  }

  if (inherits(criteria, "ecountgmifs.criterion") ||
      is.function(criteria) ||
      (is.character(criteria) && length(criteria) == 1L)) {
    criteria <- list(criteria)
  }

  if (!is.list(criteria)) {
    stop(
      "value of 'criteria' must be a criterion object, a criterion function, ",
      "a character criterion-function name, or a list containing these",
      call. = FALSE
    )
  }

  criteria.names <- names(criteria)

  out <- vector("list", length(criteria))

  for (i in seq_along(criteria)) {
    user.name <- NULL

    if (!is.null(criteria.names) &&
        length(criteria.names) == length(criteria) &&
        !is.na(criteria.names[[i]]) &&
        nzchar(criteria.names[[i]])) {
      user.name <- criteria.names[[i]]
    }

    out[[i]] <- .ecountgmifs.as.criterion(
      criteria[[i]],
      name = user.name,
      envir = envir
    )
  }

  names(out) <- vapply(
    out,
    function(x) x$name,
    character(1L)
  )

  out
}
