

#' @export
print.ecountgmifs <- function(x, ...) {
  cat("Extended count GMIFS model fit\n")
  cat("Family:", x$family, "\n")
  cat("Observations:", nrow(x$x), "\n")
  cat("Predictors:", ncol(x$x), "\n")
  invisible(x)
}
#' @export
summary.ecountgmifs <- function(object, ...) {
  out <- list(
    call = object$call,
    family = object$family,
    n = nrow(object$x),
    p = ncol(object$x),
    control = object$control,
    has_prior = !is.null(object$prior)
  )

  class(out) <- "summary.ecountgmifs"
  out
}

#' @export
print.summary.ecountgmifs <- function(x, ...) {
  cat("Summary of extended count GMIFS model fit\n")
  cat("Family:", x$family, "\n")
  cat("Observations:", x$n, "\n")
  cat("Predictors:", x$p, "\n")
  cat("Prior information:", if (x$has_prior) "yes" else "no", "\n")
  invisible(x)
}

#' @export
coef.ecountgmifs <- function(object, ...) {
  object$coefficients
}

#' @export
plot.ecountgmifs <- function(x, ...) {
  if (is.null(x$path)) {
    plot.new()
    title("No coefficient path stored in this ecountgmifs object")
    return(invisible(x))
  }

  matplot(
    x$path,
    type = "l",
    lty = 1,
    xlab = "Step",
    ylab = "Coefficient"
  )

  invisible(x)
}
