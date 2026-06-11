.ecountgmifs.family.label <- function(x) {
  x <- as.integer(x)

  switch(
    as.character(x),
    "0" = "NEGATIVE_BINOMIAL",
    "1" = "POISSON",
    stop("unknown EnumFamily value: ", x, call. = FALSE)
  )
}

.ecountgmifs.link.label <- function(x) {
  x <- as.integer(x)

  switch(
    as.character(x),
    "0" = "LOG_LINK",
    "1" = "SOFTPLUS_LINK",
    stop("unknown EnumLinkFunc value: ", x, call. = FALSE)
  )
}



#' @export
print.ecountgmifs <- function(x, ...) {
  cat("Extended count GMIFS model fit\n")
  cat("Family:", .ecountgmifs.family.label(x$input$family), "\n")
  cat("Link function:", .ecountgmifs.link.label(x$input$link), "\n")
  cat("Observations:", (x$input$n), "\n")
  cat("Predictors:", (x$input$p), "\n")
  invisible(x)
}
#' @export
summary.ecountgmifs <- function(object, ...) {
  out <- list(
    call = object$call,
    family = .ecountgmifs.family.label(object$input$family),
    link = .ecountgmifs.link.label(object$input$link),
    n = (object$input$n),
    p = (object$input$p),
    control = object$control,
    has_prior = object$input$has_prior
  )

  class(out) <- "summary.ecountgmifs"
  out
}

#' @export
print.summary.ecountgmifs <- function(x, ...) {
  cat("Summary of extended count GMIFS model fit\n")
  cat("Family:", x$family, "\n")
  cat("Link function:", x$link, "\n")
  cat("Observations:", x$n, "\n")
  cat("Predictors:", x$p, "\n")
  cat("Prior information:", if (x$has_prior) "yes" else "no", "\n")
  invisible(x)
}

#' @export
coef.ecountgmifs <- function(object, ...) {
  t(do.call("cbind", object$states$beta))
}

#' @export
plot.ecountgmifs <- function(x, ...) {
  if (is.null(x$states) || is.null(x$states$beta)) {
    plot.new()
    title("No coefficient path stored in this ecountgmifs object")
    return(invisible(x))
  }

  coef.mat = coef.ecountgmifs(x)

  matplot(
    coef.mat,
    type = "l",
    lty = 1,
    xlab = "Step",
    ylab = "Coefficient"
  )

  invisible(x)
}
