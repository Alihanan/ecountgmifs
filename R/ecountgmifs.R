


#' Fit an extended count GMIFS model
#'
#' Fits an extended generalized monotone incremental forward stagewise model
#' for high-dimensional count outcomes.
#'
#' @param X Numeric predictor matrix of dimension n * p
#' @param y Count response vector of length n
#' @param w numeric matrix or \code{NULL}. Unpenalized predictor matrix of
#'   dimension \eqn{n \times q}, where \eqn{n} is the number of observations.
#'   Columns of \code{w} are included in the model without stagewise
#'   penalization. Typical examples include an intercept, clinical covariates,
#'   batch indicators, exposure variables, or other adjustment variables. If
#'   \code{w = NULL} and \code{intercept = TRUE}, \code{w} is initialized as a
#'   one-column intercept matrix. If \code{intercept = FALSE}, \code{w} must be
#'   supplied and must contain at least one column.
#' @param intercept logical. Should an intercept column be included in the
#'   unpenalized design matrix \code{w}? If \code{TRUE}, a column of ones is
#'   added to \code{w}. If \code{w = NULL} or \code{ncol(w) = 0}, then
#'   \code{w} is initialized as a one-column intercept matrix. If
#'   \code{FALSE}, \code{w} must be supplied and must contain at least one
#'   column.
#' @param weight.vec Optional prior weight vector of length p
#' @param family Model family. Currently `"negative.binomial"` only.
#' @param control A control object from [ecountgmifs.control()].
#' @param verbose logical, if TRUE the step number is printed to the console (default is FALSE).
#' @param fixed.dispersion logical. Should the negative-binomial
#'   dispersion parameter be fixed during fitting? If \code{FALSE}, the
#'   dispersion parameter is re-estimated along the stagewise path using the
#'   internal numerical optimization routine. If \code{TRUE}, the value supplied
#'   in \code{fixed.dispersion.value} is used throughout fitting.
#' @param fixed.dispersion.value numeric. Fixed value of the
#'   negative-binomial dispersion parameter used when
#'   \code{fixed.dispersion = TRUE}. Ignored when
#'   \code{fixed.dispersion = FALSE}.
#' @param enet.alpha numeric. Elastic-net mixing parameter controlling the
#'   relative contribution of the \eqn{\ell_1} and \eqn{\ell_2} components in
#'   the stagewise update. Values must be between 0 and 1. The value
#'   \code{enet.alpha = 1} gives the lasso-type update, while
#'   \code{enet.alpha = 0} gives the ridge-type update. Intermediate values use
#'   a convex combination of the two penalties.
#' @param criteria criterion object, function, character, list, or `NULL`.
#'   Model-selection criteria evaluated along the fitted path. A criterion may
#'   be supplied as an object of class `"ecountgmifs.criterion"`, as returned by
#'   [ecountgmifs.compile.criterion()], as a zero-argument function returning
#'   such an object, such as [BIC_nnz], or as a character string naming such a
#'   function, such as `"BIC_nnz"`. A single criterion may be supplied directly,
#'   or multiple criteria may be supplied as a list. Named list elements are
#'   used as output names.
#'
#'
#' @details
#' The default `criteria` value is intentionally written as a representative
#' example:
#'
#' \preformatted{
#' criteria = list(
#'   "AIC_nnz" = AIC_nnz,
#'   "BIC_nnz" = BIC_nnz(),
#'   "SABIC_nnz" = "SABIC_nnz"
#' )
#' }
#'
#' This shows the three accepted forms: a criterion-constructor function
#' such as \code{AIC_nnz}, an already compiled criterion object such as
#' \code{BIC_nnz()}, and the character name of a criterion-constructor
#' function such as \code{"SABIC_nnz"}. Character values are resolved to
#' functions and called. This allows users to define their own
#' criterion-constructor functions and pass them by name.
#'
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
    criteria = list(
      "AIC_nnz"=AIC_nnz,
      "BIC_nnz" = BIC_nnz(),
      "SABIC_nnz" = "SABIC_nnz"
    ),
    verbose = F,
    fixed.dispersion = FALSE,
    fixed.dispersion.value = 0,
    control = ecountgmifs.control()
) {
  X <- as.matrix(X)
  y <- as.numeric(y)

  p <- ncol(X)
  n <- nrow(X)

  if (!is.numeric(X)) {
    stop("value of 'X' must be a numeric matrix")
  }

  if (!is.numeric(y)) {
    stop("value of 'y' must be numeric")
  }

  if (length(y) != nrow(X)) {
    stop("length of 'y' must be equal to 'nrow(X)'")
  }

  if (anyNA(X)) {
    stop("missing values in 'X' are not supported")
  }

  if (anyNA(y)) {
    stop("missing values in 'y' are not supported")
  }

  if (any(y < 0)) {
    stop("value of 'y' must contain non-negative counts")
  }

  if (!is.logical(intercept) || length(intercept) != 1L || is.na(intercept)) {
    stop("value of 'intercept' must be TRUE or FALSE")
  }

  if (!is.numeric(enet.alpha) ||
      length(enet.alpha) != 1L ||
      is.na(enet.alpha) ||
      enet.alpha < 0 ||
      enet.alpha > 1) {
    stop("value of 'enet.alpha' must be between 0 and 1")
  }

  if (is.null(w)) {
    if (intercept) {
      w <- matrix(1, nrow = nrow(X), ncol = 1)
      colnames(w) <- "(Intercept)"
    } else {
      stop("value of 'w' must be supplied when 'intercept' is FALSE")
    }
  } else {
    w <- as.matrix(w)

    if (nrow(w) != nrow(X)) {
      stop("number of rows in 'w' must be equal to 'nrow(X)'")
    }

    if (ncol(w) == 0L) {
      if (intercept) {
        w <- matrix(1, nrow = nrow(X), ncol = 1)
        colnames(w) <- "(Intercept)"
      } else {
        stop("value of 'w' must have at least one column when 'intercept' is FALSE")
      }
    } else if (intercept) {
      w <- cbind("(Intercept)" = 1, w)
    }
  }

  if (is.null(offset)) {
    offset <- rep(0, nrow(X))
  } else {
    offset <- as.numeric(offset)

    if (length(offset) != nrow(X)) {
      stop("length of 'offset' must be equal to 'nrow(X)'")
    }

    if (anyNA(offset)) {
      stop("missing values in 'offset' are not supported")
    }
  }


  if (is.null(weight.vec)) {
    weight.vec <- rep(1, p)
  } else {
    weight.vec <- as.numeric(weight.vec)

    if (length(weight.vec) != p) {
      stop("length of 'weight.vec' must be equal to 'ncol(X)'")
    }

    if (anyNA(weight.vec)) {
      stop("missing values in 'weight.vec' are not supported")
    }

    if (any(weight.vec <= 0)) {
      stop("values of 'weight.vec' must be positive")
    }
  }

  weight.vec <- weight.vec * p / sum(weight.vec)

  link = match.arg(link)
  link = switch(link,
                    "log" = 0,
                    "softplus" = 1
  )

  if (inherits(criteria, "ecountgmifs.criterion")) {
    criteria <- list(criteria)
  }

  if (!is.list(criteria)) {
    stop("value of 'criteria' must be a list of compiled criterion objects")
  }

  criteria <- .ecountgmifs.normalize.criteria(
    criteria,
    envir = parent.frame()
  )

  control$state.track.strategy <- switch(
    control$state.track.strategy,
    "active.set.change" = 0L,
    "all.iteration" = 1L,
    "every.k.iteration" = 2L,
    "none" = 3L
  )

  family = match.arg(family)
  family.int <- switch(
    family,
    "negative.binomial" = 0L,
    "poisson" = 1L
  )

  if(family == "poisson"){
    fixed.dispersion = T
    fixed.dispersion.value = 0.0
  }


  ecountgmifs_cpp(X = X,
                  y = y,
                  family = family.int,
                  linkfunc_int = link,
                  offset = offset,
                  weight_vec = weight.vec,
                  verbose = verbose,
                  w = w,
                  yorig = y, # TODO
                  Xtest = X, # TODO
                  ytest = y, # TODO
                  wtest = w, # TODO
                  offsettest = offset, # TODO
                  enet_alpha = enet.alpha,
                  epsilon_start = control$epsilon.start,
                  epsilon_max = control$epsilon.max,
                  epsilon_min_tol = control$epsilon.min,

                  tol = control$tol,
                  iteration_max = control$iteration.max,

                  nlopt_optim_reltol = control$nlopt.optim.reltol,
                  loglik_reltol_cutoff = control$loglik.reltol.cutoff,
                  is_fixed_disp = fixed.dispersion,
                  fixed_disp_value = fixed.dispersion.value,

                  state_track_strategy = control$state.track.strategy,
                  state_track_freq = control$state.track.freq,

                  criteria = criteria
                  )

}
