#' Normalize count matrices used by ecountgmifs examples
#'
#' @description
#' Applies sample-level count normalization to a non-negative matrix. The
#' function is intended for the preprocessing strategies used in the
#' ecountGMIFS experiments: no normalization, counts per million (CPM), and
#' trimmed mean of M-values (TMM).
#'
#' For negative-binomial ecountGMIFS models, the response should generally be
#' kept on its original count scale and sample-level normalization should be
#' represented through an offset. Set \code{return.offset = TRUE} to return the
#' log normalization factor together with the normalized matrix.
#'
#' @param x numeric matrix or data frame. Count matrix.
#' @param method character. Normalization method. One of \code{"none"},
#'   \code{"cpm"}, or \code{"tmm"}.
#' @param orientation character. Matrix orientation. Use
#'   \code{"samples_by_rows"} when rows are samples and columns are variables,
#'   which is the orientation used by \code{ecountgmifs()}. Use
#'   \code{"features_by_rows"} for matrices stored as variables by samples.
#' @param return.offset logical. If \code{TRUE}, return a list with the
#'   normalized matrix and a log offset vector. If \code{FALSE}, return only the
#'   normalized matrix.
#'
#' @return If \code{return.offset = FALSE}, a numeric matrix. If
#'   \code{return.offset = TRUE}, a list with elements:
#' \describe{
#'   \item{\code{x}}{Normalized matrix in the same orientation as the input.}
#'   \item{\code{offset}}{Log normalization factor for each sample.}
#' }
#'
#' @details
#' CPM normalization divides each sample by its library size and multiplies by
#' one million. The returned offset is \code{log(library_size / 1e6)}.
#'
#' TMM normalization uses \pkg{edgeR} to estimate effective library sizes and
#' returns CPM values based on those effective library sizes. The returned
#' offset is \code{log(effective_library_size / 1e6)}. The \pkg{edgeR} package
#' is required only when \code{method = "tmm"}.
#'
#' @examples
#' x <- matrix(rpois(20, lambda = 10), nrow = 5, ncol = 4)
#'
#' x_cpm <- normalize_counts(x, method = "cpm")
#'
#' x_cpm_with_offset <- normalize_counts(
#'   x,
#'   method = "cpm",
#'   return.offset = TRUE
#' )
#'
#' @export
normalize_counts <- function(
    x,
    method = c("none", "cpm", "tmm"),
    orientation = c("samples_by_rows", "features_by_rows"),
    return.offset = FALSE
) {
  method <- match.arg(method)
  orientation <- match.arg(orientation)

  if (!is.logical(return.offset) ||
      length(return.offset) != 1L ||
      is.na(return.offset)) {
    stop("value of 'return.offset' must be TRUE or FALSE")
  }

  x <- as.matrix(x)
  storage.mode(x) <- "numeric"

  if (anyNA(x)) {
    stop("value of 'x' must not contain missing values")
  }

  if (any(x < 0)) {
    stop("value of 'x' must be non-negative")
  }

  if (orientation == "samples_by_rows") {
    counts <- t(x)
  } else {
    counts <- x
  }

  lib.size <- colSums(counts)
  safe.lib.size <- lib.size
  safe.lib.size[is.na(safe.lib.size) | safe.lib.size <= 0] <- 1

  if (method == "none") {
    out <- counts
    offset <- rep(0, ncol(counts))
  } else if (method == "cpm") {
    scale <- safe.lib.size / 1e6
    out <- sweep(counts, 2L, scale, "/")
    out[, lib.size <= 0] <- 0
    offset <- log(scale)
  } else if (method == "tmm") {
    if (!requireNamespace("edgeR", quietly = TRUE)) {
      stop("package 'edgeR' is required for method = 'tmm'", call. = FALSE)
    }

    dge <- edgeR::DGEList(counts = counts)
    dge <- edgeR::calcNormFactors(dge, method = "TMM")

    out <- edgeR::cpm(dge, log = FALSE)
    effective.lib.size <- dge$samples$lib.size * dge$samples$norm.factors
    effective.lib.size[is.na(effective.lib.size) | effective.lib.size <= 0] <- 1
    offset <- log(effective.lib.size / 1e6)
  } else {
    stop("unknown normalization method: ", method, call. = FALSE)
  }

  if (orientation == "samples_by_rows") {
    out <- t(out)
  }

  rownames(out) <- rownames(x)
  colnames(out) <- colnames(x)

  if (!return.offset) {
    return(out)
  }

  list(
    x = out,
    offset = offset
  )
}


#' Transform predictor matrices used by ecountgmifs examples
#'
#' @description
#' Applies predictor transformations used in the RNA-seq preprocessing grid:
#' shifted logarithm, inverse-hyperbolic-sine transformation, and optional
#' standardization.
#'
#' @param x numeric matrix or data frame. Predictor matrix.
#' @param method character. Transformation method. One of \code{"none"},
#'   \code{"log1p"}, \code{"asinh"}, \code{"standardize"},
#'   \code{"standardize_log1p"}, or \code{"standardize_asinh"}.
#' @param ref optional numeric matrix or data frame used to compute
#'   standardization means and standard deviations. If \code{NULL}, these are
#'   computed from \code{x}. This is useful when applying training-set scaling
#'   to test data.
#' @param orientation character. Matrix orientation. Use
#'   \code{"samples_by_rows"} when rows are samples and columns are variables,
#'   which is the orientation used by \code{ecountgmifs()}. Use
#'   \code{"features_by_rows"} for matrices stored as variables by samples.
#'
#' @return Numeric matrix with the same orientation and dimnames as the input.
#'
#' @details
#' The \code{"log1p"} transformation computes \code{log(1 + x)}. The
#' \code{"asinh"} transformation computes \code{asinh(x)}, which is log-like
#' for large values but remains approximately linear near zero. Standardization
#' centers and scales each predictor across samples. For
#' \code{"standardize_log1p"} and \code{"standardize_asinh"}, the transformation
#' is applied first and standardization is applied second.
#'
#' @examples
#' x <- matrix(rpois(20, lambda = 10), nrow = 5, ncol = 4)
#'
#' transform_predictors(x, method = "log1p")
#' transform_predictors(x, method = "standardize_asinh")
#'
#' @export
transform_predictors <- function(
    x,
    method = c(
      "none",
      "log1p",
      "asinh",
      "standardize",
      "standardize_log1p",
      "standardize_asinh"
    ),
    ref = NULL,
    orientation = c("samples_by_rows", "features_by_rows")
) {
  method <- match.arg(method)
  orientation <- match.arg(orientation)

  x_in <- as.matrix(x)
  storage.mode(x_in) <- "numeric"

  if (anyNA(x_in)) {
    stop("value of 'x' must not contain missing values")
  }

  if (orientation == "features_by_rows") {
    x_work <- t(x_in)
  } else {
    x_work <- x_in
  }

  if (is.null(ref)) {
    ref_in <- x_in
  } else {
    ref_in <- as.matrix(ref)
    storage.mode(ref_in) <- "numeric"

    if (!identical(dim(ref_in), dim(x_in))) {
      stop("value of 'ref' must have the same dimensions as 'x'")
    }

    if (anyNA(ref_in)) {
      stop("value of 'ref' must not contain missing values")
    }
  }

  if (orientation == "features_by_rows") {
    ref_work <- t(ref_in)
  } else {
    ref_work <- ref_in
  }

  transform_one <- function(z, method) {
    if (method %in% c("log1p", "standardize_log1p")) {
      if (any(z < 0)) {
        stop("log1p transformation requires non-negative values")
      }
      return(log1p(z))
    }

    if (method %in% c("asinh", "standardize_asinh")) {
      return(asinh(z))
    }

    z
  }

  x_trans <- transform_one(x_work, method)
  ref_trans <- transform_one(ref_work, method)

  if (method %in% c("standardize", "standardize_log1p", "standardize_asinh")) {
    mu <- colMeans(ref_trans, na.rm = TRUE)
    sdv <- apply(ref_trans, 2L, stats::sd, na.rm = TRUE)
    sdv[is.na(sdv) | sdv == 0] <- 1

    x_trans <- sweep(x_trans, 2L, mu, "-")
    x_trans <- sweep(x_trans, 2L, sdv, "/")
  }

  x_trans[is.na(x_trans)] <- 0

  if (orientation == "features_by_rows") {
    out <- t(x_trans)
  } else {
    out <- x_trans
  }

  rownames(out) <- rownames(x_in)
  colnames(out) <- colnames(x_in)

  out
}


#' Transform monthly airport time-series predictors
#'
#' @description
#' Applies temporal transformations used in the T-100 airport preprocessing
#' grid. These transformations are designed for matrices whose rows are ordered
#' monthly observations and whose columns are airport-level predictor series.
#'
#' @param x numeric matrix or data frame. Rows are ordered monthly observations
#'   and columns are variables.
#' @param method character. Temporal transformation method. One of
#'   \code{"none"}, \code{"log1p_month_demean"}, \code{"month_z"},
#'   \code{"log1p_month_z"}, \code{"log1p_gaussian_smooth"},
#'   \code{"log1p_gaussian_smooth_month_demean"}, or \code{"log1p_diff1"}.
#' @param month optional integer, character, or factor vector identifying the
#'   calendar month for each row of \code{x}. If \code{NULL}, months are assumed
#'   to repeat as \code{1, 2, ..., 12}.
#' @param gaussian.variance positive numeric. Variance parameter used by the
#'   Gaussian smoothing kernel.
#'
#' @return Numeric matrix with the same dimensions and dimnames as \code{x}.
#'
#' @details
#' Month demeaning removes recurring calendar-month baselines. Month-wise
#' standardization additionally divides by the month-specific standard
#' deviation. Gaussian smoothing reduces local temporal noise by multiplying
#' each series by a normalized Gaussian kernel over time. The first-order
#' log-difference computes differences of adjacent \code{log1p}-transformed
#' observations while keeping the original number of rows by setting the first
#' difference to zero.
#'
#' @examples
#' x <- matrix(rpois(24 * 3, lambda = 100), nrow = 24, ncol = 3)
#'
#' x_md <- transform_time_series(
#'   x,
#'   method = "log1p_month_demean"
#' )
#'
#' x_diff <- transform_time_series(
#'   x,
#'   method = "log1p_diff1"
#' )
#'
#' @export
transform_time_series <- function(
    x,
    method = c(
      "none",
      "log1p_month_demean",
      "month_z",
      "log1p_month_z",
      "log1p_gaussian_smooth",
      "log1p_gaussian_smooth_month_demean",
      "log1p_diff1"
    ),
    month = NULL,
    gaussian.variance = 1
) {
  method <- match.arg(method)

  x <- as.matrix(x)
  storage.mode(x) <- "numeric"

  if (anyNA(x)) {
    stop("value of 'x' must not contain missing values")
  }

  if (any(x < 0) && grepl("log1p", method, fixed = TRUE)) {
    stop("log1p-based temporal transformations require non-negative values")
  }

  if (!is.numeric(gaussian.variance) ||
      length(gaussian.variance) != 1L ||
      is.na(gaussian.variance) ||
      gaussian.variance <= 0) {
    stop("value of 'gaussian.variance' must be a positive number")
  }

  n_time <- nrow(x)

  if (is.null(month)) {
    month <- rep(seq_len(12L), length.out = n_time)
  }

  if (length(month) != n_time) {
    stop("value of 'month' must have length equal to nrow(x)")
  }

  month <- factor(month)

  month_demean <- function(z) {
    out <- z

    for (m in levels(month)) {
      idx <- which(month == m)

      if (length(idx) > 0L) {
        mu <- colMeans(z[idx, , drop = FALSE], na.rm = TRUE)
        out[idx, ] <- sweep(z[idx, , drop = FALSE], 2L, mu, "-")
      }
    }

    out
  }

  month_z <- function(z) {
    out <- z

    for (m in levels(month)) {
      idx <- which(month == m)

      if (length(idx) > 1L) {
        mu <- colMeans(z[idx, , drop = FALSE], na.rm = TRUE)
        sdv <- apply(z[idx, , drop = FALSE], 2L, stats::sd, na.rm = TRUE)
        sdv[is.na(sdv) | sdv == 0] <- 1

        out[idx, ] <- sweep(z[idx, , drop = FALSE], 2L, mu, "-")
        out[idx, ] <- sweep(out[idx, , drop = FALSE], 2L, sdv, "/")
      }
    }

    out
  }

  gaussian_smooth <- function(z, variance) {
    time_id <- seq_len(nrow(z))
    d <- outer(time_id, time_id, "-")
    w <- exp(-(d^2) / (2 * variance))
    w <- sweep(w, 2L, colSums(w), "/")
    w[is.na(w)] <- 0
    z %*% w
  }

  if (method == "none") {
    out <- x
  } else if (method == "log1p_month_demean") {
    out <- month_demean(log1p(x))
  } else if (method == "month_z") {
    out <- month_z(x)
  } else if (method == "log1p_month_z") {
    out <- month_z(log1p(x))
  } else if (method == "log1p_gaussian_smooth") {
    out <- gaussian_smooth(log1p(x), gaussian.variance)
  } else if (method == "log1p_gaussian_smooth_month_demean") {
    out <- month_demean(gaussian_smooth(log1p(x), gaussian.variance))
  } else if (method == "log1p_diff1") {
    x_log <- log1p(x)
    out <- matrix(0, nrow = nrow(x_log), ncol = ncol(x_log))
    if (nrow(x_log) > 1L) {
      out[-1L, ] <- x_log[-1L, , drop = FALSE] -
        x_log[-nrow(x_log), , drop = FALSE]
    }
  } else {
    stop("unknown temporal transformation method: ", method, call. = FALSE)
  }

  out[is.na(out)] <- 0
  rownames(out) <- rownames(x)
  colnames(out) <- colnames(x)

  out
}
