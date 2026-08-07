# -----------------------------------------------------------------------------
# Prior-weight specifications
# -----------------------------------------------------------------------------

.ecountgmifs.weight.prior.score <- function(
    prior,
    score = NULL,
    reference = NULL
) {
  if (length(prior) == 0L) {
    stop("`prior` must contain at least one value.", call. = FALSE)
  }

  if (anyNA(prior)) {
    stop("`prior` must not contain missing values.", call. = FALSE)
  }

  if (is.logical(prior)) {
    if (!is.null(score)) {
      stop(
        "`score` is unnecessary when `prior` is logical.",
        call. = FALSE
      )
    }

    return(as.numeric(prior))
  }

  if (is.numeric(prior) && is.null(score)) {
    prior <- as.numeric(prior)

    if (any(!is.finite(prior)) || any(prior < 0)) {
      stop(
        "Numeric `prior` values must be finite and non-negative.",
        call. = FALSE
      )
    }

    return(prior)
  }

  category <- as.character(prior)

  if (!is.null(reference)) {
    if (length(reference) != 1L || is.na(reference)) {
      stop("`reference` must be one non-missing category.", call. = FALSE)
    }

    if (!is.null(score)) {
      stop(
        "Supply either `score` or `reference`, not both.",
        call. = FALSE
      )
    }

    if (!as.character(reference) %in% category) {
      stop(
        "The requested `reference` category is absent from `prior`.",
        call. = FALSE
      )
    }

    return(as.numeric(category != as.character(reference)))
  }

  if (is.null(score)) {
    stop(
      paste0(
        "Categorical `prior` values require either a named numeric `score` ",
        "mapping or a `reference` category."
      ),
      call. = FALSE
    )
  }

  if (is.numeric(score) && length(score) == length(prior) && is.null(names(score))) {
    result <- as.numeric(score)
  } else {
    if (
        !is.numeric(score) ||
        is.null(names(score)) ||
        any(names(score) == "")
    ) {
      stop(
        paste0(
          "For categorical `prior`, `score` must be a named numeric vector ",
          "mapping every category to a non-negative score."
        ),
        call. = FALSE
      )
    }

    missing_category <- setdiff(
      unique(category),
      names(score)
    )

    if (length(missing_category) > 0L) {
      stop(
        "`score` is missing prior categories: ",
        paste(missing_category, collapse = ", "),
        call. = FALSE
      )
    }

    result <- unname(score[category])
  }

  if (anyNA(result) || any(!is.finite(result)) || any(result < 0)) {
    stop(
      "Prior scores must be finite and non-negative.",
      call. = FALSE
    )
  }

  as.numeric(result)
}


.ecountgmifs.weight.prior.labels <- function(eta) {
  paste0(
    "eta=",
    format(
      eta,
      digits = 15L,
      trim = TRUE,
      scientific = FALSE
    )
  )
}


#' Construct prior-informed penalty weights
#'
#' Creates one or more predictor-weight vectors from binary, ordinal, or
#' categorical prior information. For prior score `s[j]` and strength `eta`,
#' the raw penalty weight is `eta^(-s[j])`. The main [ecountgmifs()] wrapper
#' performs its usual mean-one normalization separately for every fitted model.
#'
#' Logical prior values are converted to scores 0 and 1. Non-negative numeric
#' prior values are used directly as scores, which supports ordinal evidence
#' levels. Character or factor priors require either a named `score` mapping or
#' a `reference` category; with `reference`, the reference receives score 0 and
#' every other category receives score 1.
#'
#' @param prior Logical, non-negative numeric, factor, or character vector with
#'   one entry per penalized predictor.
#' @param eta One or more positive prior-strength values. `eta = 1` gives equal
#'   raw weights. For a binary prior, supported predictors receive raw weight
#'   `1 / eta`.
#' @param score Optional score specification. For categorical priors, use a
#'   named numeric vector such as `c(none = 0, predicted = 1, validated = 2)`.
#'   An unnamed numeric vector of length `prior` is also accepted.
#' @param reference Optional reference category for an unscored categorical
#'   prior. It receives score 0 and all other categories receive score 1.
#' @param label Optional short label describing the prior source.
#'
#' @return An object of class `ecountgmifs_weight_prior`.
#' @export
ecountgmifs.weight.prior <- function(
    prior,
    eta = 1,
    score = NULL,
    reference = NULL,
    label = deparse(substitute(prior))
) {
  eta <- as.numeric(eta)

  if (
      length(eta) == 0L ||
      anyNA(eta) ||
      any(!is.finite(eta)) ||
      any(eta <= 0)
  ) {
    stop("`eta` must contain positive finite values.", call. = FALSE)
  }

  if (anyDuplicated(eta)) {
    warning(
      "Duplicated `eta` values were removed.",
      call. = FALSE
    )
    eta <- unique(eta)
  }

  prior_names <- names(prior)
  prior_score <- .ecountgmifs.weight.prior.score(
    prior = prior,
    score = score,
    reference = reference
  )

  if (!is.null(prior_names)) {
    names(prior_score) <- prior_names
  }

  weights <- lapply(
    eta,
    function(value) {
      result <- value^(-prior_score)
      names(result) <- prior_names
      result
    }
  )
  names(weights) <- .ecountgmifs.weight.prior.labels(eta)

  out <- list(
    call = match.call(),
    label = as.character(label)[[1L]],
    prior = prior,
    score = prior_score,
    eta = eta,
    eta_labels = .ecountgmifs.weight.prior.labels(eta),
    weights = weights
  )

  class(out) <- c("ecountgmifs_weight_prior", "list")
  out
}


.ecountgmifs.prepare.weight.prior <- function(
    object,
    predictor_names,
    predictor_count
) {
  if (!inherits(object, "ecountgmifs_weight_prior")) {
    stop("Expected an `ecountgmifs_weight_prior` object.", call. = FALSE)
  }

  score <- object$score

  if (
      !is.null(names(score)) &&
      !is.null(predictor_names)
  ) {
    missing_predictor <- setdiff(
      predictor_names,
      names(score)
    )

    if (length(missing_predictor) > 0L) {
      stop(
        "The weight prior is missing predictors: ",
        paste(utils::head(missing_predictor, 20L), collapse = ", "),
        if (length(missing_predictor) > 20L) " ..." else "",
        call. = FALSE
      )
    }

    score <- score[predictor_names]
  } else if (length(score) != predictor_count) {
    stop(
      "The weight prior must contain one score per penalized predictor.",
      call. = FALSE
    )
  }

  if (length(score) != predictor_count) {
    stop(
      "The aligned weight prior has the wrong number of predictors.",
      call. = FALSE
    )
  }

  weights <- lapply(
    object$eta,
    function(value) {
      result <- value^(-as.numeric(score))
      names(result) <- predictor_names
      result
    }
  )
  names(weights) <- object$eta_labels

  object$score <- as.numeric(score)
  names(object$score) <- predictor_names
  object$weights <- weights
  object
}


#' @export
print.ecountgmifs_weight_prior <- function(
    x,
    digits = max(3L, getOption("digits") - 3L),
    ...
) {
  cat("ecountgmifs prior-weight specification\n")
  cat("  Label:          ", x$label, "\n", sep = "")
  cat("  Predictors:     ", length(x$score), "\n", sep = "")
  cat("  Positive score: ", sum(x$score > 0), "\n", sep = "")
  cat(
    "  Eta values:     ",
    paste(format(x$eta, digits = digits), collapse = ", "),
    "\n",
    sep = ""
  )
  cat(
    "  Score range:    ",
    paste(format(range(x$score), digits = digits), collapse = " to "),
    "\n",
    sep = ""
  )

  invisible(x)
}
