#' Add the ecountgmifs class and plotting metadata to a raw fit result
#'
#' This helper is useful for objects returned directly by `ecountgmifs_cpp()`
#' during development. Objects returned by [ecountgmifs()] are already
#' decorated.
#'
#' @param x A list in the current `ecountgmifs_cpp()` output format.
#' @param predictor.names Optional names for penalized predictors.
#' @param unpenalized.names Optional names for unpenalized predictors.
#' @param weight.vec Optional elastic-net prior-weight vector.
#' @param call Optional call stored in the object.
#'
#' @return `x` with class `ecountgmifs` and supplied metadata.
#' @export
as.ecountgmifs <- function(
    x,
    predictor.names = NULL,
    unpenalized.names = NULL,
    weight.vec = NULL,
    call = NULL
) {
  if (!is.list(x) || is.null(x$input) || is.null(x$path)) {
    stop(
      "`x` is not in the expected ecountgmifs result format.",
      call. = FALSE
    )
  }

  if (
      !is.null(predictor.names) &&
      length(predictor.names) != x$input$p
  ) {
    stop(
      "`predictor.names` must have length `x$input$p`.",
      call. = FALSE
    )
  }

  if (
      !is.null(unpenalized.names) &&
      length(unpenalized.names) != x$input$q
  ) {
    stop(
      "`unpenalized.names` must have length `x$input$q`.",
      call. = FALSE
    )
  }

  if (!is.null(weight.vec)) {
    weight.vec <- as.numeric(weight.vec)

    if (
        length(weight.vec) != x$input$p ||
        anyNA(weight.vec) ||
        any(!is.finite(weight.vec)) ||
        any(weight.vec <= 0)
    ) {
      stop(
        "`weight.vec` must contain `x$input$p` positive finite values.",
        call. = FALSE
      )
    }

    x$input$weight_vec <- weight.vec
    x$input$has_prior <- length(unique(weight.vec)) > 1L
  }

  x <- .ecountgmifs.assign.parameter.names(
    object = x,
    predictor.names = predictor.names,
    unpenalized.names = unpenalized.names
  )

  if (!is.null(call)) {
    x$call <- call
  }

  class(x) <- unique(c("ecountgmifs", class(x)))
  x
}


.ecountgmifs.path.x <- function(
    object,
    xvar = c("iteration", "l1", "pseudo_r2")
) {
  xvar <- match.arg(xvar)
  states <- .ecountgmifs.path.states(object)

  switch(
    xvar,
    iteration = as.numeric(states$iteration),
    l1 = rowSums(abs(.ecountgmifs.beta.matrix(object))),
    pseudo_r2 = as.numeric(states$pseudo_r2)
  )
}


.ecountgmifs.state.x <- function(
    state,
    xvar = c("iteration", "l1", "pseudo_r2")
) {
  xvar <- match.arg(xvar)

  switch(
    xvar,
    iteration = as.numeric(state$iteration),
    l1 = sum(abs(state$predictors$parameters$beta)),
    pseudo_r2 = as.numeric(state$pseudo_r2)
  )
}


.ecountgmifs.x.label <- function(xvar) {
  switch(
    xvar,
    iteration = "Iteration",
    l1 = "L1 norm of beta",
    pseudo_r2 = "Pseudo-R2"
  )
}


.ecountgmifs.selection.styles <- function(n) {
  if (n == 0L) {
    return(
      list(
        col = integer(),
        pch = integer()
      )
    )
  }

  list(
    col = rep_len(seq_len(max(1L, min(n, 8L))) + 1L, n),
    pch = rep_len(c(21L, 22L, 23L, 24L, 25L), n)
  )
}


.ecountgmifs.alpha.colors <- function(
    colors,
    alpha,
    argument
) {
  if (length(colors) == 0L) {
    return(character())
  }

  if (
      length(alpha) != 1L ||
      is.na(alpha) ||
      !is.finite(alpha) ||
      alpha < 0 ||
      alpha > 1
  ) {
    stop(
      "`", argument, "` must be one finite number in [0, 1].",
      call. = FALSE
    )
  }

  grDevices::adjustcolor(
    colors,
    alpha.f = alpha
  )
}


.ecountgmifs.plot.coefficients <- function(
    object,
    predictors = NULL,
    active.only = TRUE,
    xvar = c("iteration", "l1", "pseudo_r2"),
    criteria = NULL,
    show.criteria = TRUE,
    selection.cex = 1.8,
    selection.lwd = 1.5,
    selection.line.alpha = 0.65,
    selection.point.alpha = 0.9,
    selection.points = c("active", "all", "none"),
    selection.labels = TRUE,
    label = FALSE,
    label.n = 12L,
    label.cex = 0.7,
    zero.tol = sqrt(.Machine$double.eps),
    xlab = NULL,
    ylab = "Coefficient",
    main = "ecountgmifs coefficient paths",
    lty = 1,
    lwd = 1,
    line.alpha = 0.35,
    col = NULL,
    ...
) {
  xvar <- match.arg(xvar)
  selection.points <- match.arg(selection.points)

  beta_all <- .ecountgmifs.beta.matrix(object)
  x <- .ecountgmifs.path.x(object, xvar)

  if (is.null(predictors)) {
    plotted_indices <- seq_len(ncol(beta_all))

    if (isTRUE(active.only)) {
      plotted_indices <- which(
        apply(
          abs(beta_all) > zero.tol,
          2L,
          any
        )
      )
    }
  } else if (is.character(predictors)) {
    missing_predictors <- setdiff(
      predictors,
      colnames(beta_all)
    )

    if (length(missing_predictors) > 0L) {
      stop(
        "Unknown predictor name(s): ",
        paste(missing_predictors, collapse = ", "),
        call. = FALSE
      )
    }

    plotted_indices <- match(
      predictors,
      colnames(beta_all)
    )
  } else {
    plotted_indices <- as.integer(predictors)

    if (
        anyNA(plotted_indices) ||
        any(plotted_indices < 1L) ||
        any(plotted_indices > ncol(beta_all))
    ) {
      stop(
        "Numeric `predictors` must contain valid column indices.",
        call. = FALSE
      )
    }
  }

  if (length(plotted_indices) == 0L) {
    graphics::plot.new()
    graphics::title(
      main = main,
      sub = "No coefficient became active in the saved path"
    )
    return(invisible(object))
  }

  beta <- beta_all[, plotted_indices, drop = FALSE]

  if (length(x) != nrow(beta)) {
    stop(
      "Saved path dimensions are inconsistent.",
      call. = FALSE
    )
  }

  if (is.null(xlab)) {
    xlab <- .ecountgmifs.x.label(xvar)
  }

  if (is.null(col)) {
    col <- rep_len(seq_len(8L), ncol(beta))
  } else {
    col <- rep_len(col, ncol(beta))
  }

  path_col <- .ecountgmifs.alpha.colors(
    col,
    line.alpha,
    "line.alpha"
  )

  graphics::matplot(
    x,
    beta,
    type = "l",
    lty = lty,
    lwd = lwd,
    col = path_col,
    xlab = xlab,
    ylab = ylab,
    main = main,
    ...
  )

  if (isTRUE(show.criteria)) {
    criterion_names <- .ecountgmifs.match.criteria(
      object,
      criteria
    )

    styles <- .ecountgmifs.selection.styles(
      length(criterion_names)
    )

    selection_line_col <- .ecountgmifs.alpha.colors(
      styles$col,
      selection.line.alpha,
      "selection.line.alpha"
    )

    selection_point_col <- .ecountgmifs.alpha.colors(
      styles$col,
      selection.point.alpha,
      "selection.point.alpha"
    )

    # par("usr") is c(x_min, x_max, y_min, y_max).
    # The previous implementation sliced elements 3:4 and then
    # incorrectly requested element 4 of that two-element slice.
    label_y <- graphics::par("usr")[[4L]]

    for (i in seq_along(criterion_names)) {
      criterion_name <- criterion_names[[i]]
      selected <- object$path$best_criteria[[criterion_name]]$state
      selected_beta <- as.numeric(
        selected$predictors$parameters$beta
      )[plotted_indices]
      selected_x <- .ecountgmifs.state.x(
        selected,
        xvar
      )

      graphics::abline(
        v = selected_x,
        col = selection_line_col[[i]],
        lty = 3L,
        lwd = selection.lwd
      )

      point_indices <- switch(
        selection.points,
        active = which(abs(selected_beta) > zero.tol),
        all = seq_along(selected_beta),
        none = integer()
      )

      if (length(point_indices) > 0L) {
        graphics::points(
          rep(selected_x, length(point_indices)),
          selected_beta[point_indices],
          pch = styles$pch[[i]],
          cex = selection.cex,
          col = styles$col[[i]],
          bg = selection_point_col[[i]],
          lwd = selection.lwd
        )
      }

      if (isTRUE(selection.labels)) {
        graphics::text(
          x = selected_x,
          y = label_y,
          labels = criterion_name,
          pos = 2L + (i %% 2L),
          cex = 0.65,
          col = styles$col[[i]],
          xpd = NA
        )
      }
    }

    if (length(criterion_names) > 0L) {
      graphics::legend(
        "topleft",
        legend = criterion_names,
        col = styles$col,
        pch = styles$pch,
        pt.bg = selection_point_col,
        pt.cex = selection.cex,
        lty = 3L,
        lwd = selection.lwd,
        title = "Criterion selections",
        bty = "n",
        cex = 0.8
      )
    }
  }

  if (isTRUE(label) && ncol(beta) > 0L) {
    terminal_beta <- beta[nrow(beta), ]
    selected <- order(
      abs(terminal_beta),
      decreasing = TRUE
    )
    selected <- utils::head(selected, label.n)

    graphics::text(
      x = rep(x[[length(x)]], length(selected)),
      y = terminal_beta[selected],
      labels = colnames(beta)[selected],
      pos = 4L,
      cex = label.cex,
      col = col[selected],
      xpd = NA
    )
  }

  invisible(object)
}


.ecountgmifs.plot.criteria <- function(
    object,
    xvar = c("iteration", "l1", "pseudo_r2"),
    criteria = NULL,
    standardize = FALSE,
    selection.cex = 1.8,
    selection.lwd = 1.5,
    selection.labels = TRUE,
    xlab = NULL,
    ylab = NULL,
    xlim = NULL,
    ylim = NULL,
    main = "ecountgmifs criterion paths",
    lty = 1,
    lwd = 1.5,
    col = NULL,
    ...
) {
  xvar <- match.arg(xvar)
  values <- .ecountgmifs.criteria.path(object)

  if (ncol(values) == 0L) {
    stop(
      "No criteria are stored in this fit.",
      call. = FALSE
    )
  }

  criterion_names <- .ecountgmifs.match.criteria(
    object,
    criteria
  )

  values <- values[, criterion_names, drop = FALSE]
  plotted_values <- values

  if (isTRUE(standardize)) {
    plotted_values <- apply(
      values,
      2L,
      function(value) {
        value_range <- range(value, finite = TRUE)

        if (
            !all(is.finite(value_range)) ||
            diff(value_range) == 0
        ) {
          return(rep(0, length(value)))
        }

        (value - value_range[[1L]]) /
          diff(value_range)
      }
    )

    if (is.null(dim(plotted_values))) {
      plotted_values <- matrix(
        plotted_values,
        ncol = 1L
      )
    }

    colnames(plotted_values) <- criterion_names
  }

  x <- .ecountgmifs.path.x(object, xvar)

  if (is.null(xlab)) {
    xlab <- .ecountgmifs.x.label(xvar)
  }

  if (is.null(ylab)) {
    ylab <- if (isTRUE(standardize)) {
      "Criterion value scaled to [0, 1]"
    } else {
      "Criterion value"
    }
  }

  if (is.null(col)) {
    col <- seq_along(criterion_names) + 1L
  } else {
    col <- rep_len(col, length(criterion_names))
  }

  styles <- .ecountgmifs.selection.styles(
    length(criterion_names)
  )
  styles$col <- col

  best_x <- vapply(
    criterion_names,
    function(criterion_name) {
      .ecountgmifs.state.x(
        object$path$best_criteria[[criterion_name]]$state,
        xvar
      )
    },
    numeric(1L)
  )

  best_y <- vapply(
    criterion_names,
    function(criterion_name) {
      selected_y <- as.numeric(
        object$path$best_criteria[[criterion_name]]$value
      )

      if (!isTRUE(standardize)) {
        return(selected_y)
      }

      original <- values[, criterion_name]
      original_range <- range(original, finite = TRUE)

      if (
          all(is.finite(original_range)) &&
          diff(original_range) > 0
      ) {
        return(
          (selected_y - original_range[[1L]]) /
            diff(original_range)
        )
      }

      0
    },
    numeric(1L)
  )

  if (is.null(xlim)) {
    xlim <- range(c(x, best_x), finite = TRUE)
  }

  if (is.null(ylim)) {
    ylim <- range(c(plotted_values, best_y), finite = TRUE)
  }

  graphics::matplot(
    x,
    plotted_values,
    type = "l",
    lty = lty,
    lwd = lwd,
    col = col,
    xlim = xlim,
    ylim = ylim,
    xlab = xlab,
    ylab = ylab,
    main = main,
    ...
  )

  y_range <- graphics::par("usr")[3:4]

  for (i in seq_along(criterion_names)) {
    criterion_name <- criterion_names[[i]]
    best <- object$path$best_criteria[[criterion_name]]
    selected_x <- best_x[[i]]
    selected_y <- best_y[[i]]

    graphics::points(
      selected_x,
      selected_y,
      pch = styles$pch[[i]],
      cex = selection.cex,
      col = styles$col[[i]],
      bg = "white",
      lwd = selection.lwd
    )

    graphics::abline(
      v = selected_x,
      col = styles$col[[i]],
      lty = 3L,
      lwd = selection.lwd
    )

    if (isTRUE(selection.labels)) {
      graphics::text(
        selected_x,
        selected_y,
        labels = criterion_name,
        pos = if (selected_y > mean(y_range)) 1L else 3L,
        cex = 0.7,
        col = styles$col[[i]],
        xpd = NA
      )
    }
  }

  graphics::legend(
    "topright",
    legend = criterion_names,
    col = styles$col,
    lty = lty,
    lwd = lwd,
    pch = styles$pch,
    pt.bg = "white",
    pt.cex = selection.cex,
    bty = "n",
    cex = 0.8
  )

  invisible(object)
}


.ecountgmifs.align.binary.vector <- function(
    value,
    predictor_names,
    name
) {
  original_names <- names(value)

  if (!is.null(original_names)) {
    if (anyDuplicated(original_names)) {
      stop(
        "Named `",
        name,
        "` contains duplicated predictor names.",
        call. = FALSE
      )
    }

    missing <- setdiff(
      predictor_names,
      original_names
    )

    if (length(missing) > 0L) {
      stop(
        "Named `",
        name,
        "` is missing predictors: ",
        paste(utils::head(missing, 10L), collapse = ", "),
        call. = FALSE
      )
    }

    value <- value[predictor_names]
  }

  if (length(value) != length(predictor_names)) {
    stop(
      "`",
      name,
      "` must have one value per penalized predictor.",
      call. = FALSE
    )
  }

  if (is.logical(value)) {
    if (anyNA(value)) {
      stop(
        "`",
        name,
        "` must not contain NA.",
        call. = FALSE
      )
    }

    return(as.logical(value))
  }

  if (
      !is.numeric(value) ||
      anyNA(value) ||
      any(!is.finite(value)) ||
      any(!value %in% c(0, 1))
  ) {
    stop(
      "`",
      name,
      "` must be logical or contain only 0 and 1.",
      call. = FALSE
    )
  }

  as.logical(value)
}


.ecountgmifs.safe.ratio <- function(
    numerator,
    denominator,
    zero.division
) {
  if (denominator != 0) {
    return(numerator / denominator)
  }

  if (identical(zero.division, "zero")) {
    return(0)
  }

  if (identical(zero.division, "one")) {
    return(1)
  }

  NA_real_
}


.ecountgmifs.binary.metrics <- function(
    selected,
    ground_truth,
    zero.division = c("zero", "one", "NA")
) {
  zero.division <- match.arg(zero.division)

  selected <- as.logical(selected)
  ground_truth <- as.logical(ground_truth)

  true_positive <- sum(selected & ground_truth)
  false_positive <- sum(selected & !ground_truth)
  false_negative <- sum(!selected & ground_truth)
  true_negative <- sum(!selected & !ground_truth)

  precision <- .ecountgmifs.safe.ratio(
    true_positive,
    true_positive + false_positive,
    zero.division
  )

  recall <- .ecountgmifs.safe.ratio(
    true_positive,
    true_positive + false_negative,
    zero.division
  )

  f1 <- if (
      is.na(precision) ||
      is.na(recall)
  ) {
    NA_real_
  } else {
    .ecountgmifs.safe.ratio(
      2 * precision * recall,
      precision + recall,
      zero.division
    )
  }

  specificity <- .ecountgmifs.safe.ratio(
    true_negative,
    true_negative + false_positive,
    zero.division
  )

  accuracy <- .ecountgmifs.safe.ratio(
    true_positive + true_negative,
    length(selected),
    zero.division
  )

  data.frame(
    selected = sum(selected),
    true = sum(ground_truth),
    tp = true_positive,
    fp = false_positive,
    fn = false_negative,
    tn = true_negative,
    precision = precision,
    recall = recall,
    F1 = f1,
    specificity = specificity,
    accuracy = accuracy,
    stringsAsFactors = FALSE
  )
}


.ecountgmifs.metric.for.state <- function(
    object,
    state,
    ground.truth,
    zero.tol,
    zero.division
) {
  beta <- as.numeric(
    state$predictors$parameters$beta
  )

  result <- .ecountgmifs.binary.metrics(
    selected = abs(beta) > zero.tol,
    ground_truth = ground.truth,
    zero.division = zero.division
  )

  result$iteration <- as.integer(state$iteration)
  result$l1 <- sum(abs(beta))
  result$pseudo_r2 <- as.numeric(state$pseudo_r2)
  result$negloglik <- as.numeric(state$negloglik)

  result[, c(
    "iteration",
    "l1",
    "pseudo_r2",
    "negloglik",
    "selected",
    "true",
    "tp",
    "fp",
    "fn",
    "tn",
    "precision",
    "recall",
    "F1",
    "specificity",
    "accuracy"
  )]
}


#' Calculate pathwise variable-selection metrics
#'
#' Coefficients with `abs(beta) > zero.tol` are treated as selected. By default,
#' a prior baseline is formed from `weight_vec < prior.cutoff`, because smaller
#' elastic-net weights receive less penalization. Supply `prior.selected`
#' directly when another definition is required.
#'
#' @param object An `ecountgmifs` fit.
#' @param ground.truth Logical or binary ground-truth vector with one element per
#'   penalized predictor. A named vector is aligned to predictor names.
#' @param zero.tol Coefficient selection tolerance.
#' @param zero.division Value used for undefined precision/recall ratios.
#' @param prior.selected Optional explicit logical or binary prior selection.
#' @param prior.weight.vec Optional prior-weight vector. Defaults to the vector
#'   stored in the fit.
#' @param prior.cutoff Weight cutoff defining prior selection.
#' @param prior.direction Whether weights below or above the cutoff indicate a
#'   prior-selected predictor.
#' @param ... Unused; accepted so plotting arguments can pass through
#'   `plot.ecountgmifs(type = "metrics")`.
#'
#' @return An object of class `ecountgmifs.metrics`.
#' @export
ecountgmifs.metrics <- function(
    object,
    ground.truth,
    zero.tol = sqrt(.Machine$double.eps),
    zero.division = c("zero", "one", "NA"),
    prior.selected = NULL,
    prior.weight.vec = NULL,
    prior.cutoff = 1,
    prior.direction = c("lower", "higher"),
    ...
) {
  zero.division <- match.arg(zero.division)
  prior.direction <- match.arg(prior.direction)

  if (
      !is.numeric(zero.tol) ||
      length(zero.tol) != 1L ||
      is.na(zero.tol) ||
      !is.finite(zero.tol) ||
      zero.tol < 0
  ) {
    stop(
      "`zero.tol` must be one non-negative finite number.",
      call. = FALSE
    )
  }

  beta <- .ecountgmifs.beta.matrix(object)
  predictor_names <- colnames(beta)

  ground_truth <- .ecountgmifs.align.binary.vector(
    ground.truth,
    predictor_names,
    "ground.truth"
  )

  states <- .ecountgmifs.path.states(object)

  path_rows <- lapply(
    seq_len(nrow(beta)),
    function(index) {
      state <- list(
        predictors = list(
          parameters = list(
            beta = beta[index, ]
          )
        ),
        iteration = states$iteration[[index]],
        pseudo_r2 = states$pseudo_r2[[index]],
        negloglik = states$negloglik[[index]]
      )

      .ecountgmifs.metric.for.state(
        object = object,
        state = state,
        ground.truth = ground_truth,
        zero.tol = zero.tol,
        zero.division = zero.division
      )
    }
  )

  path_metrics <- do.call(
    rbind,
    path_rows
  )

  rownames(path_metrics) <- rownames(beta)

  best <- .ecountgmifs.best.criteria(object)

  criterion_metrics <- if (length(best) == 0L) {
    data.frame()
  } else {
    do.call(
      rbind,
      lapply(
        names(best),
        function(name) {
          row <- .ecountgmifs.metric.for.state(
            object = object,
            state = best[[name]]$state,
            ground.truth = ground_truth,
            zero.tol = zero.tol,
            zero.division = zero.division
          )

          row$criterion <- name
          row$criterion_value <- as.numeric(
            best[[name]]$value
          )

          row[, c(
            "criterion",
            "criterion_value",
            setdiff(
              names(row),
              c("criterion", "criterion_value")
            )
          )]
        }
      )
    )
  }

  prior_metrics <- NULL
  resolved_prior <- NULL

  if (!is.null(prior.selected)) {
    resolved_prior <- .ecountgmifs.align.binary.vector(
      prior.selected,
      predictor_names,
      "prior.selected"
    )
  } else {
    if (
        is.null(prior.weight.vec) &&
        isTRUE(object$input$has_prior)
    ) {
      prior.weight.vec <- object$input$weight_vec
    }

    if (!is.null(prior.weight.vec)) {
      prior.weight.vec <- as.numeric(prior.weight.vec)

      if (
          length(prior.weight.vec) != length(predictor_names) ||
          anyNA(prior.weight.vec) ||
          any(!is.finite(prior.weight.vec))
      ) {
        stop(
          "`prior.weight.vec` must contain one finite value per predictor.",
          call. = FALSE
        )
      }

      resolved_prior <- if (identical(prior.direction, "lower")) {
        prior.weight.vec < prior.cutoff
      } else {
        prior.weight.vec > prior.cutoff
      }
    }
  }

  if (!is.null(resolved_prior)) {
    prior_metrics <- .ecountgmifs.binary.metrics(
      selected = resolved_prior,
      ground_truth = ground_truth,
      zero.division = zero.division
    )

    prior_metrics$definition <- if (!is.null(prior.selected)) {
      "explicit prior.selected"
    } else {
      paste0(
        "weight_vec ",
        if (identical(prior.direction, "lower")) "<" else ">",
        " ",
        format(prior.cutoff, digits = 8L)
      )
    }

    prior_metrics <- prior_metrics[, c(
      "definition",
      setdiff(names(prior_metrics), "definition")
    )]
  }

  out <- list(
    call = match.call(),
    path = path_metrics,
    criteria = criterion_metrics,
    prior = prior_metrics,
    prior_selected = resolved_prior,
    ground_truth = ground_truth,
    predictor_names = predictor_names,
    settings = list(
      zero_tol = zero.tol,
      zero_division = zero.division,
      prior_cutoff = prior.cutoff,
      prior_direction = prior.direction
    )
  )

  class(out) <- "ecountgmifs.metrics"
  out
}


#' @export
print.ecountgmifs.metrics <- function(
    x,
    digits = max(3L, getOption("digits") - 3L),
    ...
) {
  summary_value <- summary(x)
  print(summary_value, digits = digits, ...)
  invisible(x)
}


#' @export
summary.ecountgmifs.metrics <- function(object, ...) {
  path <- object$path

  best_f1_index <- if (
      nrow(path) == 0L ||
      all(is.na(path$F1))
  ) {
    NA_integer_
  } else {
    which.max(replace(path$F1, is.na(path$F1), -Inf))
  }

  best_precision_index <- if (
      nrow(path) == 0L ||
      all(is.na(path$precision))
  ) {
    NA_integer_
  } else {
    which.max(replace(path$precision, is.na(path$precision), -Inf))
  }

  best_recall_index <- if (
      nrow(path) == 0L ||
      all(is.na(path$recall))
  ) {
    NA_integer_
  } else {
    which.max(replace(path$recall, is.na(path$recall), -Inf))
  }

  select_row <- function(index) {
    if (is.na(index)) {
      return(data.frame())
    }

    path[index, , drop = FALSE]
  }

  out <- list(
    best_f1 = select_row(best_f1_index),
    best_precision = select_row(best_precision_index),
    best_recall = select_row(best_recall_index),
    criteria = object$criteria,
    prior = object$prior,
    truth_count = sum(object$ground_truth),
    predictor_count = length(object$ground_truth),
    settings = object$settings
  )

  class(out) <- "summary.ecountgmifs.metrics"
  out
}


#' @export
print.summary.ecountgmifs.metrics <- function(
    x,
    digits = max(3L, getOption("digits") - 3L),
    ...
) {
  cat("Variable-selection metrics for an ecountgmifs path\n")
  cat("Predictors:      ", x$predictor_count, "\n", sep = "")
  cat("True positives:  ", x$truth_count, "\n", sep = "")
  cat("Zero tolerance:  ", format(x$settings$zero_tol, digits = digits), "\n", sep = "")

  print_best <- function(value, label) {
    cat("\n", label, ":\n", sep = "")

    if (nrow(value) == 0L) {
      cat("  Not available.\n")
    } else {
      print(
        value[, c(
          "iteration",
          "selected",
          "tp",
          "fp",
          "fn",
          "precision",
          "recall",
          "F1"
        ), drop = FALSE],
        row.names = FALSE,
        digits = digits
      )
    }
  }

  print_best(x$best_f1, "Best path F1")
  print_best(x$best_precision, "Best path precision")
  print_best(x$best_recall, "Best path recall")

  if (nrow(x$criteria) > 0L) {
    cat("\nCriterion-selected states:\n")
    print(
      x$criteria[, c(
        "criterion",
        "iteration",
        "selected",
        "tp",
        "fp",
        "fn",
        "precision",
        "recall",
        "F1"
      ), drop = FALSE],
      row.names = FALSE,
      digits = digits
    )
  }

  if (!is.null(x$prior)) {
    cat("\nPrior-weight baseline:\n")
    print(
      x$prior[, c(
        "definition",
        "selected",
        "tp",
        "fp",
        "fn",
        "precision",
        "recall",
        "F1"
      ), drop = FALSE],
      row.names = FALSE,
      digits = digits
    )
  }

  invisible(x)
}


#' Plot pathwise F1, precision, recall, and related metrics
#'
#' Criterion-selected states are drawn as adjustable points. When a prior
#' baseline is available, each plotted metric receives a dashed horizontal line
#' at the value obtained by thresholding `weight_vec` (or by the explicit
#' `prior.selected` vector used to build the object).
#'
#' @param x An `ecountgmifs.metrics` object.
#' @param metrics Metrics to plot.
#' @param xvar Horizontal path coordinate.
#' @param show.criteria Draw criterion-selected points.
#' @param criteria Optional subset of criterion names to mark.
#' @param show.prior Draw prior-weight horizontal baselines.
#' @param criterion.cex Criterion point size.
#' @param criterion.pch Criterion point symbols.
#' @param criterion.lwd Criterion point and vertical-line width.
#' @param label.criteria Label criterion points by name.
#' @param xlab,ylab,main Axis and title labels.
#' @param lty,lwd,col Path line styling.
#' @param prior.lty,prior.lwd Prior-baseline styling.
#' @param ... Additional arguments passed to [graphics::matplot()].
#'
#' @return `x` invisibly.
#' @export
plot.ecountgmifs.metrics <- function(
    x,
    metrics = c("F1", "precision", "recall"),
    xvar = c("iteration", "l1", "pseudo_r2"),
    show.criteria = TRUE,
    criteria = NULL,
    show.prior = TRUE,
    criterion.cex = 1.8,
    criterion.pch = c(21L, 22L, 23L, 24L, 25L),
    criterion.lwd = 1.5,
    label.criteria = FALSE,
    xlab = NULL,
    ylab = "Selection metric",
    main = "ecountgmifs selection metrics",
    lty = 1,
    lwd = 2,
    col = NULL,
    prior.lty = 2,
    prior.lwd = 1.5,
    ...
) {
  xvar <- match.arg(xvar)

  available_metrics <- c(
    "F1",
    "precision",
    "recall",
    "specificity",
    "accuracy"
  )

  metrics <- match.arg(
    metrics,
    available_metrics,
    several.ok = TRUE
  )

  path <- x$path

  if (nrow(path) == 0L) {
    stop(
      "No path states are available in this metrics object.",
      call. = FALSE
    )
  }

  if (is.null(xlab)) {
    xlab <- .ecountgmifs.x.label(xvar)
  }

  if (is.null(col)) {
    col <- seq_along(metrics) + 1L
  } else {
    col <- rep_len(col, length(metrics))
  }

  y <- as.matrix(
    path[, metrics, drop = FALSE]
  )

  graphics::matplot(
    path[[xvar]],
    y,
    type = "l",
    lty = lty,
    lwd = lwd,
    col = col,
    ylim = c(0, 1),
    xlab = xlab,
    ylab = ylab,
    main = main,
    ...
  )

  metric_legend <- metrics
  metric_legend_col <- col
  metric_legend_lty <- rep(lty, length(metrics))
  metric_legend_lwd <- rep(lwd, length(metrics))

  if (isTRUE(show.prior) && !is.null(x$prior)) {
    for (i in seq_along(metrics)) {
      metric_name <- metrics[[i]]
      prior_value <- as.numeric(
        x$prior[[metric_name]]
      )

      if (length(prior_value) == 1L && is.finite(prior_value)) {
        graphics::abline(
          h = prior_value,
          col = col[[i]],
          lty = prior.lty,
          lwd = prior.lwd
        )
      }
    }

    metric_legend <- c(
      metric_legend,
      paste0(metrics, " prior")
    )
    metric_legend_col <- c(
      metric_legend_col,
      col
    )
    metric_legend_lty <- c(
      metric_legend_lty,
      rep(prior.lty, length(metrics))
    )
    metric_legend_lwd <- c(
      metric_legend_lwd,
      rep(prior.lwd, length(metrics))
    )
  }

  graphics::legend(
    "bottomright",
    legend = metric_legend,
    col = metric_legend_col,
    lty = metric_legend_lty,
    lwd = metric_legend_lwd,
    title = "Metrics",
    bty = "n",
    cex = 0.8
  )

  if (
      isTRUE(show.criteria) &&
      !is.null(x$criteria) &&
      nrow(x$criteria) > 0L
  ) {
    criterion_values <- x$criteria

    if (!is.null(criteria)) {
      criteria <- as.character(criteria)
      unknown <- setdiff(
        criteria,
        criterion_values$criterion
      )

      if (length(unknown) > 0L) {
        stop(
          "Unknown criterion name(s): ",
          paste(unknown, collapse = ", "),
          call. = FALSE
        )
      }

      criterion_values <- criterion_values[
        match(criteria, criterion_values$criterion),
        ,
        drop = FALSE
      ]
    }

    criterion_names <- criterion_values$criterion
    point_symbols <- rep_len(
      criterion.pch,
      length(criterion_names)
    )

    for (criterion_index in seq_along(criterion_names)) {
      for (metric_index in seq_along(metrics)) {
        metric_name <- metrics[[metric_index]]

        graphics::points(
          criterion_values[[xvar]][[criterion_index]],
          criterion_values[[metric_name]][[criterion_index]],
          pch = point_symbols[[criterion_index]],
          cex = criterion.cex,
          col = col[[metric_index]],
          bg = "white",
          lwd = criterion.lwd
        )
      }

      graphics::abline(
        v = criterion_values[[xvar]][[criterion_index]],
        col = "grey70",
        lty = 3L,
        lwd = criterion.lwd
      )

      if (isTRUE(label.criteria)) {
        label_metric <- if ("F1" %in% metrics) "F1" else metrics[[1L]]

        graphics::text(
          criterion_values[[xvar]][[criterion_index]],
          criterion_values[[label_metric]][[criterion_index]],
          labels = criterion_names[[criterion_index]],
          pos = 3L,
          cex = 0.7,
          xpd = NA
        )
      }
    }

    graphics::legend(
      "topleft",
      legend = criterion_names,
      pch = point_symbols,
      pt.bg = "white",
      pt.cex = criterion.cex,
      col = "black",
      title = "Criterion selections",
      bty = "n",
      cex = 0.75
    )
  }

  invisible(x)
}
