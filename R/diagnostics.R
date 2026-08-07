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
    selection.cex = 0.75,
    selection.lwd = 1.5,
    selection.line.alpha = 0.65,
    selection.point.alpha = 0.9,
    selection.points = c("active", "all", "none"),
    selection.labels = TRUE,
    selection.label.cex = NULL,
    selection.label.offset = 0.35,
    criterion.legend = c("right", "none"),
    criterion.legend.cex = 1.0,
    criterion.legend.point.cex = 1.0,
    criterion.legend.margin = 7,
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
  criterion.legend <- match.arg(criterion.legend)
  criterion.legend.cex <- as.numeric(criterion.legend.cex[[1L]])
  if (!is.finite(criterion.legend.cex) || criterion.legend.cex <= 0) {
    criterion.legend.cex <- 1.0
  }

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

  criterion_names <- if (isTRUE(show.criteria)) {
    .ecountgmifs.match.criteria(
      object,
      criteria
    )
  } else {
    character()
  }

  outside_legend <-
    identical(criterion.legend, "right") &&
    length(criterion_names) > 0L

  if (outside_legend) {
    old_par <- graphics::par(no.readonly = TRUE)
    legend_width <- max(
      1.45,
      min(2.8, as.numeric(criterion.legend.margin[[1L]]) * 0.22)
    )

    on.exit(
      {
        graphics::layout(matrix(1L))
        graphics::par(old_par)
      },
      add = TRUE
    )

    graphics::layout(
      matrix(c(1L, 2L), nrow = 1L),
      widths = c(4.6, legend_width)
    )
    graphics::par(
      mar = c(5.1, 5.1, 4.1, 1.0) + 0.1
    )
  }

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

  if (length(criterion_names) > 0L) {
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

    selected_x_values <- vapply(
      criterion_names,
      function(criterion_name) {
        .ecountgmifs.state.x(
          object$path$best_criteria[[criterion_name]]$state,
          xvar
        )
      },
      numeric(1L)
    )

    usr <- graphics::par("usr")

    criterion_label_cex <- selection.label.cex

    if (is.null(criterion_label_cex)) {
      # Match the text size used by the bottom-axis tick labels exactly.
      criterion_label_cex <- graphics::par("cex.axis")
    }

    if (
        length(criterion_label_cex) != 1L ||
          is.na(criterion_label_cex) ||
          !is.finite(criterion_label_cex) ||
          criterion_label_cex <= 0
    ) {
      stop(
        "`selection.label.cex` must be NULL or one positive finite number.",
        call. = FALSE
      )
    }

    # Place criterion names on the upper axis rather than inside the panel.
    # Assign extra margin lines only when the rendered label intervals overlap.
    label_lane <- integer(length(criterion_names))

    if (isTRUE(selection.labels)) {
      label_width <- graphics::strwidth(
        criterion_names,
        units = "user",
        cex = criterion_label_cex,
        font = 2L
      )

      x_padding <- 0.01 * diff(usr[1:2])
      ordered <- order(selected_x_values)
      lane_right <- numeric()

      for (index in ordered) {
        label_left <-
          selected_x_values[[index]] - 0.5 * label_width[[index]]
        label_right <-
          selected_x_values[[index]] + 0.5 * label_width[[index]]

        lane <- which(
          lane_right + x_padding < label_left
        )

        if (length(lane) == 0L) {
          lane <- length(lane_right) + 1L
          lane_right <- c(lane_right, label_right)
        } else {
          lane <- lane[[1L]]
          lane_right[[lane]] <- label_right
        }

        label_lane[[index]] <- lane - 1L
      }
    }

    # Use segments rather than abline so selection lines stay clipped to the
    # plotting box even when labels are allowed into the figure margin.
    for (i in seq_along(criterion_names)) {
      criterion_name <- criterion_names[[i]]
      selected <- object$path$best_criteria[[criterion_name]]$state
      selected_beta <- as.numeric(
        selected$predictors$parameters$beta
      )[plotted_indices]
      selected_x <- selected_x_values[[i]]

      graphics::segments(
        x0 = selected_x,
        y0 = usr[[3L]],
        x1 = selected_x,
        y1 = usr[[4L]],
        col = selection_line_col[[i]],
        lty = 3L,
        lwd = selection.lwd,
        xpd = FALSE
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
          lwd = selection.lwd,
          xpd = FALSE
        )
      }

      if (isTRUE(selection.labels)) {
        graphics::mtext(
          text = criterion_name,
          side = 3L,
          at = selected_x,
          line =
            selection.label.offset +
              0.8 * label_lane[[i]],
          adj = 0.5,
          cex = criterion_label_cex,
          col = styles$col[[i]],
          font = 2L,
          padj = 0.5
        )
      }
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
      xpd = TRUE
    )
  }

  if (outside_legend) {
    graphics::par(
      mar = c(0.5, 0.2, 0.5, 0.2) + 0.1
    )
    .ecountgmifs.legend.panel(
      legend = criterion_names,
      col = styles$col,
      pch = styles$pch,
      pt.bg = selection_point_col,
      pt.cex = criterion.legend.point.cex,
      lty = 3L,
      lwd = selection.lwd,
      title = "Criterion selections",
      bty = "n",
      cex = criterion.legend.cex,
      cex.min = min(criterion.legend.cex, 0.92),
      max.columns = 2L
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
    legend.position = "outside",
    legend.margin = 7,
    ...
) {
  xvar <- match.arg(xvar)
  legend.position <- .ecountgmifs.legend.position(legend.position)
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

  outside_legend <- identical(legend.position, "outside")

  if (outside_legend) {
    old_par <- graphics::par(no.readonly = TRUE)
    legend_width <- max(
      1.45,
      min(2.8, as.numeric(legend.margin[[1L]]) * 0.22)
    )

    on.exit(
      {
        graphics::layout(matrix(1L))
        graphics::par(old_par)
      },
      add = TRUE
    )

    graphics::layout(
      matrix(c(1L, 2L), nrow = 1L),
      widths = c(4.6, legend_width)
    )
    graphics::par(
      mar = c(5.1, 5.1, 4.1, 1.0) + 0.1
    )
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
        xpd = TRUE
      )
    }
  }

  if (outside_legend) {
    graphics::par(
      mar = c(0.5, 0.2, 0.5, 0.2) + 0.1
    )
    .ecountgmifs.legend.panel(
      legend = criterion_names,
      col = styles$col,
      lty = lty,
      lwd = lwd,
      pch = styles$pch,
      pt.bg = "white",
      pt.cex = selection.cex,
      title = "Criteria",
      bty = "n",
      cex = 1.05,
      cex.min = 0.92,
      max.columns = 2L
    )
  } else {
    .ecountgmifs.draw.legend(
      legend.position,
      legend = criterion_names,
      col = styles$col,
      lty = lty,
      lwd = lwd,
      pch = styles$pch,
      pt.bg = "white",
      pt.cex = selection.cex,
      bty = "n",
      cex = 1.0
    )
  }

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
  } else if (!is.null(object$input$prior_selected)) {
    resolved_prior <- .ecountgmifs.align.binary.vector(
      object$input$prior_selected,
      predictor_names,
      "stored prior selection"
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
    } else if (!is.null(object$input$prior_selected)) {
      "stored prior specification"
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
#' @param criterion.cex Criterion point size in the plot.
#' @param criterion.legend.point.cex Criterion symbol size in the outside
#'   criterion legend.
#' @param criterion.pch Criterion point symbols.
#' @param criterion.lwd Criterion point-border width.
#' @param criterion.line.lwd,criterion.line.alpha Criterion vertical-line style.
#' @param criterion.label.cex,criterion.label.offset Criterion label size and
#'   distance above the upper plot border. When `criterion.label.cex = NULL`,
#'   labels use the same size as the axis tick text.
#' @param label.criteria Label criterion selections on the upper plot axis.
#' @param legend.position Whether to place the legend in a dedicated right-side
#'   panel, inside the plot, or suppress it.
#' @param legend.cex Target legend-text size. Long labels are wrapped and the
#'   size is reduced only when required to fit the dedicated panel.
#' @param legend.margin Relative width control for the dedicated outside legend
#'   panel. Larger values reserve more horizontal space.
#' @param xlab,ylab,main Axis and title labels.
#' @param xlim,ylim Axis limits.
#' @param lty,lwd,line.alpha,col Path-line styling.
#' @param prior.lty,prior.lwd,prior.alpha Prior-baseline styling.
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
    criterion.cex = 0.9,
    criterion.pch = c(21L, 22L, 23L, 24L, 25L),
    criterion.lwd = 1.4,
    criterion.line.lwd = 1.2,
    criterion.line.alpha = 0.6,
    criterion.label.cex = NULL,
    criterion.label.offset = 0.35,
    label.criteria = TRUE,
    legend.position = "outside",
    legend.cex = 1.0,
    criterion.legend.point.cex = 1.0,
    legend.margin = 9,
    xlab = NULL,
    ylab = "Selection metric",
    main = "ecountgmifs selection metrics",
    xlim = NULL,
    ylim = c(0, 1),
    lty = 1,
    lwd = 2.4,
    line.alpha = 0.9,
    col = NULL,
    prior.lty = 5L,
    prior.lwd = 1.8,
    prior.alpha = 0.7,
    ...
) {
  xvar <- match.arg(xvar)
  legend.position <- .ecountgmifs.legend.position(legend.position)
  legend.cex <- as.numeric(legend.cex[[1L]])
  if (!is.finite(legend.cex) || legend.cex <= 0) {
    legend.cex <- 1.0
  }

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

  default_metric_colors <- c(
    F1 = "#D55E00",
    precision = "#009E73",
    recall = "#0072B2",
    specificity = "#CC79A7",
    accuracy = "#56B4E9"
  )

  if (is.null(col)) {
    col <- unname(default_metric_colors[metrics])
  } else {
    col <- rep_len(col, length(metrics))
  }

  path_col <- .ecountgmifs.alpha.colors(
    col,
    line.alpha,
    "line.alpha"
  )

  prior_col <- .ecountgmifs.alpha.colors(
    col,
    prior.alpha,
    "prior.alpha"
  )

  criterion_values <- NULL
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
  }

  x_values <- as.numeric(path[[xvar]])

  if (is.null(xlim)) {
    x_for_range <- x_values

    if (!is.null(criterion_values)) {
      x_for_range <- c(
        x_for_range,
        as.numeric(criterion_values[[xvar]])
      )
    }

    xlim <- range(x_for_range, finite = TRUE)
    x_span <- diff(xlim)

    if (!is.finite(x_span) || x_span == 0) {
      x_span <- max(1, abs(xlim[[1L]]))
    }

    xlim <- xlim + c(-1, 1) * 0.02 * x_span
  }

  y <- as.matrix(
    path[, metrics, drop = FALSE]
  )

  outside_legend <- identical(legend.position, "outside")

  if (outside_legend) {
    old_par <- graphics::par(no.readonly = TRUE)
    legend_width <- max(
      1.65,
      min(3.0, as.numeric(legend.margin[[1L]]) * 0.22)
    )

    on.exit(
      {
        graphics::layout(matrix(1L))
        graphics::par(old_par)
      },
      add = TRUE
    )

    graphics::layout(
      matrix(c(1L, 2L), nrow = 1L),
      widths = c(4.6, legend_width)
    )
    graphics::par(
      mar = c(5.1, 5.1, 4.1, 1.0) + 0.1
    )
  }

  graphics::matplot(
    x_values,
    y,
    type = "l",
    lty = lty,
    lwd = lwd,
    col = path_col,
    xlim = xlim,
    ylim = ylim,
    xlab = xlab,
    ylab = ylab,
    main = main,
    ...
  )

  if (isTRUE(show.prior) && !is.null(x$prior)) {
    for (i in seq_along(metrics)) {
      metric_name <- metrics[[i]]
      prior_value <- as.numeric(
        x$prior[[metric_name]]
      )

      if (length(prior_value) == 1L && is.finite(prior_value)) {
        usr <- graphics::par("usr")

        graphics::segments(
          x0 = usr[[1L]],
          y0 = prior_value,
          x1 = usr[[2L]],
          y1 = prior_value,
          col = prior_col[[i]],
          lty = prior.lty,
          lwd = prior.lwd,
          xpd = FALSE
        )
      }
    }
  }

  criterion_names <- character()
  criterion_styles <- NULL

  if (!is.null(criterion_values)) {
    criterion_names <- as.character(
      criterion_values$criterion
    )

    criterion_styles <- .ecountgmifs.selection.styles(
      length(criterion_names)
    )

    criterion_styles$pch <- rep_len(
      criterion.pch,
      length(criterion_names)
    )

    criterion_line_col <- .ecountgmifs.alpha.colors(
      criterion_styles$col,
      criterion.line.alpha,
      "criterion.line.alpha"
    )

    selected_x_values <- as.numeric(
      criterion_values[[xvar]]
    )

    usr <- graphics::par("usr")

    criterion_label_cex <- criterion.label.cex

    if (is.null(criterion_label_cex)) {
      # Match the text size used by the bottom-axis tick labels exactly.
      criterion_label_cex <- graphics::par("cex.axis")
    }

    if (
        length(criterion_label_cex) != 1L ||
          is.na(criterion_label_cex) ||
          !is.finite(criterion_label_cex) ||
          criterion_label_cex <= 0
    ) {
      stop(
        "`criterion.label.cex` must be NULL or one positive finite number.",
        call. = FALSE
      )
    }

    # Place criterion names on the upper axis. Use additional margin lanes only
    # when the rendered label intervals overlap.
    label_lane <- integer(length(criterion_names))

    if (isTRUE(label.criteria)) {
      label_width <- graphics::strwidth(
        criterion_names,
        units = "user",
        cex = criterion_label_cex,
        font = 2L
      )

      x_padding <- 0.01 * diff(usr[1:2])
      ordered <- order(selected_x_values)
      lane_right <- numeric()

      for (index in ordered) {
        label_left <-
          selected_x_values[[index]] - 0.5 * label_width[[index]]
        label_right <-
          selected_x_values[[index]] + 0.5 * label_width[[index]]

        lane <- which(
          lane_right + x_padding < label_left
        )

        if (length(lane) == 0L) {
          lane <- length(lane_right) + 1L
          lane_right <- c(lane_right, label_right)
        } else {
          lane <- lane[[1L]]
          lane_right[[lane]] <- label_right
        }

        label_lane[[index]] <- lane - 1L
      }
    }

    for (criterion_index in seq_along(criterion_names)) {
      selected_x <- selected_x_values[[criterion_index]]

      # Clip criterion lines to the plotting box. This prevents them from
      # extending through the title, labels, or the outside legends.
      graphics::segments(
        x0 = selected_x,
        y0 = usr[[3L]],
        x1 = selected_x,
        y1 = usr[[4L]],
        col = criterion_line_col[[criterion_index]],
        lty = 3L,
        lwd = criterion.line.lwd,
        xpd = FALSE
      )

      for (metric_index in seq_along(metrics)) {
        metric_name <- metrics[[metric_index]]

        graphics::points(
          selected_x,
          criterion_values[[metric_name]][[criterion_index]],
          pch = criterion_styles$pch[[criterion_index]],
          cex = criterion.cex,
          col = criterion_styles$col[[criterion_index]],
          bg = path_col[[metric_index]],
          lwd = criterion.lwd,
          xpd = FALSE
        )
      }

      if (isTRUE(label.criteria)) {
        graphics::mtext(
          text = criterion_names[[criterion_index]],
          side = 3L,
          at = selected_x,
          line =
            criterion.label.offset +
              0.8 * label_lane[[criterion_index]],
          adj = 0.5,
          cex = criterion_label_cex,
          col = criterion_styles$col[[criterion_index]],
          font = 2L,
          padj = 0.5
        )
      }
    }
  }

  if (!identical(legend.position, "none")) {
    metric_labels <- metrics
    metric_colors <- path_col
    metric_lty <- rep(lty, length(metrics))
    metric_lwd <- rep(lwd, length(metrics))

    if (isTRUE(show.prior) && !is.null(x$prior)) {
      metric_labels <- c(
        paste0(metrics, " path"),
        paste0(metrics, " prior")
      )
      metric_colors <- c(path_col, prior_col)
      metric_lty <- c(
        rep(lty, length(metrics)),
        rep(prior.lty, length(metrics))
      )
      metric_lwd <- c(
        rep(lwd, length(metrics)),
        rep(prior.lwd, length(metrics))
      )
    }

    legend_labels <- metric_labels
    legend_col <- metric_colors
    legend_lty <- metric_lty
    legend_lwd <- metric_lwd
    legend_pch <- rep(NA_integer_, length(metric_labels))
    legend_bg <- rep(NA_character_, length(metric_labels))
    legend_title <- "Metrics"

    if (length(criterion_names) > 0L) {
      criterion_legend_fill <- .ecountgmifs.alpha.colors(
        criterion_styles$col,
        0.9,
        "criterion legend point alpha"
      )

      legend_labels <- c(legend_labels, criterion_names)
      legend_col <- c(legend_col, criterion_styles$col)
      legend_lty <- c(
        legend_lty,
        rep(3L, length(criterion_names))
      )
      legend_lwd <- c(
        legend_lwd,
        rep(criterion.lwd, length(criterion_names))
      )
      legend_pch <- c(
        legend_pch,
        criterion_styles$pch
      )
      legend_bg <- c(
        legend_bg,
        criterion_legend_fill
      )
      legend_title <- "Metrics and criterion selections"
    }

    if (outside_legend) {
      graphics::par(
        mar = c(0.5, 0.2, 0.5, 0.2) + 0.1
      )
      .ecountgmifs.legend.panel(
        legend = legend_labels,
        col = legend_col,
        lty = legend_lty,
        lwd = legend_lwd,
        pch = legend_pch,
        pt.bg = legend_bg,
        pt.cex = criterion.legend.point.cex,
        title = legend_title,
        bty = "n",
        cex = legend.cex,
        cex.min = min(legend.cex, 0.92),
        max.columns = 2L
      )
    } else {
      graphics::legend(
        legend.position,
        legend = legend_labels,
        col = legend_col,
        lty = legend_lty,
        lwd = legend_lwd,
        pch = legend_pch,
        pt.bg = legend_bg,
        pt.cex = criterion.legend.point.cex,
        title = legend_title,
        bty = "n",
        cex = legend.cex
      )
    }
  }

  invisible(x)
}

