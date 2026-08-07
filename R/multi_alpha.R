.ecountgmifs.alpha.labels <- function(alpha) {
  format(
    as.numeric(alpha),
    digits = 15L,
    trim = TRUE,
    scientific = FALSE
  )
}


.ecountgmifs.fit.multi.alpha <- function(
    fit_call,
    alpha,
    envir
) {
  alpha <- as.numeric(alpha)
  labels <- .ecountgmifs.alpha.labels(alpha)

  fits <- lapply(
    alpha,
    function(alpha_value) {
      current_call <- fit_call
      current_call$enet.alpha <- alpha_value
      eval(current_call, envir = envir)
    }
  )

  names(fits) <- paste0("alpha=", labels)

  out <- list(
    call = fit_call,
    alpha = alpha,
    alpha_labels = labels,
    fits = fits
  )

  class(out) <- c("ecountgmifs_multi_alpha", "list")
  out
}


.ecountgmifs.multi.alpha.select <- function(
    object,
    alpha = NULL
) {
  if (is.null(alpha)) {
    return(seq_along(object$fits))
  }

  if (is.character(alpha)) {
    match_index <- match(alpha, object$alpha_labels)

    if (anyNA(match_index)) {
      match_index <- match(alpha, names(object$fits))
    }

    if (anyNA(match_index)) {
      stop(
        "Unknown alpha label(s): ",
        paste(alpha[is.na(match_index)], collapse = ", "),
        call. = FALSE
      )
    }

    return(as.integer(match_index))
  }

  if (
      !is.numeric(alpha) ||
      length(alpha) == 0L ||
      anyNA(alpha) ||
      any(!is.finite(alpha))
  ) {
    stop(
      "`alpha` must be NULL, finite numeric values, or stored alpha labels.",
      call. = FALSE
    )
  }

  vapply(
    as.numeric(alpha),
    function(value) {
      difference <- abs(object$alpha - value)
      index <- which.min(difference)
      tolerance <- 100 * .Machine$double.eps * max(1, abs(value))

      if (difference[[index]] > tolerance) {
        stop(
          "Alpha ",
          format(value, digits = 15L),
          " is not present. Available values: ",
          paste(object$alpha_labels, collapse = ", "),
          call. = FALSE
        )
      }

      as.integer(index)
    },
    integer(1L)
  )
}


.ecountgmifs.multi.alpha.overview <- function(object) {
  do.call(
    rbind,
    lapply(
      seq_along(object$fits),
      function(index) {
        fit <- object$fits[[index]]
        terminal <- .ecountgmifs.terminal.state(fit)
        beta <- terminal$predictors$parameters$beta

        data.frame(
          alpha = object$alpha[[index]],
          iteration = as.integer(terminal$iteration),
          nonzero = sum(beta != 0),
          negloglik = as.numeric(terminal$negloglik),
          pseudo_r2 = as.numeric(terminal$pseudo_r2),
          total_seconds = as.numeric(fit$path$total_time),
          termination = as.character(fit$path$message),
          stringsAsFactors = FALSE
        )
      }
    )
  )
}


.ecountgmifs.multi.alpha.criteria <- function(object) {
  rows <- lapply(
    seq_along(object$fits),
    function(index) {
      table <- .ecountgmifs.criterion.table(
        object$fits[[index]]
      )

      if (nrow(table) == 0L) {
        return(NULL)
      }

      table$alpha <- object$alpha[[index]]
      table[, c(
        "alpha",
        "criterion",
        "value",
        "iteration",
        "nonzero",
        "negloglik",
        "pseudo_r2"
      ), drop = FALSE]
    }
  )

  rows <- Filter(Negate(is.null), rows)

  if (length(rows) == 0L) {
    return(
      data.frame(
        alpha = numeric(),
        criterion = character(),
        value = numeric(),
        iteration = integer(),
        nonzero = integer(),
        negloglik = numeric(),
        pseudo_r2 = numeric(),
        stringsAsFactors = FALSE
      )
    )
  }

  do.call(rbind, rows)
}


#' @export
print.ecountgmifs_multi_alpha <- function(
    x,
    digits = max(3L, getOption("digits") - 3L),
    ...
) {
  cat("ecountgmifs multi-alpha fit\n")

  if (!is.null(x$call)) {
    cat("\nCall:\n")
    print(x$call)
  }

  first <- x$fits[[1L]]

  cat("\nModel:\n")
  cat("  Family:            ", .ecountgmifs.family.label(first), "\n", sep = "")
  cat("  Link:              ", .ecountgmifs.link.label(first), "\n", sep = "")
  cat("  Family/link mode:  ", .ecountgmifs.interface.label(first), "\n", sep = "")
  cat("  Observations:      ", first$input$n, "\n", sep = "")
  cat("  Penalized terms:   ", first$input$p, "\n", sep = "")
  cat("  Unpenalized terms: ", first$input$q, "\n", sep = "")
  cat("  Alpha values:      ", paste(x$alpha_labels, collapse = ", "), "\n", sep = "")

  cat("\nTerminal states by alpha:\n")
  print(
    .ecountgmifs.multi.alpha.overview(x)[, c(
      "alpha",
      "iteration",
      "nonzero",
      "negloglik",
      "pseudo_r2",
      "total_seconds"
    ), drop = FALSE],
    row.names = FALSE,
    digits = digits
  )

  criterion_table <- .ecountgmifs.multi.alpha.criteria(x)

  if (nrow(criterion_table) > 0L) {
    cat("\nCriterion-selected states by alpha:\n")
    print(
      criterion_table,
      row.names = FALSE,
      digits = digits
    )
  }

  invisible(x)
}


#' @export
summary.ecountgmifs_multi_alpha <- function(
    object,
    ground.truth = NULL,
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

  metrics <- NULL

  if (!is.null(ground.truth)) {
    metrics <- lapply(
      object$fits,
      ecountgmifs.metrics,
      ground.truth = ground.truth,
      zero.tol = zero.tol,
      zero.division = zero.division,
      prior.selected = prior.selected,
      prior.weight.vec = prior.weight.vec,
      prior.cutoff = prior.cutoff,
      prior.direction = prior.direction
    )
  }

  out <- list(
    call = object$call,
    alpha = object$alpha,
    alpha_labels = object$alpha_labels,
    overview = .ecountgmifs.multi.alpha.overview(object),
    criteria = .ecountgmifs.multi.alpha.criteria(object),
    metrics = metrics
  )

  class(out) <- "summary.ecountgmifs_multi_alpha"
  out
}


#' @export
print.summary.ecountgmifs_multi_alpha <- function(
    x,
    digits = max(3L, getOption("digits") - 3L),
    ...
) {
  cat("ecountgmifs multi-alpha summary\n")

  if (!is.null(x$call)) {
    cat("\nCall:\n")
    print(x$call)
  }

  cat("\nTerminal states by alpha:\n")
  print(
    x$overview,
    row.names = FALSE,
    digits = digits
  )

  if (nrow(x$criteria) > 0L) {
    cat("\nCriterion-selected states by alpha:\n")
    print(
      x$criteria,
      row.names = FALSE,
      digits = digits
    )
  }

  if (!is.null(x$metrics)) {
    metric_rows <- do.call(
      rbind,
      lapply(
        seq_along(x$metrics),
        function(index) {
          value <- x$metrics[[index]]$criteria

          if (is.null(value) || nrow(value) == 0L) {
            return(NULL)
          }

          value$alpha <- x$alpha[[index]]
          value[, c(
            "alpha",
            "criterion",
            "iteration",
            "selected",
            "precision",
            "recall",
            "F1"
          ), drop = FALSE]
        }
      )
    )

    if (!is.null(metric_rows) && nrow(metric_rows) > 0L) {
      cat("\nGround-truth metrics by alpha and criterion:\n")
      print(
        metric_rows,
        row.names = FALSE,
        digits = digits
      )
    }
  }

  invisible(x)
}


#' Extract coefficients from a multi-alpha fit
#'
#' @param object An `ecountgmifs_multi_alpha` object.
#' @param criterion,iteration,include.theta,state,drop Passed to
#'   [coef.ecountgmifs()].
#' @param alpha Optional stored alpha value or values, or their labels.
#' @param ... Additional arguments passed to [coef.ecountgmifs()].
#'
#' @return A named list for multiple alpha values. When one alpha is explicitly
#'   selected and `drop = TRUE`, the coefficient result for that fit is returned
#'   directly.
#' @export
coef.ecountgmifs_multi_alpha <- function(
    object,
    criterion = NULL,
    iteration = NULL,
    include.theta = FALSE,
    alpha = NULL,
    state = NULL,
    drop = TRUE,
    ...
) {
  indices <- .ecountgmifs.multi.alpha.select(
    object,
    alpha = alpha
  )

  out <- lapply(
    indices,
    function(index) {
      coef(
        object$fits[[index]],
        criterion = criterion,
        iteration = iteration,
        include.theta = include.theta,
        state = state,
        drop = drop,
        ...
      )
    }
  )

  names(out) <- paste0(
    "alpha=",
    object$alpha_labels[indices]
  )

  if (!is.null(alpha) && length(out) == 1L && isTRUE(drop)) {
    return(out[[1L]])
  }

  out
}


.ecountgmifs.multi.alpha.metric.data <- function(
    object,
    ground.truth,
    zero.tol,
    zero.division,
    prior.selected,
    prior.weight.vec,
    prior.cutoff,
    prior.direction
) {
  metric_objects <- lapply(
    object$fits,
    ecountgmifs.metrics,
    ground.truth = ground.truth,
    zero.tol = zero.tol,
    zero.division = zero.division,
    prior.selected = prior.selected,
    prior.weight.vec = prior.weight.vec,
    prior.cutoff = prior.cutoff,
    prior.direction = prior.direction
  )

  selected <- do.call(
    rbind,
    lapply(
      seq_along(metric_objects),
      function(index) {
        value <- metric_objects[[index]]$criteria

        if (is.null(value) || nrow(value) == 0L) {
          return(NULL)
        }

        value$alpha <- object$alpha[[index]]
        value
      }
    )
  )

  safe_maximum <- function(value) {
    value <- value[is.finite(value)]

    if (length(value) == 0L) {
      return(NA_real_)
    }

    max(value)
  }

  maximum <- do.call(
    rbind,
    lapply(
      seq_along(metric_objects),
      function(index) {
        value <- metric_objects[[index]]$path

        data.frame(
          alpha = object$alpha[[index]],
          F1 = safe_maximum(value$F1),
          precision = safe_maximum(value$precision),
          recall = safe_maximum(value$recall),
          stringsAsFactors = FALSE
        )
      }
    )
  )

  if (is.null(selected) || nrow(selected) == 0L) {
    stop(
      "No criterion-selected metric values are available across alpha values.",
      call. = FALSE
    )
  }

  list(
    objects = metric_objects,
    selected = selected,
    maximum = maximum,
    prior = metric_objects[[1L]]$prior
  )
}


.ecountgmifs.best.alpha.rows <- function(
    criterion_table,
    criteria
) {
  rows <- lapply(
    criteria,
    function(criterion_name) {
      current <- criterion_table[
        criterion_table$criterion == criterion_name &
        is.finite(criterion_table$value),
        ,
        drop = FALSE
      ]

      if (nrow(current) == 0L) {
        return(NULL)
      }

      current[which.min(current$value), , drop = FALSE]
    }
  )

  rows <- Filter(Negate(is.null), rows)

  if (length(rows) == 0L) {
    return(criterion_table[0, , drop = FALSE])
  }

  do.call(rbind, rows)
}



.ecountgmifs.plot.alpha.comparison <- function(
    object,
    measure = NULL,
    ground.truth = NULL,
    criteria = NULL,
    zero.tol = sqrt(.Machine$double.eps),
    zero.division = c("zero", "one", "NA"),
    prior.selected = NULL,
    prior.weight.vec = NULL,
    prior.cutoff = 1,
    prior.direction = c("lower", "higher"),
    ask = interactive(),
    .manage.ask = TRUE,
    .manage.par = TRUE,
    context.label = NULL,
    col = NULL,
    pch = c(21L, 22L, 23L, 24L, 25L),
    lwd = 2,
    point.cex = 1,
    best.alpha.cex = 1.8,
    best.alpha.lwd = 2,
    best.label = "Best alpha by criterion",
    comparison.label = "alpha",
    maximum.lty = 3L,
    prior.lty = 2L,
    legend.position = "outside",
    legend.width = 1.55,
    legend.cex = 1.00,
    xlab = "Elastic-net alpha",
    ylab = NULL,
    main = NULL,
    ...
) {
  if (length(object$fits) < 2L) {
    stop(
      "The alpha-comparison plot requires at least two fitted alpha values.",
      call. = FALSE
    )
  }

  zero.division <- match.arg(zero.division)
  prior.direction <- match.arg(prior.direction)
  legend.position <- .ecountgmifs.legend.position(legend.position)
  legend.cex <- as.numeric(legend.cex[[1L]])
  if (!is.finite(legend.cex) || legend.cex <= 0) {
    legend.cex <- 1.00
  }

  has_ground_truth <- !is.null(ground.truth)

  if (is.null(measure)) {
    measure <- if (has_ground_truth) "F1" else "nonzero"
  }

  allowed <- if (has_ground_truth) {
    c("F1", "precision", "recall")
  } else {
    c(
      "nonzero",
      "negloglik",
      "pseudo_r2",
      "value",
      "iteration"
    )
  }

  measure <- match.arg(
    measure,
    allowed,
    several.ok = TRUE
  )

  criterion_table <- .ecountgmifs.multi.alpha.criteria(object)

  if (nrow(criterion_table) == 0L) {
    stop(
      "The alpha-comparison plot requires at least one fitted criterion.",
      call. = FALSE
    )
  }

  available_criteria <- unique(criterion_table$criterion)

  if (is.null(criteria)) {
    criteria <- available_criteria
  } else {
    criteria <- as.character(criteria)
    unknown <- setdiff(criteria, available_criteria)

    if (length(unknown) > 0L) {
      stop(
        "Unknown criterion name(s): ",
        paste(unknown, collapse = ", "),
        call. = FALSE
      )
    }
  }

  styles <- .ecountgmifs.selection.styles(length(criteria))
  styles$pch <- rep_len(pch, length(criteria))

  if (!is.null(col)) {
    styles$col <- rep_len(col, length(criteria))
  }

  alpha_position <- seq_along(object$alpha)
  best_alpha <- .ecountgmifs.best.alpha.rows(
    criterion_table,
    criteria
  )

  metric_data <- NULL

  if (has_ground_truth) {
    metric_data <- .ecountgmifs.multi.alpha.metric.data(
      object = object,
      ground.truth = ground.truth,
      zero.tol = zero.tol,
      zero.division = zero.division,
      prior.selected = prior.selected,
      prior.weight.vec = prior.weight.vec,
      prior.cutoff = prior.cutoff,
      prior.direction = prior.direction
    )
  }

  outside_legend <- identical(legend.position, "outside")

  if (isTRUE(.manage.par)) {
    old_par <- graphics::par(no.readonly = TRUE)

    on.exit(
      {
        if (outside_legend) {
          graphics::layout(matrix(1L))
        }
        graphics::par(old_par)
      },
      add = TRUE
    )
  }

  if (isTRUE(.manage.ask)) {
    old_ask <- grDevices::devAskNewPage(
      ask = isTRUE(ask) && length(measure) > 1L
    )
    on.exit(
      grDevices::devAskNewPage(old_ask),
      add = TRUE
    )
  }

  for (current_measure in measure) {
    if (outside_legend) {
      layout.legend.labels <- c(
        criteria,
        best.label,
        if (has_ground_truth) {
          paste0("Maximum ", current_measure, " along path")
        } else {
          character()
        },
        if (has_ground_truth && !is.null(metric_data$prior)) {
          "Prior reference"
        } else {
          character()
        }
      )
      current.legend.width <- .ecountgmifs.relative.legend.width(
        labels = layout.legend.labels,
        cex = legend.cex,
        plot.width = 4.6
      )

      graphics::layout(
        matrix(c(1L, 2L), nrow = 1L),
        widths = c(4.6, current.legend.width)
      )
      graphics::par(
        mar = c(5.1, 5.1, 4.1, 1.0) + 0.1
      )
    }

    plot_data <- if (has_ground_truth) {
      metric_data$selected
    } else {
      criterion_table
    }

    plot_data <- plot_data[
      plot_data$criterion %in% criteria,
      ,
      drop = FALSE
    ]

    y_values <- as.numeric(plot_data[[current_measure]])
    extra_values <- numeric()

    if (has_ground_truth) {
      extra_values <- metric_data$maximum[[current_measure]]

      if (!is.null(metric_data$prior)) {
        extra_values <- c(
          extra_values,
          as.numeric(metric_data$prior[[current_measure]])
        )
      }
    }

    plot_range <- range(
      c(y_values, extra_values),
      finite = TRUE
    )

    if (!all(is.finite(plot_range))) {
      plot_range <- c(0, 1)
    }

    padding <- 0.05 * max(1, diff(plot_range))
    plot_range <- plot_range + c(-padding, padding)

    current_ylab <- if (is.null(ylab)) {
      switch(
        current_measure,
        F1 = "F1-score",
        precision = "Precision",
        recall = "Recall",
        nonzero = "Selected penalized coefficients",
        negloglik = "Negative log-likelihood",
        pseudo_r2 = "Pseudo-R2",
        value = "Criterion value",
        iteration = "Selected iteration"
      )
    } else {
      ylab
    }

    current_main <- if (is.null(main)) {
      paste0(
        "ecountgmifs ",
        comparison.label,
        " comparison: ",
        current_measure,
        if (
            !is.null(context.label) &&
            nzchar(as.character(context.label[[1L]]))
        ) {
          paste0(" (", as.character(context.label[[1L]]), ")")
        } else {
          ""
        }
      )
    } else {
      main
    }

    graphics::plot(
      alpha_position,
      rep(NA_real_, length(alpha_position)),
      type = "n",
      xaxt = "n",
      xlim = c(0.5, length(alpha_position) + 0.5),
      ylim = plot_range,
      xlab = xlab,
      ylab = current_ylab,
      main = current_main,
      ...
    )
    graphics::axis(
      side = 1,
      at = alpha_position,
      labels = object$alpha_labels
    )

    for (criterion_index in seq_along(criteria)) {
      criterion_name <- criteria[[criterion_index]]
      current <- plot_data[
        plot_data$criterion == criterion_name,
        ,
        drop = FALSE
      ]
      current_position <- match(
        current$alpha,
        object$alpha
      )
      current_order <- order(current_position)
      current <- current[current_order, , drop = FALSE]
      current_position <- current_position[current_order]

      graphics::lines(
        current_position,
        current[[current_measure]],
        col = styles$col[[criterion_index]],
        lwd = lwd
      )
      graphics::points(
        current_position,
        current[[current_measure]],
        pch = styles$pch[[criterion_index]],
        col = styles$col[[criterion_index]],
        bg = styles$col[[criterion_index]],
        cex = point.cex
      )

      best <- best_alpha[
        best_alpha$criterion == criterion_name,
        ,
        drop = FALSE
      ]

      if (nrow(best) > 0L) {
        best_index <- match(
          best$alpha[[1L]],
          current$alpha
        )

        if (!is.na(best_index)) {
          graphics::points(
            current_position[[best_index]],
            current[[current_measure]][[best_index]],
            pch = 1L,
            col = styles$col[[criterion_index]],
            cex = best.alpha.cex,
            lwd = best.alpha.lwd
          )
        }
      }
    }

    legend_labels <- criteria
    legend_col <- styles$col
    legend_lty <- rep(1L, length(criteria))
    legend_pch <- styles$pch

    legend_labels <- c(
      legend_labels,
      best.label
    )
    legend_col <- c(legend_col, "black")
    legend_lty <- c(legend_lty, NA_integer_)
    legend_pch <- c(legend_pch, 1L)

    if (has_ground_truth) {
      maximum_position <- match(
        metric_data$maximum$alpha,
        object$alpha
      )
      maximum_order <- order(maximum_position)
      maximum <- metric_data$maximum[
        maximum_order,
        ,
        drop = FALSE
      ]
      maximum_position <- maximum_position[maximum_order]

      graphics::lines(
        maximum_position,
        maximum[[current_measure]],
        col = "black",
        lty = maximum.lty,
        lwd = lwd
      )

      legend_labels <- c(
        legend_labels,
        paste0(
          "Maximum ",
          current_measure,
          " along path"
        )
      )
      legend_col <- c(legend_col, "black")
      legend_lty <- c(legend_lty, maximum.lty)
      legend_pch <- c(legend_pch, NA_integer_)

      if (!is.null(metric_data$prior)) {
        prior_value <- as.numeric(
          metric_data$prior[[current_measure]]
        )

        if (
            length(prior_value) == 1L &&
            is.finite(prior_value)
        ) {
          graphics::abline(
            h = prior_value,
            col = "#D62728",
            lty = prior.lty,
            lwd = lwd
          )

          legend_labels <- c(
            legend_labels,
            "Prior reference"
          )
          legend_col <- c(
            legend_col,
            "#D62728"
          )
          legend_lty <- c(
            legend_lty,
            prior.lty
          )
          legend_pch <- c(
            legend_pch,
            NA_integer_
          )
        }
      }
    }

    if (outside_legend) {
      graphics::par(
        mar = c(0.5, 0.2, 0.5, 0.2) + 0.1
      )
      .ecountgmifs.legend.panel(
        legend = legend_labels,
        col = legend_col,
        lty = legend_lty,
        lwd = lwd,
        pch = legend_pch,
        pt.bg = legend_col,
        bty = "n",
        cex = legend.cex
      )
    } else {
      .ecountgmifs.draw.legend(
        legend.position,
        legend = legend_labels,
        col = legend_col,
        lty = legend_lty,
        lwd = lwd,
        pch = legend_pch,
        pt.bg = legend_col,
        bty = "n",
        cex = legend.cex
      )
    }
  }

  invisible(
    if (has_ground_truth) metric_data else criterion_table
  )
}




.ecountgmifs.plot.alpha.distribution <- function(
    object,
    ground.truth,
    measure = c("F1", "precision", "recall"),
    criteria = NULL,
    zero.tol = sqrt(.Machine$double.eps),
    zero.division = c("zero", "one", "NA"),
    prior.selected = NULL,
    prior.weight.vec = NULL,
    prior.cutoff = 1,
    prior.direction = c("lower", "higher"),
    ask = interactive(),
    .manage.ask = TRUE,
    .manage.par = TRUE,
    context.label = NULL,
    col = NULL,
    pch = c(21L, 22L, 23L, 24L, 25L),
    box.col = "grey85",
    box.border = "grey35",
    path.point.col = "grey35",
    path.point.alpha = 0.22,
    path.point.cex = 0.45,
    criterion.point.cex = 1,
    best.alpha.cex = 1.8,
    best.alpha.lwd = 2,
    best.label = "Best alpha by criterion",
    comparison.label = "alpha",
    maximum.pch = 4L,
    maximum.cex = 1,
    prior.lty = 2L,
    legend.position = "outside",
    legend.width = 1.70,
    legend.cex = 0.95,
    xlab = "Elastic-net alpha",
    ylab = NULL,
    main = NULL,
    ...
) {
  if (length(object$fits) < 2L) {
    stop(
      "The alpha-distribution plot requires at least two fitted alpha values.",
      call. = FALSE
    )
  }

  if (missing(ground.truth) || is.null(ground.truth)) {
    stop(
      "`ground.truth` is required for the alpha-distribution plot.",
      call. = FALSE
    )
  }

  zero.division <- match.arg(zero.division)
  prior.direction <- match.arg(prior.direction)
  measure <- match.arg(
    measure,
    c("F1", "precision", "recall"),
    several.ok = TRUE
  )
  legend.position <- .ecountgmifs.legend.position(legend.position)
  legend.cex <- as.numeric(legend.cex[[1L]])
  if (!is.finite(legend.cex) || legend.cex <= 0) {
    legend.cex <- 0.95
  }

  metric_data <- .ecountgmifs.multi.alpha.metric.data(
    object = object,
    ground.truth = ground.truth,
    zero.tol = zero.tol,
    zero.division = zero.division,
    prior.selected = prior.selected,
    prior.weight.vec = prior.weight.vec,
    prior.cutoff = prior.cutoff,
    prior.direction = prior.direction
  )

  criterion_table <- .ecountgmifs.multi.alpha.criteria(object)
  available_criteria <- unique(metric_data$selected$criterion)

  if (is.null(criteria)) {
    criteria <- available_criteria
  } else {
    criteria <- as.character(criteria)
    unknown <- setdiff(criteria, available_criteria)

    if (length(unknown) > 0L) {
      stop(
        "Unknown criterion name(s): ",
        paste(unknown, collapse = ", "),
        call. = FALSE
      )
    }
  }

  styles <- .ecountgmifs.selection.styles(length(criteria))
  styles$pch <- rep_len(pch, length(criteria))

  if (!is.null(col)) {
    styles$col <- rep_len(col, length(criteria))
  }

  best_alpha <- .ecountgmifs.best.alpha.rows(
    criterion_table,
    criteria
  )

  outside_legend <- identical(legend.position, "outside")

  if (isTRUE(.manage.par)) {
    old_par <- graphics::par(no.readonly = TRUE)

    on.exit(
      {
        if (outside_legend) {
          graphics::layout(matrix(1L))
        }
        graphics::par(old_par)
      },
      add = TRUE
    )
  }

  if (isTRUE(.manage.ask)) {
    old_ask <- grDevices::devAskNewPage(
      ask = isTRUE(ask) && length(measure) > 1L
    )
    on.exit(
      grDevices::devAskNewPage(old_ask),
      add = TRUE
    )
  }

  for (current_measure in measure) {
    if (outside_legend) {
      layout.legend.labels <- c(
        criteria,
        best.label,
        paste0("Maximum ", current_measure, " along path"),
        "Saved path states",
        if (!is.null(metric_data$prior)) {
          "Prior reference"
        } else {
          character()
        }
      )
      current.legend.width <- .ecountgmifs.relative.legend.width(
        labels = layout.legend.labels,
        cex = legend.cex,
        plot.width = 4.5
      )

      graphics::layout(
        matrix(c(1L, 2L), nrow = 1L),
        widths = c(4.5, current.legend.width)
      )
      graphics::par(
        mar = c(5.1, 5.1, 5.6, 1.0) + 0.1
      )
    }

    path_values <- lapply(
      metric_data$objects,
      function(value) {
        current <- as.numeric(
          value$path[[current_measure]]
        )
        current[is.finite(current)]
      }
    )

    names(path_values) <- object$alpha_labels

    current_ylab <- if (is.null(ylab)) {
      switch(
        current_measure,
        F1 = "F1-score",
        precision = "Precision",
        recall = "Recall"
      )
    } else {
      ylab
    }

    current_main <- if (is.null(main)) {
      paste0(
        "Path-state distribution across ",
        comparison.label,
        ": ",
        current_measure,
        if (
            !is.null(context.label) &&
            nzchar(as.character(context.label[[1L]]))
        ) {
          paste0(" (", as.character(context.label[[1L]]), ")")
        } else {
          ""
        }
      )
    } else {
      main
    }

    boxplot.arguments <- list(...)
    requested.main.cex <- if (is.null(boxplot.arguments$cex.main)) {
      1.2
    } else {
      as.numeric(boxplot.arguments$cex.main[[1L]])
    }
    if (!is.finite(requested.main.cex) || requested.main.cex <= 0) {
      requested.main.cex <- 1.2
    }
    boxplot.arguments$cex.main <- NULL
    boxplot.arguments$main <- NULL

    title.layout <- .ecountgmifs.relative.title(
      current_main,
      cex = requested.main.cex,
      max.lines = 2L
    )

    do.call(
      graphics::boxplot,
      c(
        list(
          x = path_values,
          names = object$alpha_labels,
          outline = FALSE,
          col = box.col,
          border = box.border,
          xlab = xlab,
          ylab = current_ylab,
          main = title.layout$text,
          cex.main = title.layout$cex
        ),
        boxplot.arguments
      )
    )

    for (index in seq_along(path_values)) {
      values <- path_values[[index]]

      if (length(values) == 0L) {
        next
      }

      offsets <- rep(
        seq(
          -0.12,
          0.12,
          length.out = 11L
        ),
        length.out = length(values)
      )

      graphics::points(
        index + offsets,
        values,
        pch = 16L,
        cex = path.point.cex,
        col = grDevices::adjustcolor(
          path.point.col,
          alpha.f = path.point.alpha
        )
      )
    }

    selected <- metric_data$selected[
      metric_data$selected$criterion %in% criteria,
      ,
      drop = FALSE
    ]

    for (criterion_index in seq_along(criteria)) {
      criterion_name <- criteria[[criterion_index]]
      current <- selected[
        selected$criterion == criterion_name,
        ,
        drop = FALSE
      ]

      alpha_index <- match(
        current$alpha,
        object$alpha
      )

      graphics::points(
        alpha_index,
        current[[current_measure]],
        pch = styles$pch[[criterion_index]],
        col = styles$col[[criterion_index]],
        bg = styles$col[[criterion_index]],
        cex = criterion.point.cex
      )

      best <- best_alpha[
        best_alpha$criterion == criterion_name,
        ,
        drop = FALSE
      ]

      if (nrow(best) > 0L) {
        best_index <- match(
          best$alpha[[1L]],
          current$alpha
        )

        if (!is.na(best_index)) {
          graphics::points(
            alpha_index[[best_index]],
            current[[current_measure]][[best_index]],
            pch = 1L,
            col = styles$col[[criterion_index]],
            cex = best.alpha.cex,
            lwd = best.alpha.lwd
          )
        }
      }
    }

    maximum_position <- match(
      metric_data$maximum$alpha,
      object$alpha
    )

    graphics::points(
      maximum_position,
      metric_data$maximum[[current_measure]],
      pch = maximum.pch,
      cex = maximum.cex,
      lwd = 1.5
    )

    legend_labels <- c(
      criteria,
      best.label,
      paste0(
        "Maximum ",
        current_measure,
        " along path"
      ),
      "Saved path states"
    )
    legend_col <- c(
      styles$col,
      "black",
      "black",
      grDevices::adjustcolor(
        path.point.col,
        alpha.f = max(path.point.alpha, 0.45)
      )
    )
    legend_pch <- c(
      styles$pch,
      1L,
      maximum.pch,
      16L
    )
    legend_lty <- rep(
      NA_integer_,
      length(legend_labels)
    )

    if (!is.null(metric_data$prior)) {
      prior_value <- as.numeric(
        metric_data$prior[[current_measure]]
      )

      if (
          length(prior_value) == 1L &&
          is.finite(prior_value)
      ) {
        graphics::abline(
          h = prior_value,
          col = "#D62728",
          lty = prior.lty,
          lwd = 1.5
        )

        legend_labels <- c(
          legend_labels,
          "Prior reference"
        )
        legend_col <- c(
          legend_col,
          "#D62728"
        )
        legend_pch <- c(
          legend_pch,
          NA_integer_
        )
        legend_lty <- c(
          legend_lty,
          prior.lty
        )
      }
    }

    if (outside_legend) {
      graphics::par(
        mar = c(0.5, 0.2, 0.5, 0.2) + 0.1
      )
      .ecountgmifs.legend.panel(
        legend = legend_labels,
        col = legend_col,
        pch = legend_pch,
        lty = legend_lty,
        pt.bg = legend_col,
        bty = "n",
        cex = legend.cex,
        align = "auto"
      )
    } else {
      .ecountgmifs.draw.legend(
        legend.position,
        legend = legend_labels,
        col = legend_col,
        pch = legend_pch,
        lty = legend_lty,
        pt.bg = legend_col,
        bty = "n",
        cex = legend.cex
      )
    }
  }

  invisible(metric_data)
}



# -----------------------------------------------------------------------------
# Multi-alpha diagnostic page layouts
# -----------------------------------------------------------------------------

.ecountgmifs.multi.diagnostic.combine <- function(
    combine,
    diagnostic_count
) {
  if (is.logical(combine) && length(combine) == 1L && !is.na(combine)) {
    if (!combine) {
      return("none")
    }

    # For a single diagnostic, a combined page is most useful when it compares
    # all alpha values. For several diagnostics, the natural plot.lm-like page
    # is one diagnostic panel set per alpha.
    if (diagnostic_count == 1L) {
      return("by.diagnostic")
    }

    return("by.alpha")
  }

  combine <- as.character(combine)

  if (identical(combine, "grid")) {
    combine <- "all"
  }

  match.arg(
    combine,
    c("none", "by.alpha", "by.diagnostic", "all")
  )
}



.ecountgmifs.plot.diagnostic.across.alpha <- function(
    diagnostic_objects,
    diagnostic,
    alpha_labels,
    page.main,
    plot.arguments
) {
  plot.arguments$which <- NULL
  plot.arguments$ask <- NULL
  plot.arguments$combine <- NULL
  plot.arguments$legend.position <- NULL
  plot.arguments$.manage.ask <- NULL
  plot.arguments$main <- NULL

  legend.cex <- if (is.null(plot.arguments$legend.cex)) {
    1.08
  } else {
    plot.arguments$legend.cex
  }

  first <- diagnostic_objects[[1L]]
  legend_panel <- !is.null(
    .ecountgmifs.diagnostic.legend.spec(
      first,
      diagnostic
    )
  )

  page_layout <- .ecountgmifs.panel.layout(
    panel.count = length(diagnostic_objects),
    legend = legend_panel
  )

  old_par <- graphics::par(no.readonly = TRUE)
  on.exit(
    {
      graphics::layout(matrix(1L))
      graphics::par(old_par)
    },
    add = TRUE
  )

  graphics::layout(
    page_layout$matrix,
    widths = page_layout$widths
  )
  graphics::par(
    oma = c(0, 0, 2.2, 0),
    mar = c(4.2, 4.2, 3.0, 1.2) + 0.1
  )

  for (index in seq_along(diagnostic_objects)) {
    arguments <- c(
      list(
        x = diagnostic_objects[[index]],
        which = diagnostic,
        ask = FALSE,
        combine = FALSE,
        legend.position = "none",
        .manage.ask = FALSE,
        main = paste0(
          "alpha = ",
          alpha_labels[[index]]
        )
      ),
      plot.arguments
    )

    do.call(
      plot,
      arguments
    )
  }

  if (legend_panel) {
    .ecountgmifs.draw.diagnostic.legend.panel(
      first,
      diagnostic,
      cex = legend.cex
    )
  }

  graphics::mtext(
    page.main,
    side = 3,
    outer = TRUE,
    line = 0.5,
    font = 2
  )

  invisible(NULL)
}




.ecountgmifs.plot.diagnostics.matrix <- function(
    diagnostic_objects,
    diagnostics,
    alpha_labels,
    page.main,
    plot.arguments
) {
  alpha_count <- length(diagnostic_objects)
  diagnostic_count <- length(diagnostics)

  if (alpha_count == 0L || diagnostic_count == 0L) {
    return(invisible(NULL))
  }

  point.cex <- if (is.null(plot.arguments$point.cex)) {
    0.48
  } else {
    as.numeric(plot.arguments$point.cex[[1L]])
  }
  smooth <- if (is.null(plot.arguments$smooth)) {
    TRUE
  } else {
    isTRUE(plot.arguments$smooth)
  }
  log.mean.variance <- if (is.null(plot.arguments$log.mean.variance)) {
    TRUE
  } else {
    isTRUE(plot.arguments$log.mean.variance)
  }
  log.overdispersion.x <- if (
      is.null(plot.arguments$log.overdispersion.x)
  ) {
    TRUE
  } else {
    isTRUE(plot.arguments$log.overdispersion.x)
  }
  legend.cex <- if (is.null(plot.arguments$legend.cex)) {
    1.08
  } else {
    plot.arguments$legend.cex
  }

  alpha_styles <- .ecountgmifs.selection.styles(alpha_count)
  alpha_col <- alpha_styles$col
  names(alpha_col) <- alpha_labels

  safe_range <- function(value, fallback = c(0, 1), padding = 0.04) {
    value <- as.numeric(value)
    value <- value[is.finite(value)]

    if (length(value) == 0L) {
      return(fallback)
    }

    out <- range(value)

    if (diff(out) == 0) {
      delta <- max(1, abs(out[[1L]])) * padding
    } else {
      delta <- diff(out) * padding
    }

    out + c(-delta, delta)
  }

  first <- diagnostic_objects[[1L]]
  single.legend.width <- if (diagnostic_count == 1L) {
    current.spec <- .ecountgmifs.diagnostic.legend.spec(
      first,
      diagnostics[[1L]]
    )
    alpha.labels <- paste0("alpha = ", alpha_labels)
    spec.labels <- if (is.null(current.spec)) {
      character()
    } else {
      current.spec$legend
    }
    .ecountgmifs.relative.legend.width(
      labels = c(alpha.labels, spec.labels),
      title = if (is.null(current.spec)) NULL else current.spec$section,
      cex = legend.cex,
      plot.width = 4.6
    )
  } else {
    1.05
  }

  # A five-diagnostic page naturally leaves one cell in a 2 x 3 layout.
  # That cell is used for the shared alpha and line-type legend.
  page_layout <- .ecountgmifs.panel.layout(
    panel.count = diagnostic_count,
    legend = TRUE,
    legend.width = single.legend.width
  )

  # A single diagnostic otherwise inherits full-device base text sizes, which
  # are visually oversized beside its dedicated legend. Scale the complete
  # one-panel page while leaving the five-panel overview unchanged.
  page_cex <- if (diagnostic_count == 1L) 0.82 else 1.0

  old_par <- graphics::par(no.readonly = TRUE)
  on.exit(
    {
      graphics::layout(matrix(1L))
      graphics::par(old_par)
    },
    add = TRUE
  )

  graphics::layout(
    page_layout$matrix,
    widths = page_layout$widths
  )
  graphics::par(
    oma = c(0, 0, 2.3, 0),
    mar = c(4.0, 4.2, 2.8, 1.0) + 0.1,
    cex = page_cex
  )

  for (diagnostic in diagnostics) {
    if (identical(diagnostic, "mean.variance")) {
      value <- first$mean_variance
      keep <-
        value$nonnegative &
        is.finite(value$mean) &
        is.finite(value$variance) &
        value$mean >= 0 &
        value$variance >= 0
      value <- value[keep, , drop = FALSE]

      if (nrow(value) == 0L) {
        graphics::plot.new()
        graphics::title("Mean-variance diagnostic")
        graphics::text(0.5, 0.5, "No finite non-negative data")
        next
      }

      x_value <- value$mean
      y_value <- value$variance
      prefix <- ""

      if (log.mean.variance) {
        x_value <- log1p(x_value)
        y_value <- log1p(y_value)
        prefix <- "Log1p "
      }

      role <- factor(
        value$role,
        levels = c("penalized predictor", "response")
      )
      point_col <- c("#D55E00", "#0072B2")[as.integer(role)]
      point_pch <- c(16L, 17L)[as.integer(role)]
      limit <- safe_range(c(x_value, y_value), c(0, 1), 0.02)

      graphics::plot(
        x_value,
        y_value,
        xlim = limit,
        ylim = limit,
        pch = point_pch,
        col = grDevices::adjustcolor(
          point_col,
          alpha.f = 0.55
        ),
        cex = point.cex,
        xlab = paste0(prefix, "mean of observed values"),
        ylab = paste0(prefix, "variance of observed values"),
        main = "Mean-variance (common data)"
      )
      graphics::abline(
        0,
        1,
        lty = 3L,
        lwd = 1.5
      )
    }

    if (identical(diagnostic, "dispersion.path")) {
      paths <- lapply(
        diagnostic_objects,
        function(value) {
          out <- value$dispersion_path
          out <- out[
            out$source %in% c("null", "path") &
            is.finite(out$iteration) &
            is.finite(out$dispersion),
            ,
            drop = FALSE
          ]
          out
        }
      )

      xlim <- safe_range(
        unlist(lapply(paths, `[[`, "iteration")),
        c(0, 1)
      )
      ylim <- safe_range(
        unlist(lapply(paths, `[[`, "dispersion")),
        c(0, 1)
      )

      graphics::plot(
        NA_real_,
        NA_real_,
        type = "n",
        xlim = xlim,
        ylim = ylim,
        xlab = "Iteration",
        ylab = if (
            grepl(
              "nb2|negative",
              tolower(first$family)
            )
        ) {
          "NB2 dispersion parameter"
        } else {
          "First family parameter"
        },
        main = "Family-parameter paths"
      )

      for (index in seq_along(paths)) {
        current <- paths[[index]]

        if (nrow(current) == 0L) {
          next
        }

        graphics::lines(
          current$iteration,
          current$dispersion,
          col = alpha_col[[index]],
          lwd = 2
        )
      }
    }

    if (identical(diagnostic, "residuals.fitted")) {
      values <- lapply(
        diagnostic_objects,
        `[[`,
        "residuals"
      )

      xlim <- safe_range(
        unlist(lapply(values, `[[`, "fitted")),
        c(0, 1)
      )
      ylim <- safe_range(
        unlist(lapply(values, `[[`, "residual")),
        c(-1, 1)
      )

      graphics::plot(
        NA_real_,
        NA_real_,
        type = "n",
        xlim = xlim,
        ylim = ylim,
        xlab = "Fitted mean",
        ylab = paste0(
          first$residual_type,
          " residual"
        ),
        main = "Residuals vs fitted"
      )
      graphics::abline(h = 0, lty = 3L)

      for (index in seq_along(values)) {
        current <- values[[index]]
        current_col <- alpha_col[[index]]

        graphics::points(
          current$fitted,
          current$residual,
          pch = 16L,
          cex = point.cex,
          col = grDevices::adjustcolor(
            current_col,
            alpha.f = 0.15
          )
        )

        if (smooth && nrow(current) >= 3L) {
          lowess_value <- stats::lowess(
            current$fitted,
            current$residual
          )
          graphics::lines(
            lowess_value,
            col = current_col,
            lwd = 2
          )
        }
      }
    }

    if (identical(diagnostic, "scale.location")) {
      values <- lapply(
        diagnostic_objects,
        `[[`,
        "residuals"
      )

      xlim <- safe_range(
        unlist(lapply(values, `[[`, "fitted")),
        c(0, 1)
      )
      ylim <- safe_range(
        unlist(
          lapply(
            values,
            `[[`,
            "sqrt_abs_residual"
          )
        ),
        c(0, 1)
      )

      graphics::plot(
        NA_real_,
        NA_real_,
        type = "n",
        xlim = xlim,
        ylim = ylim,
        xlab = "Fitted mean",
        ylab = paste0(
          "sqrt(|",
          first$residual_type,
          " residual|)"
        ),
        main = "Scale-location"
      )

      for (index in seq_along(values)) {
        current <- values[[index]]
        current_col <- alpha_col[[index]]

        graphics::points(
          current$fitted,
          current$sqrt_abs_residual,
          pch = 16L,
          cex = point.cex,
          col = grDevices::adjustcolor(
            current_col,
            alpha.f = 0.15
          )
        )

        if (smooth && nrow(current) >= 3L) {
          lowess_value <- stats::lowess(
            current$fitted,
            current$sqrt_abs_residual
          )
          graphics::lines(
            lowess_value,
            col = current_col,
            lwd = 2
          )
        }
      }
    }

    if (identical(diagnostic, "overdispersion")) {
      values <- lapply(
        diagnostic_objects,
        function(value) {
          out <- value$overdispersion
          out <- out[
            is.finite(out$fitted_mean) &
            is.finite(out$empirical),
            ,
            drop = FALSE
          ]
          out
        }
      )

      x_values <- lapply(
        values,
        function(value) {
          if (log.overdispersion.x) {
            log1p(value$fitted_mean)
          } else {
            value$fitted_mean
          }
        }
      )

      xlim <- safe_range(
        unlist(x_values),
        c(0, 1)
      )
      ylim <- safe_range(
        c(
          unlist(lapply(values, `[[`, "empirical")),
          unlist(lapply(values, `[[`, "expected")),
          if (
              identical(
                first$overdispersion_kind,
                "count"
              )
          ) 1 else numeric()
        ),
        c(0, 1)
      )

      graphics::plot(
        NA_real_,
        NA_real_,
        type = "n",
        xlim = xlim,
        ylim = ylim,
        xlab = if (log.overdispersion.x) {
          "Log1p fitted mean"
        } else {
          "Fitted mean"
        },
        ylab = if (
            identical(
              first$overdispersion_kind,
              "count"
            )
        ) {
          "Empirical variance-to-mean ratio"
        } else {
          "Binned residual variance"
        },
        main = "Overdispersion / variance"
      )

      if (
          identical(
            first$overdispersion_kind,
            "count"
          )
      ) {
        graphics::abline(
          h = 1,
          lty = 3L,
          lwd = 1.5
        )
      }

      for (index in seq_along(values)) {
        current <- values[[index]]
        current_x <- x_values[[index]]
        current_col <- alpha_col[[index]]

        if (nrow(current) == 0L) {
          next
        }

        graphics::lines(
          current_x,
          current$empirical,
          col = current_col,
          lwd = 1.5
        )
        graphics::points(
          current_x,
          current$empirical,
          pch = 16L,
          cex = point.cex,
          col = current_col
        )

        expected <- is.finite(current$expected)

        if (any(expected)) {
          graphics::lines(
            current_x[expected],
            current$expected[expected],
            col = current_col,
            lty = 2L,
            lwd = 2
          )
        }
      }
    }
  }

  # Shared legend: alpha is encoded by colour, while line type retains the
  # diagnostic meaning.  It occupies the unused sixth panel for five
  # diagnostics, so no data panel is shrunk or covered.
  legend_labels <- paste0(
    "alpha = ",
    alpha_labels
  )
  legend_col <- alpha_col
  legend_lty <- rep(1L, alpha_count)
  legend_pch <- rep(NA_integer_, alpha_count)

  if ("mean.variance" %in% diagnostics) {
    legend_labels <- c(
      legend_labels,
      "Mean-variance: predictors",
      "Mean-variance: response",
      "Mean-variance: Poisson line"
    )
    legend_col <- c(
      legend_col,
      "#D55E00",
      "#0072B2",
      "black"
    )
    legend_lty <- c(
      legend_lty,
      NA_integer_,
      NA_integer_,
      3L
    )
    legend_pch <- c(
      legend_pch,
      16L,
      17L,
      NA_integer_
    )
  }

  if ("overdispersion" %in% diagnostics) {
    legend_labels <- c(
      legend_labels,
      "Overdispersion: empirical",
      "Overdispersion: fitted variance"
    )
    legend_col <- c(
      legend_col,
      "black",
      "black"
    )
    legend_lty <- c(
      legend_lty,
      1L,
      2L
    )
    legend_pch <- c(
      legend_pch,
      16L,
      NA_integer_
    )

    if (
        identical(
          first$overdispersion_kind,
          "count"
        )
    ) {
      legend_labels <- c(
        legend_labels,
        "Overdispersion: Poisson line"
      )
      legend_col <- c(
        legend_col,
        "black"
      )
      legend_lty <- c(
        legend_lty,
        3L
      )
      legend_pch <- c(
        legend_pch,
        NA_integer_
      )
    }
  }

  .ecountgmifs.legend.panel(
    legend = legend_labels,
    col = legend_col,
    lty = legend_lty,
    pch = legend_pch,
    lwd = 2,
    bty = "n",
    cex = legend.cex,
    cex.min = 0.55,
    max.columns = 2L,
    align = "top-left"
  )

  graphics::mtext(
    page.main,
    side = 3,
    outer = TRUE,
    line = 0.45,
    font = 2
  )

  invisible(NULL)
}



#' @export
plot.ecountgmifs_multi_alpha <- function(
    x,
    type = c(
      "coefficients",
      "criteria",
      "metrics",
      "diagnostics",
      "overdispersion",
      "alpha",
      "alpha.distribution",
      "alpha.boxplot"
    ),
    alpha = NULL,
    ask = interactive(),
    combine = FALSE,
    ground.truth = NULL,
    ...
) {
  type <- match.arg(type)

  if (identical(type, "alpha")) {
    return(
      .ecountgmifs.plot.alpha.comparison(
        object = x,
        ground.truth = ground.truth,
        ask = ask,
        ...
      )
    )
  }

  if (identical(type, "alpha.boxplot")) {
    warning(
      "`type = \"alpha.boxplot\"` is deprecated; use ",
      "`type = \"alpha.distribution\"`.",
      call. = FALSE
    )
    type <- "alpha.distribution"
  }

  if (identical(type, "alpha.distribution")) {
    return(
      .ecountgmifs.plot.alpha.distribution(
        object = x,
        ground.truth = ground.truth,
        ask = ask,
        ...
      )
    )
  }

  indices <- .ecountgmifs.multi.alpha.select(
    x,
    alpha = alpha
  )
  supplied_dots <- list(...)

  if (type %in% c("diagnostics", "overdispersion")) {
    diagnostic_which <- if (identical(type, "overdispersion")) {
      "overdispersion"
    } else if (is.null(supplied_dots$which)) {
      1:5
    } else {
      supplied_dots$which
    }
    diagnostic_which <- .ecountgmifs.diagnostic.which(diagnostic_which)
    combine_mode <- .ecountgmifs.multi.diagnostic.combine(
      combine,
      length(diagnostic_which)
    )

    diagnostic_criterion <- supplied_dots$criterion
    diagnostic_iteration <- supplied_dots$iteration
    supplied_dots$which <- NULL
    supplied_dots$ask <- NULL
    supplied_dots$combine <- NULL
    supplied_dots$criterion <- NULL
    supplied_dots$iteration <- NULL

    results <- vector("list", length(indices))
    names(results) <- names(x$fits)[indices]

    for (position in seq_along(indices)) {
      index <- indices[[position]]
      results[[position]] <- ecountgmifs.diagnostics(
        x$fits[[index]],
        criterion = diagnostic_criterion,
        iteration = diagnostic_iteration
      )
    }

    if (identical(combine_mode, "none")) {
      total_plots <- length(indices) * length(diagnostic_which)
      old_ask <- grDevices::devAskNewPage(
        ask = isTRUE(ask) && total_plots > 1L
      )
      on.exit(
        grDevices::devAskNewPage(old_ask),
        add = TRUE
      )

      for (position in seq_along(indices)) {
        index <- indices[[position]]

        for (current_which in diagnostic_which) {
          dots <- supplied_dots

          if (is.null(dots$main)) {
            dots$main <- paste0(
              "ecountgmifs diagnostics (alpha = ",
              x$alpha_labels[[index]],
              ")"
            )
          }

          do.call(
            plot,
            c(
              list(
                x = results[[position]],
                which = current_which,
                ask = FALSE,
                combine = FALSE,
                .manage.ask = FALSE
              ),
              dots
            )
          )
        }
      }

      return(invisible(results))
    }

    if (identical(combine_mode, "all")) {
      page_main <- supplied_dots$main
      page_arguments <- supplied_dots
      page_arguments$main <- NULL

      .ecountgmifs.plot.diagnostics.matrix(
        diagnostic_objects = results,
        diagnostics = diagnostic_which,
        alpha_labels = x$alpha_labels[indices],
        page.main = if (is.null(page_main)) {
          "ecountgmifs diagnostics across alpha values"
        } else {
          page_main
        },
        plot.arguments = page_arguments
      )

      return(invisible(results))
    }

    if (identical(combine_mode, "by.alpha")) {
      page_main <- supplied_dots$main

      old_ask <- grDevices::devAskNewPage(
        ask = isTRUE(ask) && length(indices) > 1L
      )
      on.exit(
        grDevices::devAskNewPage(old_ask),
        add = TRUE
      )

      for (position in seq_along(indices)) {
        index <- indices[[position]]
        dots <- supplied_dots

        dots$main <- if (is.null(page_main)) {
          paste0(
            "ecountgmifs diagnostics (alpha = ",
            x$alpha_labels[[index]],
            ")"
          )
        } else {
          paste0(
            page_main,
            "; alpha = ",
            x$alpha_labels[[index]]
          )
        }

        do.call(
          plot,
          c(
            list(
              x = results[[position]],
              which = diagnostic_which,
              ask = FALSE,
              combine = TRUE,
              .manage.ask = FALSE
            ),
            dots
          )
        )
      }

      return(invisible(results))
    }

    # One page per diagnostic, with all requested alpha values shown together.
    old_ask <- grDevices::devAskNewPage(
      ask = isTRUE(ask) && length(diagnostic_which) > 1L
    )
    on.exit(
      grDevices::devAskNewPage(old_ask),
      add = TRUE
    )

    page_main <- supplied_dots$main
    page_arguments <- supplied_dots
    page_arguments$main <- NULL

    for (current_which in diagnostic_which) {
      .ecountgmifs.plot.diagnostic.across.alpha(
        diagnostic_objects = results,
        diagnostic = current_which,
        alpha_labels = x$alpha_labels[indices],
        page.main = if (is.null(page_main)) {
          paste0(
            "ecountgmifs diagnostics across alpha: ",
            current_which
          )
        } else {
          paste0(
            page_main,
            ": ",
            current_which
          )
        },
        plot.arguments = page_arguments
      )
    }

    return(invisible(results))
  }

  old_ask <- grDevices::devAskNewPage(
    ask = isTRUE(ask) && length(indices) > 1L
  )
  on.exit(
    grDevices::devAskNewPage(old_ask),
    add = TRUE
  )

  results <- vector("list", length(indices))
  names(results) <- names(x$fits)[indices]

  for (position in seq_along(indices)) {
    index <- indices[[position]]
    fit <- x$fits[[index]]
    dots <- supplied_dots

    if (is.null(dots$main)) {
      dots$main <- paste0(
        switch(
          type,
          coefficients = "ecountgmifs coefficient paths",
          criteria = "ecountgmifs criterion paths",
          metrics = "ecountgmifs selection metrics"
        ),
        " (alpha = ",
        x$alpha_labels[[index]],
        ")"
      )
    }

    arguments <- c(
      list(
        x = fit,
        type = type,
        ground.truth = ground.truth
      ),
      dots
    )

    results[[position]] <- do.call(
      plot,
      arguments
    )
  }

  invisible(results)
}
