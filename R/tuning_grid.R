# -----------------------------------------------------------------------------
# Joint alpha and prior-strength fitting grid
# -----------------------------------------------------------------------------

.ecountgmifs.clean.eta.labels <- function(
    labels = NULL,
    eta = NULL
) {
  if (is.null(labels)) {
    if (is.null(eta)) {
      return(character())
    }

    labels <- format(
      as.numeric(eta),
      digits = 15L,
      trim = TRUE,
      scientific = FALSE
    )
  }

  labels <- trimws(as.character(labels))

  sub(
    "^[[:space:]]*eta[[:space:]]*=[[:space:]]*",
    "",
    labels,
    ignore.case = TRUE
  )
}


.ecountgmifs.eta.context.labels <- function(
    labels = NULL,
    eta = NULL
) {
  paste0(
    "eta = ",
    .ecountgmifs.clean.eta.labels(
      labels = labels,
      eta = eta
    )
  )
}


.ecountgmifs.fit.tuning.grid <- function(
    fit_call,
    alpha,
    weight_prior,
    envir
) {
  alpha <- as.numeric(alpha)
  alpha_labels <- .ecountgmifs.alpha.labels(alpha)
  eta <- as.numeric(weight_prior$eta)
  eta_labels <- .ecountgmifs.clean.eta.labels(
    labels = weight_prior$eta_labels,
    eta = eta
  )

  grid <- expand.grid(
    alpha_index = seq_along(alpha),
    eta_index = seq_along(eta),
    KEEP.OUT.ATTRS = FALSE,
    stringsAsFactors = FALSE
  )
  grid <- grid[order(grid$eta_index, grid$alpha_index), , drop = FALSE]
  rownames(grid) <- NULL
  grid$alpha <- alpha[grid$alpha_index]
  grid$alpha_label <- alpha_labels[grid$alpha_index]
  grid$eta <- eta[grid$eta_index]
  grid$eta_label <- eta_labels[grid$eta_index]

  fits <- vector("list", nrow(grid))

  for (index in seq_len(nrow(grid))) {
    current_call <- fit_call
    current_call$enet.alpha <- grid$alpha[[index]]
    current_call$weight.vec <- weight_prior$weights[[grid$eta_index[[index]]]]

    fit <- eval(
      current_call,
      envir = envir
    )

    fit$input$prior_eta <- grid$eta[[index]]
    fit$input$prior_eta_label <- grid$eta_label[[index]]
    fit$input$prior_selected <- weight_prior$score > 0
    names(fit$input$prior_selected) <- names(weight_prior$score)
    fit$weight_prior <- list(
      label = weight_prior$label,
      eta = grid$eta[[index]],
      eta_label = grid$eta_label[[index]]
    )

    # Keep the stored fit call readable rather than printing a p-length vector.
    fit$call$weight.vec <- substitute(
      weight_prior[eta == ETA],
      list(ETA = grid$eta[[index]])
    )

    fits[[index]] <- fit
  }

  names(fits) <- paste0(
    "alpha=",
    grid$alpha_label,
    ", eta=",
    grid$eta_label
  )

  out <- list(
    call = fit_call,
    alpha = alpha,
    alpha_labels = alpha_labels,
    eta = eta,
    eta_labels = eta_labels,
    grid = grid,
    fits = fits,
    weight_prior = weight_prior
  )

  class(out) <- c("ecountgmifs_tuning_grid", "list")
  out
}


.ecountgmifs.tuning.grid.match <- function(
    values,
    stored,
    labels,
    name
) {
  if (is.null(values)) {
    return(seq_along(stored))
  }

  if (is.character(values)) {
    match_values <- as.character(values)
    match_labels <- as.character(labels)

    if (identical(name, "eta")) {
      match_values <- .ecountgmifs.clean.eta.labels(match_values)
      match_labels <- .ecountgmifs.clean.eta.labels(match_labels)
    }

    index <- match(match_values, match_labels)

    if (anyNA(index)) {
      stop(
        "Unknown ", name, " label(s): ",
        paste(values[is.na(index)], collapse = ", "),
        call. = FALSE
      )
    }

    return(as.integer(index))
  }

  if (
      !is.numeric(values) ||
      length(values) == 0L ||
      anyNA(values) ||
      any(!is.finite(values))
  ) {
    stop(
      "`", name, "` must be NULL, finite numeric values, or stored labels.",
      call. = FALSE
    )
  }

  vapply(
    as.numeric(values),
    function(value) {
      difference <- abs(stored - value)
      index <- which.min(difference)
      tolerance <- 100 * .Machine$double.eps * max(1, abs(value))

      if (difference[[index]] > tolerance) {
        stop(
          name,
          " ",
          format(value, digits = 15L),
          " is not present. Available values: ",
          paste(labels, collapse = ", "),
          call. = FALSE
        )
      }

      as.integer(index)
    },
    integer(1L)
  )
}


.ecountgmifs.tuning.grid.select <- function(
    object,
    alpha = NULL,
    eta = NULL
) {
  alpha_index <- .ecountgmifs.tuning.grid.match(
    alpha,
    object$alpha,
    object$alpha_labels,
    "alpha"
  )
  eta_index <- .ecountgmifs.tuning.grid.match(
    eta,
    object$eta,
    object$eta_labels,
    "eta"
  )

  which(
    object$grid$alpha_index %in% alpha_index &
    object$grid$eta_index %in% eta_index
  )
}


.ecountgmifs.tuning.grid.overview <- function(object) {
  do.call(
    rbind,
    lapply(
      seq_along(object$fits),
      function(index) {
        fit <- object$fits[[index]]
        terminal <- .ecountgmifs.terminal.state(fit)
        beta <- terminal$predictors$parameters$beta

        data.frame(
          alpha = object$grid$alpha[[index]],
          eta = object$grid$eta[[index]],
          iteration = as.integer(terminal$iteration),
          nonzero = sum(beta != 0),
          negloglik = as.numeric(terminal$negloglik),
          pseudo_r2 = as.numeric(terminal$pseudo_r2),
          total_seconds = as.numeric(fit$path$total_time),
          stringsAsFactors = FALSE
        )
      }
    )
  )
}


#' @export
print.ecountgmifs_tuning_grid <- function(
    x,
    digits = max(3L, getOption("digits") - 3L),
    ...
) {
  cat("ecountgmifs alpha x prior-strength tuning grid\n")

  if (!is.null(x$call)) {
    cat("\nCall:\n")
    print(x$call)
  }

  cat("\nGrid:\n")
  cat("  Alpha values: ", paste(x$alpha_labels, collapse = ", "), "\n", sep = "")
  cat(
    "  Eta values:   ",
    paste(
      .ecountgmifs.clean.eta.labels(
        labels = x$eta_labels,
        eta = x$eta
      ),
      collapse = ", "
    ),
    "\n",
    sep = ""
  )
  cat("  Fits:         ", length(x$fits), "\n", sep = "")

  cat("\nTerminal states:\n")
  print(
    .ecountgmifs.tuning.grid.overview(x),
    row.names = FALSE,
    digits = digits
  )

  invisible(x)
}


#' @export
summary.ecountgmifs_tuning_grid <- function(object, ...) {
  out <- list(
    call = object$call,
    alpha = object$alpha,
    eta = object$eta,
    overview = .ecountgmifs.tuning.grid.overview(object)
  )
  class(out) <- "summary.ecountgmifs_tuning_grid"
  out
}


#' @export
print.summary.ecountgmifs_tuning_grid <- function(
    x,
    digits = max(3L, getOption("digits") - 3L),
    ...
) {
  cat("ecountgmifs tuning-grid summary\n")

  if (!is.null(x$call)) {
    cat("\nCall:\n")
    print(x$call)
  }

  cat("\nTerminal states:\n")
  print(
    x$overview,
    row.names = FALSE,
    digits = digits
  )

  invisible(x)
}


#' Extract coefficients from an alpha by prior-strength tuning grid
#'
#' @param object An `ecountgmifs_tuning_grid` object.
#' @param criterion,iteration,include.theta,state,drop Passed to
#'   [coef.ecountgmifs()].
#' @param alpha Optional stored alpha value or values, or their labels.
#' @param eta Optional stored prior-strength value or values, or their labels.
#' @param ... Additional arguments passed to [coef.ecountgmifs()].
#'
#' @return A named list for multiple alpha by eta fits. When exactly one fit is
#'   selected and `drop = TRUE`, its coefficient result is returned directly.
#' @export
coef.ecountgmifs_tuning_grid <- function(
    object,
    criterion = NULL,
    iteration = NULL,
    include.theta = FALSE,
    alpha = NULL,
    eta = NULL,
    state = NULL,
    drop = TRUE,
    ...
) {
  indices <- .ecountgmifs.tuning.grid.select(
    object,
    alpha = alpha,
    eta = eta
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

  names(out) <- names(object$fits)[indices]

  if (length(out) == 1L && isTRUE(drop)) {
    return(out[[1L]])
  }

  out
}


.ecountgmifs.tuning.as.multi.alpha <- function(
    object,
    eta_index
) {
  rows <- which(object$grid$eta_index == eta_index)
  rows <- rows[order(object$grid$alpha_index[rows])]

  out <- list(
    call = object$call,
    alpha = object$grid$alpha[rows],
    alpha_labels = object$grid$alpha_label[rows],
    fits = object$fits[rows]
  )
  class(out) <- c("ecountgmifs_multi_alpha", "list")
  out
}


.ecountgmifs.tuning.as.multi.eta <- function(
    object,
    alpha_index
) {
  rows <- which(object$grid$alpha_index == alpha_index)
  rows <- rows[order(object$grid$eta_index[rows])]

  out <- list(
    call = object$call,
    alpha = object$grid$eta[rows],
    alpha_labels = .ecountgmifs.clean.eta.labels(
      labels = object$grid$eta_label[rows],
      eta = object$grid$eta[rows]
    ),
    fits = object$fits[rows]
  )
  class(out) <- c("ecountgmifs_multi_alpha", "list")
  out
}


.ecountgmifs.tuning.legend.spec <- function(
    object,
    ground.truth,
    measure,
    distribution = FALSE,
    criteria = NULL,
    col = NULL,
    pch = c(21L, 22L, 23L, 24L, 25L),
    prior.available = FALSE
) {
  criterion_table <- .ecountgmifs.multi.alpha.criteria(object)
  available <- unique(criterion_table$criterion)

  if (is.null(criteria)) {
    criteria <- available
  }

  styles <- .ecountgmifs.selection.styles(length(criteria))
  styles$pch <- rep_len(pch, length(criteria))

  if (!is.null(col)) {
    styles$col <- rep_len(col, length(criteria))
  }

  labels <- c(
    criteria,
    "Best value by criterion"
  )
  legend_col <- c(
    styles$col,
    "black"
  )
  legend_pch <- c(
    styles$pch,
    1L
  )
  legend_lty <- c(
    rep(1L, length(criteria)),
    NA_integer_
  )

  if (!is.null(ground.truth)) {
    labels <- c(
      labels,
      paste0("Maximum ", measure, " along path")
    )
    legend_col <- c(
      legend_col,
      "black"
    )
    legend_pch <- c(
      legend_pch,
      if (distribution) 4L else NA_integer_
    )
    legend_lty <- c(
      legend_lty,
      if (distribution) NA_integer_ else 3L
    )

    if (distribution) {
      labels <- c(
        labels,
        "Saved path states"
      )
      legend_col <- c(
        legend_col,
        grDevices::adjustcolor(
          "grey35",
          alpha.f = 0.55
        )
      )
      legend_pch <- c(
        legend_pch,
        16L
      )
      legend_lty <- c(
        legend_lty,
        NA_integer_
      )
    }

    if (isTRUE(prior.available)) {
      labels <- c(
        labels,
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
        2L
      )
    }
  }

  list(
    legend = labels,
    col = legend_col,
    pch = legend_pch,
    lty = legend_lty,
    pt.bg = legend_col
  )
}


.ecountgmifs.plot.tuning.combined <- function(
    objects,
    labels,
    comparison = c("alpha", "prior"),
    distribution = FALSE,
    ground.truth = NULL,
    measures,
    ask,
    supplied.dots
) {
  comparison <- match.arg(comparison)
  measures <- as.character(measures)

  if (length(objects) == 0L) {
    return(invisible(list()))
  }

  legend.cex <- if (is.null(supplied.dots$legend.cex)) {
    1.08
  } else {
    supplied.dots$legend.cex
  }

  prior.available <-
    !is.null(ground.truth) &&
    (
      !is.null(supplied.dots$prior.selected) ||
      !is.null(supplied.dots$prior.weight.vec) ||
      !is.null(objects[[1L]]$fits[[1L]]$input$prior_selected) ||
      isTRUE(objects[[1L]]$fits[[1L]]$input$has_prior)
    )

  legend_spec <- .ecountgmifs.tuning.legend.spec(
    object = objects[[1L]],
    ground.truth = ground.truth,
    measure = measures[[1L]],
    distribution = distribution,
    criteria = supplied.dots$criteria,
    col = supplied.dots$col,
    pch = if (is.null(supplied.dots$pch)) {
      c(21L, 22L, 23L, 24L, 25L)
    } else {
      supplied.dots$pch
    },
    prior.available = prior.available
  )

  # Three comparison panels fit more naturally into a 2 x 2 page with the
  # shared legend in the fourth cell. The previous 3 x 1 plus external legend
  # layout left a large horizontal gap between the panels and their key.
  combined_mfrow <- if (length(objects) == 3L) {
    c(2L, 2L)
  } else {
    NULL
  }

  page_layout <- .ecountgmifs.panel.layout(
    panel.count = length(objects),
    mfrow = combined_mfrow,
    legend = TRUE
  )

  old_par <- graphics::par(no.readonly = TRUE)
  on.exit(
    {
      graphics::layout(matrix(1L))
      graphics::par(old_par)
    },
    add = TRUE
  )

  old_ask <- grDevices::devAskNewPage(
    ask = isTRUE(ask) && length(measures) > 1L
  )
  on.exit(
    grDevices::devAskNewPage(old_ask),
    add = TRUE
  )

  results <- vector("list", length(measures))
  names(results) <- measures

  for (measure_index in seq_along(measures)) {
    current_measure <- measures[[measure_index]]

    graphics::layout(
      page_layout$matrix,
      widths = page_layout$widths
    )
    graphics::par(
      oma = c(0, 0, 3.4, 0),
      mar = c(4.4, 4.5, 3.0, 1.0) + 0.1
    )

    page_results <- vector(
      "list",
      length(objects)
    )
    names(page_results) <- labels

    for (index in seq_along(objects)) {
      current_dots <- supplied.dots
      current_dots$measure <- NULL
      current_dots$legend.position <- NULL
      current_dots$main <- NULL
      current_dots$ask <- NULL
      current_dots$context.label <- NULL
      current_dots$.manage.ask <- NULL
      current_dots$.manage.par <- NULL

      helper <- if (distribution) {
        .ecountgmifs.plot.alpha.distribution
      } else {
        .ecountgmifs.plot.alpha.comparison
      }

      page_results[[index]] <- do.call(
        helper,
        c(
          list(
            object = objects[[index]],
            ground.truth = ground.truth,
            measure = current_measure,
            ask = FALSE,
            .manage.ask = FALSE,
            .manage.par = FALSE,
            legend.position = "none",
            main = labels[[index]]
          ),
          current_dots
        )
      )
    }

    # The shared legend occupies a naturally empty cell where possible.
    legend_spec <- .ecountgmifs.tuning.legend.spec(
      object = objects[[1L]],
      ground.truth = ground.truth,
      measure = current_measure,
      distribution = distribution,
      criteria = supplied.dots$criteria,
      col = supplied.dots$col,
      pch = if (is.null(supplied.dots$pch)) {
        c(21L, 22L, 23L, 24L, 25L)
      } else {
        supplied.dots$pch
      },
      prior.available = prior.available
    )

    .ecountgmifs.legend.panel(
      legend = legend_spec$legend,
      col = legend_spec$col,
      pch = legend_spec$pch,
      lty = legend_spec$lty,
      pt.bg = legend_spec$pt.bg,
      lwd = 2,
      bty = "n",
      cex = legend.cex,
      cex.min = 0.55,
      max.columns = 2L,
      align = "auto"
    )

    page.title <- paste0(
      if (comparison == "alpha") {
        "Alpha comparison across prior strengths: "
      } else {
        "Prior-strength comparison across alpha values: "
      },
      current_measure
    )
    title.layout <- .ecountgmifs.relative.title(
      page.title,
      cex = 1,
      outer = TRUE,
      max.lines = 2L
    )
    .ecountgmifs.draw.outer.title(
      title.layout$text,
      line = 0.45,
      font = 2,
      cex = title.layout$cex
    )

    results[[measure_index]] <- page_results
  }

  invisible(results)
}


.ecountgmifs.tuning.selected.table <- function(
    object,
    ground.truth = NULL,
    measures = NULL,
    criteria = NULL,
    zero.tol = sqrt(.Machine$double.eps),
    zero.division = c("zero", "one", "NA"),
    prior.selected = NULL,
    prior.weight.vec = NULL,
    prior.cutoff = 1,
    prior.direction = c("lower", "higher")
) {
  zero.division <- match.arg(zero.division)
  prior.direction <- match.arg(prior.direction)
  has_ground_truth <- !is.null(ground.truth)

  if (is.null(measures)) {
    measures <- if (has_ground_truth) {
      "F1"
    } else {
      "pseudo_r2"
    }
  }

  allowed <- if (has_ground_truth) {
    c("F1", "precision", "recall")
  } else {
    c(
      "nonzero",
      "negloglik",
      "pseudo_r2",
      "iteration",
      "value"
    )
  }

  measures <- match.arg(
    measures,
    allowed,
    several.ok = TRUE
  )

  rows <- lapply(
    seq_along(object$fits),
    function(index) {
      fit <- object$fits[[index]]

      if (has_ground_truth) {
        metric <- ecountgmifs.metrics(
          fit,
          ground.truth = ground.truth,
          zero.tol = zero.tol,
          zero.division = zero.division,
          prior.selected = prior.selected,
          prior.weight.vec = prior.weight.vec,
          prior.cutoff = prior.cutoff,
          prior.direction = prior.direction
        )
        current <- metric$criteria

        if (is.null(current) || nrow(current) == 0L) {
          return(NULL)
        }

        current$value <- current$criterion_value
      } else {
        current <- .ecountgmifs.criterion.table(fit)

        if (nrow(current) == 0L) {
          return(NULL)
        }
      }

      current$alpha <- object$grid$alpha[[index]]
      current$alpha_label <- object$grid$alpha_label[[index]]
      current$alpha_index <- object$grid$alpha_index[[index]]
      current$eta <- object$grid$eta[[index]]
      current$eta_label <- object$grid$eta_label[[index]]
      current$eta_index <- object$grid$eta_index[[index]]
      current
    }
  )

  rows <- Filter(Negate(is.null), rows)

  if (length(rows) == 0L) {
    stop(
      "No criterion-selected states are available in the tuning grid.",
      call. = FALSE
    )
  }

  out <- do.call(rbind, rows)
  rownames(out) <- NULL

  available_criteria <- unique(out$criterion)

  if (!is.null(criteria)) {
    criteria <- as.character(criteria)
    unknown <- setdiff(
      criteria,
      available_criteria
    )

    if (length(unknown) > 0L) {
      stop(
        "Unknown criterion name(s): ",
        paste(unknown, collapse = ", "),
        call. = FALSE
      )
    }

    out <- out[
      out$criterion %in% criteria,
      ,
      drop = FALSE
    ]
  }

  attr(out, "measures") <- measures
  out
}


.ecountgmifs.plot.tuning.surface <- function(
    object,
    ground.truth = NULL,
    measure = NULL,
    criteria = NULL,
    ask = interactive(),
    zero.tol = sqrt(.Machine$double.eps),
    zero.division = c("zero", "one", "NA"),
    prior.selected = NULL,
    prior.weight.vec = NULL,
    prior.cutoff = 1,
    prior.direction = c("lower", "higher"),
    palette = "YlOrRd",
    reverse.palette = FALSE,
    cell.labels = TRUE,
    cell.cex = 0.85,
    best.cex = 2,
    best.lwd = 2,
    key.width = 0.9,
    main = NULL,
    ...
) {
  table <- .ecountgmifs.tuning.selected.table(
    object = object,
    ground.truth = ground.truth,
    measures = measure,
    criteria = criteria,
    zero.tol = zero.tol,
    zero.division = zero.division,
    prior.selected = prior.selected,
    prior.weight.vec = prior.weight.vec,
    prior.cutoff = prior.cutoff,
    prior.direction = prior.direction
  )

  measures <- attr(table, "measures")
  criteria <- unique(table$criterion)
  pages <- expand.grid(
    criterion = criteria,
    measure = measures,
    KEEP.OUT.ATTRS = FALSE,
    stringsAsFactors = FALSE
  )

  old_par <- graphics::par(no.readonly = TRUE)
  on.exit(
    {
      graphics::layout(matrix(1L))
      graphics::par(old_par)
    },
    add = TRUE
  )

  old_ask <- grDevices::devAskNewPage(
    ask = isTRUE(ask) && nrow(pages) > 1L
  )
  on.exit(
    grDevices::devAskNewPage(old_ask),
    add = TRUE
  )

  colours <- grDevices::hcl.colors(
    40L,
    palette = palette,
    rev = isTRUE(reverse.palette)
  )

  for (page_index in seq_len(nrow(pages))) {
    criterion_name <- pages$criterion[[page_index]]
    current_measure <- pages$measure[[page_index]]
    current <- table[
      table$criterion == criterion_name,
      ,
      drop = FALSE
    ]

    z <- matrix(
      NA_real_,
      nrow = length(object$alpha),
      ncol = length(object$eta)
    )

    for (row_index in seq_len(nrow(current))) {
      z[
        current$alpha_index[[row_index]],
        current$eta_index[[row_index]]
      ] <- as.numeric(
        current[[current_measure]][[row_index]]
      )
    }

    finite <- z[is.finite(z)]

    if (length(finite) == 0L) {
      next
    }

    zlim <- range(finite)

    if (diff(zlim) == 0) {
      zlim <- zlim + c(-0.5, 0.5)
    }

    graphics::layout(
      matrix(c(1L, 2L), nrow = 1L),
      widths = c(4.8, key.width)
    )
    graphics::par(
      mar = c(5.1, 8.0, 4.1, 1.0) + 0.1
    )

    graphics::image(
      x = seq_along(object$alpha),
      y = seq_along(object$eta),
      z = z,
      xaxt = "n",
      yaxt = "n",
      xlab = "Elastic-net alpha",
      ylab = "",
      col = colours,
      zlim = zlim,
      main = if (is.null(main)) {
        paste0(
          "ecountgmifs tuning surface: ",
          current_measure,
          " selected by ",
          criterion_name
        )
      } else {
        main
      },
      ...
    )
    graphics::axis(
      1,
      at = seq_along(object$alpha),
      labels = object$alpha_labels
    )
    graphics::axis(
      2,
      at = seq_along(object$eta),
      labels = .ecountgmifs.clean.eta.labels(
        labels = object$eta_labels,
        eta = object$eta
      ),
      las = 1
    )
    graphics::mtext(
      "Prior strength eta",
      side = 2,
      line = 6.2
    )

    if (isTRUE(cell.labels)) {
      for (alpha_index in seq_along(object$alpha)) {
        for (eta_index in seq_along(object$eta)) {
          value <- z[alpha_index, eta_index]

          if (!is.finite(value)) {
            next
          }

          graphics::text(
            alpha_index,
            eta_index,
            labels = format(
              value,
              digits = 3L
            ),
            cex = cell.cex
          )
        }
      }
    }

    best_row <- current[
      is.finite(current$value),
      ,
      drop = FALSE
    ]

    if (nrow(best_row) > 0L) {
      best_row <- best_row[
        which.min(best_row$value),
        ,
        drop = FALSE
      ]

      graphics::points(
        best_row$alpha_index[[1L]],
        best_row$eta_index[[1L]],
        pch = 1L,
        cex = best.cex,
        lwd = best.lwd
      )
    }

    graphics::par(
      mar = c(5.1, 0.8, 4.1, 3.4) + 0.1
    )
    key_y <- seq(
      zlim[[1L]],
      zlim[[2L]],
      length.out = length(colours)
    )
    graphics::image(
      x = c(0, 1),
      y = key_y,
      z = matrix(
        rep(key_y, each = 2L),
        nrow = 2L
      ),
      xaxt = "n",
      yaxt = "n",
      xlab = "",
      ylab = "",
      col = colours,
      zlim = zlim
    )
    graphics::axis(
      4,
      las = 1
    )
    graphics::mtext(
      current_measure,
      side = 3,
      line = 1.0,
      font = 2,
      cex = 0.9
    )
  }

  invisible(table)
}


.ecountgmifs.tuning.standard.legend <- function(
    fit,
    type,
    supplied.dots
) {
  legend.cex <- if (is.null(supplied.dots$legend.cex)) {
    1.08
  } else {
    supplied.dots$legend.cex
  }

  criterion_names <- character()
  show_criteria <- if (is.null(supplied.dots$show.criteria)) {
    TRUE
  } else {
    isTRUE(supplied.dots$show.criteria)
  }

  if (show_criteria) {
    criterion_names <- .ecountgmifs.match.criteria(
      fit,
      supplied.dots$criteria
    )
  }

  if (identical(type, "coefficients")) {
    if (length(criterion_names) == 0L) {
      graphics::plot.new()
      return(invisible(NULL))
    }

    styles <- .ecountgmifs.selection.styles(
      length(criterion_names)
    )

    .ecountgmifs.legend.panel(
      legend = criterion_names,
      col = styles$col,
      pch = styles$pch,
      pt.bg = styles$col,
      lty = 3L,
      lwd = 1.5,
      title = "Criterion selections",
      bty = "n",
      cex = legend.cex,
      cex.min = 0.55,
      max.columns = 2L
    )

    return(invisible(NULL))
  }

  if (identical(type, "criteria")) {
    if (length(criterion_names) == 0L) {
      graphics::plot.new()
      return(invisible(NULL))
    }

    criterion_col <- if (is.null(supplied.dots$col)) {
      seq_along(criterion_names) + 1L
    } else {
      rep_len(
        supplied.dots$col,
        length(criterion_names)
      )
    }

    styles <- .ecountgmifs.selection.styles(
      length(criterion_names)
    )

    .ecountgmifs.legend.panel(
      legend = criterion_names,
      col = criterion_col,
      pch = styles$pch,
      pt.bg = criterion_col,
      lty = 1L,
      lwd = 1.5,
      title = "Criteria",
      bty = "n",
      cex = legend.cex,
      cex.min = 0.55,
      max.columns = 2L
    )

    return(invisible(NULL))
  }

  metrics <- if (is.null(supplied.dots$metrics)) {
    c("F1", "precision", "recall")
  } else {
    as.character(supplied.dots$metrics)
  }

  default_metric_col <- c(
    F1 = "#D55E00",
    precision = "#009E73",
    recall = "#0072B2",
    specificity = "#CC79A7",
    accuracy = "#56B4E9"
  )

  metric_col <- if (is.null(supplied.dots$col)) {
    unname(default_metric_col[metrics])
  } else {
    rep_len(
      supplied.dots$col,
      length(metrics)
    )
  }

  show_prior <- if (is.null(supplied.dots$show.prior)) {
    TRUE
  } else {
    isTRUE(supplied.dots$show.prior)
  }

  has_prior <-
    !is.null(fit$input$prior_selected) ||
    isTRUE(fit$input$has_prior)

  labels <- paste0(metrics, " path")
  legend_col <- metric_col
  legend_lty <- rep(1L, length(metrics))
  legend_lwd <- rep(2.4, length(metrics))
  legend_pch <- rep(NA_integer_, length(metrics))

  if (show_prior && has_prior) {
    labels <- c(
      labels,
      paste0(metrics, " prior")
    )
    legend_col <- c(
      legend_col,
      metric_col
    )
    legend_lty <- c(
      legend_lty,
      rep(5L, length(metrics))
    )
    legend_lwd <- c(
      legend_lwd,
      rep(1.8, length(metrics))
    )
    legend_pch <- c(
      legend_pch,
      rep(NA_integer_, length(metrics))
    )
  }

  if (length(criterion_names) > 0L) {
    styles <- .ecountgmifs.selection.styles(
      length(criterion_names)
    )
    criterion_pch <- if (is.null(supplied.dots$criterion.pch)) {
      c(21L, 22L, 23L, 24L, 25L)
    } else {
      supplied.dots$criterion.pch
    }
    criterion_pch <- rep_len(
      criterion_pch,
      length(criterion_names)
    )

    labels <- c(labels, criterion_names)
    legend_col <- c(
      legend_col,
      styles$col
    )
    legend_lty <- c(
      legend_lty,
      rep(3L, length(criterion_names))
    )
    legend_lwd <- c(
      legend_lwd,
      rep(1.4, length(criterion_names))
    )
    legend_pch <- c(
      legend_pch,
      criterion_pch
    )
  }

  .ecountgmifs.legend.panel(
    legend = labels,
    col = legend_col,
    lty = legend_lty,
    lwd = legend_lwd,
    pch = legend_pch,
    pt.bg = legend_col,
    title = "Metrics and selections",
    bty = "n",
    cex = legend.cex,
    cex.min = 0.55,
    max.columns = 2L,
    y.intersp = 1.05
  )

  invisible(NULL)
}


.ecountgmifs.plot.tuning.standard.pages <- function(
    object,
    rows,
    type,
    ground.truth,
    ask,
    supplied.dots
) {
  eta_indices <- seq_along(object$eta)
  eta_indices <- eta_indices[
    eta_indices %in% object$grid$eta_index[rows]
  ]

  eta_labels <- .ecountgmifs.clean.eta.labels(
    labels = object$eta_labels,
    eta = object$eta
  )

  old_par <- graphics::par(no.readonly = TRUE)
  on.exit(
    {
      graphics::layout(matrix(1L))
      graphics::par(old_par)
    },
    add = TRUE
  )

  old_ask <- grDevices::devAskNewPage(
    ask = isTRUE(ask) && length(eta_indices) > 1L
  )
  on.exit(
    grDevices::devAskNewPage(old_ask),
    add = TRUE
  )

  results <- vector("list", length(eta_indices))
  names(results) <- paste0(
    "eta=",
    eta_labels[eta_indices]
  )

  for (eta_position in seq_along(eta_indices)) {
    eta_index <- eta_indices[[eta_position]]
    eta_rows <- rows[
      object$grid$eta_index[rows] == eta_index
    ]
    eta_rows <- eta_rows[
      order(object$grid$alpha_index[eta_rows])
    ]

    first_fit <- object$fits[[eta_rows[[1L]]]]
    show_criteria <- if (is.null(supplied.dots$show.criteria)) {
      TRUE
    } else {
      isTRUE(supplied.dots$show.criteria)
    }
    criterion_count <- if (show_criteria) {
      length(
        .ecountgmifs.match.criteria(
          first_fit,
          supplied.dots$criteria
        )
      )
    } else {
      0L
    }
    show_legend <-
      identical(type, "metrics") ||
      criterion_count > 0L

    standard_legend_width <- switch(
      type,
      coefficients = 0.70,
      criteria = 0.60,
      metrics = 0.88
    )

    page_layout <- .ecountgmifs.panel.layout(
      panel.count = length(eta_rows),
      legend = show_legend,
      legend.width = standard_legend_width
    )

    graphics::layout(
      page_layout$matrix,
      widths = page_layout$widths
    )
    graphics::par(
      oma = c(0, 0, 3.4, 0),
      mar = c(4.3, 4.3, 3.0, 1.0) + 0.1
    )

    page_result <- vector(
      "list",
      length(eta_rows)
    )
    names(page_result) <- object$grid$alpha_label[eta_rows]

    for (alpha_position in seq_along(eta_rows)) {
      row <- eta_rows[[alpha_position]]
      fit <- object$fits[[row]]
      current_dots <- supplied.dots

      current_dots$main <- paste0(
        "alpha = ",
        object$grid$alpha_label[[row]]
      )
      current_dots$ask <- NULL
      current_dots$combine <- NULL

      if (
          identical(type, "metrics") &&
          is.null(current_dots$criterion.label.cex)
      ) {
        # Multi-panel metric pages need smaller top-axis selection labels than
        # a full-device single-fit plot. This avoids collisions with the alpha
        # panel title while keeping user-supplied values unchanged.
        current_dots$criterion.label.cex <- 0.72
      }

      if (identical(type, "coefficients")) {
        current_dots$criterion.legend <- "none"
      } else {
        current_dots$legend.position <- "none"
      }

      page_result[[alpha_position]] <- do.call(
        plot,
        c(
          list(
            x = fit,
            type = type,
            ground.truth = ground.truth
          ),
          current_dots
        )
      )
    }

    if (show_legend) {
      .ecountgmifs.tuning.standard.legend(
        fit = first_fit,
        type = type,
        supplied.dots = supplied.dots
      )
    }

    page.title <- paste0(
      switch(
        type,
        coefficients = "ecountgmifs coefficient paths",
        criteria = "ecountgmifs criterion paths",
        metrics = "ecountgmifs selection metrics"
      ),
      " (eta = ",
      eta_labels[[eta_index]],
      ")"
    )
    title.layout <- .ecountgmifs.relative.title(
      page.title,
      cex = 1,
      outer = TRUE,
      max.lines = 2L
    )
    .ecountgmifs.draw.outer.title(
      title.layout$text,
      line = 0.55,
      font = 2,
      cex = title.layout$cex
    )

    results[[eta_position]] <- page_result
  }

  invisible(results)
}


.ecountgmifs.plot.tuning.diagnostic.pages <- function(
    object,
    rows,
    type,
    ground.truth,
    ask,
    combine,
    supplied.dots
) {
  eta_indices <- seq_along(object$eta)
  eta_indices <- eta_indices[
    eta_indices %in% object$grid$eta_index[rows]
  ]

  eta_labels <- .ecountgmifs.clean.eta.labels(
    labels = object$eta_labels,
    eta = object$eta
  )

  old_ask <- grDevices::devAskNewPage(
    ask = isTRUE(ask) && length(eta_indices) > 1L
  )
  on.exit(
    grDevices::devAskNewPage(old_ask),
    add = TRUE
  )

  results <- vector("list", length(eta_indices))
  names(results) <- paste0(
    "eta=",
    eta_labels[eta_indices]
  )

  for (eta_position in seq_along(eta_indices)) {
    eta_index <- eta_indices[[eta_position]]
    current <- .ecountgmifs.tuning.as.multi.alpha(
      object,
      eta_index
    )
    current_dots <- supplied.dots
    current_dots$ask <- NULL
    current_dots$combine <- NULL
    current_dots$main <- NULL

    # A tuning-grid diagnostic page always compares all selected alpha
    # values for one eta. This prevents alpha and eta from being interleaved
    # across unrelated graphics pages.
    current_combine <- "all"

    results[[eta_position]] <- do.call(
      plot,
      c(
        list(
          x = current,
          type = type,
          ask = FALSE,
          combine = current_combine,
          ground.truth = ground.truth,
          main = paste0(
            "ecountgmifs diagnostics (eta = ",
            eta_labels[[eta_index]],
            ")"
          )
        ),
        current_dots
      )
    )
  }

  invisible(results)
}


#' @export
plot.ecountgmifs_tuning_grid <- function(
    x,
    type = c(
      "coefficients",
      "criteria",
      "metrics",
      "diagnostics",
      "overdispersion",
      "alpha",
      "alpha.distribution",
      "prior",
      "prior.distribution",
      "tuning",
      "grid"
    ),
    alpha = NULL,
    eta = NULL,
    ask = interactive(),
    combine = FALSE,
    ground.truth = NULL,
    ...
) {
  type <- match.arg(type)

  if (identical(type, "grid")) {
    type <- "tuning"
  }

  supplied_dots <- list(...)

  if (
      type %in% c(
        "alpha.distribution",
        "prior.distribution"
      ) &&
      is.null(ground.truth)
  ) {
    stop(
      "`ground.truth` is required for distribution comparisons.",
      call. = FALSE
    )
  }

  if (
      !is.null(ground.truth) &&
      is.null(supplied_dots$prior.selected) &&
      !is.null(x$weight_prior$score)
  ) {
    supplied_dots$prior.selected <-
      x$weight_prior$score > 0
  }

  if (identical(type, "tuning")) {
    return(
      do.call(
        .ecountgmifs.plot.tuning.surface,
        c(
          list(
            object = x,
            ground.truth = ground.truth,
            ask = ask
          ),
          supplied_dots
        )
      )
    )
  }

  if (type %in% c("alpha", "alpha.distribution")) {
    eta_indices <- .ecountgmifs.tuning.grid.match(
      eta,
      x$eta,
      x$eta_labels,
      "eta"
    )

    objects <- lapply(
      eta_indices,
      function(index) {
        .ecountgmifs.tuning.as.multi.alpha(
          x,
          index
        )
      }
    )
    eta_value_labels <- .ecountgmifs.clean.eta.labels(
      labels = x$eta_labels[eta_indices],
      eta = x$eta[eta_indices]
    )
    labels <- .ecountgmifs.eta.context.labels(
      labels = eta_value_labels
    )
    names(objects) <- paste0(
      "eta=",
      eta_value_labels
    )

    measures <- supplied_dots$measure

    if (is.null(measures)) {
      measures <- if (!is.null(ground.truth)) {
        "F1"
      } else {
        "nonzero"
      }
    }

    if (isTRUE(combine)) {
      return(
        .ecountgmifs.plot.tuning.combined(
          objects = objects,
          labels = labels,
          comparison = "alpha",
          distribution = identical(
            type,
            "alpha.distribution"
          ),
          ground.truth = ground.truth,
          measures = measures,
          ask = ask,
          supplied.dots = supplied_dots
        )
      )
    }

    pages <- expand.grid(
      object_index = seq_along(objects),
      measure = as.character(measures),
      KEEP.OUT.ATTRS = FALSE,
      stringsAsFactors = FALSE
    )

    old_ask <- grDevices::devAskNewPage(
      ask = isTRUE(ask) && nrow(pages) > 1L
    )
    on.exit(
      grDevices::devAskNewPage(old_ask),
      add = TRUE
    )

    result <- vector("list", nrow(pages))
    names(result) <- paste0(
      labels[pages$object_index],
      ": ",
      pages$measure
    )

    for (page_index in seq_len(nrow(pages))) {
      object_index <- pages$object_index[[page_index]]
      current_measure <- pages$measure[[page_index]]
      current_dots <- supplied_dots
      current_dots$measure <- NULL
      current_dots$ask <- NULL
      current_dots$.manage.ask <- NULL
      current_dots$.manage.par <- NULL
      current_dots$context.label <- NULL

      helper <- if (identical(type, "alpha")) {
        .ecountgmifs.plot.alpha.comparison
      } else {
        .ecountgmifs.plot.alpha.distribution
      }

      result[[page_index]] <- do.call(
        helper,
        c(
          list(
            object = objects[[object_index]],
            ground.truth = ground.truth,
            measure = current_measure,
            ask = FALSE,
            .manage.ask = FALSE,
            context.label = labels[[object_index]]
          ),
          current_dots
        )
      )
    }

    return(invisible(result))
  }

  if (type %in% c("prior", "prior.distribution")) {
    alpha_indices <- .ecountgmifs.tuning.grid.match(
      alpha,
      x$alpha,
      x$alpha_labels,
      "alpha"
    )

    objects <- lapply(
      alpha_indices,
      function(index) {
        .ecountgmifs.tuning.as.multi.eta(
          x,
          index
        )
      }
    )
    labels <- paste0(
      "alpha = ",
      x$alpha_labels[alpha_indices]
    )
    names(objects) <- x$alpha_labels[alpha_indices]

    measures <- supplied_dots$measure

    if (is.null(measures)) {
      measures <- if (!is.null(ground.truth)) {
        "F1"
      } else {
        "nonzero"
      }
    }

    supplied_dots$xlab <- if (
      is.null(supplied_dots$xlab)
    ) {
      "Prior strength eta"
    } else {
      supplied_dots$xlab
    }
    supplied_dots$best.label <- if (
      is.null(supplied_dots$best.label)
    ) {
      "Best eta by criterion"
    } else {
      supplied_dots$best.label
    }
    supplied_dots$comparison.label <- if (
      is.null(supplied_dots$comparison.label)
    ) {
      "prior strength"
    } else {
      supplied_dots$comparison.label
    }

    if (isTRUE(combine)) {
      return(
        .ecountgmifs.plot.tuning.combined(
          objects = objects,
          labels = labels,
          comparison = "prior",
          distribution = identical(
            type,
            "prior.distribution"
          ),
          ground.truth = ground.truth,
          measures = measures,
          ask = ask,
          supplied.dots = supplied_dots
        )
      )
    }

    pages <- expand.grid(
      object_index = seq_along(objects),
      measure = as.character(measures),
      KEEP.OUT.ATTRS = FALSE,
      stringsAsFactors = FALSE
    )

    old_ask <- grDevices::devAskNewPage(
      ask = isTRUE(ask) && nrow(pages) > 1L
    )
    on.exit(
      grDevices::devAskNewPage(old_ask),
      add = TRUE
    )

    result <- vector("list", nrow(pages))
    names(result) <- paste0(
      labels[pages$object_index],
      ": ",
      pages$measure
    )

    for (page_index in seq_len(nrow(pages))) {
      object_index <- pages$object_index[[page_index]]
      current_measure <- pages$measure[[page_index]]
      current_dots <- supplied_dots
      current_dots$measure <- NULL
      current_dots$ask <- NULL
      current_dots$.manage.ask <- NULL
      current_dots$.manage.par <- NULL
      current_dots$context.label <- NULL

      helper <- if (identical(type, "prior")) {
        .ecountgmifs.plot.alpha.comparison
      } else {
        .ecountgmifs.plot.alpha.distribution
      }

      result[[page_index]] <- do.call(
        helper,
        c(
          list(
            object = objects[[object_index]],
            ground.truth = ground.truth,
            measure = current_measure,
            ask = FALSE,
            .manage.ask = FALSE,
            context.label = labels[[object_index]]
          ),
          current_dots
        )
      )
    }

    return(invisible(result))
  }

  rows <- .ecountgmifs.tuning.grid.select(
    x,
    alpha = alpha,
    eta = eta
  )

  if (
      type %in% c(
        "coefficients",
        "criteria",
        "metrics"
      )
  ) {
    return(
      .ecountgmifs.plot.tuning.standard.pages(
        object = x,
        rows = rows,
        type = type,
        ground.truth = ground.truth,
        ask = ask,
        supplied.dots = supplied_dots
      )
    )
  }

  .ecountgmifs.plot.tuning.diagnostic.pages(
    object = x,
    rows = rows,
    type = type,
    ground.truth = ground.truth,
    ask = ask,
    combine = combine,
    supplied.dots = supplied_dots
  )
}

