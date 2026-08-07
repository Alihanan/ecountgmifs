.ecountgmifs.diagnostic.input <- function(object) {
  value <- object$diagnostics

  if (!is.null(value)) {
    return(value)
  }

  input <- object$input

  if (is.null(input$y)) {
    stop(
      paste0(
        "Diagnostic data are unavailable. Refit with the current ",
        "ecountgmifs() wrapper, or fit with include.data = TRUE."
      ),
      call. = FALSE
    )
  }

  out <- list(
    response = as.numeric(input$y),
    response_name = if (is.null(input$response_name)) {
      "response"
    } else {
      input$response_name
    },
    response_mean = mean(input$y),
    response_variance = stats::var(input$y)
  )

  if (!is.null(input$X)) {
    X <- as.matrix(input$X)
    predictor_names <- .ecountgmifs.predictor.names(object, ncol(X))

    out$predictor_mean <- colMeans(X)
    out$predictor_variance <- apply(X, 2L, stats::var)
    out$predictor_nonnegative <- apply(
      X,
      2L,
      function(value) all(value >= 0)
    )

    names(out$predictor_mean) <- predictor_names
    names(out$predictor_variance) <- predictor_names
    names(out$predictor_nonnegative) <- predictor_names
  }

  out
}


.ecountgmifs.variance.function <- function(
    object,
    mu,
    family_parameters
) {
  family <- tolower(.ecountgmifs.family.label(object))

  if (
      grepl("nb2", family, fixed = TRUE) ||
      grepl("negative", family, fixed = TRUE)
  ) {
    if (length(family_parameters) < 1L) {
      return(
        list(
          variance = rep(NA_real_, length(mu)),
          residual_type = "raw",
          family_kind = "count",
          variance_label = "NB2 variance unavailable"
        )
      )
    }

    dispersion <- as.numeric(family_parameters[[1L]])

    return(
      list(
        variance = mu + dispersion * mu^2,
        residual_type = "Pearson NB2",
        family_kind = "count",
        variance_label = "Fitted NB2 variance"
      )
    )
  }

  if (grepl("poisson", family, fixed = TRUE)) {
    return(
      list(
        variance = mu,
        residual_type = "Pearson Poisson",
        family_kind = "count",
        variance_label = "Poisson variance"
      )
    )
  }

  if (
      grepl("gaussian", family, fixed = TRUE) ||
      grepl("normal", family, fixed = TRUE)
  ) {
    return(
      list(
        variance = rep(NA_real_, length(mu)),
        residual_type = "raw Gaussian",
        family_kind = "continuous",
        variance_label = "Constant residual variance"
      )
    )
  }

  list(
    variance = rep(NA_real_, length(mu)),
    residual_type = "raw",
    family_kind = "other",
    variance_label = "Empirical residual variance"
  )
}


.ecountgmifs.overdispersion.table <- function(
    y,
    mu,
    fitted_variance,
    family_kind,
    bins = 10L
) {
  keep <-
    is.finite(y) &
    is.finite(mu) &
    mu >= 0

  if (sum(keep) < 2L) {
    return(data.frame())
  }

  y <- y[keep]
  mu <- mu[keep]

  if (length(fitted_variance) == length(keep)) {
    fitted_variance <- fitted_variance[keep]
  } else {
    fitted_variance <- rep(NA_real_, length(mu))
  }

  bins <- as.integer(bins[[1L]])
  if (is.na(bins) || bins < 1L) {
    bins <- 10L
  }

  bins <- min(
    bins,
    max(1L, floor(length(mu) / 5L))
  )

  probabilities <- seq(
    0,
    1,
    length.out = bins + 1L
  )

  breaks <- unique(
    as.numeric(
      stats::quantile(
        mu,
        probs = probabilities,
        names = FALSE,
        type = 8L,
        na.rm = TRUE
      )
    )
  )

  if (length(breaks) < 2L) {
    group <- factor(rep(1L, length(mu)))
  } else {
    breaks[[1L]] <- -Inf
    breaks[[length(breaks)]] <- Inf
    group <- cut(
      mu,
      breaks = breaks,
      include.lowest = TRUE,
      labels = FALSE
    )
  }

  indices <- split(
    seq_along(mu),
    group
  )

  rows <- lapply(
    indices,
    function(index) {
      current_y <- y[index]
      current_mu <- mu[index]
      current_variance <- fitted_variance[index]
      raw_residual <- current_y - current_mu

      observed_mean <- mean(current_y)
      observed_variance <- if (length(index) > 1L) {
        stats::var(current_y)
      } else {
        NA_real_
      }

      residual_variance <- if (length(index) > 1L) {
        stats::var(raw_residual)
      } else {
        NA_real_
      }

      model_variance <- if (
          length(current_variance) > 0L &&
          any(is.finite(current_variance))
      ) {
        mean(current_variance[is.finite(current_variance)])
      } else {
        NA_real_
      }

      if (identical(family_kind, "count")) {
        empirical <- if (
            is.finite(observed_mean) &&
            observed_mean > 0
        ) {
          observed_variance / observed_mean
        } else {
          NA_real_
        }

        expected <- if (
            is.finite(model_variance) &&
            mean(current_mu) > 0
        ) {
          model_variance / mean(current_mu)
        } else {
          NA_real_
        }
      } else {
        empirical <- residual_variance
        expected <- model_variance
      }

      data.frame(
        fitted_mean = mean(current_mu),
        observed_mean = observed_mean,
        observed_variance = observed_variance,
        residual_variance = residual_variance,
        empirical = empirical,
        expected = expected,
        observations = length(index),
        stringsAsFactors = FALSE
      )
    }
  )

  out <- do.call(rbind, rows)
  rownames(out) <- NULL
  out[order(out$fitted_mean), , drop = FALSE]
}


#' Calculate ecountgmifs data and fitted-model diagnostics
#'
#' Creates compact data used for mean-variance, dispersion-path, residual-vs-
#' fitted, and scale-location diagnostics. The diagnostic object can be plotted
#' directly or produced implicitly by `plot(fit, type = "diagnostics")`.
#'
#' @param object An `ecountgmifs` fit.
#' @param criterion Optional criterion selecting the fitted state used by
#'   residual diagnostics.
#' @param iteration Optional iteration selecting the nearest saved state.
#'
#' @return An object of class `ecountgmifs.diagnostics`.
#' @export
ecountgmifs.diagnostics <- function(
    object,
    criterion = NULL,
    iteration = NULL
) {
  selected <- .ecountgmifs.resolve.state(
    object,
    criterion = criterion,
    iteration = iteration
  )
  state <- selected$state
  parameters <- state$predictors$parameters
  diagnostic_input <- .ecountgmifs.diagnostic.input(object)

  y <- as.numeric(diagnostic_input$response)
  mu <- as.numeric(state$predictors$mu)

  if (length(y) != length(mu)) {
    stop(
      "Stored response and fitted-mean vectors have different lengths.",
      call. = FALSE
    )
  }

  variance_result <- .ecountgmifs.variance.function(
    object = object,
    mu = mu,
    family_parameters = parameters$family_parameters
  )

  raw_residual <- y - mu
  fitted_variance <- variance_result$variance

  if (
      length(fitted_variance) == length(mu) &&
      all(is.finite(fitted_variance)) &&
      all(fitted_variance > 0)
  ) {
    residual <- raw_residual / sqrt(fitted_variance)
  } else {
    residual <- raw_residual
  }

  predictor_table <- data.frame(
    variable = character(),
    mean = numeric(),
    variance = numeric(),
    nonnegative = logical(),
    stringsAsFactors = FALSE
  )

  if (!is.null(diagnostic_input$predictor_mean)) {
    predictor_table <- data.frame(
      variable = names(diagnostic_input$predictor_mean),
      mean = as.numeric(diagnostic_input$predictor_mean),
      variance = as.numeric(diagnostic_input$predictor_variance),
      nonnegative = as.logical(diagnostic_input$predictor_nonnegative),
      stringsAsFactors = FALSE
    )
  }

  predictor_table$role <- rep(
    "penalized predictor",
    nrow(predictor_table)
  )

  mean_variance <- rbind(
    predictor_table[, c(
      "variable",
      "role",
      "mean",
      "variance",
      "nonnegative"
    ), drop = FALSE],
    data.frame(
      variable = diagnostic_input$response_name,
      role = "response",
      mean = as.numeric(diagnostic_input$response_mean),
      variance = as.numeric(diagnostic_input$response_variance),
      nonnegative = all(y >= 0),
      stringsAsFactors = FALSE
    )
  )

  states <- .ecountgmifs.path.states(object, required = FALSE)
  dispersion_path <- data.frame(
    iteration = numeric(),
    dispersion = numeric(),
    source = character(),
    stringsAsFactors = FALSE
  )

  null_family_parameters <- object$path$null_family_parameters
  if (!is.null(null_family_parameters) && length(null_family_parameters) > 0L) {
    dispersion_path <- rbind(
      dispersion_path,
      data.frame(
        iteration = 0,
        dispersion = as.numeric(null_family_parameters[[1L]]),
        source = "null",
        stringsAsFactors = FALSE
      )
    )
  }

  if (
      !is.null(states) &&
      !is.null(states$family_parameters) &&
      length(states$family_parameters) > 0L
  ) {
    path_dispersion <- vapply(
      states$family_parameters,
      function(value) {
        if (length(value) == 0L) NA_real_ else as.numeric(value[[1L]])
      },
      numeric(1L)
    )

    dispersion_path <- rbind(
      dispersion_path,
      data.frame(
        iteration = as.numeric(states$iteration),
        dispersion = path_dispersion,
        source = "path",
        stringsAsFactors = FALSE
      )
    )
  }

  saturated_family_parameters <- object$path$saturated_family_parameters
  if (
      !is.null(saturated_family_parameters) &&
      length(saturated_family_parameters) > 0L
  ) {
    final_iteration <- if (nrow(dispersion_path) == 0L) {
      as.numeric(state$iteration)
    } else {
      max(dispersion_path$iteration, na.rm = TRUE)
    }

    dispersion_path <- rbind(
      dispersion_path,
      data.frame(
        iteration = final_iteration,
        dispersion = as.numeric(saturated_family_parameters[[1L]]),
        source = "saturated reference",
        stringsAsFactors = FALSE
      )
    )
  }

  overdispersion <- .ecountgmifs.overdispersion.table(
    y = y,
    mu = mu,
    fitted_variance = fitted_variance,
    family_kind = variance_result$family_kind
  )

  out <- list(
    call = object$call,
    family = .ecountgmifs.family.label(object),
    link = .ecountgmifs.link.label(object),
    state_label = selected$label,
    iteration = as.integer(state$iteration),
    mean_variance = mean_variance,
    overdispersion = overdispersion,
    overdispersion_kind = variance_result$family_kind,
    variance_label = variance_result$variance_label,
    dispersion_path = dispersion_path,
    residuals = data.frame(
      observation = seq_along(y),
      observed = y,
      fitted = mu,
      raw_residual = raw_residual,
      residual = residual,
      sqrt_abs_residual = sqrt(abs(residual)),
      stringsAsFactors = FALSE
    ),
    residual_type = variance_result$residual_type
  )

  class(out) <- "ecountgmifs.diagnostics"
  out
}


.ecountgmifs.diagnostic.which <- function(which) {
  choices <- c(
    "mean.variance",
    "dispersion.path",
    "residuals.fitted",
    "scale.location",
    "overdispersion"
  )

  aliases <- c(
    "mean-variance" = "mean.variance",
    "dispersion" = "dispersion.path",
    "residuals" = "residuals.fitted",
    "scale-location" = "scale.location",
    "overdispersion" = "overdispersion"
  )

  if (is.numeric(which)) {
    if (
        length(which) == 0L ||
        anyNA(which) ||
        any(which < 1L) ||
        any(which > length(choices)) ||
        any(which != as.integer(which))
    ) {
      stop(
        "Numeric `which` values must be integers from 1 to 5.",
        call. = FALSE
      )
    }

    return(choices[as.integer(which)])
  }

  value <- as.character(which)
  matched_alias <- aliases[value]
  value[!is.na(matched_alias)] <- matched_alias[!is.na(matched_alias)]

  match.arg(
    value,
    choices,
    several.ok = TRUE
  )
}


# -----------------------------------------------------------------------------
# Shared legend and diagnostic-page helpers
# -----------------------------------------------------------------------------

.ecountgmifs.legend.position <- function(position) {
  match.arg(
    position,
    c(
      "outside",
      "bottomright",
      "bottom",
      "bottomleft",
      "left",
      "topleft",
      "top",
      "topright",
      "right",
      "center",
      "none"
    )
  )
}


.ecountgmifs.clean.legend.labels <- function(labels) {
  labels <- as.character(labels)
  labels <- gsub("[[:cntrl:]]+", " ", labels)
  labels <- gsub("[[:space:]]+", " ", labels)
  trimws(labels)
}


.ecountgmifs.is.rstudio.graphics.device <- function() {
  current <- grDevices::dev.cur()
  device.name <- names(current)

  length(device.name) == 1L &&
    !is.na(device.name) &&
    identical(tolower(device.name), "rstudiogd")
}


.ecountgmifs.clamp <- function(value, lower, upper) {
  pmin(upper, pmax(lower, value))
}


.ecountgmifs.figure.size.inches <- function(outer = FALSE) {
  device.size <- graphics::par("din")

  if (
      length(device.size) != 2L ||
      any(!is.finite(device.size)) ||
      any(device.size <= 0)
  ) {
    device.size <- c(7, 7)
  }

  if (isTRUE(outer)) {
    return(device.size)
  }

  figure <- graphics::par("fig")

  if (
      length(figure) != 4L ||
      any(!is.finite(figure))
  ) {
    return(device.size)
  }

  c(
    device.size[[1L]] * diff(figure[1:2]),
    device.size[[2L]] * diff(figure[3:4])
  )
}


.ecountgmifs.wrap.lines <- function(
    text,
    width,
    max.lines = 3L
) {
  text <- .ecountgmifs.clean.legend.labels(text)
  width <- as.integer(width[[1L]])
  max.lines <- as.integer(max.lines[[1L]])

  if (is.na(width) || width < 1L) {
    width <- 1L
  }
  if (is.na(max.lines) || max.lines < 1L) {
    max.lines <- 1L
  }

  vapply(
    text,
    function(label) {
      if (!nzchar(label)) {
        return(label)
      }

      current.width <- max(width, ceiling(nchar(label, type = "width") / max.lines))
      lines <- strwrap(
        label,
        width = current.width,
        simplify = FALSE
      )[[1L]]

      if (length(lines) > max.lines) {
        current.width <- max(
          current.width,
          ceiling(nchar(label, type = "width") / max.lines)
        )
        lines <- strwrap(
          label,
          width = current.width,
          simplify = FALSE
        )[[1L]]
      }

      paste(lines, collapse = "\n")
    },
    character(1L),
    USE.NAMES = FALSE
  )
}


.ecountgmifs.relative.title <- function(
    label,
    cex = 1.2,
    outer = FALSE,
    padding = 0.90,
    max.lines = 2L,
    minimum.characters = 18L
) {
  label <- .ecountgmifs.clean.legend.labels(label)
  cex <- as.numeric(cex[[1L]])
  padding <- as.numeric(padding[[1L]])
  max.lines <- as.integer(max.lines[[1L]])
  minimum.characters <- as.integer(minimum.characters[[1L]])

  if (!is.finite(cex) || cex <= 0) {
    cex <- 1.2
  }
  if (!is.finite(padding) || padding <= 0 || padding > 1) {
    padding <- 0.90
  }
  if (is.na(max.lines) || max.lines < 1L) {
    max.lines <- 2L
  }
  if (is.na(minimum.characters) || minimum.characters < 1L) {
    minimum.characters <- 18L
  }

  figure.size <- .ecountgmifs.figure.size.inches(outer = outer)
  character.size <- graphics::par("cin")
  character.width <- if (
      length(character.size) >= 1L &&
      is.finite(character.size[[1L]]) &&
      character.size[[1L]] > 0
  ) {
    character.size[[1L]]
  } else {
    0.15
  }

  available.width <- max(0.1, figure.size[[1L]] * padding)
  estimated.character.width <- max(
    1e-8,
    character.width * 0.58 * cex
  )
  characters.per.line <- max(
    minimum.characters,
    floor(available.width / estimated.character.width)
  )

  list(
    text = .ecountgmifs.wrap.lines(
      label,
      width = characters.per.line,
      max.lines = max.lines
    ),
    cex = cex
  )
}


.ecountgmifs.relative.title.cex <- function(...) {
  .ecountgmifs.relative.title(...)$cex
}


.ecountgmifs.draw.outer.title <- function(
    title,
    line = 0.45,
    cex = 1,
    font = 2
) {
  lines <- strsplit(
    as.character(title[[1L]]),
    "\n",
    fixed = TRUE
  )[[1L]]
  lines <- lines[nzchar(lines)]

  if (length(lines) == 0L) {
    return(invisible(NULL))
  }

  line.positions <- line + rev(seq_along(lines) - 1L) * 1.05

  for (index in seq_along(lines)) {
    graphics::mtext(
      lines[[index]],
      side = 3,
      outer = TRUE,
      line = line.positions[[index]],
      font = font,
      cex = cex
    )
  }

  invisible(NULL)
}


.ecountgmifs.relative.legend.width <- function(
    labels,
    title = NULL,
    cex = 1,
    plot.width = 4.6,
    minimum.fraction = 0.18,
    maximum.fraction = 0.38
) {
  text <- .ecountgmifs.clean.legend.labels(c(labels, title))
  text <- text[nzchar(text)]

  if (length(text) == 0L) {
    return(plot.width * minimum.fraction / (1 - minimum.fraction))
  }

  cex <- as.numeric(cex[[1L]])
  if (!is.finite(cex) || cex <= 0) {
    cex <- 1
  }

  device.size <- .ecountgmifs.figure.size.inches(outer = TRUE)
  character.size <- graphics::par("cin")
  character.width <- if (
      length(character.size) >= 1L &&
      is.finite(character.size[[1L]]) &&
      character.size[[1L]] > 0
  ) {
    character.size[[1L]]
  } else {
    0.15
  }

  # Reserve enough room for a readable wrapped legend rather than sizing
  # the whole column for the longest unbroken label. Long entries are wrapped
  # by `.ecountgmifs.legend.fit()` before any text-size reduction is attempted.
  longest <- max(nchar(text, type = "width"), 1L)
  target.characters <- min(longest, 28L)
  required.width <- (
    target.characters * character.width * 0.58 * cex +
      5 * character.width * cex
  )
  fraction <- required.width / max(device.size[[1L]], 1e-8)
  fraction <- .ecountgmifs.clamp(
    fraction,
    minimum.fraction,
    maximum.fraction
  )

  plot.width * fraction / (1 - fraction)
}


.ecountgmifs.wrap.legend.text <- function(
    labels,
    available.width,
    cex,
    ncol = 1L,
    max.lines = 3L,
    minimum.characters = 12L
) {
  labels <- .ecountgmifs.clean.legend.labels(labels)
  ncol <- max(1L, as.integer(ncol[[1L]]))
  max.lines <- max(1L, as.integer(max.lines[[1L]]))
  minimum.characters <- max(1L, as.integer(minimum.characters[[1L]]))

  em.width <- tryCatch(
    graphics::strwidth("M", units = "user", cex = cex),
    error = function(error) NA_real_
  )
  if (!is.finite(em.width) || em.width <= 0) {
    em.width <- available.width / 30
  }

  # Keep space for the key symbol, line segment, and inter-column separation.
  text.width <- max(
    em.width * minimum.characters,
    available.width / ncol * 0.72
  )
  characters.per.line <- max(
    minimum.characters,
    floor(text.width / em.width)
  )

  .ecountgmifs.wrap.lines(
    labels,
    width = characters.per.line,
    max.lines = max.lines
  )
}


.ecountgmifs.legend.fit <- function(
    arguments,
    available.width,
    available.height,
    cex,
    cex.min = 0.55,
    ncol = NULL,
    max.columns = 3L
) {
  labels <- .ecountgmifs.clean.legend.labels(arguments$legend)
  arguments$legend <- labels

  if (!is.null(arguments$title)) {
    arguments$title <- .ecountgmifs.clean.legend.labels(arguments$title)
  }

  cex <- as.numeric(cex[[1L]])
  cex.min <- as.numeric(cex.min[[1L]])

  if (!is.finite(cex) || cex <= 0) {
    cex <- 1
  }
  if (!is.finite(cex.min) || cex.min <= 0) {
    cex.min <- 0.55
  }
  cex.min <- min(cex, cex.min)

  panel.size <- graphics::par("pin")
  panel.area.scale <- if (
      length(panel.size) == 2L &&
      all(is.finite(panel.size)) &&
      all(panel.size > 0)
  ) {
    min(1, sqrt(prod(panel.size) / 6))
  } else {
    1
  }
  cex.min <- min(
    cex.min,
    max(0.35, cex * 0.55 * panel.area.scale)
  )

  label.count <- length(labels)

  if (is.null(ncol)) {
    max.columns <- as.integer(max.columns[[1L]])
    if (is.na(max.columns) || max.columns < 1L) {
      max.columns <- 1L
    }
    column.candidates <- seq_len(min(max.columns, label.count))
  } else {
    ncol <- as.integer(ncol[[1L]])
    if (is.na(ncol) || ncol < 1L) {
      stop("`ncol` must be a positive integer.", call. = FALSE)
    }
    column.candidates <- min(ncol, label.count)
  }

  cex.candidates <- unique(
    c(
      seq(cex, cex.min, length.out = 25L),
      cex.min
    )
  )
  cex.candidates <- sort(cex.candidates, decreasing = TRUE)

  fitting <- list()
  best <- NULL
  best.overflow <- Inf

  for (current.cex in cex.candidates) {
    for (current.ncol in column.candidates) {
      current <- arguments
      current$legend <- .ecountgmifs.wrap.legend.text(
        labels,
        available.width = available.width,
        cex = current.cex,
        ncol = current.ncol,
        max.lines = 3L
      )
      if (!is.null(arguments$title)) {
        current$title <- .ecountgmifs.wrap.legend.text(
          arguments$title,
          available.width = available.width,
          cex = current.cex,
          ncol = 1L,
          max.lines = 2L,
          minimum.characters = 10L
        )
      }
      current$cex <- current.cex
      current$ncol <- current.ncol
      current$plot <- FALSE

      measured <- tryCatch(
        do.call(
          graphics::legend,
          c(list(x = "center"), current)
        ),
        error = function(error) NULL
      )

      if (
          is.null(measured) ||
          !is.finite(measured$rect$w) ||
          !is.finite(measured$rect$h)
      ) {
        next
      }

      overflow <- max(
        measured$rect$w / available.width,
        measured$rect$h / available.height
      )

      if (overflow < best.overflow) {
        best.overflow <- overflow
        best <- current
      }

      if (overflow <= 1) {
        fitting[[length(fitting) + 1L]] <- list(
          arguments = current,
          cex = current.cex,
          ncol = current.ncol
        )
      }
    }
  }

  if (length(fitting) > 0L) {
    fitting.cex <- vapply(fitting, `[[`, numeric(1L), "cex")
    largest.cex <- max(fitting.cex)
    candidates <- which(abs(fitting.cex - largest.cex) < 1e-12)
    fitting.ncol <- vapply(fitting[candidates], `[[`, integer(1L), "ncol")
    selected.index <- candidates[[which.min(fitting.ncol)]]
    selected <- fitting[[selected.index]]$arguments
    selected$plot <- FALSE

    measured <- tryCatch(
      do.call(
        graphics::legend,
        c(list(x = "center"), selected)
      ),
      error = function(error) NULL
    )

    if (!is.null(measured)) {
      overflow <- max(
        measured$rect$w / (available.width * 0.92),
        measured$rect$h / (available.height * 0.92)
      )

      if (is.finite(overflow) && overflow > 1) {
        selected$cex <- max(
          0.30,
          selected$cex * 0.90 / overflow
        )
      }
    }

    return(selected)
  }

  if (is.null(best)) {
    best <- arguments
    best$cex <- cex.min
    best$ncol <- 1L
    return(best)
  }

  # Legend dimensions are approximately proportional to cex. This final
  # reduction is used only when even the lowest candidate does not fit.
  best$cex <- max(
    0.30,
    best$cex * 0.90 / best.overflow
  )
  best
}


.ecountgmifs.draw.legend <- function(
    position = "outside",
    ...,
    outside.inset = 0.035
) {
  position <- .ecountgmifs.legend.position(position)

  if (identical(position, "none")) {
    return(invisible(NULL))
  }

  arguments <- list(...)

  if (is.null(arguments$legend) || length(arguments$legend) == 0L) {
    return(invisible(NULL))
  }

  arguments$legend <- .ecountgmifs.clean.legend.labels(arguments$legend)
  if (!is.null(arguments$title)) {
    arguments$title <- .ecountgmifs.clean.legend.labels(arguments$title)
  }

  requested.cex <- if (is.null(arguments$cex)) {
    0.95
  } else {
    as.numeric(arguments$cex[[1L]])
  }
  if (!is.finite(requested.cex) || requested.cex <= 0) {
    requested.cex <- 0.95
  }

  if (identical(position, "outside")) {
    figure <- graphics::par("fig")
    plot.region <- graphics::par("plt")
    device.inches <- graphics::par("din")
    plot.right <- figure[[1L]] +
      plot.region[[2L]] * (figure[[2L]] - figure[[1L]])
    available.width.inches <- device.inches[[1L]] * (1 - plot.right) - 0.08

    if (!is.finite(available.width.inches) || available.width.inches < 0.7) {
      arguments$cex <- requested.cex
      arguments$xpd <- FALSE

      do.call(
        graphics::legend,
        c(list(x = "topright"), arguments)
      )

      return(invisible(NULL))
    }

    usr <- graphics::par("usr")
    pin <- graphics::par("pin")
    available.width.user <- diff(usr[1:2]) *
      max(0.05, available.width.inches / pin[[1L]])
    available.height.user <- diff(usr[3:4]) * 0.96

    fitted <- .ecountgmifs.legend.fit(
      arguments = arguments,
      available.width = available.width.user,
      available.height = available.height.user,
      cex = requested.cex,
      cex.min = 0.55,
      max.columns = 1L
    )
    fitted$plot <- NULL
    fitted$xpd <- TRUE

    do.call(
      graphics::legend,
      c(
        list(
          x = usr[[2L]] + outside.inset * diff(usr[1:2]),
          y = usr[[4L]],
          xjust = 0,
          yjust = 1
        ),
        fitted
      )
    )
  } else {
    usr <- graphics::par("usr")
    fitted <- .ecountgmifs.legend.fit(
      arguments = arguments,
      available.width = diff(usr[1:2]) * 0.96,
      available.height = diff(usr[3:4]) * 0.96,
      cex = requested.cex,
      cex.min = 0.55,
      max.columns = 1L
    )
    fitted$plot <- NULL
    fitted$xpd <- FALSE

    do.call(
      graphics::legend,
      c(list(x = position), fitted)
    )
  }

  invisible(NULL)
}


.ecountgmifs.diagnostic.legend.spec <- function(
    x,
    diagnostic
) {
  if (identical(diagnostic, "mean.variance")) {
    return(
      list(
        section = "Mean-variance",
        legend = c(
          "Penalized predictors",
          "Response",
          "Poisson equidispersion"
        ),
        col = c("#D55E00", "#0072B2", "black"),
        pch = c(16L, 17L, NA_integer_),
        lty = c(NA_integer_, NA_integer_, 3L),
        lwd = c(NA_real_, NA_real_, 1.5)
      )
    )
  }

  if (identical(diagnostic, "overdispersion")) {
    if (identical(x$overdispersion_kind, "count")) {
      return(
        list(
          section = "Overdispersion",
          legend = c(
            "Empirical binned ratio",
            x$variance_label,
            "Poisson equidispersion"
          ),
          col = c("black", "#D55E00", "black"),
          pch = c(16L, NA_integer_, NA_integer_),
          lty = c(1L, 2L, 3L),
          lwd = c(1.5, 2, 1.5)
        )
      )
    }

    return(
      list(
        section = "Variance",
        legend = c(
          "Empirical binned residual variance",
          x$variance_label
        ),
        col = c("black", "#D55E00"),
        pch = c(16L, NA_integer_),
        lty = c(1L, 2L),
        lwd = c(1.5, 2)
      )
    )
  }

  NULL
}


.ecountgmifs.draw.diagnostic.legend.panel <- function(
    x,
    which,
    cex = 1.08,
    align = c("auto", "top-left", "center", "left-center")
) {
  align <- match.arg(align)
  cex <- as.numeric(cex[[1L]])
  if (!is.finite(cex) || cex <= 0) {
    cex <- 1.08
  }

  specs <- lapply(
    unique(which),
    function(diagnostic) {
      .ecountgmifs.diagnostic.legend.spec(
        x,
        diagnostic
      )
    }
  )
  specs <- Filter(Negate(is.null), specs)

  if (length(specs) == 0L) {
    graphics::plot.new()
    return(invisible(NULL))
  }

  labels <- if (length(specs) == 1L) {
    specs[[1L]]$legend
  } else {
    unlist(
      lapply(
        specs,
        function(spec) {
          prefix <- rep("", length(spec$legend))
          prefix[[1L]] <- paste0(spec$section, ": ")
          paste0(prefix, spec$legend)
        }
      ),
      use.names = FALSE
    )
  }
  col <- unlist(lapply(specs, `[[`, "col"), use.names = FALSE)
  pch <- unlist(lapply(specs, `[[`, "pch"), use.names = FALSE)
  lty <- unlist(lapply(specs, `[[`, "lty"), use.names = FALSE)
  lwd <- unlist(lapply(specs, `[[`, "lwd"), use.names = FALSE)

  .ecountgmifs.legend.panel(
    legend = labels,
    col = col,
    pch = pch,
    lty = lty,
    lwd = lwd,
    title = if (length(specs) == 1L) {
      specs[[1L]]$section
    } else {
      "Diagnostic keys"
    },
    bty = "n",
    x.intersp = 0.8,
    y.intersp = 1.15,
    cex = cex,
    cex.min = min(as.numeric(cex[[1L]]), 0.55),
    max.columns = 2L,
    align = align
  )

  invisible(NULL)
}


.ecountgmifs.panel.layout <- function(
    panel.count,
    mfrow = NULL,
    legend = FALSE,
    legend.width = 1.05,
    byrow = TRUE
) {
  panel.count <- as.integer(panel.count[[1L]])

  if (is.na(panel.count) || panel.count < 1L) {
    stop(
      "`panel.count` must be a positive integer.",
      call. = FALSE
    )
  }

  if (is.null(mfrow)) {
    mfrow <- grDevices::n2mfrow(panel.count)
  } else {
    mfrow <- as.integer(mfrow)

    if (
        length(mfrow) != 2L ||
        anyNA(mfrow) ||
        any(mfrow < 1L) ||
        prod(mfrow) < panel.count
    ) {
      stop(
        "`mfrow` must contain two positive integers with enough panels.",
        call. = FALSE
      )
    }
  }

  slots <- prod(mfrow)
  panel.ids <- c(
    seq_len(panel.count),
    rep(0L, slots - panel.count)
  )

  layout.matrix <- matrix(
    panel.ids,
    nrow = mfrow[[1L]],
    ncol = mfrow[[2L]],
    byrow = byrow
  )

  widths <- rep(1, ncol(layout.matrix))
  legend.id <- NA_integer_
  legend.in.empty <- FALSE

  if (isTRUE(legend)) {
    legend.id <- panel.count + 1L
    empty <- which(layout.matrix == 0L)

    if (length(empty) > 0L) {
      # Prefer the last unused panel. With the conventional 2 x 3
      # diagnostic layout this places the shared legend in the lower-right
      # cell, like a compact plot.lm()-style diagnostic page.
      layout.matrix[empty[[length(empty)]]] <- legend.id
      legend.in.empty <- TRUE
    } else {
      layout.matrix <- cbind(
        layout.matrix,
        rep(legend.id, nrow(layout.matrix))
      )
      widths <- c(widths, legend.width)
    }
  }

  list(
    matrix = layout.matrix,
    widths = widths,
    mfrow = mfrow,
    legend.id = legend.id,
    legend.in.empty = legend.in.empty
  )
}


.ecountgmifs.legend.panel <- function(
    ...,
    cex = 1.0,
    cex.min = 0.55,
    ncol = NULL,
    max.columns = 3L,
    align = c("auto", "top-left", "center", "left-center")
) {
  align <- match.arg(align)
  arguments <- list(...)
  graphics::par(
    mar = c(0.5, 0.2, 0.5, 0.2) + 0.1
  )
  graphics::plot.new()

  if (is.null(arguments$legend) || length(arguments$legend) == 0L) {
    return(invisible(NULL))
  }

  user.range <- graphics::par("usr")
  selected <- .ecountgmifs.legend.fit(
    arguments = arguments,
    available.width = diff(user.range[1:2]) * 0.96,
    available.height = diff(user.range[3:4]) * 0.96,
    cex = cex,
    cex.min = cex.min,
    ncol = ncol,
    max.columns = max.columns
  )

  selected$plot <- NULL
  selected$xpd <- FALSE

  if (identical(align, "auto")) {
    measured <- tryCatch(
      do.call(
        graphics::legend,
        c(list(x = "center", plot = FALSE), selected)
      ),
      error = function(error) NULL
    )
    panel.size <- graphics::par("pin")
    panel.aspect <- if (
        length(panel.size) == 2L &&
        all(is.finite(panel.size)) &&
        panel.size[[2L]] > 0
    ) {
      panel.size[[1L]] / panel.size[[2L]]
    } else {
      1
    }
    height.ratio <- if (!is.null(measured)) {
      measured$rect$h / diff(user.range[3:4])
    } else {
      0.5
    }

    align <- if (panel.aspect >= 1.05) {
      "center"
    } else if (is.finite(height.ratio) && height.ratio >= 0.34) {
      "top-left"
    } else {
      "left-center"
    }
  }

  legend.position <- switch(
    align,
    center = list(x = "center"),
    `top-left` = list(
      x = user.range[[1L]] + 0.02 * diff(user.range[1:2]),
      y = user.range[[4L]] - 0.02 * diff(user.range[3:4]),
      xjust = 0,
      yjust = 1
    ),
    `left-center` = list(
      x = user.range[[1L]] + 0.02 * diff(user.range[1:2]),
      y = mean(user.range[3:4]),
      xjust = 0,
      yjust = 0.5
    )
  )

  do.call(
    graphics::legend,
    c(legend.position, selected)
  )

  invisible(NULL)
}


#' Plot ecountgmifs data and fitted-model diagnostics
#'
#' @param x An `ecountgmifs.diagnostics` object.
#' @param which Diagnostic names or numeric indices.
#' @param ask Logical. Ask before drawing each additional page, following the
#'   style of `plot.lm()`.
#' @param combine Logical. Draw all requested diagnostics on one page. A
#'   sensible panel arrangement is computed by [grDevices::n2mfrow()].
#' @param mfrow Optional two-element panel arrangement used when
#'   `combine = TRUE`.
#' @param legend.position Legend placement. The default, `"outside"`, uses a
#'   dedicated right-side legend panel. Combined pages use an unused panel cell
#'   when available. Long labels are wrapped and fitted to the available panel.
#' @param legend.margin Relative width control for the dedicated outside
#'   legend panel. Larger values reserve more horizontal space.
#' @param legend.cex Target legend-text size. Labels are wrapped and arranged
#'   in columns before the size is reduced.
#' @param log.mean.variance Logical. Use `log1p` mean and variance axes.
#' @param log.overdispersion.x Logical. Use `log1p` fitted means on the
#'   overdispersion x-axis.
#' @param point.cex Point size.
#' @param smooth Logical. Add a lowess curve to residual diagnostics.
#' @param main Optional page or plot title.
#' @param ... Additional graphical arguments.
#'
#' @return `x` invisibly.
#' @export
plot.ecountgmifs.diagnostics <- function(
    x,
    which = 1:5,
    ask = interactive(),
    combine = FALSE,
    mfrow = NULL,
    legend.position = "outside",
    legend.margin = 9,
    legend.cex = 1.08,
    .manage.ask = TRUE,
    log.mean.variance = TRUE,
    log.overdispersion.x = TRUE,
    point.cex = 0.7,
    smooth = TRUE,
    main = NULL,
    ...
) {
  which <- .ecountgmifs.diagnostic.which(which)
  legend.position <- .ecountgmifs.legend.position(legend.position)
  combine <- isTRUE(combine)
  legend.cex <- as.numeric(legend.cex[[1L]])
  if (!is.finite(legend.cex) || legend.cex <= 0) {
    legend.cex <- 1.08
  }

  if (combine) {
    legend_panel <-
      !identical(legend.position, "none") &&
      any(
        vapply(
          which,
          function(value) {
            !is.null(
              .ecountgmifs.diagnostic.legend.spec(
                x,
                value
              )
            )
          },
          logical(1L)
        )
      )

    diagnostic_count <- length(which)

    if (is.null(mfrow)) {
      mfrow <- grDevices::n2mfrow(diagnostic_count)
    } else {
      mfrow <- as.integer(mfrow)

      if (
          length(mfrow) != 2L ||
          anyNA(mfrow) ||
          any(mfrow < 1L) ||
          prod(mfrow) < diagnostic_count
      ) {
        stop(
          "`mfrow` must contain two positive integers with enough diagnostic panels.",
          call. = FALSE
        )
      }
    }

    page_layout <- .ecountgmifs.panel.layout(
      panel.count = diagnostic_count,
      mfrow = mfrow,
      legend = legend_panel
    )
    layout_matrix <- page_layout$matrix
    layout_widths <- page_layout$widths

    old_par <- graphics::par(no.readonly = TRUE)
    on.exit(
      {
        graphics::layout(matrix(1L))
        graphics::par(old_par)
      },
      add = TRUE
    )

    graphics::layout(
      layout_matrix,
      widths = layout_widths
    )
    graphics::par(
      oma = c(0, 0, 2.2, 0),
      mar = c(4.2, 4.2, 3.0, 1.2) + 0.1
    )
  } else {
    if (isTRUE(.manage.ask)) {
      old_ask <- grDevices::devAskNewPage(
        ask = isTRUE(ask) && length(which) > 1L
      )
      on.exit(
        grDevices::devAskNewPage(old_ask),
        add = TRUE
      )
    }

    has_legend <-
      !identical(legend.position, "none") &&
      any(
        vapply(
          which,
          function(value) {
            !is.null(
              .ecountgmifs.diagnostic.legend.spec(
                x,
                value
              )
            )
          },
          logical(1L)
        )
      )

    if (identical(legend.position, "outside")) {
      old_par_outside <- graphics::par(no.readonly = TRUE)
      on.exit(
        {
          graphics::layout(matrix(1L))
          graphics::par(old_par_outside)
        },
        add = TRUE
      )
    }
  }

  panel_main <- if (combine) NULL else main
  panel_legend_position <- if (
      combine ||
      identical(legend.position, "outside")
  ) {
    "none"
  } else {
    legend.position
  }

  for (diagnostic in which) {
    current_outside_legend <-
      !combine &&
      identical(legend.position, "outside") &&
      !is.null(
        .ecountgmifs.diagnostic.legend.spec(
          x,
          diagnostic
        )
      )

    if (current_outside_legend) {
      current.spec <- .ecountgmifs.diagnostic.legend.spec(
        x,
        diagnostic
      )
      legend_width <- .ecountgmifs.relative.legend.width(
        labels = current.spec$legend,
        title = current.spec$section,
        cex = legend.cex,
        plot.width = 4.6
      )
      graphics::layout(
        matrix(c(1L, 2L), nrow = 1L),
        widths = c(4.6, legend_width)
      )
      graphics::par(
        mar = c(5.1, 5.1, 4.1, 1.0) + 0.1
      )
    } else if (
        !combine &&
        identical(legend.position, "outside")
    ) {
      graphics::layout(matrix(1L))
      graphics::par(
        mar = c(5.1, 5.1, 4.1, 2.1) + 0.1
      )
    }
    if (identical(diagnostic, "mean.variance")) {
      value <- x$mean_variance
      keep <-
        value$nonnegative &
        is.finite(value$mean) &
        is.finite(value$variance) &
        value$mean >= 0 &
        value$variance >= 0
      value <- value[keep, , drop = FALSE]

      if (nrow(value) == 0L) {
        stop(
          "No non-negative finite variables are available for the mean-variance diagnostic.",
          call. = FALSE
        )
      }

      x_value <- value$mean
      y_value <- value$variance
      axis_label_prefix <- ""

      if (isTRUE(log.mean.variance)) {
        x_value <- log1p(x_value)
        y_value <- log1p(y_value)
        axis_label_prefix <- "Log1p "
      }

      role <- factor(
        value$role,
        levels = c("penalized predictor", "response")
      )
      point_col <- c("#D55E00", "#0072B2")[as.integer(role)]
      point_pch <- c(16L, 17L)[as.integer(role)]
      limit <- range(c(x_value, y_value), finite = TRUE)

      graphics::plot(
        x_value,
        y_value,
        xlim = limit,
        ylim = limit,
        pch = point_pch,
        col = grDevices::adjustcolor(point_col, alpha.f = 0.55),
        cex = point.cex,
        xlab = paste0(axis_label_prefix, "mean of observed values"),
        ylab = paste0(axis_label_prefix, "variance of observed values"),
        main = if (is.null(panel_main)) {
          "Mean-variance diagnostic"
        } else {
          paste0(panel_main, ": mean-variance")
        },
        ...
      )
      graphics::abline(0, 1, lty = 3L, lwd = 1.5)

      spec <- .ecountgmifs.diagnostic.legend.spec(
        x,
        diagnostic
      )
      .ecountgmifs.draw.legend(
        panel_legend_position,
        legend = spec$legend,
        col = spec$col,
        pch = spec$pch,
        lty = spec$lty,
        lwd = spec$lwd,
        bty = "n",
        cex = legend.cex
      )
    }

    if (identical(diagnostic, "dispersion.path")) {
      value <- x$dispersion_path
      path <- value[value$source %in% c("null", "path"), , drop = FALSE]
      path <- path[
        is.finite(path$iteration) & is.finite(path$dispersion),
        ,
        drop = FALSE
      ]

      if (nrow(path) == 0L) {
        graphics::plot.new()
        graphics::title(
          main = if (is.null(panel_main)) {
            "Family-parameter path"
          } else {
            paste0(panel_main, ": family-parameter path")
          }
        )
        graphics::text(
          0.5,
          0.5,
          paste0(
            "No estimated family-parameter path is available for ",
            x$family,
            "."
          )
        )
      } else {
        is_nb <- grepl(
          "nb2|negative",
          tolower(x$family)
        )

        graphics::plot(
          path$iteration,
          path$dispersion,
          type = "l",
          lwd = 2,
          xlab = "Iteration",
          ylab = if (is_nb) {
            "NB2 dispersion parameter"
          } else {
            "First family parameter"
          },
          main = if (is.null(panel_main)) {
            if (is_nb) "NB2 dispersion path" else "Family-parameter path"
          } else {
            paste0(panel_main, ": family-parameter path")
          },
          ...
        )

        null <- value[value$source == "null", , drop = FALSE]
        if (nrow(null) > 0L) {
          graphics::points(
            null$iteration[[1L]],
            null$dispersion[[1L]],
            pch = 19L
          )
          graphics::text(
            null$iteration[[1L]],
            null$dispersion[[1L]],
            labels = "Null",
            pos = 4L,
            cex = 0.8
          )
        }

        saturated <- value[
          value$source == "saturated reference",
          ,
          drop = FALSE
        ]
        if (nrow(saturated) > 0L) {
          graphics::points(
            saturated$iteration[[1L]],
            saturated$dispersion[[1L]],
            pch = 1L
          )
          graphics::text(
            saturated$iteration[[1L]],
            saturated$dispersion[[1L]],
            labels = "Saturated reference",
            pos = 2L,
            cex = 0.8
          )
        }
      }
    }

    if (identical(diagnostic, "residuals.fitted")) {
      value <- x$residuals

      graphics::plot(
        value$fitted,
        value$residual,
        pch = 16L,
        col = grDevices::adjustcolor("black", alpha.f = 0.45),
        cex = point.cex,
        xlab = "Fitted mean",
        ylab = paste0(x$residual_type, " residual"),
        main = if (is.null(panel_main)) {
          paste0("Residuals vs fitted (", x$state_label, ")")
        } else {
          paste0(panel_main, ": residuals vs fitted")
        },
        ...
      )
      graphics::abline(h = 0, lty = 3L)

      if (isTRUE(smooth) && nrow(value) >= 3L) {
        lowess_value <- stats::lowess(
          value$fitted,
          value$residual
        )
        graphics::lines(
          lowess_value,
          col = "#D55E00",
          lwd = 2
        )
      }
    }

    if (identical(diagnostic, "scale.location")) {
      value <- x$residuals

      graphics::plot(
        value$fitted,
        value$sqrt_abs_residual,
        pch = 16L,
        col = grDevices::adjustcolor("black", alpha.f = 0.45),
        cex = point.cex,
        xlab = "Fitted mean",
        ylab = paste0("sqrt(|", x$residual_type, " residual|)"),
        main = if (is.null(panel_main)) {
          paste0("Scale-location (", x$state_label, ")")
        } else {
          paste0(panel_main, ": scale-location")
        },
        ...
      )

      if (isTRUE(smooth) && nrow(value) >= 3L) {
        lowess_value <- stats::lowess(
          value$fitted,
          value$sqrt_abs_residual
        )
        graphics::lines(
          lowess_value,
          col = "#D55E00",
          lwd = 2
        )
      }
    }

    if (identical(diagnostic, "overdispersion")) {
      value <- x$overdispersion
      value <- value[
        is.finite(value$fitted_mean) &
        is.finite(value$empirical),
        ,
        drop = FALSE
      ]

      if (nrow(value) == 0L) {
        stop(
          "No finite binned values are available for the overdispersion diagnostic.",
          call. = FALSE
        )
      }

      x_value <- value$fitted_mean
      x_label <- "Fitted mean"

      if (isTRUE(log.overdispersion.x)) {
        x_value <- log1p(x_value)
        x_label <- "Log1p fitted mean"
      }

      if (identical(x$overdispersion_kind, "count")) {
        graphics::plot(
          x_value,
          value$empirical,
          type = "b",
          pch = 16L,
          cex = point.cex,
          lwd = 1.5,
          xlab = x_label,
          ylab = "Empirical variance-to-mean ratio",
          main = if (is.null(panel_main)) {
            paste0(x$family, " overdispersion diagnostic")
          } else {
            paste0(panel_main, ": overdispersion")
          },
          ...
        )
        graphics::abline(h = 1, lty = 3L, lwd = 1.5)

        expected <- is.finite(value$expected)
        if (any(expected)) {
          graphics::lines(
            x_value[expected],
            value$expected[expected],
            col = "#D55E00",
            lty = 2L,
            lwd = 2
          )
        }
      } else {
        overall_variance <- stats::var(
          x$residuals$raw_residual,
          na.rm = TRUE
        )

        graphics::plot(
          x_value,
          value$empirical,
          type = "b",
          pch = 16L,
          cex = point.cex,
          lwd = 1.5,
          xlab = x_label,
          ylab = "Binned residual variance",
          main = if (is.null(panel_main)) {
            paste0(x$family, " variance/heteroscedasticity diagnostic")
          } else {
            paste0(panel_main, ": variance diagnostic")
          },
          ...
        )

        expected <- is.finite(value$expected)
        if (any(expected)) {
          graphics::lines(
            x_value[expected],
            value$expected[expected],
            col = "#D55E00",
            lty = 2L,
            lwd = 2
          )
        } else if (is.finite(overall_variance)) {
          graphics::abline(
            h = overall_variance,
            col = "#D55E00",
            lty = 2L,
            lwd = 2
          )
        }
      }

      spec <- .ecountgmifs.diagnostic.legend.spec(
        x,
        diagnostic
      )
      .ecountgmifs.draw.legend(
        panel_legend_position,
        legend = spec$legend,
        col = spec$col,
        pch = spec$pch,
        lty = spec$lty,
        lwd = spec$lwd,
        bty = "n",
        cex = legend.cex
      )
    }

    if (current_outside_legend) {
      graphics::par(
        mar = c(0.5, 0.2, 0.5, 0.2) + 0.1
      )
      .ecountgmifs.draw.diagnostic.legend.panel(
        x,
        diagnostic,
        cex = legend.cex,
        align = "auto"
      )
    }
  }

  if (combine) {
    if (legend_panel) {
      .ecountgmifs.draw.diagnostic.legend.panel(
        x,
        which,
        cex = legend.cex
      )
    }

    page_title <- if (is.null(main)) {
      paste0(
        "ecountgmifs diagnostics: ",
        x$state_label
      )
    } else {
      main
    }

    graphics::mtext(
      page_title,
      side = 3,
      outer = TRUE,
      line = 0.5,
      font = 2
    )
  }

  invisible(x)
}
