# Source this file after installing the revised package. Sourcing only defines
# functions; it does not launch the 500-replicate study or write any files.
#
# library(ecountgmifs)
# source(system.file("experiments", "nb2_noise_study.R", package = "ecountgmifs"))
# pilot <- run.nb2.noise.study(repetitions = 5)
# study <- run.nb2.noise.study(repetitions = 500,
#                              output.directory = "nb2_noise_results")
#
# The experiment is univariate and unpenalized. It isolates observation noise
# and the NB2 objective; it does not test high-dimensional variable selection.

nb2.study.noises <- function() {
  list(
    none = ecountgmifs::noise.none(),
    uniform_positive = ecountgmifs::noise.uniform.positive(),
    uniform_centered = ecountgmifs::noise.uniform.centered(),
    gaussian_matched = ecountgmifs::noise.gaussian(sd = 1 / sqrt(12)),
    gaussian_sd1 = ecountgmifs::noise.gaussian(sd = 1)
  )
}

nb2.study.grid <- function() {
  expand.grid(
    mu = c(0.5, 5, 50),
    dispersion = c(0.1, 0.5, 2),
    n = c(100L, 1000L),
    KEEP.OUT.ATTRS = FALSE
  )
}

# Intercept-only fitting: for any fixed dispersion, the interior optimum of
# mu is mean(y). Thus only dispersion needs a numerical search. All objective
# evaluations, including the Poisson boundary, call the package's C++ family.
#
# Search the positive interval on a log scale, refine every grid-local maximum,
# and compare with alpha = 0 and both search endpoints. Report search-boundary
# hits instead of silently treating them as interior estimates. The small gap
# (0, dispersion.bounds[1]) is not searched; reduce the lower bound cautiously
# if it is selected. The existing gamma subtraction can lose precision near 0.
fit.nb2.intercept <- function(y, dispersion.bounds = c(1e-6, 100),
                             search.points = 41L,
                             poisson.fallback.eps = 1e-8) {
  check.nb2.search(dispersion.bounds, search.points, poisson.fallback.eps)

  # This validates the data and numerical settings before computing the mean.
  ecountgmifs::nb2.loglik(y, mu = 1, dispersion = 0,
                         poisson.fallback.eps = poisson.fallback.eps)
  sample.mean <- mean(y)
  if (sample.mean == 0) {
    return(list(
      mu = 0, dispersion = NA_real_,
      loglik = ecountgmifs::nb2.loglik(y, 0, 0),
      status = "all_zero", mean.capped = TRUE,
      message = "Mean is zero; dispersion is not identifiable at that boundary."
    ))
  }
  fitted.mean <- min(max(sample.mean, 1e-12), 1e12)

  objective <- function(dispersion) {
    ecountgmifs::nb2.loglik(y, fitted.mean, dispersion,
                           poisson.fallback.eps = poisson.fallback.eps)
  }
  log.dispersions <- seq(log(dispersion.bounds[1]),
                         log(dispersion.bounds[2]), length.out = search.points)
  candidates <- exp(log.dispersions)
  candidate.loglik <- vapply(candidates, objective, numeric(1))
  if (any(!is.finite(candidate.loglik))) {
    stop("The C++ objective was non-finite inside the dispersion search.")
  }

  for (i in seq.int(2L, search.points - 1L)) {
    is.local.maximum <- candidate.loglik[i] >= candidate.loglik[i - 1L] &&
      candidate.loglik[i] >= candidate.loglik[i + 1L]
    if (is.local.maximum) {
      refined <- stats::optimize(
        f = function(log.dispersion) objective(exp(log.dispersion)),
        interval = log.dispersions[c(i - 1L, i + 1L)],
        maximum = TRUE,
        tol = 1e-7
      )
      candidates <- c(candidates, exp(refined$maximum))
      candidate.loglik <- c(candidate.loglik, refined$objective)
    }
  }

  # Put zero first: exact numerical ties prefer the simpler Poisson boundary.
  candidates <- c(0, candidates)
  candidate.loglik <- c(objective(0), candidate.loglik)
  best <- which.max(candidate.loglik)
  fitted.dispersion <- candidates[best]

  status <- "ok"
  if (fitted.dispersion == 0) status <- "poisson_boundary"
  if (best == 2L) status <- "positive_lower_bound"
  if (best == search.points + 1L) status <- "upper_bound"

  list(
    mu = fitted.mean,
    dispersion = fitted.dispersion,
    loglik = candidate.loglik[best],
    status = status,
    mean.capped = fitted.mean != sample.mean,
    message = ""
  )
}

# Evaluate a two-dimensional objective surface on one fixed response vector.
# Each surface is centered by its own grid maximum and divided by sample size.
# These relative objective values are diagnostics, not confidence regions.
nb2.objective.surface <- function(y, mu.values, dispersion.values) {
  if (!length(mu.values) || !length(dispersion.values)) {
    stop("Both parameter grids must be non-empty.")
  }
  surface <- expand.grid(mu = mu.values, dispersion = dispersion.values,
                         KEEP.OUT.ATTRS = FALSE)
  surface$loglik <- vapply(seq_len(nrow(surface)), function(i) {
    ecountgmifs::nb2.loglik(y, surface$mu[i], surface$dispersion[i])
  }, numeric(1))
  surface$relative.loglik.per.observation <-
    (surface$loglik - max(surface$loglik)) / length(y)
  surface
}

run.nb2.noise.study <- function(grid = nb2.study.grid(),
                                noises = nb2.study.noises(),
                                repetitions = 500L, test.n = 1000L,
                                seed = 20260923L,
                                dispersion.bounds = c(1e-6, 100),
                                search.points = 41L,
                                output.directory = NULL,
                                verbose = TRUE) {
  if (!is.data.frame(grid) || nrow(grid) == 0L ||
      !all(c("mu", "dispersion", "n") %in% names(grid))) {
    stop("`grid` must be a non-empty data frame with mu, dispersion, and n.")
  }
  for (column in c("mu", "dispersion", "n")) {
    if (!is.numeric(grid[[column]]) || any(!is.finite(grid[[column]]))) {
      stop("Grid columns must be finite numeric vectors.")
    }
  }
  if (any(grid$mu < 0) || any(grid$dispersion < 0) || any(grid$n < 2) ||
      any(grid$n != floor(grid$n)) || any(grid$n > .Machine$integer.max)) {
    stop("Grid parameters must be non-negative and n must be an integer >= 2.")
  }
  if (!is.list(noises) || !length(noises) || is.null(names(noises)) ||
      anyNA(names(noises)) || any(!nzchar(names(noises))) ||
      anyDuplicated(names(noises))) {
    stop("`noises` must be a non-empty list with unique, non-empty names.")
  }
  for (noise in noises) {
    if (!inherits(noise, "ecountgmifs_noise")) stop("Invalid noise object.")
  }
  check.study.integer(repetitions, "repetitions")
  check.study.integer(test.n, "test.n")
  check.study.integer(seed, "seed", minimum = 0)
  check.nb2.search(dispersion.bounds, search.points, poisson.fallback.eps = 1e-8)
  if (!is.null(output.directory)) {
    if (length(output.directory) != 1L || is.na(output.directory) ||
        !is.character(output.directory) || !nzchar(output.directory)) {
      stop("`output.directory` must be NULL or a non-empty path.")
    }
    # A fresh directory prevents an accidental overwrite of an earlier study.
    if (file.exists(output.directory)) stop("Output directory already exists.")
    if (!dir.create(output.directory, recursive = TRUE)) {
      stop("Could not create the output directory.")
    }
  }

  # Save and restore the caller's RNG state, including RNGkind. The result
  # records the kind as well as the seed, R version, and package version.
  previous.kind <- RNGkind()
  had.seed <- exists(".Random.seed", envir = .GlobalEnv, inherits = FALSE)
  if (had.seed) previous.seed <- get(".Random.seed", envir = .GlobalEnv)
  on.exit({
    do.call(RNGkind, as.list(previous.kind))
    if (had.seed) {
      assign(".Random.seed", previous.seed, envir = .GlobalEnv)
    } else if (exists(".Random.seed", envir = .GlobalEnv, inherits = FALSE)) {
      rm(".Random.seed", envir = .GlobalEnv)
    }
  }, add = TRUE)
  set.seed(seed)

  all.rows <- list()
  theory.rows <- list()
  representative.data <- list()
  row.number <- 0L

  safe.fit <- function(y) {
    tryCatch(
      fit.nb2.intercept(y, dispersion.bounds, search.points),
      error = function(error) list(
        mu = NA_real_, dispersion = NA_real_, loglik = NA_real_,
        status = "failed", mean.capped = NA,
        message = conditionMessage(error)
      )
    )
  }

  for (cell in seq_len(nrow(grid))) {
    mu <- grid$mu[cell]
    dispersion <- grid$dispersion[cell]
    n <- grid$n[cell]
    if (verbose) message("Cell ", cell, "/", nrow(grid),
                         ": mu=", mu, ", dispersion=", dispersion, ", n=", n)
    cell.rows <- list()

    for (noise.name in names(noises)) {
      noise <- noises[[noise.name]]
      if (is.null(noise$moments)) {
        population <- list(mean = NA_real_, variance = NA_real_,
                           dispersion.moment = NA_real_,
                           zero.probability = NA_real_)
      } else {
        population <- ecountgmifs::nb2.noise.moments(mu, dispersion, noise)
      }
      theory.rows[[length(theory.rows) + 1L]] <- data.frame(
        cell = cell, mu = mu, dispersion = dispersion, n = n,
        noise = noise.name,
        observed.mean = population$mean,
        observed.variance = population$variance,
        observed.dispersion.moment = population$dispersion.moment,
        observed.zero.probability = population$zero.probability
      )
    }

    for (replication in seq_len(repetitions)) {
      # Both draws occur before any noise arm. Every method/arm in this
      # replication sees the same training counts and independent test counts.
      latent <- ecountgmifs::generate.nb2(n, mu, dispersion)$count
      test.counts <- ecountgmifs::generate.nb2(test.n, mu, dispersion)$count
      oracle <- safe.fit(latent)
      oracle.score <- score.nb2.test(test.counts, oracle)

      for (noise.name in names(noises)) {
        data <- ecountgmifs::generate.nb2(
          n, mu, dispersion, noises[[noise.name]], latent.counts = latent
        )
        if (replication == 1L) {
          key <- paste0("cell_", cell, "_", noise.name)
          representative.data[[key]] <- data
        }
        responses <- list(oracle = data$count, untransformed = data$y,
                          floored = data$floored, rounded = data$rounded)

        for (method in names(responses)) {
          response <- responses[[method]]
          # Reusing an exactly identical response fit preserves pairing and
          # avoids fitting the oracle repeatedly. It does not change a score.
          fit <- if (all(response == latent)) oracle else safe.fit(response)
          sample.mean <- mean(response)
          sample.variance <- stats::var(response)
          sample.dispersion <- if (sample.mean == 0) NA_real_ else
            (sample.variance - sample.mean) / sample.mean^2
          test.score <- score.nb2.test(test.counts, fit)
          loglik.truth <- ecountgmifs::nb2.loglik(response, mu, dispersion)

          row <- data.frame(
            cell = cell, mu = mu, dispersion = dispersion, n = n,
            replication = replication, noise = noise.name, method = method,
            sample.mean = sample.mean, sample.variance = sample.variance,
            sample.dispersion.moment = sample.dispersion,
            zero.fraction = mean(response == 0),
            clamped.fraction = mean(data$clamped),
            changed.from.latent.fraction = mean(response != latent),
            loglik.at.truth = loglik.truth,
            loglik.at.truth.per.observation = loglik.truth / n,
            fitted.mu = fit$mu, fitted.dispersion = fit$dispersion,
            loglik.at.fit = fit$loglik,
            loglik.at.fit.per.observation = fit$loglik / n,
            mu.error = fit$mu - mu,
            dispersion.error = fit$dispersion - dispersion,
            mu.difference.from.oracle = fit$mu - oracle$mu,
            dispersion.difference.from.oracle = fit$dispersion - oracle$dispersion,
            test.log.score = test.score,
            test.log.score.difference.from.oracle = test.score - oracle.score,
            fit.status = fit$status, mean.capped = fit$mean.capped,
            fit.message = fit$message,
            stringsAsFactors = FALSE
          )
          row.number <- row.number + 1L
          all.rows[[row.number]] <- row
          cell.rows[[length(cell.rows) + 1L]] <- row
        }
      }
    }
    # Optional per-cell checkpoints remain readable if a later cell fails.
    if (!is.null(output.directory)) {
      utils::write.csv(do.call(rbind, cell.rows),
                       file.path(output.directory, sprintf("cell_%02d.csv", cell)),
                       row.names = FALSE)
    }
  }

  results <- do.call(rbind, all.rows)
  study <- list(
    results = results,
    summary = summarize.nb2.noise.study(results),
    theoretical.moments = do.call(rbind, theory.rows),
    representative.data = representative.data,
    settings = list(
      grid = grid, noises = noises, repetitions = repetitions, test.n = test.n,
      seed = seed, rng.kind = RNGkind(), dispersion.bounds = dispersion.bounds,
      search.points = search.points, poisson.fallback.eps = 1e-8,
      mu.min.cap = 1e-12, mu.max.cap = 1e12,
      R.version = R.version.string,
      package.version = as.character(utils::packageVersion("ecountgmifs"))
    ),
    session.info = utils::sessionInfo()
  )
  if (!is.null(output.directory)) {
    saveRDS(study, file.path(output.directory, "study.rds"))
    utils::write.csv(study$summary, file.path(output.directory, "summary.csv"),
                     row.names = FALSE)
    utils::write.csv(study$theoretical.moments,
                     file.path(output.directory, "theoretical_moments.csv"),
                     row.names = FALSE)
    writeLines(capture.output(study$session.info),
               file.path(output.directory, "sessionInfo.txt"))
  }
  study
}

score.nb2.test <- function(test.counts, fit) {
  if (!is.finite(fit$mu) || !is.finite(fit$dispersion)) return(NA_real_)
  ecountgmifs::nb2.loglik(test.counts, fit$mu, fit$dispersion) / length(test.counts)
}

check.study.integer <- function(value, name, minimum = 1) {
  if (!is.numeric(value) || length(value) != 1L || !is.finite(value) ||
      value != floor(value) || value < minimum || value > .Machine$integer.max) {
    stop("`", name, "` must be a finite integer >= ", minimum, ".")
  }
}

check.nb2.search <- function(dispersion.bounds, search.points, poisson.fallback.eps) {
  if (!is.numeric(poisson.fallback.eps) || length(poisson.fallback.eps) != 1L ||
      !is.finite(poisson.fallback.eps) || poisson.fallback.eps < 0) {
    stop("`poisson.fallback.eps` must be finite and non-negative.")
  }
  if (!is.numeric(dispersion.bounds) || length(dispersion.bounds) != 2L ||
      any(!is.finite(dispersion.bounds)) ||
      dispersion.bounds[1] <= poisson.fallback.eps ||
      dispersion.bounds[2] <= dispersion.bounds[1]) {
    stop("Positive dispersion bounds must be ordered and above the fallback.")
  }
  check.study.integer(search.points, "search.points", minimum = 3)
}

# Monte Carlo SEs use independent replications within each scenario and method.
# Repeated oracle rows across different arms are paired, not extra replications.
summarize.nb2.noise.study <- function(results) {
  key <- paste(results$cell, results$noise, results$method, sep = ":")
  groups <- split(results, key)
  summaries <- lapply(groups, function(rows) {
    summary <- rows[1, c("cell", "mu", "dispersion", "n", "noise", "method")]
    summary$replications <- nrow(rows)
    for (status in c("failed", "all_zero", "poisson_boundary",
                     "positive_lower_bound", "upper_bound")) {
      frequency <- mean(rows$fit.status == status)
      summary[[paste0(status, ".rate")]] <- frequency
      summary[[paste0(status, ".mcse")]] <-
        sqrt(frequency * (1 - frequency) / nrow(rows))
    }
    summary$mean.capped.rate <- mean(rows$mean.capped, na.rm = TRUE)

    for (parameter in c("mu", "dispersion")) {
      error <- rows[[paste0(parameter, ".error")]]
      error <- error[is.finite(error)]
      count <- length(error)
      bias <- if (count) mean(error) else NA_real_
      rmse <- if (count) sqrt(mean(error^2)) else NA_real_
      summary[[paste0(parameter, ".valid.replications")]] <- count
      summary[[paste0(parameter, ".bias")]] <- bias
      summary[[paste0(parameter, ".bias.mcse")]] <- study.mcse(error)
      summary[[paste0(parameter, ".rmse")]] <- rmse
      # Delta-method MCSE for RMSE. At zero RMSE the derivative is undefined.
      summary[[paste0(parameter, ".rmse.mcse")]] <-
        if (count >= 2L && rmse > 0) study.mcse(error^2) / (2 * rmse) else NA_real_
    }

    for (metric in c("sample.mean", "sample.variance", "sample.dispersion.moment",
                     "zero.fraction", "clamped.fraction",
                     "changed.from.latent.fraction",
                     "loglik.at.truth.per.observation", "loglik.at.fit.per.observation",
                     "mu.difference.from.oracle", "dispersion.difference.from.oracle",
                     "test.log.score", "test.log.score.difference.from.oracle")) {
      values <- rows[[metric]]
      values <- values[is.finite(values)]
      summary[[paste0(metric, ".mean")]] <- if (length(values)) mean(values) else NA_real_
      summary[[paste0(metric, ".mcse")]] <- study.mcse(values)
      summary[[paste0(metric, ".valid.replications")]] <- length(values)
    }
    summary
  })
  result <- do.call(rbind, summaries)
  rownames(result) <- NULL
  result
}

study.mcse <- function(values) {
  if (length(values) < 2L) return(NA_real_)
  stats::sd(values) / sqrt(length(values))
}
