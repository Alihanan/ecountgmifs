#' Additive noise generators for NB2 experiments
#'
#' A noise generator is a named list of callbacks, analogous to an R family
#' object. Pass it to [generate.nb2()]. It is independent of the model's C++
#' family plugins: noise describes observation error, not the fitted family.
#'
#' @param name A descriptive, non-empty name.
#' @param draw Function of `n` returning `n` independent noise draws.
#' @param moments Optional function of `mu` and `dispersion`, returning the
#'   population moments of `pmax(0, count + noise)`. See Details.
#' @param parameters Named list of fixed noise parameters, for reporting.
#' @param sd Positive Gaussian standard deviation. The default matches the
#'   variance of either uniform noise, before clamping.
#' @param x A noise generator.
#' @param ... Unused.
#'
#' @details
#' `noise.none()` is the no-noise control; `noise.uniform.positive()` draws
#' U(0, 1); `noise.uniform.centered()` draws U(-0.5, 0.5); and
#' `noise.gaussian(sd)` draws N(0, sd^2). Negative noisy responses are always
#' clamped by [generate.nb2()], not by the noise callback.
#'
#' A custom `moments` callback returns a list with `mean`, `variance`, and
#' `zero.probability`. Optional numerical-error bounds can also be returned.
#' [nb2.noise.moments()] adds `dispersion.moment = (variance - mean) / mean^2`.
#' Negative moment dispersion is retained. It means that an NB2 distribution
#' cannot reproduce both moments. It is not an NB likelihood estimate.
#'
#' Gaussian moments use published left-censored normal moments conditional
#' on the latent integer count. The code averages the clipping corrections
#' over NB probabilities up to `ceiling(10 * sd)`, adding the exact unclipped
#' moments. Returned bounds control the omitted corrections; no NB tail is
#' renormalized. The centered-uniform calculation and the NB mixture of normal
#' moments are elementary calculations for this experiment, not results
#' claimed by the cited papers.
#'
#' @return An object of class `ecountgmifs_noise`, containing `name`, `draw`,
#'   `moments`, and `parameters`.
#' @references
#' Foi, A., Trimeche, M., Katkovnik, V., and Egiazarian, K. (2008).
#' Practical Poissonian-Gaussian noise modeling and fitting for single-image
#' raw-data. IEEE Transactions on Image Processing, 17(10), 1737--1754.
#' doi:10.1109/TIP.2008.2001399. Equations (32)--(33).
#'
#' Ouimet, F. (2023). A refined continuity correction for the negative binomial
#' distribution and asymptotics of the median. Metrika, 86, 827--849.
#' doi:10.1007/s00184-023-00897-2. Studies uniform jitter without our clamping.
#' @examples
#' noise <- noise.gaussian(sd = 1)
#' print(noise)
#' set.seed(42)
#' generate.nb2(5, mu = 2, dispersion = 0.5, noise = noise)
#'
#' custom.noise <- noise.generator(
#'   name = "small uniform",
#'   draw = function(n) stats::runif(n, -0.1, 0.1)
#' )
#' @export
noise.generator <- function(name, draw, moments = NULL, parameters = list()) {
  .r.scalar.string(name, "name")
  .r.callback(draw, "draw")
  if (!is.null(moments)) .r.callback(moments, "moments")
  if (!is.list(parameters)) {
    stop("`parameters` must be a list.", call. = FALSE)
  }

  structure(
    list(name = name, draw = draw, moments = moments, parameters = parameters),
    class = "ecountgmifs_noise"
  )
}

#' @rdname noise.generator
#' @export
noise.none <- function() {
  noise.generator(
    name = "none",
    draw = function(n) rep(0, n),
    moments = function(mu, dispersion) {
      list(
        mean = mu,
        variance = mu + dispersion * mu^2,
        zero.probability = .nb2.zero.probability(mu, dispersion)
      )
    }
  )
}

#' @rdname noise.generator
#' @export
noise.uniform.positive <- function() {
  noise.generator(
    name = "uniform_positive",
    draw = function(n) stats::runif(n, min = 0, max = 1),
    parameters = list(min = 0, max = 1),
    moments = function(mu, dispersion) {
      list(
        mean = mu + 1 / 2,
        variance = mu + dispersion * mu^2 + 1 / 12,
        zero.probability = 0
      )
    }
  )
}

#' @rdname noise.generator
#' @export
noise.uniform.centered <- function() {
  noise.generator(
    name = "uniform_centered",
    draw = function(n) stats::runif(n, min = -0.5, max = 0.5),
    parameters = list(min = -0.5, max = 0.5),
    moments = function(mu, dispersion) {
      probability.zero <- .nb2.zero.probability(mu, dispersion)
      latent.variance <- mu + dispersion * mu^2

      # Only latent zeros can cross the lower boundary. At a latent zero,
      # the clamped noise has first moment 1/8 and second moment 1/24.
      observed.mean <- mu + probability.zero / 8
      observed.variance <- latent.variance + 1 / 12 -
        probability.zero / 24 - mu * probability.zero / 4 -
        probability.zero^2 / 64

      list(
        mean = observed.mean,
        variance = observed.variance,
        zero.probability = probability.zero / 2
      )
    }
  )
}

#' @rdname noise.generator
#' @export
noise.gaussian <- function(sd = 1 / sqrt(12)) {
  .nb2.check.scalar(sd, "sd", lower = 0, strictly.greater = TRUE)
  force(sd)

  noise.generator(
    name = paste0("gaussian_sd_", format(sd, digits = 8, trim = TRUE)),
    draw = function(n) stats::rnorm(n, mean = 0, sd = sd),
    parameters = list(sd = sd),
    moments = function(mu, dispersion) {
      .nb2.gaussian.moments(mu, dispersion, sd)
    }
  )
}

#' @rdname noise.generator
#' @export
print.ecountgmifs_noise <- function(x, ...) {
  cat("Additive noise generator:", x$name, "\n")
  if (length(x$parameters)) print(x$parameters)
  cat("Negative observations are clamped to zero after adding noise.\n")
  invisible(x)
}

#' Population moments of a noisy, zero-clamped NB2 response
#'
#' @param mu Non-negative scalar latent mean.
#' @param dispersion Non-negative scalar NB2 overdispersion, with latent
#'   variance `mu + dispersion * mu^2`. Zero specifies a Poisson distribution.
#' @param noise A [noise.generator()] object with a `moments` callback.
#' @return A named list containing `mean`, `variance`, `zero.probability`,
#'   `dispersion.moment`, `latent.mean`, `latent.variance`, and
#'   `latent.dispersion`. Gaussian noise also returns absolute error bounds
#'   for the mean, second moment, variance, and zero probability. Moment
#'   dispersion is `NA_real_` when the mean is zero.
#' @seealso [noise.generator()], [generate.nb2()]
#' @examples
#' nb2.noise.moments(1, 1, noise.gaussian(sd = 1))
#' @export
nb2.noise.moments <- function(mu, dispersion, noise = noise.none()) {
  .nb2.check.parameters(mu, dispersion)
  .nb2.check.noise(noise)
  if (is.null(noise$moments)) {
    stop("This noise generator has no population-moments callback.", call. = FALSE)
  }

  result <- noise$moments(mu, dispersion)
  if (!is.list(result)) stop("`moments` must return a list.", call. = FALSE)
  for (field in c("mean", "variance", "zero.probability")) {
    .nb2.check.scalar(result[[field]], paste0("moments$", field), lower = 0)
  }
  if (result$zero.probability > 1) {
    stop("Zero probability cannot exceed one.", call. = FALSE)
  }

  result$dispersion.moment <- .nb2.moment.dispersion(result$mean, result$variance)
  result$latent.mean <- mu
  result$latent.variance <- mu + dispersion * mu^2
  result$latent.dispersion <- dispersion
  result
}

#' Generate latent NB2 counts and noisy non-negative observations
#'
#' @param n Positive integer number of observations.
#' @param mu Non-negative scalar latent mean.
#' @param dispersion Non-negative scalar NB2 overdispersion. The latent
#'   variance is `mu + dispersion * mu^2`, and R's NB `size` is `1/dispersion`.
#'   This is not the elastic-net mixing parameter.
#' @param noise A [noise.generator()] object.
#' @param latent.counts Optional non-negative integer-valued vector of length
#'   `n`. Use it to share the same latent counts across noise arms. Its claimed
#'   generating parameters are recorded but cannot be verified from the data.
#'
#' @details
#' Counts are drawn with `stats::rnbinom(n, mu = mu, size = 1/dispersion)`.
#' At exactly zero dispersion, `stats::rpois()` is used. Positive dispersion
#' is never silently replaced by zero in the generator. In contrast,
#' [nb2.loglik()] deliberately follows the C++ family's configurable Poisson
#' fallback threshold. This numerical distinction is immaterial on the
#' default study grid but matters very near zero dispersion.
#'
#' Each noise draw is added to its count. Every negative result is set to
#' zero before `floor()` and `round()` are applied. R's ties-to-even rounding
#' is used; ties have probability zero under these continuous noise laws.
#' Uniform-positive flooring and centered-uniform rounding recover the latent
#' counts almost surely. Gaussian rounding does not generally recover them.
#' The function uses the caller's RNG; call `set.seed()` for reproducibility.
#'
#' @return A data frame with columns `count`, `noise`, `before.clamp`, `y`,
#'   `floored`, `rounded`, and logical `clamped`. Attributes `generator` and
#'   `noise.generator` record the parameters and the noise object.
#' @examples
#' set.seed(1)
#' data <- generate.nb2(100, 5, 0.5, noise.uniform.positive())
#' stopifnot(all(data$floored == data$count))
#' gaussian.data <- generate.nb2(
#'   100, 5, 0.5, noise.gaussian(sd = 1), latent.counts = data$count
#' )
#' @export
generate.nb2 <- function(n, mu, dispersion, noise = noise.none(),
                         latent.counts = NULL) {
  .nb2.check.scalar(n, "n", lower = 1)
  if (n != floor(n) || n > .Machine$integer.max) {
    stop("`n` must be a positive integer within R's integer range.", call. = FALSE)
  }
  .nb2.check.parameters(mu, dispersion)
  .nb2.check.noise(noise)

  if (is.null(latent.counts)) {
    if (dispersion == 0) {
      latent.counts <- stats::rpois(n, lambda = mu)
    } else {
      latent.counts <- stats::rnbinom(n, mu = mu, size = 1 / dispersion)
    }
  }
  .nb2.check.vector(latent.counts, "latent.counts", n, nonnegative = TRUE)
  if (any(latent.counts != floor(latent.counts))) {
    stop("`latent.counts` must be integer-valued.", call. = FALSE)
  }

  noise.values <- noise$draw(n)
  .nb2.check.vector(noise.values, "noise draws", n)
  before.clamp <- latent.counts + noise.values
  if (any(!is.finite(before.clamp))) {
    stop("Adding the noise produced a non-finite observation.", call. = FALSE)
  }
  observed <- pmax(0, before.clamp)

  result <- data.frame(
    count = latent.counts,
    noise = noise.values,
    before.clamp = before.clamp,
    y = observed,
    floored = floor(observed),
    rounded = round(observed),
    clamped = before.clamp < 0
  )
  attr(result, "generator") <- list(n = n, mu = mu, dispersion = dispersion)
  attr(result, "noise.generator") <- noise
  result
}

# Internal validation is deliberately explicit: no partial recycling and no
# silent coercion of factors, character values, matrices, or missing values.
.nb2.check.scalar <- function(value, name, lower = -Inf,
                              strictly.greater = FALSE) {
  valid <- is.numeric(value) && is.null(dim(value)) && length(value) == 1L &&
    !is.na(value) && is.finite(value)
  if (valid) {
    valid <- if (strictly.greater) value > lower else value >= lower
  }
  if (!valid) {
    relation <- if (strictly.greater) "greater than" else "at least"
    stop("`", name, "` must be one finite number ", relation, " ", lower, ".",
         call. = FALSE)
  }
  invisible(NULL)
}

.nb2.check.parameters <- function(mu, dispersion) {
  .nb2.check.scalar(mu, "mu", lower = 0)
  .nb2.check.scalar(dispersion, "dispersion", lower = 0)
  if (!is.finite(mu + dispersion * mu^2)) {
    stop("The requested latent variance exceeds numeric range.", call. = FALSE)
  }
}

.nb2.check.vector <- function(value, name, n, nonnegative = FALSE) {
  if (!is.numeric(value) || !is.null(dim(value)) || length(value) != n ||
      anyNA(value) || any(!is.finite(value))) {
    stop("`", name, "` must be a finite numeric vector of length ", n, ".",
         call. = FALSE)
  }
  if (nonnegative && any(value < 0)) {
    stop("`", name, "` must be non-negative.", call. = FALSE)
  }
}

.nb2.check.noise <- function(noise) {
  if (!inherits(noise, "ecountgmifs_noise") || !is.function(noise$draw)) {
    stop("`noise` must be created by noise.generator().", call. = FALSE)
  }
}

.nb2.zero.probability <- function(mu, dispersion) {
  if (dispersion == 0) return(exp(-mu))
  exp(-log1p(dispersion * mu) / dispersion)
}

.nb2.moment.dispersion <- function(mean, variance) {
  if (mean == 0) return(NA_real_)
  (variance - mean) / mean^2
}

.nb2.gaussian.moments <- function(mu, dispersion, sd) {
  last.count <- ceiling(10 * sd)
  if (!is.finite(last.count) || last.count > 1000000) {
    stop("Gaussian sd is too large for the moment summation.", call. = FALSE)
  }
  count <- seq.int(0, last.count)
  probability <- if (dispersion == 0) {
    stats::dpois(count, lambda = mu)
  } else {
    stats::dnbinom(count, mu = mu, size = 1 / dispersion)
  }

  standardized.count <- count / sd
  normal.density <- stats::dnorm(standardized.count)
  lower.probability <- stats::pnorm(-standardized.count)

  # Published conditional censored-normal moments, written as corrections
  # to the unclipped mean k and second moment k^2 + sd^2. This avoids having
  # to truncate the (potentially very long) upper tail of the NB itself.
  mean.correction <- sd * normal.density - count * lower.probability
  second.correction <- count * sd * normal.density -
    (count^2 + sd^2) * lower.probability

  change.in.mean <- sum(probability * mean.correction)
  change.in.second.moment <- sum(probability * second.correction)
  observed.mean <- mu + change.in.mean
  # Algebraically equal to E[Y^2] - E[Y]^2, without subtracting two large
  # mu^2 terms when the latent mean is high.
  observed.variance <- mu + dispersion * mu^2 + sd^2 +
    change.in.second.moment - 2 * mu * change.in.mean - change.in.mean^2

  # For every omitted k, k >= last.count + 1. Bound the mean correction
  # by E[|epsilon| I(epsilon < -k)] and the second-moment correction by
  # E[epsilon^2 I(epsilon < -k)]. The omitted NB probabilities sum to <= 1.
  cutoff <- (last.count + 1) / sd
  tail.probability <- stats::pnorm(-cutoff)
  tail.density <- stats::dnorm(cutoff)
  mean.error <- sd * tail.density
  second.error <- sd^2 * (cutoff * tail.density + tail.probability)

  list(
    mean = observed.mean,
    variance = observed.variance,
    zero.probability = sum(probability * lower.probability),
    mean.error.bound = mean.error,
    second.moment.error.bound = second.error,
    variance.error.bound = second.error +
      2 * observed.mean * mean.error + mean.error^2,
    zero.probability.error.bound = tail.probability
  )
}
