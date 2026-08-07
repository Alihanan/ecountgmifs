.plugin.family.poisson.live.code <- paste(
  c(
    "#include <RcppArmadillo.h>",
    "// [[Rcpp::depends(RcppArmadillo)]]",
    "// [[Rcpp::depends(nloptr)]]",
    "// [[Rcpp::depends(ecountgmifs)]]",
    "// [[Rcpp::plugins(cpp17)]]",
    "#include <ecountgmifs/api.h>",
    "#include <algorithm>",
    "#include <cmath>",
    "#include <stdexcept>",
    "#include <string>",
    "",
    "class LivePoissonFamily final : public IEcountgmifsFamily",
    "{",
    "  double mu_min_;",
    "  double mu_max_;",
    "public:",
    "  LivePoissonFamily(double mu_min, double mu_max) : mu_min_(mu_min), mu_max_(mu_max)",
    "  {",
    "    if (!std::isfinite(mu_min_) || !std::isfinite(mu_max_) || mu_min_ <= 0.0 || mu_min_ >= mu_max_)",
    "      throw std::invalid_argument(\"Invalid Poisson mean clamps.\");",
    "  }",
    "  std::string name() const override { return \"Poisson.live\"; }",
    "  arma::uword parameter_count() const noexcept override { return 0; }",
    "  void prepare(const EcountgmifsInput& input, const EcountgmifsControl& control) const override",
    "  {",
    "    (void) input;",
    "    (void) control;",
    "  }",
    "  void negloglik(const arma::vec& y, const arma::vec& mu, const arma::vec& parameters, double& value) const override",
    "  {",
    "    if (!parameters.empty() || y.n_elem != mu.n_elem) throw std::invalid_argument(\"Invalid Poisson arguments.\");",
    "    value = 0.0;",
    "    for (arma::uword i = 0; i < y.n_elem; ++i) {",
    "      const double m = std::min(std::max(mu[i], mu_min_), mu_max_);",
    "      value += m - y[i] * std::log(m) + std::lgamma(y[i] + 1.0);",
    "    }",
    "  }",
    "  void grad(const arma::vec& y, const arma::vec& mu, const arma::vec& parameters, arma::vec& d_mu, arma::vec& d_parameters) const override",
    "  {",
    "    if (!parameters.empty() || y.n_elem != mu.n_elem) throw std::invalid_argument(\"Invalid Poisson arguments.\");",
    "    d_mu = 1.0 - y / mu;",
    "    d_parameters.reset();",
    "  }",
    "};",
    "",
    "// [[Rcpp::export]]",
    "SEXP create_live_poisson_family(double mu_min_cap, double mu_max_cap)",
    "{",
    "  return Rcpp::XPtr<IEcountgmifsFamily>(new LivePoissonFamily(mu_min_cap, mu_max_cap), true);",
    "}"
  ),
  collapse = "\n"
)

.plugin.family.nb2.live.code <- paste(
  c(
    "#include <RcppArmadillo.h>",
    "// [[Rcpp::depends(RcppArmadillo)]]",
    "// [[Rcpp::depends(nloptr)]]",
    "// [[Rcpp::depends(ecountgmifs)]]",
    "// [[Rcpp::plugins(cpp17)]]",
    "#include <ecountgmifs/api.h>",
    "#include <algorithm>",
    "#include <cmath>",
    "#include <limits>",
    "#include <stdexcept>",
    "#include <string>",
    "",
    "class LiveNB2Family final : public IEcountgmifsFamily",
    "{",
    "  double mu_min_;",
    "  double mu_max_;",
    "  double fallback_;",
    "  double initial_;",
    "  double lower_;",
    "  double upper_;",
    "  mutable arma::vec log_factorial_;",
    "",
    "  double dispersion(const arma::vec& parameters) const",
    "  {",
    "    if (parameters.n_elem != 1 || !std::isfinite(parameters[0]))",
    "      throw std::invalid_argument(\"NB2 requires one finite dispersion.\");",
    "    return parameters[0];",
    "  }",
    "public:",
    "  LiveNB2Family(double mu_min, double mu_max, double fallback, double initial, double lower, double upper)",
    "    : mu_min_(mu_min), mu_max_(mu_max), fallback_(fallback), initial_(initial), lower_(lower), upper_(upper)",
    "  {",
    "    if (!std::isfinite(mu_min_) || !std::isfinite(mu_max_) || mu_min_ <= 0.0 || mu_min_ >= mu_max_)",
    "      throw std::invalid_argument(\"Invalid NB2 mean clamps.\");",
    "    if (!std::isfinite(fallback_) || fallback_ < 0.0 || lower_ < 0.0 || lower_ >= upper_ || initial_ < lower_ || initial_ > upper_)",
    "      throw std::invalid_argument(\"Invalid NB2 dispersion configuration.\");",
    "  }",
    "  std::string name() const override { return \"NB2.live\"; }",
    "  arma::uword parameter_count() const noexcept override { return 1; }",
    "  arma::vec initial_parameters() const override { return arma::vec({initial_}); }",
    "  arma::vec parameter_lower_bounds() const override { return arma::vec({lower_}); }",
    "  arma::vec parameter_upper_bounds() const override { return arma::vec({upper_}); }",
    "  void prepare(const EcountgmifsInput& input, const EcountgmifsControl& control) const override",
    "  {",
    "    (void) control;",
    "    log_factorial_ = arma::lgamma(input.y + 1.0);",
    "  }",
    "",
    "  void negloglik(const arma::vec& y, const arma::vec& mu, const arma::vec& parameters, double& value) const override",
    "  {",
    "    if (y.n_elem != mu.n_elem) throw std::invalid_argument(\"NB2 y/mu lengths differ.\");",
    "    if (log_factorial_.n_elem != y.n_elem)",
    "      throw std::runtime_error(\"LiveNB2Family::prepare() must be called before negloglik().\");",
    "    const double a = dispersion(parameters);",
    "    value = 0.0;",
    "    if (a <= fallback_) {",
    "      for (arma::uword i = 0; i < y.n_elem; ++i) {",
    "        const double m = std::min(std::max(mu[i], mu_min_), mu_max_);",
    "        value += m - y[i] * std::log(m) + log_factorial_[i];",
    "      }",
    "      return;",
    "    }",
    "    if (a <= 0.0) { value = std::numeric_limits<double>::infinity(); return; }",
    "    const double r = 1.0 / a;",
    "    const double lgamma_r = std::lgamma(r);",
    "    for (arma::uword i = 0; i < y.n_elem; ++i) {",
    "      const double m = std::min(std::max(mu[i], mu_min_), mu_max_);",
    "      const double am = a * m;",
    "      const double lp = std::log1p(am);",
    "      value -= y[i] * (std::log(am) - lp) - r * lp + std::lgamma(y[i] + r) - log_factorial_[i] - lgamma_r;",
    "    }",
    "  }",
    "",
    "  void grad(const arma::vec& y, const arma::vec& mu, const arma::vec& parameters, arma::vec& d_mu, arma::vec& d_parameters) const override",
    "  {",
    "    if (y.n_elem != mu.n_elem) throw std::invalid_argument(\"NB2 y/mu lengths differ.\");",
    "    const double a = dispersion(parameters);",
    "    d_parameters.set_size(1);",
    "    if (a <= fallback_) { d_mu = 1.0 - y / mu; d_parameters[0] = 0.0; return; }",
    "    const double r = 1.0 / a;",
    "    d_mu = -y / mu + (1.0 + a * y) / (1.0 + a * mu);",
    "    double d_r = 0.0;",
    "    for (arma::uword i = 0; i < y.n_elem; ++i) {",
    "      const double rpm = r + mu[i];",
    "      d_r += R::digamma(r) - R::digamma(y[i] + r) - std::log(r / rpm) - 1.0 + (r + y[i]) / rpm;",
    "    }",
    "    d_parameters[0] = -d_r / (a * a);",
    "  }",
    "};",
    "",
    "// [[Rcpp::export]]",
    "SEXP create_live_nb2_family(double mu_min_cap, double mu_max_cap, double poisson_fallback_eps, double dispersion_initial, double dispersion_lower_bound, double dispersion_upper_bound)",
    "{",
    "  return Rcpp::XPtr<IEcountgmifsFamily>(new LiveNB2Family(mu_min_cap, mu_max_cap, poisson_fallback_eps, dispersion_initial, dispersion_lower_bound, dispersion_upper_bound), true);",
    "}"
  ),
  collapse = "\n"
)

#' Family plugins in four implementation variants
#'
#' @name family.plugins
#' @aliases family-plugins
#'
#' @description
#' A family plugin evaluates negative log-likelihood and its derivatives with
#' respect to `mu` and optimized family parameters. [r.family()] supports an
#' optional `prepare(input, control, environment)` callback invoked once before
#' fitting, followed by `negloglik(y, mu, family.parameters, environment)` and
#' `grad(y, mu, family.parameters, environment)`.
#'
#' `negloglik()` returns one finite numeric value. `grad()` returns a list with
#' `d_negloglik_d_mu`, a vector of length `length(y)`, and
#' `d_negloglik_d_family_parameters`, a vector with one element per optimized
#' family parameter.
#'
#' @section Current examples:
#' Poisson has no optimized family parameters. Its positive `mu.min.cap` and
#' `mu.max.cap` are fixed auxiliary configuration. NB2 uses dispersion `a` in
#' `Var(Y) = mu * (1 + a * mu)` as one optimized family parameter. Its initial
#' value and lower/upper bounds are constructor arguments. `mu.min.cap`,
#' `mu.max.cap`, and `poisson.fallback.eps` are fixed configuration and are not
#' optimized. The NB2 examples use `prepare()` to cache `lgamma(y + 1)` once
#' per fit and reuse it in both the NB2 and Poisson-limit likelihoods.
#'
#' @section Four variants:
#' `.builtin()` uses classes from `src/example.h`; `.live()` compiles complete
#' C++ children of `IEcountgmifsFamily`; `.r.environment()` stores fixed data in
#' one explicit environment; `.r.closure()` captures fixed data in the outer R
#' factory. All return ordinary owning external pointers.
#'
#' @section Performance and thread safety:
#' Native and live C++ are suitable for repeated likelihood and gradient calls.
#' Their prepared NB2 cache is mutable and fit-specific, so the same plugin
#' pointer must not be evaluated concurrently by multiple fits.
#' R callbacks are vectorized but pay one R boundary crossing per evaluation.
#' R callbacks must execute on the R main thread. Mutable retained state is not
#' thread-safe and should normally be reserved for diagnostics.
#'
#' @param mu.min.cap,mu.max.cap Positive fixed clamps for `mu`.
#' @param poisson.fallback.eps Non-negative fixed NB2 threshold below which the
#'   Poisson limit is used.
#' @param dispersion.initial Initial optimized NB2 dispersion.
#' @param dispersion.lower.bound,dispersion.upper.bound Bounds for dispersion.
#' @param cache Reuse the live-compiled object in the current R session.
#' @return An external pointer to an `IEcountgmifsFamily` child.
#'
#' @examples
#' families <- list(
#'   poisson.builtin = plugin.family.poisson.builtin(),
#'   poisson.live = plugin.family.poisson.live(),
#'   nb2.environment = plugin.family.nb2.r.environment(),
#'   nb2.closure = plugin.family.nb2.r.closure()
#' )
#'
#' \dontrun{
#' set.seed(1)
#' X <- matrix(rnorm(80), 20, 4)
#' y <- rnbinom(20, mu = 2, size = 4)
#' fit <- ecountgmifs(
#'   X = X,
#'   y = y,
#'   family = plugin.family.nb2.r.environment(
#'     mu.min.cap = 1e-12,
#'     mu.max.cap = 1e12,
#'     poisson.fallback.eps = 1e-8,
#'     dispersion.initial = 0.25,
#'     dispersion.lower.bound = 1e-12,
#'     dispersion.upper.bound = 1e12
#'   ),
#'   link = plugin.link.log.builtin(),
#'   control = ecountgmifs.control(stagewise.iteration.max = 20L)
#' )
#' }
#'
#' @seealso [custom.plugins], [link.plugins], [family.link.plugins], [r.family()]
NULL

#' @rdname family.plugins
#' @export
plugin.family.poisson.builtin <- function(mu.min.cap = 1e-12, mu.max.cap = 1e12) {
  example_create_poisson_family(mu_min_cap = mu.min.cap, mu_max_cap = mu.max.cap)
}

#' @rdname family.plugins
#' @export
plugin.family.poisson.live <- function(
    mu.min.cap = 1e-12,
    mu.max.cap = 1e12,
    cache = TRUE
) {
  .r.validate.caps(mu.min.cap, mu.max.cap)
  compile.family(
    code = .plugin.family.poisson.live.code,
    factory = "create_live_poisson_family",
    name = "Poisson.live",
    arguments = list(mu_min_cap = mu.min.cap, mu_max_cap = mu.max.cap),
    cache = cache
  )
}

#' @rdname family.plugins
#' @export
plugin.family.poisson.r.environment <- function(mu.min.cap = 1e-12, mu.max.cap = 1e12) {
  .r.validate.caps(mu.min.cap, mu.max.cap)
  workspace <- r.environment(mu.min.cap = mu.min.cap, mu.max.cap = mu.max.cap)
  r.family(
    name = "Poisson.r.environment",
    environment = workspace,
    negloglik = function(y, mu, family.parameters, environment) {
      .r.poisson.negloglik(y, mu, environment$mu.min.cap, environment$mu.max.cap)
    },
    grad = function(y, mu, family.parameters, environment) .r.poisson.grad(y, mu)
  )
}

#' @rdname family.plugins
#' @export
plugin.family.poisson.r.closure <- function(mu.min.cap = 1e-12, mu.max.cap = 1e12) {
  .r.validate.caps(mu.min.cap, mu.max.cap)
  r.family(
    name = "Poisson.r.closure",
    negloglik = function(y, mu, family.parameters, environment) {
      .r.poisson.negloglik(y, mu, mu.min.cap, mu.max.cap)
    },
    grad = function(y, mu, family.parameters, environment) .r.poisson.grad(y, mu)
  )
}

.nb2.arguments <- function(
    mu.min.cap,
    mu.max.cap,
    poisson.fallback.eps,
    dispersion.initial,
    dispersion.lower.bound,
    dispersion.upper.bound
) {
  .r.validate.nb2(
    mu.min.cap, mu.max.cap, poisson.fallback.eps,
    dispersion.initial, dispersion.lower.bound, dispersion.upper.bound
  )
  list(
    mu.min.cap = mu.min.cap,
    mu.max.cap = mu.max.cap,
    poisson.fallback.eps = poisson.fallback.eps,
    dispersion.initial = dispersion.initial,
    dispersion.lower.bound = dispersion.lower.bound,
    dispersion.upper.bound = dispersion.upper.bound
  )
}

#' @rdname family.plugins
#' @export
plugin.family.nb2.builtin <- function(
    mu.min.cap = 1e-12,
    mu.max.cap = 1e12,
    poisson.fallback.eps = 1e-8,
    dispersion.initial = 1e-4,
    dispersion.lower.bound = 1e-12,
    dispersion.upper.bound = 1e12
) {
  settings <- .nb2.arguments(
    mu.min.cap, mu.max.cap, poisson.fallback.eps,
    dispersion.initial, dispersion.lower.bound, dispersion.upper.bound
  )
  example_create_nb2_family(
    mu_min_cap = settings$mu.min.cap,
    mu_max_cap = settings$mu.max.cap,
    poisson_fallback_eps = settings$poisson.fallback.eps,
    dispersion_initial = settings$dispersion.initial,
    dispersion_lower_bound = settings$dispersion.lower.bound,
    dispersion_upper_bound = settings$dispersion.upper.bound
  )
}

#' @rdname family.plugins
#' @export
plugin.family.nb2.live <- function(
    mu.min.cap = 1e-12,
    mu.max.cap = 1e12,
    poisson.fallback.eps = 1e-8,
    dispersion.initial = 1e-4,
    dispersion.lower.bound = 1e-12,
    dispersion.upper.bound = 1e12,
    cache = TRUE
) {
  settings <- .nb2.arguments(
    mu.min.cap, mu.max.cap, poisson.fallback.eps,
    dispersion.initial, dispersion.lower.bound, dispersion.upper.bound
  )
  compile.family(
    code = .plugin.family.nb2.live.code,
    factory = "create_live_nb2_family",
    name = "NB2.live",
    arguments = list(
      mu_min_cap = settings$mu.min.cap,
      mu_max_cap = settings$mu.max.cap,
      poisson_fallback_eps = settings$poisson.fallback.eps,
      dispersion_initial = settings$dispersion.initial,
      dispersion_lower_bound = settings$dispersion.lower.bound,
      dispersion_upper_bound = settings$dispersion.upper.bound
    ),
    cache = cache
  )
}

.make.nb2.r <- function(settings, name, use.environment) {
  workspace <- if (use.environment) {
    r.environment(
      mu.min.cap = settings$mu.min.cap,
      mu.max.cap = settings$mu.max.cap,
      poisson.fallback.eps = settings$poisson.fallback.eps
    )
  } else {
    NULL
  }

  current <- function(environment) {
    if (use.environment) {
      list(
        mu.min.cap = environment$mu.min.cap,
        mu.max.cap = environment$mu.max.cap,
        poisson.fallback.eps = environment$poisson.fallback.eps
      )
    } else {
      settings
    }
  }

  r.family(
    name = name,
    prepare = function(input, control, environment) {
      environment$log.factorial <- lgamma(input$y + 1)
      invisible(NULL)
    },
    initial.parameters = settings$dispersion.initial,
    lower.bounds = settings$dispersion.lower.bound,
    upper.bounds = settings$dispersion.upper.bound,
    environment = workspace,
    negloglik = function(y, mu, family.parameters, environment) {
      fixed <- current(environment)
      .r.nb2.negloglik(
        y, mu, family.parameters[[1L]], fixed$mu.min.cap,
        fixed$mu.max.cap, fixed$poisson.fallback.eps,
        log.factorial = environment$log.factorial
      )
    },
    grad = function(y, mu, family.parameters, environment) {
      fixed <- current(environment)
      .r.nb2.grad(y, mu, family.parameters[[1L]], fixed$poisson.fallback.eps)
    }
  )
}

#' @rdname family.plugins
#' @export
plugin.family.nb2.r.environment <- function(
    mu.min.cap = 1e-12,
    mu.max.cap = 1e12,
    poisson.fallback.eps = 1e-8,
    dispersion.initial = 1e-4,
    dispersion.lower.bound = 1e-12,
    dispersion.upper.bound = 1e12
) {
  settings <- .nb2.arguments(
    mu.min.cap, mu.max.cap, poisson.fallback.eps,
    dispersion.initial, dispersion.lower.bound, dispersion.upper.bound
  )
  .make.nb2.r(settings, "NB2.r.environment", TRUE)
}

#' @rdname family.plugins
#' @export
plugin.family.nb2.r.closure <- function(
    mu.min.cap = 1e-12,
    mu.max.cap = 1e12,
    poisson.fallback.eps = 1e-8,
    dispersion.initial = 1e-4,
    dispersion.lower.bound = 1e-12,
    dispersion.upper.bound = 1e12
) {
  settings <- .nb2.arguments(
    mu.min.cap, mu.max.cap, poisson.fallback.eps,
    dispersion.initial, dispersion.lower.bound, dispersion.upper.bound
  )
  .make.nb2.r(settings, "NB2.r.closure", FALSE)
}

# Compatibility aliases.
#' @rdname family.plugins
#' @export
example.poisson.family <- plugin.family.poisson.builtin
#' @rdname family.plugins
#' @export
example.nb2.family <- plugin.family.nb2.builtin
#' @rdname family.plugins
#' @export
example.r.poisson.family <- plugin.family.poisson.r.environment
#' @rdname family.plugins
#' @export
example.r.nb2.family <- plugin.family.nb2.r.environment
#' @rdname family.plugins
#' @export
example.r.poisson.family.environment <- plugin.family.poisson.r.environment
#' @rdname family.plugins
#' @export
example.r.poisson.family.closure <- plugin.family.poisson.r.closure
#' @rdname family.plugins
#' @export
example.r.nb2.family.environment <- plugin.family.nb2.r.environment
#' @rdname family.plugins
#' @export
example.r.nb2.family.closure <- plugin.family.nb2.r.closure
