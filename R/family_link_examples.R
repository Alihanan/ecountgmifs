.plugin.family.link.nb2.log.live.code <- paste(
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
    "",
    "class LiveNB2LogFamilyLink final : public IEcountgmifsFamilyLink",
    "{",
    "  double mu_min_;",
    "  double mu_max_;",
    "  double fallback_;",
    "  double initial_;",
    "  double lower_;",
    "  double upper_;",
    "",
    "  double dispersion(const arma::vec& parameters) const",
    "  {",
    "    if (parameters.n_elem != 1 || !std::isfinite(parameters[0]))",
    "      throw std::invalid_argument(\"NB2-log requires one finite dispersion.\");",
    "    return parameters[0];",
    "  }",
    "public:",
    "  LiveNB2LogFamilyLink(double mu_min, double mu_max, double fallback, double initial, double lower, double upper)",
    "    : mu_min_(mu_min), mu_max_(mu_max), fallback_(fallback), initial_(initial), lower_(lower), upper_(upper)",
    "  {",
    "    if (!std::isfinite(mu_min_) || !std::isfinite(mu_max_) || mu_min_ <= 0.0 || mu_min_ >= mu_max_)",
    "      throw std::invalid_argument(\"Invalid NB2-log mean clamps.\");",
    "    if (!std::isfinite(fallback_) || fallback_ < 0.0 || lower_ < 0.0 || lower_ >= upper_ || initial_ < lower_ || initial_ > upper_)",
    "      throw std::invalid_argument(\"Invalid NB2-log dispersion configuration.\");",
    "  }",
    "  std::string family_name() const override { return \"NB2.live\"; }",
    "  std::string link_name() const override { return \"Log.live\"; }",
    "  arma::uword family_parameter_count() const noexcept override { return 1; }",
    "  arma::uword link_parameter_count() const noexcept override { return 0; }",
    "  arma::vec family_initial_parameters() const override { return arma::vec({initial_}); }",
    "  arma::vec family_parameter_lower_bounds() const override { return arma::vec({lower_}); }",
    "  arma::vec family_parameter_upper_bounds() const override { return arma::vec({upper_}); }",
    "",
    "  void inverse(const arma::vec& eta, const arma::vec& link_parameters, arma::vec& mu) const override",
    "  {",
    "    if (!link_parameters.empty()) throw std::invalid_argument(\"Log link expects no parameters.\");",
    "    mu = arma::exp(eta);",
    "  }",
    "",
    "  void negloglik(const arma::vec& y, const arma::vec& mu, const arma::vec& family_parameters, double& value) const override",
    "  {",
    "    if (y.n_elem != mu.n_elem) throw std::invalid_argument(\"NB2-log y/mu lengths differ.\");",
    "    const double a = dispersion(family_parameters);",
    "    value = 0.0;",
    "    if (a <= fallback_) {",
    "      for (arma::uword i = 0; i < y.n_elem; ++i) {",
    "        const double m = std::min(std::max(mu[i], mu_min_), mu_max_);",
    "        value += m - y[i] * std::log(m) + std::lgamma(y[i] + 1.0);",
    "      }",
    "      return;",
    "    }",
    "    if (a <= 0.0) { value = std::numeric_limits<double>::infinity(); return; }",
    "    const double r = 1.0 / a;",
    "    for (arma::uword i = 0; i < y.n_elem; ++i) {",
    "      const double m = std::min(std::max(mu[i], mu_min_), mu_max_);",
    "      const double am = a * m;",
    "      const double lp = std::log1p(am);",
    "      value -= y[i] * (std::log(am) - lp) - r * lp + std::lgamma(y[i] + r) - std::lgamma(y[i] + 1.0) - std::lgamma(r);",
    "    }",
    "  }",
    "",
    "  void grad(",
    "      const arma::vec& y, const arma::vec& eta, const arma::vec& mu,",
    "      const arma::vec& family_parameters, const arma::vec& link_parameters,",
    "      arma::vec& d_nll_d_mu, arma::vec& d_mu_d_eta, arma::mat& d_mu_d_link,",
    "      arma::vec& d_nll_d_eta, arma::vec& d_nll_d_family, arma::vec& d_nll_d_link",
    "  ) const override",
    "  {",
    "    if (y.n_elem != mu.n_elem || eta.n_elem != mu.n_elem || !link_parameters.empty())",
    "      throw std::invalid_argument(\"Invalid NB2-log gradient arguments.\");",
    "    const double a = dispersion(family_parameters);",
    "    d_mu_d_eta = mu;",
    "    d_mu_d_link.set_size(mu.n_elem, 0);",
    "    d_nll_d_link.reset();",
    "    d_nll_d_family.set_size(1);",
    "    if (a <= fallback_) {",
    "      d_nll_d_mu = 1.0 - y / mu;",
    "      d_nll_d_eta = mu - y;",
    "      d_nll_d_family[0] = 0.0;",
    "      return;",
    "    }",
    "    const double r = 1.0 / a;",
    "    d_nll_d_mu = -y / mu + (1.0 + a * y) / (1.0 + a * mu);",
    "    d_nll_d_eta = (mu - y) / (1.0 + a * mu);",
    "    double d_r = 0.0;",
    "    for (arma::uword i = 0; i < y.n_elem; ++i) {",
    "      const double rpm = r + mu[i];",
    "      d_r += R::digamma(r) - R::digamma(y[i] + r) - std::log(r / rpm) - 1.0 + (r + y[i]) / rpm;",
    "    }",
    "    d_nll_d_family[0] = -d_r / (a * a);",
    "  }",
    "};",
    "",
    "// [[Rcpp::export]]",
    "SEXP create_live_nb2_log_family_link(double mu_min_cap, double mu_max_cap, double poisson_fallback_eps, double dispersion_initial, double dispersion_lower_bound, double dispersion_upper_bound)",
    "{",
    "  return Rcpp::XPtr<IEcountgmifsFamilyLink>(new LiveNB2LogFamilyLink(mu_min_cap, mu_max_cap, poisson_fallback_eps, dispersion_initial, dispersion_lower_bound, dispersion_upper_bound), true);",
    "}"
  ),
  collapse = "\n"
)

#' Fused family-link plugins in four implementation variants
#'
#' @name family.link.plugins
#' @aliases family-link-plugins
#'
#' @description
#' A fused family-link plugin combines inverse-link, likelihood, and derivative
#' calculations and may provide `d(negative log-likelihood)/d(eta)` directly.
#' It takes precedence over separately supplied `family` and `link` plugins.
#'
#' [r.family.link()] uses callbacks
#' `inverse(eta, link.parameters, environment)`,
#' `negloglik(y, mu, family.parameters, environment)`, and
#' `grad(y, eta, mu, family.parameters, link.parameters, environment)`.
#' The gradient callback returns all six named objects required by
#' `IEcountgmifsFamilyLink`: `d_negloglik_d_mu`, `d_mu_d_eta`,
#' `d_mu_d_link_parameters`, `d_negloglik_d_eta`,
#' `d_negloglik_d_family_parameters`, and
#' `d_negloglik_d_link_parameters`. Vector lengths and matrix dimensions are
#' checked by the compiled adapter.
#'
#' @section Current NB2-log example:
#' Dispersion is one optimized family parameter with configurable initial value
#' and bounds. The log link has zero optimized parameters. `mu.min.cap`,
#' `mu.max.cap`, and `poisson.fallback.eps` are fixed auxiliary configuration.
#' The fused implementation supplies the stable eta-scale derivative
#' `(mu - y) / (1 + dispersion * mu)` outside the Poisson fallback region.
#'
#' @section Four variants:
#' `.builtin()` uses `NB2LogFamilyLink` from `src/example.h`; `.live()` compiles
#' a complete C++ child; `.r.environment()` retains explicit settings; and
#' `.r.closure()` captures immutable settings lexically.
#'
#' @section Performance and thread safety:
#' Fused built-in or live C++ avoids R callbacks and can avoid intermediate
#' chain-rule work. R variants are vectorized but execute on the R main thread.
#' Mutable R state is not thread-safe.
#'
#' @inheritParams plugin.family.nb2.builtin
#' @param cache Reuse the live-compiled object in the current R session.
#' @return An external pointer to an `IEcountgmifsFamilyLink` child.
#'
#' @examples
#' implementations <- list(
#'   builtin = plugin.family.link.nb2.log.builtin(),
#'   live = plugin.family.link.nb2.log.live(),
#'   environment = plugin.family.link.nb2.log.r.environment(),
#'   closure = plugin.family.link.nb2.log.r.closure()
#' )
#'
#' \dontrun{
#' set.seed(1)
#' X <- matrix(rnorm(80), 20, 4)
#' y <- rnbinom(20, mu = 2, size = 4)
#' fit <- ecountgmifs(
#'   X = X,
#'   y = y,
#'   family.link = plugin.family.link.nb2.log.builtin(),
#'   control = ecountgmifs.control(stagewise.iteration.max = 20L)
#' )
#' }
#'
#' @seealso [custom.plugins], [family.plugins], [link.plugins], [r.family.link()]
NULL

#' @rdname family.link.plugins
#' @export
plugin.family.link.nb2.log.builtin <- function(
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
  example_create_nb2_log_family_link(
    mu_min_cap = settings$mu.min.cap,
    mu_max_cap = settings$mu.max.cap,
    poisson_fallback_eps = settings$poisson.fallback.eps,
    dispersion_initial = settings$dispersion.initial,
    dispersion_lower_bound = settings$dispersion.lower.bound,
    dispersion_upper_bound = settings$dispersion.upper.bound
  )
}

#' @rdname family.link.plugins
#' @export
plugin.family.link.nb2.log.live <- function(
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
  compile.family.link(
    code = .plugin.family.link.nb2.log.live.code,
    factory = "create_live_nb2_log_family_link",
    name = "NB2/Log.live",
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

.make.nb2.log.r <- function(settings, name, use.environment) {
  workspace <- if (use.environment) {
    r.environment(
      mu.min.cap = settings$mu.min.cap,
      mu.max.cap = settings$mu.max.cap,
      poisson.fallback.eps = settings$poisson.fallback.eps
    )
  } else {
    NULL
  }

  fixed <- function(environment) {
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

  r.family.link(
    family.name = name,
    link.name = "Log",
    family.initial.parameters = settings$dispersion.initial,
    family.lower.bounds = settings$dispersion.lower.bound,
    family.upper.bounds = settings$dispersion.upper.bound,
    environment = workspace,
    inverse = function(eta, link.parameters, environment) exp(eta),
    negloglik = function(y, mu, family.parameters, environment) {
      current <- fixed(environment)
      .r.nb2.negloglik(
        y, mu, family.parameters[[1L]], current$mu.min.cap,
        current$mu.max.cap, current$poisson.fallback.eps
      )
    },
    grad = function(y, eta, mu, family.parameters, link.parameters, environment) {
      current <- fixed(environment)
      dispersion <- family.parameters[[1L]]
      family.gradient <- .r.nb2.grad(
        y, mu, dispersion, current$poisson.fallback.eps
      )
      list(
        d_negloglik_d_mu = family.gradient$d_negloglik_d_mu,
        d_mu_d_eta = mu,
        d_mu_d_link_parameters = matrix(numeric(), length(y), 0L),
        d_negloglik_d_eta = if (dispersion <= current$poisson.fallback.eps) {
          mu - y
        } else {
          (mu - y) / (1 + dispersion * mu)
        },
        d_negloglik_d_family_parameters =
          family.gradient$d_negloglik_d_family_parameters,
        d_negloglik_d_link_parameters = numeric()
      )
    }
  )
}

#' @rdname family.link.plugins
#' @export
plugin.family.link.nb2.log.r.environment <- function(
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
  .make.nb2.log.r(settings, "NB2.r.environment", TRUE)
}

#' @rdname family.link.plugins
#' @export
plugin.family.link.nb2.log.r.closure <- function(
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
  .make.nb2.log.r(settings, "NB2.r.closure", FALSE)
}

# Compatibility aliases.
#' @rdname family.link.plugins
#' @export
example.nb2.log.family.link <- plugin.family.link.nb2.log.builtin
#' @rdname family.link.plugins
#' @export
example.r.nb2.log.family.link <- plugin.family.link.nb2.log.r.environment
#' @rdname family.link.plugins
#' @export
example.r.nb2.log.family.link.environment <- plugin.family.link.nb2.log.r.environment
#' @rdname family.link.plugins
#' @export
example.r.nb2.log.family.link.closure <- plugin.family.link.nb2.log.r.closure
