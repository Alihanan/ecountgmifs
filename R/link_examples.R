.plugin.link.log.live.code <- paste(
  c(
    "#include <RcppArmadillo.h>",
    "// [[Rcpp::depends(RcppArmadillo)]]",
    "// [[Rcpp::depends(nloptr)]]",
    "// [[Rcpp::depends(ecountgmifs)]]",
    "// [[Rcpp::plugins(cpp17)]]",
    "#include <ecountgmifs/api.h>",
    "#include <stdexcept>",
    "",
    "class LiveLogLink final : public IEcountgmifsLinkFunc",
    "{",
    "public:",
    "  std::string name() const override { return \"Log.live\"; }",
    "  arma::uword parameter_count() const noexcept override { return 0; }",
    "",
    "  void inverse(",
    "      const arma::vec& eta,",
    "      const arma::vec& link_parameters,",
    "      arma::vec& mu",
    "  ) const override",
    "  {",
    "    if (!link_parameters.empty())",
    "      throw std::invalid_argument(\"LiveLogLink expects no parameters.\");",
    "    mu = arma::exp(eta);",
    "  }",
    "",
    "  void grad(",
    "      const arma::vec& eta,",
    "      const arma::vec& link_parameters,",
    "      arma::vec& d_mu_d_eta,",
    "      arma::mat& d_mu_d_link_parameters",
    "  ) const override",
    "  {",
    "    if (!link_parameters.empty())",
    "      throw std::invalid_argument(\"LiveLogLink expects no parameters.\");",
    "    d_mu_d_eta = arma::exp(eta);",
    "    d_mu_d_link_parameters.set_size(eta.n_elem, 0);",
    "  }",
    "};",
    "",
    "// [[Rcpp::export]]",
    "SEXP create_live_log_link()",
    "{",
    "  return Rcpp::XPtr<IEcountgmifsLinkFunc>(new LiveLogLink(), true);",
    "}"
  ),
  collapse = "\n"
)

.plugin.link.softplus.live.code <- paste(
  c(
    "#include <RcppArmadillo.h>",
    "// [[Rcpp::depends(RcppArmadillo)]]",
    "// [[Rcpp::depends(nloptr)]]",
    "// [[Rcpp::depends(ecountgmifs)]]",
    "// [[Rcpp::plugins(cpp17)]]",
    "#include <ecountgmifs/api.h>",
    "#include <cmath>",
    "#include <stdexcept>",
    "",
    "class LiveSoftplusLink final : public IEcountgmifsLinkFunc",
    "{",
    "  static double log1pexp(double x)",
    "  {",
    "    return x > 0.0 ? x + std::log1p(std::exp(-x)) : std::log1p(std::exp(x));",
    "  }",
    "  static double logistic(double x)",
    "  {",
    "    return x >= 0.0 ? 1.0 / (1.0 + std::exp(-x)) : std::exp(x) / (1.0 + std::exp(x));",
    "  }",
    "  static void validate(const arma::vec& parameters)",
    "  {",
    "    if (parameters.n_elem != 1 || !std::isfinite(parameters[0]) || parameters[0] <= 0.0)",
    "      throw std::invalid_argument(\"Softplus requires one positive parameter.\");",
    "  }",
    "public:",
    "  std::string name() const override { return \"Softplus.live\"; }",
    "  arma::uword parameter_count() const noexcept override { return 1; }",
    "  arma::vec initial_parameters() const override { return arma::vec({1.0}); }",
    "  arma::vec parameter_lower_bounds() const override { return arma::vec({1e-8}); }",
    "",
    "  void inverse(const arma::vec& eta, const arma::vec& parameters, arma::vec& mu) const override",
    "  {",
    "    validate(parameters);",
    "    const double a = parameters[0];",
    "    mu.set_size(eta.n_elem);",
    "    for (arma::uword i = 0; i < eta.n_elem; ++i) mu[i] = log1pexp(a * eta[i]) / a;",
    "  }",
    "",
    "  void grad(const arma::vec& eta, const arma::vec& parameters, arma::vec& d_mu_d_eta, arma::mat& d_mu_d_parameters) const override",
    "  {",
    "    validate(parameters);",
    "    const double a = parameters[0];",
    "    d_mu_d_eta.set_size(eta.n_elem);",
    "    d_mu_d_parameters.set_size(eta.n_elem, 1);",
    "    for (arma::uword i = 0; i < eta.n_elem; ++i) {",
    "      const double a_eta = a * eta[i];",
    "      const double s = logistic(a_eta);",
    "      const double mu = log1pexp(a_eta) / a;",
    "      d_mu_d_eta[i] = s;",
    "      d_mu_d_parameters(i, 0) = (eta[i] * s - mu) / a;",
    "    }",
    "  }",
    "};",
    "",
    "// [[Rcpp::export]]",
    "SEXP create_live_softplus_link()",
    "{",
    "  return Rcpp::XPtr<IEcountgmifsLinkFunc>(new LiveSoftplusLink(), true);",
    "}"
  ),
  collapse = "\n"
)

#' Link plugins in four implementation variants
#'
#' @name link.plugins
#' @aliases link-plugins
#'
#' @description
#' A link plugin represents the inverse link `mu = g^{-1}(eta)` and its
#' derivatives. [r.link()] callbacks have signatures
#' `inverse(eta, link.parameters, environment)` and
#' `grad(eta, link.parameters, environment)`.
#'
#' `inverse()` returns a numeric vector of length `length(eta)`. `grad()` returns
#' a list with `d_mu_d_eta`, a vector of that length, and
#' `d_mu_d_link_parameters`, a matrix with one row per observation and one
#' column per optimized link parameter. For zero link parameters, return an
#' `n x 0` matrix.
#'
#' @section Current examples:
#' The log link has no optimized parameters and computes `mu = exp(eta)`. The
#' softplus link computes `log(1 + exp(a * eta)) / a`; `a` is an optimized link
#' parameter with initial value `1`, lower bound `1e-8`, and upper bound `Inf`.
#' The capped-log R examples have no optimized parameters: `mu.min.cap` and
#' `mu.max.cap` are fixed auxiliary configuration retained in an environment or
#' closure.
#'
#' @section Four variants:
#' `plugin.link.log.builtin()` and `plugin.link.softplus.builtin()` use native classes from
#' `src/example.h`. `.live()` constructors compile complete C++ children of
#' `IEcountgmifsLinkFunc`. `.r.environment()` constructors retain explicit
#' inspectable data. `.r.closure()` constructors retain fixed values lexically.
#'
#' @section Performance and thread safety:
#' Built-in and live C++ execute without entering R and are the fast variants.
#' R-backed callbacks cross the R/C++ boundary once per vectorized evaluation,
#' never once per observation. They must run on the R main thread and are not
#' suitable for parallel calls into R. Explicit mutable environments and
#' closure mutation are not thread-safe; immutable closure or environment data
#' is preferable during fitting.
#'
#' @param cache Reuse the live-compiled object in the current R session.
#' @param mu.min.cap,mu.max.cap Positive fixed clamps for capped-log examples.
#' @return An external pointer to an `IEcountgmifsLinkFunc` child.
#'
#' @examples
#' links <- list(
#'   builtin = plugin.link.log.builtin(),
#'   live = plugin.link.log.live(),
#'   environment = plugin.link.capped.log.r.environment(1e-12, 1e12),
#'   closure = plugin.link.capped.log.r.closure(1e-12, 1e12)
#' )
#'
#' \dontrun{
#' set.seed(1)
#' X <- matrix(rnorm(80), 20, 4)
#' y <- rpois(20, 2)
#' fit <- ecountgmifs(
#'   X = X,
#'   y = y,
#'   family = plugin.family.poisson.builtin(),
#'   link = plugin.link.capped.log.r.closure(1e-12, 1e12),
#'   control = ecountgmifs.control(stagewise.iteration.max = 20L)
#' )
#' }
#'
#' @seealso [custom.plugins], [custom.plugins.live], [r.link()]
NULL

#' @rdname link.plugins
#' @export
plugin.link.log.builtin <- function() example_create_log_link()

#' @rdname link.plugins
#' @export
plugin.link.log.live <- function(cache = TRUE) {
  compile.link(
    code = .plugin.link.log.live.code,
    factory = "create_live_log_link",
    name = "Log.live",
    cache = cache
  )
}

#' @rdname link.plugins
#' @export
plugin.link.log.r.environment <- function() {
  r.link(
    name = "Log.r.environment",
    environment = r.environment(),
    inverse = function(eta, link.parameters, environment) exp(eta),
    grad = function(eta, link.parameters, environment) {
      list(
        d_mu_d_eta = exp(eta),
        d_mu_d_link_parameters = matrix(numeric(), length(eta), 0L)
      )
    }
  )
}

#' @rdname link.plugins
#' @export
plugin.link.log.r.closure <- function() {
  make.link <- function(link.name) {
    r.link(
      name = link.name,
      inverse = function(eta, link.parameters, environment) exp(eta),
      grad = function(eta, link.parameters, environment) {
        list(
          d_mu_d_eta = exp(eta),
          d_mu_d_link_parameters = matrix(numeric(), length(eta), 0L)
        )
      }
    )
  }
  make.link("Log.r.closure")
}

#' @rdname link.plugins
#' @export
plugin.link.softplus.builtin <- function() example_create_softplus_link()

#' @rdname link.plugins
#' @export
plugin.link.softplus.live <- function(cache = TRUE) {
  compile.link(
    code = .plugin.link.softplus.live.code,
    factory = "create_live_softplus_link",
    name = "Softplus.live",
    cache = cache
  )
}

.make.softplus.r <- function(name, environment = NULL) {
  r.link(
    name = name,
    initial.parameters = 1,
    lower.bounds = 1e-8,
    upper.bounds = Inf,
    environment = environment,
    inverse = function(eta, link.parameters, environment) {
      a <- link.parameters[[1L]]
      .r.log1pexp(a * eta) / a
    },
    grad = function(eta, link.parameters, environment) {
      a <- link.parameters[[1L]]
      a.eta <- a * eta
      s <- .r.logistic(a.eta)
      mu <- .r.log1pexp(a.eta) / a
      list(
        d_mu_d_eta = s,
        d_mu_d_link_parameters = matrix((eta * s - mu) / a, length(eta), 1L)
      )
    }
  )
}

#' @rdname link.plugins
#' @export
plugin.link.softplus.r.environment <- function() {
  .make.softplus.r("Softplus.r.environment", r.environment())
}

#' @rdname link.plugins
#' @export
plugin.link.softplus.r.closure <- function() {
  make.link <- function(link.name) .make.softplus.r(link.name)
  make.link("Softplus.r.closure")
}

#' @rdname link.plugins
#' @export
plugin.link.capped.log.r.environment <- function(
    mu.min.cap = 1e-12,
    mu.max.cap = 1e12
) {
  .r.validate.caps(mu.min.cap, mu.max.cap)
  workspace <- r.environment(mu.min.cap = mu.min.cap, mu.max.cap = mu.max.cap)

  r.link(
    name = "CappedLog.r.environment",
    environment = workspace,
    inverse = function(eta, link.parameters, environment) {
      .r.capped.exp(
        eta,
        environment$mu.min.cap,
        environment$mu.max.cap
      )$mu
    },
    grad = function(eta, link.parameters, environment) {
      capped <- .r.capped.exp(
        eta,
        environment$mu.min.cap,
        environment$mu.max.cap
      )
      list(
        d_mu_d_eta = capped$d.mu.d.eta,
        d_mu_d_link_parameters = matrix(numeric(), length(eta), 0L)
      )
    }
  )
}

#' @rdname link.plugins
#' @export
plugin.link.capped.log.r.closure <- function(
    mu.min.cap = 1e-12,
    mu.max.cap = 1e12
) {
  .r.validate.caps(mu.min.cap, mu.max.cap)

  r.link(
    name = "CappedLog.r.closure",
    inverse = function(eta, link.parameters, environment) {
      .r.capped.exp(eta, mu.min.cap, mu.max.cap)$mu
    },
    grad = function(eta, link.parameters, environment) {
      capped <- .r.capped.exp(eta, mu.min.cap, mu.max.cap)
      list(
        d_mu_d_eta = capped$d.mu.d.eta,
        d_mu_d_link_parameters = matrix(numeric(), length(eta), 0L)
      )
    }
  )
}

# Compatibility aliases.
#' @rdname link.plugins
#' @export
example.log.link <- plugin.link.log.builtin
#' @rdname link.plugins
#' @export
example.softplus.link <- plugin.link.softplus.builtin
#' @rdname link.plugins
#' @export
example.r.log.link <- plugin.link.log.r.environment
#' @rdname link.plugins
#' @export
example.r.softplus.link <- plugin.link.softplus.r.environment
#' @rdname link.plugins
#' @export
example.r.capped.log.link.environment <- plugin.link.capped.log.r.environment
#' @rdname link.plugins
#' @export
example.r.capped.log.link.closure <- plugin.link.capped.log.r.closure
