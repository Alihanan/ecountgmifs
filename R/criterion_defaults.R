# Internal information-criterion helpers ----------------------------------

.information.penalty.code <- c(
  AIC = 0L,
  BIC = 1L,
  SABIC = 2L,
  EBIC = 3L
)

.information.df.code <- c(
  full = 0L,
  nnz = 1L,
  hedf = 2L
)

.information.parameter.count <- function(state, input, definition) {
  nnz <- sum(state$beta != 0)

  switch(
    definition,
    full = nnz +
      length(state$theta) +
      length(state$family_parameters) +
      length(state$link_parameters),
    nnz = nnz,
    hedf = {
      if (input$p == 0L) {
        stop("HEDF requires at least one penalized predictor.", call. = FALSE)
      }
      nnz * input$n / input$p
    },
    stop("Unknown degrees-of-freedom definition.", call. = FALSE)
  )
}

.information.value <- function(
    input,
    state,
    penalty,
    definition,
    gamma = 0.5
) {
  if (input$n == 0L) {
    stop("Information criteria require at least one observation.", call. = FALSE)
  }

  df <- .information.parameter.count(state, input, definition)
  value <- switch(
    penalty,
    AIC = 2 * state$negloglik + 2 * df,
    BIC = 2 * state$negloglik + log(input$n) * df,
    SABIC = 2 * state$negloglik + log((input$n + 2) / 24) * df,
    EBIC = {
      nnz <- sum(state$beta != 0)
      2 * state$negloglik +
        log(input$n) * df +
        2 * gamma * lchoose(input$p, nnz)
    },
    stop("Unknown information-criterion penalty.", call. = FALSE)
  )

  as.numeric(value)
}

.validate.information.arguments <- function(penalty, definition, gamma) {
  penalty <- match.arg(penalty, names(.information.penalty.code))
  definition <- match.arg(definition, names(.information.df.code))

  if (length(gamma) != 1L || !is.finite(gamma) || gamma < 0) {
    stop("`gamma` must be one finite non-negative value.", call. = FALSE)
  }

  list(
    penalty = penalty,
    definition = definition,
    gamma = as.numeric(gamma)
  )
}

.information.name <- function(penalty, definition, variant) {
  if (identical(definition, "full")) {
    paste(penalty, variant, sep = ".")
  } else {
    paste(penalty, definition, variant, sep = ".")
  }
}

.information.builtin <- function(
    penalty,
    definition = "full",
    gamma = 0.5
) {
  settings <- .validate.information.arguments(penalty, definition, gamma)

  example_create_information_criterion(
    name = .information.name(
      settings$penalty,
      settings$definition,
      "builtin"
    ),
    penalty_type = unname(.information.penalty.code[[settings$penalty]]),
    df_type = unname(.information.df.code[[settings$definition]]),
    gamma = settings$gamma
  )
}

.information.live.code <- paste(
  c(
    "#include <RcppArmadillo.h>",
    "// [[Rcpp::depends(RcppArmadillo)]]",
    "// [[Rcpp::depends(nloptr)]]",
    "// [[Rcpp::depends(ecountgmifs)]]",
    "// [[Rcpp::plugins(cpp17)]]",
    "",
    "#include <ecountgmifs/api.h>",
    "#include <cmath>",
    "#include <stdexcept>",
    "#include <string>",
    "#include <utility>",
    "",
    "class LiveInformationCriterion final : public IEcountgmifsCriterion",
    "{",
    "private:",
    "  std::string name_;",
    "  int penalty_type_;",
    "  int df_type_;",
    "  double gamma_;",
    "",
    "  static double nnz(const EcountgmifsState& state)",
    "  {",
    "    return static_cast<double>(",
    "      arma::accu(state.param.param.beta != 0.0)",
    "    );",
    "  }",
    "",
    "  double degrees_of_freedom(",
    "      const EcountgmifsInput& input,",
    "      const EcountgmifsState& state",
    "  ) const",
    "  {",
    "    const double beta_df = nnz(state);",
    "",
    "    if (df_type_ == 0) {",
    "      return beta_df +",
    "        static_cast<double>(state.param.param.theta.n_elem) +",
    "        static_cast<double>(state.param.param.family_parameters.n_elem) +",
    "        static_cast<double>(state.param.param.link_parameters.n_elem);",
    "    }",
    "",
    "    if (df_type_ == 1) {",
    "      return beta_df;",
    "    }",
    "",
    "    if (df_type_ == 2) {",
    "      if (input.X.n_cols == 0) {",
    "        throw std::runtime_error(\"HEDF requires at least one predictor.\");",
    "      }",
    "",
    "      return beta_df *",
    "        static_cast<double>(input.X.n_rows) /",
    "        static_cast<double>(input.X.n_cols);",
    "    }",
    "",
    "    throw std::runtime_error(\"Unknown degrees-of-freedom type.\");",
    "  }",
    "",
    "public:",
    "  LiveInformationCriterion(",
    "      std::string name,",
    "      int penalty_type,",
    "      int df_type,",
    "      double gamma",
    "  ) :",
    "    name_(std::move(name)),",
    "    penalty_type_(penalty_type),",
    "    df_type_(df_type),",
    "    gamma_(gamma)",
    "  {",
    "    if (penalty_type_ < 0 || penalty_type_ > 3) {",
    "      throw std::invalid_argument(\"Invalid penalty type.\");",
    "    }",
    "    if (df_type_ < 0 || df_type_ > 2) {",
    "      throw std::invalid_argument(\"Invalid degrees-of-freedom type.\");",
    "    }",
    "    if (!std::isfinite(gamma_) || gamma_ < 0.0) {",
    "      throw std::invalid_argument(\"gamma must be finite and non-negative.\");",
    "    }",
    "  }",
    "",
    "  std::string name() const override",
    "  {",
    "    return name_;",
    "  }",
    "",
    "  void prepare(const EcountgmifsInput& input, const EcountgmifsControl& control) const override",
    "  {",
    "    (void) input;",
    "    (void) control;",
    "  }",
    "",
    "  double evaluate(",
    "      const EcountgmifsInput& input,",
    "      const EcountgmifsControl&,",
    "      const EcountgmifsState& state",
    "  ) const override",
    "  {",
    "    const double n = static_cast<double>(input.y.n_elem);",
    "    if (n <= 0.0) {",
    "      throw std::runtime_error(\"Information criterion: zero observations.\");",
    "    }",
    "",
    "    const double df = degrees_of_freedom(input, state);",
    "",
    "    if (penalty_type_ == 0) {",
    "      return 2.0 * state.negloglik + 2.0 * df;",
    "    }",
    "    if (penalty_type_ == 1) {",
    "      return 2.0 * state.negloglik + std::log(n) * df;",
    "    }",
    "    if (penalty_type_ == 2) {",
    "      return 2.0 * state.negloglik +",
    "        std::log((n + 2.0) / 24.0) * df;",
    "    }",
    "",
    "    const double p = static_cast<double>(input.X.n_cols);",
    "    const double beta_df = nnz(state);",
    "    const double log_choose =",
    "      std::lgamma(p + 1.0) -",
    "      std::lgamma(beta_df + 1.0) -",
    "      std::lgamma(p - beta_df + 1.0);",
    "",
    "    return 2.0 * state.negloglik +",
    "      std::log(n) * df +",
    "      2.0 * gamma_ * log_choose;",
    "  }",
    "};",
    "",
    "// [[Rcpp::export]]",
    "SEXP create_live_information_criterion(",
    "    std::string name,",
    "    int penalty_type,",
    "    int df_type,",
    "    double gamma",
    ")",
    "{",
    "  return Rcpp::XPtr<IEcountgmifsCriterion>(",
    "    new LiveInformationCriterion(",
    "      std::move(name),",
    "      penalty_type,",
    "      df_type,",
    "      gamma",
    "    ),",
    "    true",
    "  );",
    "}"
  ),
  collapse = "\n"
)

.information.live <- function(
    penalty,
    definition = "full",
    gamma = 0.5,
    cache = TRUE,
    name = NULL
) {
  settings <- .validate.information.arguments(penalty, definition, gamma)
  if (is.null(name)) {
    name <- .information.name(settings$penalty, settings$definition, "live")
  } else {
    if (!is.character(name) || length(name) != 1L || is.na(name) || !nzchar(name)) {
      stop("`name` must be one non-empty character value.", call. = FALSE)
    }
  }

  compile.criterion(
    code = .information.live.code,
    factory = "create_live_information_criterion",
    name = name,
    arguments = list(
      name = name,
      penalty_type = unname(.information.penalty.code[[settings$penalty]]),
      df_type = unname(.information.df.code[[settings$definition]]),
      gamma = settings$gamma
    ),
    cache = cache
  )
}

.information.r.environment <- function(
    penalty,
    definition = "full",
    gamma = 0.5
) {
  settings <- .validate.information.arguments(penalty, definition, gamma)
  workspace <- r.environment(
    penalty = settings$penalty,
    definition = settings$definition,
    gamma = settings$gamma
  )

  r.criterion(
    name = .information.name(settings$penalty, settings$definition, "r.environment"),
    environment = workspace,
    evaluate = function(input, control, state, environment) {
      .information.value(
        input = input,
        state = state,
        penalty = environment$penalty,
        definition = environment$definition,
        gamma = environment$gamma
      )
    }
  )
}

.information.r.closure <- function(
    penalty,
    definition = "full",
    gamma = 0.5
) {
  settings <- .validate.information.arguments(penalty, definition, gamma)
  penalty <- settings$penalty
  definition <- settings$definition
  gamma <- settings$gamma

  r.criterion(
    name = .information.name(penalty, definition, "r.closure"),
    evaluate = function(input, control, state, environment) {
      .information.value(
        input = input,
        state = state,
        penalty = penalty,
        definition = definition,
        gamma = gamma
      )
    }
  )
}

#' Information criteria in four implementation variants
#'
#' @name information.criteria
#' @aliases information-criteria AIC_nnz BIC_nnz SABIC_nnz AIC_hedf BIC_hedf SABIC_hedf
#'
#' @description
#' Every criterion on this page is available as package-built C++,
#' live-compiled C++, an R callback with explicit environment data, and an R
#' callback using closure-captured data. Constructors use the common pattern
#' `plugin.criterion.<name>[.<df>].<variant>()`. The leading `plugin`
#' namespace keeps constructor names from being misregistered as S3 methods for
#' the base `AIC()` and `BIC()` generics.
#'
#' The implementation suffixes are:
#'
#' * `.builtin`: native C++ compiled with the package;
#' * `.live`: C++ compiled by [compile.criterion()] in the current session;
#' * `.r.environment`: an R callback retaining explicit environment data;
#' * `.r.closure`: an R callback retaining fixed values lexically.
#'
#' @section Degrees of freedom:
#' Let `NNZ` be the number of nonzero penalized coefficients. The implemented
#' definitions are:
#'
#' * full: `NNZ + length(theta) + length(family_parameters) +
#'   length(link_parameters)`;
#' * nnz: `NNZ`;
#' * hedf: `NNZ * n / p`.
#'
#' Full-count constructors are provided for AIC, BIC, SABIC, and EBIC. NNZ and
#' HEDF constructors are provided for AIC, BIC, and SABIC.
#'
#' @section Formulas:
#' With negative log-likelihood `L` and selected complexity `df`,
#' \\deqn{AIC = 2L + 2df,}
#' \\deqn{BIC = 2L + \\log(n)df,}
#' \\deqn{SABIC = 2L + \\log((n + 2)/24)df,}
#' and
#' \\deqn{EBIC = 2L + \\log(n)df + 2\\gamma\\log {p \\choose NNZ}.}
#'
#' @section Historical compatibility:
#' [AIC_nnz()], [BIC_nnz()], [SABIC_nnz()], [AIC_hedf()], [BIC_hedf()], and
#' [SABIC_hedf()] remain aliases to the corresponding `.live` constructors.
#' They therefore retain the historical live-compilation behavior rather than
#' using R callback adapters.
#'
#' @param cache Reuse an identical live-compiled criterion in the current R
#'   session.
#' @param gamma Finite non-negative EBIC hyperparameter.
#'
#' @return An external pointer to an `IEcountgmifsCriterion` child.
#'
#' @examples
#' criteria <- list(
#'   AIC = plugin.criterion.AIC.builtin(),
#'   BIC.nnz = plugin.criterion.BIC.nnz.live(),
#'   SABIC.hedf.environment = plugin.criterion.SABIC.hedf.r.environment(),
#'   EBIC.closure = plugin.criterion.EBIC.r.closure(gamma = 0.5)
#' )
#'
#' \dontrun{
#' set.seed(1)
#' X <- matrix(rnorm(80), 20, 4)
#' y <- rpois(20, lambda = 2)
#' fit <- ecountgmifs(
#'   X = X,
#'   y = y,
#'   family = plugin.family.poisson.builtin(),
#'   link = plugin.link.log.builtin(),
#'   criteria = criteria,
#'   control = ecountgmifs.control(stagewise.iteration.max = 20L)
#' )
#' }
#'
#' @seealso [custom.plugins], [custom.plugins.builtin],
#'   [custom.plugins.live], [custom.plugins.r.environment],
#'   [custom.plugins.r.closure]
NULL

# Full parameter-count criteria -------------------------------------------

#' @rdname information.criteria
#' @export
plugin.criterion.AIC.builtin <- function() .information.builtin("AIC", "full")

#' @rdname information.criteria
#' @export
plugin.criterion.AIC.live <- function(cache = TRUE) {
  .information.live("AIC", "full", cache = cache)
}

#' @rdname information.criteria
#' @export
plugin.criterion.AIC.r.environment <- function() {
  .information.r.environment("AIC", "full")
}

#' @rdname information.criteria
#' @export
plugin.criterion.AIC.r.closure <- function() .information.r.closure("AIC", "full")

#' @rdname information.criteria
#' @export
plugin.criterion.BIC.builtin <- function() .information.builtin("BIC", "full")

#' @rdname information.criteria
#' @export
plugin.criterion.BIC.live <- function(cache = TRUE) {
  .information.live("BIC", "full", cache = cache)
}

#' @rdname information.criteria
#' @export
plugin.criterion.BIC.r.environment <- function() {
  .information.r.environment("BIC", "full")
}

#' @rdname information.criteria
#' @export
plugin.criterion.BIC.r.closure <- function() .information.r.closure("BIC", "full")

#' @rdname information.criteria
#' @export
plugin.criterion.SABIC.builtin <- function() .information.builtin("SABIC", "full")

#' @rdname information.criteria
#' @export
plugin.criterion.SABIC.live <- function(cache = TRUE) {
  .information.live("SABIC", "full", cache = cache)
}

#' @rdname information.criteria
#' @export
plugin.criterion.SABIC.r.environment <- function() {
  .information.r.environment("SABIC", "full")
}

#' @rdname information.criteria
#' @export
plugin.criterion.SABIC.r.closure <- function() {
  .information.r.closure("SABIC", "full")
}

#' @rdname information.criteria
#' @export
plugin.criterion.EBIC.builtin <- function(gamma = 0.5) {
  .information.builtin("EBIC", "full", gamma)
}

#' @rdname information.criteria
#' @export
plugin.criterion.EBIC.live <- function(gamma = 0.5, cache = TRUE) {
  .information.live("EBIC", "full", gamma, cache)
}

#' @rdname information.criteria
#' @export
plugin.criterion.EBIC.r.environment <- function(gamma = 0.5) {
  .information.r.environment("EBIC", "full", gamma)
}

#' @rdname information.criteria
#' @export
plugin.criterion.EBIC.r.closure <- function(gamma = 0.5) {
  .information.r.closure("EBIC", "full", gamma)
}

# NNZ criteria ------------------------------------------------------------

#' @rdname information.criteria
#' @export
plugin.criterion.AIC.nnz.builtin <- function() .information.builtin("AIC", "nnz")

#' @rdname information.criteria
#' @export
plugin.criterion.AIC.nnz.live <- function(cache = TRUE) {
  .information.live("AIC", "nnz", cache = cache)
}

#' @rdname information.criteria
#' @export
plugin.criterion.AIC.nnz.r.environment <- function() {
  .information.r.environment("AIC", "nnz")
}

#' @rdname information.criteria
#' @export
plugin.criterion.AIC.nnz.r.closure <- function() {
  .information.r.closure("AIC", "nnz")
}

#' @rdname information.criteria
#' @export
plugin.criterion.BIC.nnz.builtin <- function() .information.builtin("BIC", "nnz")

#' @rdname information.criteria
#' @export
plugin.criterion.BIC.nnz.live <- function(cache = TRUE) {
  .information.live("BIC", "nnz", cache = cache)
}

#' @rdname information.criteria
#' @export
plugin.criterion.BIC.nnz.r.environment <- function() {
  .information.r.environment("BIC", "nnz")
}

#' @rdname information.criteria
#' @export
plugin.criterion.BIC.nnz.r.closure <- function() {
  .information.r.closure("BIC", "nnz")
}

#' @rdname information.criteria
#' @export
plugin.criterion.SABIC.nnz.builtin <- function() .information.builtin("SABIC", "nnz")

#' @rdname information.criteria
#' @export
plugin.criterion.SABIC.nnz.live <- function(cache = TRUE) {
  .information.live("SABIC", "nnz", cache = cache)
}

#' @rdname information.criteria
#' @export
plugin.criterion.SABIC.nnz.r.environment <- function() {
  .information.r.environment("SABIC", "nnz")
}

#' @rdname information.criteria
#' @export
plugin.criterion.SABIC.nnz.r.closure <- function() {
  .information.r.closure("SABIC", "nnz")
}

# HEDF criteria -----------------------------------------------------------

#' @rdname information.criteria
#' @export
plugin.criterion.AIC.hedf.builtin <- function() .information.builtin("AIC", "hedf")

#' @rdname information.criteria
#' @export
plugin.criterion.AIC.hedf.live <- function(cache = TRUE) {
  .information.live("AIC", "hedf", cache = cache)
}

#' @rdname information.criteria
#' @export
plugin.criterion.AIC.hedf.r.environment <- function() {
  .information.r.environment("AIC", "hedf")
}

#' @rdname information.criteria
#' @export
plugin.criterion.AIC.hedf.r.closure <- function() {
  .information.r.closure("AIC", "hedf")
}

#' @rdname information.criteria
#' @export
plugin.criterion.BIC.hedf.builtin <- function() .information.builtin("BIC", "hedf")

#' @rdname information.criteria
#' @export
plugin.criterion.BIC.hedf.live <- function(cache = TRUE) {
  .information.live("BIC", "hedf", cache = cache)
}

#' @rdname information.criteria
#' @export
plugin.criterion.BIC.hedf.r.environment <- function() {
  .information.r.environment("BIC", "hedf")
}

#' @rdname information.criteria
#' @export
plugin.criterion.BIC.hedf.r.closure <- function() {
  .information.r.closure("BIC", "hedf")
}

#' @rdname information.criteria
#' @export
plugin.criterion.SABIC.hedf.builtin <- function() .information.builtin("SABIC", "hedf")

#' @rdname information.criteria
#' @export
plugin.criterion.SABIC.hedf.live <- function(cache = TRUE) {
  .information.live("SABIC", "hedf", cache = cache)
}

#' @rdname information.criteria
#' @export
plugin.criterion.SABIC.hedf.r.environment <- function() {
  .information.r.environment("SABIC", "hedf")
}

#' @rdname information.criteria
#' @export
plugin.criterion.SABIC.hedf.r.closure <- function() {
  .information.r.closure("SABIC", "hedf")
}

# Historical compatibility constructors. These remain live-compiled C++
# criteria and preserve their historical criterion names.

#' @rdname information.criteria
#' @export
AIC_nnz <- function(cache = TRUE) {
  .information.live("AIC", "nnz", cache = cache, name = "AIC_nnz")
}

#' @rdname information.criteria
#' @export
BIC_nnz <- function(cache = TRUE) {
  .information.live("BIC", "nnz", cache = cache, name = "BIC_nnz")
}

#' @rdname information.criteria
#' @export
SABIC_nnz <- function(cache = TRUE) {
  .information.live("SABIC", "nnz", cache = cache, name = "SABIC_nnz")
}

#' @rdname information.criteria
#' @export
AIC_hedf <- function(cache = TRUE) {
  .information.live("AIC", "hedf", cache = cache, name = "AIC_hedf")
}

#' @rdname information.criteria
#' @export
BIC_hedf <- function(cache = TRUE) {
  .information.live("BIC", "hedf", cache = cache, name = "BIC_hedf")
}

#' @rdname information.criteria
#' @export
SABIC_hedf <- function(cache = TRUE) {
  .information.live("SABIC", "hedf", cache = cache, name = "SABIC_hedf")
}

# Compatibility aliases for earlier example names.
#' @rdname information.criteria
#' @export
example.aic.criterion <- plugin.criterion.AIC.builtin

#' @rdname information.criteria
#' @export
example.bic.criterion <- plugin.criterion.BIC.builtin

#' @rdname information.criteria
#' @export
example.r.aic.criterion <- plugin.criterion.AIC.r.environment

#' @rdname information.criteria
#' @export
example.r.bic.criterion <- plugin.criterion.BIC.r.environment

#' @rdname information.criteria
#' @export
example.r.ebic.criterion.environment <- plugin.criterion.EBIC.r.environment

#' @rdname information.criteria
#' @export
example.r.ebic.criterion.closure <- plugin.criterion.EBIC.r.closure
