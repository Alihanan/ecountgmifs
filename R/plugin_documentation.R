#' Custom family, link, family-link, and criterion plugins
#'
#' @name custom.plugins
#' @aliases custom-plugins
#'
#' @description
#' `ecountgmifs` consumes ordinary owning external pointers to four public C++
#' interfaces. The fitting core does not branch on whether an implementation is
#' package-built C++, live-compiled C++, or an R callback adapter.
#'
#' The implementation pages are [custom.plugins.builtin],
#' [custom.plugins.live], [custom.plugins.r.environment], and
#' [custom.plugins.r.closure]. Component-specific pages are [link.plugins],
#' [family.plugins], [family.link.plugins], and [information.criteria].
#'
#' Public example constructors use the S3-safe dotted pattern
#' `plugin.<component>.<implementation>.<variant>()`, for example
#' `plugin.link.log.builtin()`, `plugin.family.nb2.live()`,
#' `plugin.family.link.nb2.log.r.environment()`, and
#' `plugin.criterion.AIC.nnz.r.closure()`. The shared `plugin` prefix avoids
#' accidental registration as methods for base generics such as `AIC()`,
#' `BIC()`, `log()`, and `family()`.
#'
#' @section Complete public link interface:
#' Copied verbatim from `inst/include/ecountgmifs/api.h`:
#'
#' ```cpp
#' struct IEcountgmifsLinkFunc
#' {
#'   virtual ~IEcountgmifsLinkFunc() = default;
#'
#'   virtual std::string name() const = 0;
#'
#'   virtual arma::uword parameter_count() const noexcept = 0;
#'
#'   virtual void inverse(
#'       const arma::vec& eta,
#'       const arma::vec& link_parameters,
#'       arma::vec& mu
#'   ) const = 0;
#'
#'   virtual void grad(
#'       const arma::vec& eta,
#'       const arma::vec& link_parameters,
#'       arma::vec& d_mu_d_eta,
#'       arma::mat& d_mu_d_link_parameters
#'   ) const = 0;
#'
#'   virtual arma::vec initial_parameters() const
#'   {
#'     return arma::vec(parameter_count(), arma::fill::zeros);
#'   }
#'
#'   virtual arma::vec parameter_lower_bounds() const
#'   {
#'     arma::vec lower(parameter_count());
#'     lower.fill(-arma::datum::inf);
#'     return lower;
#'   }
#'
#'   virtual arma::vec parameter_upper_bounds() const
#'   {
#'     arma::vec upper(parameter_count());
#'     upper.fill(arma::datum::inf);
#'     return upper;
#'   }
#' };
#' ```
#'
#' @section Complete public family interface:
#' Copied verbatim from `inst/include/ecountgmifs/api.h`:
#'
#' ```cpp
#' struct IEcountgmifsFamily
#' {
#'   virtual ~IEcountgmifsFamily() = default;
#'
#'   virtual std::string name() const = 0;
#'
#'   virtual arma::uword parameter_count() const noexcept = 0;
#'
#'   virtual void negloglik(
#'       const arma::vec& y,
#'       const arma::vec& mu,
#'       const arma::vec& family_parameters,
#'       double& negloglik
#'   ) const = 0;
#'
#'   virtual void grad(
#'       const arma::vec& y,
#'       const arma::vec& mu,
#'       const arma::vec& family_parameters,
#'       arma::vec& d_negloglik_d_mu,
#'       arma::vec& d_negloglik_d_family_parameters
#'   ) const = 0;
#'
#'   virtual arma::vec initial_parameters() const
#'   {
#'     return arma::vec(parameter_count(), arma::fill::zeros);
#'   }
#'
#'   virtual arma::vec parameter_lower_bounds() const
#'   {
#'     arma::vec lower(parameter_count());
#'     lower.fill(-arma::datum::inf);
#'     return lower;
#'   }
#'
#'   virtual arma::vec parameter_upper_bounds() const
#'   {
#'     arma::vec upper(parameter_count());
#'     upper.fill(arma::datum::inf);
#'     return upper;
#'   }
#' };
#' ```
#'
#' @section Complete public fused family-link interface:
#' Copied verbatim from `inst/include/ecountgmifs/api.h`:
#'
#' ```cpp
#' struct IEcountgmifsFamilyLink
#' {
#'   virtual ~IEcountgmifsFamilyLink() = default;
#'
#'   virtual std::string family_name() const = 0;
#'   virtual std::string link_name() const = 0;
#'
#'   virtual arma::uword family_parameter_count() const noexcept = 0;
#'   virtual arma::uword link_parameter_count() const noexcept = 0;
#'
#'   virtual arma::vec family_initial_parameters() const
#'   {
#'     return arma::vec(
#'       family_parameter_count(),
#'       arma::fill::zeros
#'     );
#'   }
#'
#'   virtual arma::vec family_parameter_lower_bounds() const
#'   {
#'     arma::vec lower(family_parameter_count());
#'     lower.fill(-arma::datum::inf);
#'     return lower;
#'   }
#'
#'   virtual arma::vec family_parameter_upper_bounds() const
#'   {
#'     arma::vec upper(family_parameter_count());
#'     upper.fill(arma::datum::inf);
#'     return upper;
#'   }
#'
#'   virtual arma::vec link_initial_parameters() const
#'   {
#'     return arma::vec(
#'       link_parameter_count(),
#'       arma::fill::zeros
#'     );
#'   }
#'
#'   virtual arma::vec link_parameter_lower_bounds() const
#'   {
#'     arma::vec lower(link_parameter_count());
#'     lower.fill(-arma::datum::inf);
#'     return lower;
#'   }
#'
#'   virtual arma::vec link_parameter_upper_bounds() const
#'   {
#'     arma::vec upper(link_parameter_count());
#'     upper.fill(arma::datum::inf);
#'     return upper;
#'   }
#'
#'   virtual void inverse(
#'       const arma::vec& eta,
#'       const arma::vec& link_parameters,
#'       arma::vec& mu
#'   ) const = 0;
#'
#'   virtual void negloglik(
#'       const arma::vec& y,
#'       const arma::vec& mu,
#'       const arma::vec& family_parameters,
#'       double& negloglik
#'   ) const = 0;
#'
#'   /*
#'    * Fill all derivatives needed by the fitting code.
#'    *
#'    * d_negloglik_d_eta is the derivative used for beta and theta. A custom
#'    * combined implementation may calculate it directly rather than as
#'    * d_negloglik_d_mu % d_mu_d_eta.
#'    */
#'   virtual void grad(
#'       const arma::vec& y,
#'       const arma::vec& eta,
#'       const arma::vec& mu,
#'       const arma::vec& family_parameters,
#'       const arma::vec& link_parameters,
#'       arma::vec& d_negloglik_d_mu,
#'       arma::vec& d_mu_d_eta,
#'       arma::mat& d_mu_d_link_parameters,
#'       arma::vec& d_negloglik_d_eta,
#'       arma::vec& d_negloglik_d_family_parameters,
#'       arma::vec& d_negloglik_d_link_parameters
#'   ) const = 0;
#' };
#' ```
#'
#' @section Complete public criterion interface:
#' Copied verbatim from `inst/include/ecountgmifs/api.h`:
#'
#' ```cpp
#' struct IEcountgmifsCriterion
#' {
#'   virtual ~IEcountgmifsCriterion() = default;
#'
#'   virtual std::string name() const = 0;
#'
#'   virtual double evaluate(
#'       const EcountgmifsInput& input,
#'       const EcountgmifsControl& control,
#'       const EcountgmifsState& state
#'   ) const = 0;
#' };
#' ```
#'
#' @section Parameter and auxiliary-data categories:
#' An optimized parameter is owned by the fitter and passed through
#' `family_parameters` or `link_parameters`; it therefore requires an initial
#' value and bounds. Fixed immutable configuration, such as caps or EBIC
#' `gamma`, belongs to the plugin object, explicit environment, or closure.
#' Large retained auxiliary data, such as test matrices, should be stored once
#' in the object/environment/closure rather than passed on every call. Mutable
#' shared state, such as counters or caches, is best placed in an explicit
#' environment because it is inspectable, but it is not thread-safe.
#'
#' @seealso [custom.plugins.builtin], [custom.plugins.live],
#'   [custom.plugins.r.environment], [custom.plugins.r.closure]
NULL

#' Package-built native C++ plugins
#'
#' @name custom.plugins.builtin
#'
#' @description
#' Built-in plugins are concrete children compiled with the package, currently
#' in `src/example.h`. Constructor arguments become native class members;
#' optimized family and link parameters are still initialized and bounded
#' through the virtual interface. These are the normal high-performance
#' implementations and do not enter the R interpreter during fitting.
#'
#' Built-in examples include log and softplus links, Poisson and NB2 families,
#' the fused NB2-log family-link, and information criteria. Constructors are
#' documented on [link.plugins], [family.plugins], [family.link.plugins], and
#' [information.criteria]. Native plugin objects are immutable after
#' construction and are safe for concurrent read-only evaluation provided the
#' implementation itself does not introduce mutable shared state.
#'
#' @examples
#' \dontrun{
#' set.seed(1)
#' X <- matrix(rnorm(80), 20, 4)
#' y <- rpois(20, lambda = 2)
#'
#' family <- plugin.family.poisson.builtin(
#'   mu.min.cap = 1e-12,
#'   mu.max.cap = 1e12
#' )
#' link <- plugin.link.log.builtin()
#' criteria <- list(
#'   AIC.nnz = plugin.criterion.AIC.nnz.builtin(),
#'   BIC.nnz = plugin.criterion.BIC.nnz.builtin()
#' )
#'
#' fit <- ecountgmifs(
#'   X = X,
#'   y = y,
#'   family = family,
#'   link = link,
#'   criteria = criteria,
#'   control = ecountgmifs.control(stagewise.iteration.max = 20L)
#' )
#'
#' fused.fit <- ecountgmifs(
#'   X = X,
#'   y = y,
#'   family.link = plugin.family.link.nb2.log.builtin(),
#'   criteria = list(AIC = plugin.criterion.AIC.builtin()),
#'   control = ecountgmifs.control(stagewise.iteration.max = 20L)
#' )
#' }
#'
#' @seealso [custom.plugins], [custom.plugins.live]
NULL

#' Live-compiled custom C++ plugins
#'
#' @name custom.plugins.live
#'
#' @description
#' A live plugin is a complete C++ child of one public interface plus an
#' exported factory returning the corresponding owning `Rcpp::XPtr`.
#' [compile.link()], [compile.family()], [compile.family.link()], and
#' [compile.criterion()] call `Rcpp::sourceCpp()`, invoke the exported R factory
#' directly, validate the returned interface metadata, and cache the result.
#' File-based variants read a complete `.cpp` file and use the same path.
#'
#' The source must include the public API and all dependencies needed by
#' `api.h`. Live C++ has native evaluation speed after the one-time compilation
#' cost. A live plugin must not retain references to temporary R objects unless
#' it explicitly preserves them. Pure native implementations are suitable for
#' concurrent read-only evaluation; implementations that call R are not.
#'
#' @examples
#' \dontrun{
#' live.log.source <- paste(c(
#'   "#include <RcppArmadillo.h>",
#'   "// [[Rcpp::depends(RcppArmadillo)]]",
#'   "// [[Rcpp::depends(nloptr)]]",
#'   "// [[Rcpp::depends(ecountgmifs)]]",
#'   "// [[Rcpp::plugins(cpp17)]]",
#'   "#include <ecountgmifs/api.h>",
#'   "#include <stdexcept>",
#'   "",
#'   "class UserLogLink final : public IEcountgmifsLinkFunc",
#'   "{",
#'   "public:",
#'   "  ~UserLogLink() override = default;",
#'   "  std::string name() const override { return \"UserLog\"; }",
#'   "  arma::uword parameter_count() const noexcept override { return 0; }",
#'   "  void inverse(",
#'   "      const arma::vec& eta,",
#'   "      const arma::vec& link_parameters,",
#'   "      arma::vec& mu",
#'   "  ) const override",
#'   "  {",
#'   "    if (!link_parameters.empty())",
#'   "      throw std::invalid_argument(\"UserLog expects no parameters.\");",
#'   "    mu = arma::exp(eta);",
#'   "  }",
#'   "  void grad(",
#'   "      const arma::vec& eta,",
#'   "      const arma::vec& link_parameters,",
#'   "      arma::vec& d_mu_d_eta,",
#'   "      arma::mat& d_mu_d_link_parameters",
#'   "  ) const override",
#'   "  {",
#'   "    if (!link_parameters.empty())",
#'   "      throw std::invalid_argument(\"UserLog expects no parameters.\");",
#'   "    d_mu_d_eta = arma::exp(eta);",
#'   "    d_mu_d_link_parameters.set_size(eta.n_elem, 0);",
#'   "  }",
#'   "};",
#'   "",
#'   "// [[Rcpp::export]]",
#'   "SEXP create_user_log_link()",
#'   "{",
#'   "  return Rcpp::XPtr<IEcountgmifsLinkFunc>(new UserLogLink(), true);",
#'   "}"
#' ), collapse = "\\n")
#'
#' user.link <- compile.link(
#'   code = live.log.source,
#'   factory = "create_user_log_link",
#'   name = "UserLog"
#' )
#'
#' set.seed(1)
#' X <- matrix(rnorm(80), 20, 4)
#' y <- rpois(20, 2)
#' fit <- ecountgmifs(
#'   X = X,
#'   y = y,
#'   family = plugin.family.poisson.builtin(),
#'   link = user.link,
#'   control = ecountgmifs.control(stagewise.iteration.max = 20L)
#' )
#' }
#'
#' @seealso [custom.plugins], [compile.link()], [compile.family()],
#'   [compile.family.link()], [compile.criterion()]
NULL

#' R callback plugins with an explicit environment
#'
#' @name custom.plugins.r.environment
#'
#' @description
#' [r.link()], [r.family()], [r.family.link()], and [r.criterion()] create
#' package-compiled C++ adapter children that retain `Rcpp::Function` callbacks
#' and one `Rcpp::Environment`. Every callback receives that exact environment
#' as its final argument. This is appropriate for inspectable fixed settings,
#' large retained data, caches, and deliberately mutable shared state.
#'
#' Callbacks are vectorized and are never invoked observation by observation.
#' Return dimensions and finiteness are checked in C++. R errors propagate out
#' of fitting. Because callbacks enter R, they must execute on the R main thread;
#' mutable environments are not thread-safe.
#'
#' @examples
#' \dontrun{
#' workspace <- r.environment(
#'   mu.min = 1e-12,
#'   mu.max = 1e12,
#'   evaluation.count = 0L
#' )
#'
#' capped.link <- r.link(
#'   name = "CappedLogEnvironment",
#'   environment = workspace,
#'   inverse = function(eta, link.parameters, environment) {
#'     eta.safe <- pmin(
#'       pmax(eta, log(environment$mu.min)),
#'       log(environment$mu.max)
#'     )
#'     exp(eta.safe)
#'   },
#'   grad = function(eta, link.parameters, environment) {
#'     eta.min <- log(environment$mu.min)
#'     eta.max <- log(environment$mu.max)
#'     mu <- exp(pmin(pmax(eta, eta.min), eta.max))
#'     list(
#'       d_mu_d_eta = mu * (eta > eta.min & eta < eta.max),
#'       d_mu_d_link_parameters = matrix(numeric(), length(eta), 0L)
#'     )
#'   }
#' )
#'
#' counted.aic <- r.criterion(
#'   name = "CountedAICEnvironment",
#'   environment = workspace,
#'   evaluate = function(input, control, state, environment) {
#'     environment$evaluation.count <- environment$evaluation.count + 1L
#'     df <- sum(state$beta != 0) + length(state$theta) +
#'       length(state$family_parameters) + length(state$link_parameters)
#'     2 * state$negloglik + 2 * df
#'   }
#' )
#'
#' set.seed(1)
#' X <- matrix(rnorm(80), 20, 4)
#' y <- rpois(20, 2)
#' fit <- ecountgmifs(
#'   X = X,
#'   y = y,
#'   family = plugin.family.poisson.builtin(),
#'   link = capped.link,
#'   criteria = list(AIC = counted.aic),
#'   control = ecountgmifs.control(stagewise.iteration.max = 20L)
#' )
#' r.plugin.environment(counted.aic)$evaluation.count
#'
#' Xtest <- matrix(rnorm(40), 10, 4)
#' ytest <- rpois(10, 2)
#' test.data <- r.environment(Xtest = Xtest, ytest = ytest)
#' test.criterion <- r.criterion(
#'   name = "PoissonTestNegloglikEnvironment",
#'   environment = test.data,
#'   evaluate = function(input, control, state, environment) {
#'     mu <- exp(drop(environment$Xtest %*% state$beta) + state$theta[[1L]])
#'     sum(mu - environment$ytest * log(mu) + lgamma(environment$ytest + 1))
#'   }
#' )
#' }
#'
#' @seealso [custom.plugins], [custom.plugins.r.closure], [r.environment()]
NULL

#' R callback plugins using closures
#'
#' @name custom.plugins.r.closure
#'
#' @description
#' A closure variant creates callbacks inside an outer factory. Fixed arguments
#' and retained objects remain available through lexical scope. This is concise
#' for immutable auxiliary configuration such as caps, NB2 fallback thresholds,
#' EBIC `gamma`, or retained test data. Use an explicit environment instead when
#' state must be inspected or intentionally shared and mutated.
#'
#' The compiled adapter still retains the `Rcpp::Function` objects, so their
#' closure environments remain alive. Calls are vectorized, checked in C++, and
#' restricted to the R main thread. Mutation with `<<-` is possible but is less
#' inspectable and not thread-safe.
#'
#' @examples
#' \dontrun{
#' make.capped.log <- function(mu.min, mu.max) {
#'   stopifnot(is.finite(mu.min), is.finite(mu.max), 0 < mu.min, mu.min < mu.max)
#'
#'   inverse <- function(eta, link.parameters, environment) {
#'     exp(pmin(pmax(eta, log(mu.min)), log(mu.max)))
#'   }
#'   grad <- function(eta, link.parameters, environment) {
#'     eta.min <- log(mu.min)
#'     eta.max <- log(mu.max)
#'     mu <- exp(pmin(pmax(eta, eta.min), eta.max))
#'     list(
#'       d_mu_d_eta = mu * (eta > eta.min & eta < eta.max),
#'       d_mu_d_link_parameters = matrix(numeric(), length(eta), 0L)
#'     )
#'   }
#'
#'   r.link(
#'     name = "CappedLogClosure",
#'     inverse = inverse,
#'     grad = grad
#'   )
#' }
#'
#' make.ebic <- function(gamma) {
#'   stopifnot(length(gamma) == 1L, is.finite(gamma), gamma >= 0)
#'   r.criterion(
#'     name = "EBICClosure",
#'     evaluate = function(input, control, state, environment) {
#'       nnz <- sum(state$beta != 0)
#'       df <- nnz + length(state$theta) + length(state$family_parameters) +
#'         length(state$link_parameters)
#'       2 * state$negloglik + log(input$n) * df +
#'         2 * gamma * lchoose(input$p, nnz)
#'     }
#'   )
#' }
#'
#' set.seed(1)
#' X <- matrix(rnorm(80), 20, 4)
#' y <- rpois(20, 2)
#' fit <- ecountgmifs(
#'   X = X,
#'   y = y,
#'   family = plugin.family.poisson.builtin(),
#'   link = make.capped.log(1e-12, 1e12),
#'   criteria = list(EBIC = make.ebic(0.5)),
#'   control = ecountgmifs.control(stagewise.iteration.max = 20L)
#' )
#' }
#'
#' @seealso [custom.plugins], [custom.plugins.r.environment]
NULL
