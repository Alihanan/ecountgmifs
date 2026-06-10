#' Compile AIC criterion using nonzero coefficients
#'
#' @description
#' Compiles an example C++ criterion for the Akaike information criterion using
#' the number of nonzero penalized coefficients as the complexity term.
#'
#' \deqn{
#'   \mathrm{AIC}_{\mathrm{NNZ}} = 2L + 2\,\mathrm{NNZ}
#' }
#'
#' Here \eqn{L} is the negative log-likelihood and \eqn{\mathrm{NNZ}} is the
#' number of nonzero penalized coefficients.
#'
#' @param cache logical. If \code{TRUE}, reuse an already compiled version of
#'   the criterion when the same C++ source and symbol were compiled earlier in
#'   the current R session. If \code{FALSE}, compile a new version.
#'
#' @return An object of class \code{"ecountgmifs.criterion"}.
#' @export
AIC_nnz <- function(cache = TRUE) {
  code <- '
#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::depends(ecountgmifs)]]

#include <ecountgmifs/api.h>

extern "C" double cpp_score_AIC_nnz(const EcountgmifsContext* ctx)
{
  double negloglik = ctx->state.negloglik;
  double nnz = arma::accu(ctx->state.beta != 0.0);

  return 2.0 * negloglik + 2.0 * nnz;
}

// [[Rcpp::export]]
int cpp_anchor_AIC_nnz()
{
  return 0;
}
'

  ecountgmifs.compile.criterion(
    code = code,
    symbol = "cpp_score_AIC_nnz",
    name = "AIC_nnz",
    use.cache = cache
  )
}


#' Compile BIC criterion using nonzero coefficients
#'
#' @description
#' Compiles an example C++ criterion for the Bayesian information criterion
#' using the number of nonzero penalized coefficients as the complexity term.
#'
#' \deqn{
#'   \mathrm{BIC}_{\mathrm{NNZ}} = 2L + \log(n)\,\mathrm{NNZ}
#' }
#'
#' Here \eqn{L} is the negative log-likelihood, \eqn{n} is the sample size, and
#' \eqn{\mathrm{NNZ}} is the number of nonzero penalized coefficients.
#'
#' @param cache logical. If \code{TRUE}, reuse an already compiled version of
#'   the criterion when the same C++ source and symbol were compiled earlier in
#'   the current R session. If \code{FALSE}, compile a new version.
#'
#' @return An object of class \code{"ecountgmifs.criterion"}.
#' @export
BIC_nnz <- function(cache = TRUE) {
  code <- '
#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::depends(ecountgmifs)]]

#include <ecountgmifs/api.h>
#include <cmath>

extern "C" double cpp_score_BIC_nnz(const EcountgmifsContext* ctx)
{
  double negloglik = ctx->state.negloglik;
  double n = static_cast<double>(ctx->input.X.n_rows);
  double nnz = arma::accu(ctx->state.beta != 0.0);

  return 2.0 * negloglik + std::log(n) * nnz;
}

// [[Rcpp::export]]
int cpp_anchor_BIC_nnz()
{
  return 0;
}
'

  ecountgmifs.compile.criterion(
    code = code,
    symbol = "cpp_score_BIC_nnz",
    name = "BIC_nnz",
    use.cache = cache
  )
}


#' Compile sample-size adjusted BIC criterion using nonzero coefficients
#'
#' @description
#' Compiles an example C++ criterion for the sample-size adjusted BIC using the
#' number of nonzero penalized coefficients as the complexity term.
#'
#' \deqn{
#'   \mathrm{SABIC}_{\mathrm{NNZ}}
#'   =
#'   2L + \log\left(\frac{n + 2}{24}\right)\mathrm{NNZ}
#' }
#'
#' Here \eqn{L} is the negative log-likelihood, \eqn{n} is the sample size, and
#' \eqn{\mathrm{NNZ}} is the number of nonzero penalized coefficients.
#'
#' @param cache logical. If \code{TRUE}, reuse an already compiled version of
#'   the criterion when the same C++ source and symbol were compiled earlier in
#'   the current R session. If \code{FALSE}, compile a new version.
#'
#' @return An object of class \code{"ecountgmifs.criterion"}.
#' @export
SABIC_nnz <- function(cache = TRUE) {
  code <- '
#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::depends(ecountgmifs)]]

#include <ecountgmifs/api.h>
#include <cmath>

extern "C" double cpp_score_SABIC_nnz(const EcountgmifsContext* ctx)
{
  double negloglik = ctx->state.negloglik;
  double n = static_cast<double>(ctx->input.X.n_rows);
  double nnz = arma::accu(ctx->state.beta != 0.0);

  return 2.0 * negloglik + std::log((n + 2.0) / 24.0) * nnz;
}

// [[Rcpp::export]]
int cpp_anchor_SABIC_nnz()
{
  return 0;
}
'

  ecountgmifs.compile.criterion(
    code = code,
    symbol = "cpp_score_SABIC_nnz",
    name = "SABIC_nnz",
    use.cache = cache
  )
}


#' Compile AIC criterion using heuristic effective degrees of freedom
#'
#' @description
#' Compiles an example C++ criterion for the Akaike information criterion using
#' a heuristic effective degrees of freedom, abbreviated HEDF.
#'
#' \deqn{
#'   \mathrm{AIC}_{\mathrm{HEDF}} = 2L + 2\,\mathrm{HEDF}
#' }
#'
#' In this example,
#'
#' \deqn{
#'   \mathrm{HEDF} = \frac{\mathrm{NNZ}\,n}{p}
#' }
#'
#' where \eqn{L} is the negative log-likelihood, \eqn{\mathrm{NNZ}} is the
#' number of nonzero penalized coefficients, \eqn{n} is the sample size, and
#' \eqn{p} is the total number of penalized predictors.
#'
#' @param cache logical. If \code{TRUE}, reuse an already compiled version of
#'   the criterion when the same C++ source and symbol were compiled earlier in
#'   the current R session. If \code{FALSE}, compile a new version.
#'
#' @return An object of class \code{"ecountgmifs.criterion"}.
#' @export
AIC_hedf <- function(cache = TRUE) {
  code <- '
#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::depends(ecountgmifs)]]

#include <ecountgmifs/api.h>

extern "C" double cpp_score_AIC_hedf(const EcountgmifsContext* ctx)
{
  double negloglik = ctx->state.negloglik;
  double n = static_cast<double>(ctx->input.X.n_rows);
  double p = static_cast<double>(ctx->input.X.n_cols);
  double nnz = arma::accu(ctx->state.beta != 0.0);
  double hedf = nnz * n / p;

  return 2.0 * negloglik + 2.0 * hedf;
}

// [[Rcpp::export]]
int cpp_anchor_AIC_hedf()
{
  return 0;
}
'

  ecountgmifs.compile.criterion(
    code = code,
    symbol = "cpp_score_AIC_hedf",
    name = "AIC_hedf",
    use.cache = cache
  )
}


#' Compile BIC criterion using heuristic effective degrees of freedom
#'
#' @description
#' Compiles an example C++ criterion for the Bayesian information criterion
#' using a heuristic effective degrees of freedom, abbreviated HEDF.
#'
#' \deqn{
#'   \mathrm{BIC}_{\mathrm{HEDF}} = 2L + \log(n)\,\mathrm{HEDF}
#' }
#'
#' In this example,
#'
#' \deqn{
#'   \mathrm{HEDF} = \frac{\mathrm{NNZ}\,n}{p}
#' }
#'
#' where \eqn{L} is the negative log-likelihood, \eqn{n} is the sample size,
#' \eqn{p} is the total number of penalized predictors, and \eqn{\mathrm{NNZ}}
#' is the number of nonzero penalized coefficients.
#'
#' @param cache logical. If \code{TRUE}, reuse an already compiled version of
#'   the criterion when the same C++ source and symbol were compiled earlier in
#'   the current R session. If \code{FALSE}, compile a new version.
#'
#' @return An object of class \code{"ecountgmifs.criterion"}.
#' @export
BIC_hedf <- function(cache = TRUE) {
  code <- '
#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::depends(ecountgmifs)]]

#include <ecountgmifs/api.h>
#include <cmath>

extern "C" double cpp_score_BIC_hedf(const EcountgmifsContext* ctx)
{
  double negloglik = ctx->state.negloglik;
  double n = static_cast<double>(ctx->input.X.n_rows);
  double p = static_cast<double>(ctx->input.X.n_cols);
  double nnz = arma::accu(ctx->state.beta != 0.0);
  double hedf = nnz * n / p;

  return 2.0 * negloglik + std::log(n) * hedf;
}

// [[Rcpp::export]]
int cpp_anchor_BIC_hedf()
{
  return 0;
}
'

  ecountgmifs.compile.criterion(
    code = code,
    symbol = "cpp_score_BIC_hedf",
    name = "BIC_hedf",
    use.cache = cache
  )
}


#' Compile sample-size adjusted BIC criterion using HEDF
#'
#' @description
#' Compiles an example C++ criterion for the sample-size adjusted BIC using a
#' heuristic effective degrees of freedom, abbreviated HEDF.
#'
#' \deqn{
#'   \mathrm{SABIC}_{\mathrm{HEDF}}
#'   =
#'   2L + \log\left(\frac{n + 2}{24}\right)\mathrm{HEDF}
#' }
#'
#' In this example,
#'
#' \deqn{
#'   \mathrm{HEDF} = \frac{\mathrm{NNZ}\,n}{p}
#' }
#'
#' where \eqn{L} is the negative log-likelihood, \eqn{n} is the sample size,
#' \eqn{p} is the total number of penalized predictors, and \eqn{\mathrm{NNZ}}
#' is the number of nonzero penalized coefficients.
#'
#' @param cache logical. If \code{TRUE}, reuse an already compiled version of
#'   the criterion when the same C++ source and symbol were compiled earlier in
#'   the current R session. If \code{FALSE}, compile a new version.
#'
#' @return An object of class \code{"ecountgmifs.criterion"}.
#' @export
SABIC_hedf <- function(cache = TRUE) {
  code <- '
#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::depends(ecountgmifs)]]

#include <ecountgmifs/api.h>
#include <cmath>

extern "C" double cpp_score_SABIC_hedf(const EcountgmifsContext* ctx)
{
  double negloglik = ctx->state.negloglik;
  double n = static_cast<double>(ctx->input.X.n_rows);
  double p = static_cast<double>(ctx->input.X.n_cols);
  double nnz = arma::accu(ctx->state.beta != 0.0);
  double hedf = nnz * n / p;

  return 2.0 * negloglik + std::log((n + 2.0) / 24.0) * hedf;
}

// [[Rcpp::export]]
int cpp_anchor_SABIC_hedf()
{
  return 0;
}
'

  ecountgmifs.compile.criterion(
    code = code,
    symbol = "cpp_score_SABIC_hedf",
    name = "SABIC_hedf",
    use.cache = cache
  )
}
