#pragma once

#include <RcppArmadillo.h>
#include <cmath>
#include <limits>

#include "context.h"

inline double poisson_negloglik(
    const arma::vec& mu,
    const arma::vec& y,
    const arma::vec& y_one_lgamma
) {
  arma::vec logp = y % arma::log(mu) - mu + y_one_lgamma;
  return -arma::accu(logp);
}

inline double nb_negloglik(
    const arma::vec& mu,
    const arma::vec& y,
    double dispersion,
    const arma::vec& y_one_lgamma
) {
  if (dispersion <= 0.0) {
    return std::numeric_limits<double>::infinity();
  }

  const double a = dispersion;
  const double one_over_a = 1.0 / a;

  arma::vec a_mu = a * mu;
  arma::vec log_a_mu = arma::log(a_mu);
  arma::vec log_one_plus_a_mu = arma::log1p(a_mu);

  arma::vec logp =
    y % (log_a_mu - log_one_plus_a_mu) -
    one_over_a * log_one_plus_a_mu +
    arma::lgamma(y + one_over_a) +
    y_one_lgamma -
    std::lgamma(one_over_a);

  return -arma::accu(logp);
}

inline double negloglik(
    const EcountgmifsContext& ctx,
    const arma::vec& mu,
    double dispersion
) {
  switch (ctx.input.family) {
  case EnumFamily::POISSON:
    return poisson_negloglik(
      mu,
      ctx.input.y,
      ctx.input.train_y_one_lgamma
    );

  case EnumFamily::NEGATIVE_BINOMIAL:
    return nb_negloglik(
      mu,
      ctx.input.y,
      dispersion,
      ctx.input.train_y_one_lgamma
    );
  }

  Rcpp::stop("unknown family");
}
