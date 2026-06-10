#pragma once

#include <RcppArmadillo.h>
#include <cmath>
#include <limits>

#include "context.h"
#include "../inst/include/ecountgmifs/api.h"

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


inline arma::vec sigmoid_vec(
    const arma::vec& x
) {
  arma::vec out(x.n_elem);

  for (arma::uword i = 0; i < x.n_elem; ++i) {
    if (x[i] >= 0.0) {
      const double z = std::exp(-x[i]);
      out[i] = 1.0 / (1.0 + z);
    } else {
      const double z = std::exp(x[i]);
      out[i] = z / (1.0 + z);
    }
  }

  return out;
}

inline arma::vec d_negloglik_d_mu(
    const arma::vec& y,
    const arma::vec& mu,
    EnumFamily family,
    double dispersion
) {
  switch (family) {
  case POISSON:
    return 1.0 - y / mu;

  case NEGATIVE_BINOMIAL: {
    if (dispersion <= 0.0 || !std::isfinite(dispersion)) {
    Rcpp::stop("d_negloglik_d_mu(): invalid dispersion");
  }

    return -y / mu +
      (1.0 + dispersion * y) / (1.0 + dispersion * mu);
  }
  }

  Rcpp::stop("unknown family");
}

inline arma::vec d_mu_d_eta(
    const arma::vec& eta,
    const arma::vec& mu,
    const arma::vec& offset,
    EnumLinkFunc link_func
) {
  switch (link_func) {
  case LOG_LINK:
    return mu;

  case SOFTPLUS_LINK:
    return arma::exp(offset) % sigmoid_vec(eta);
  }

  Rcpp::stop("unknown link function");
}

inline arma::vec gradient_eta(
    const arma::vec& y,
    const arma::vec& eta,
    const arma::vec& mu,
    const arma::vec& offset,
    EnumFamily family,
    EnumLinkFunc link_func,
    double dispersion
) {
  return d_negloglik_d_mu(
    y,
    mu,
    family,
    dispersion
  ) %
    d_mu_d_eta(
      eta,
      mu,
      offset,
      link_func
    );
}

inline arma::vec gradient_beta(
    const arma::mat& X,
    const arma::vec& y,
    const arma::vec& eta,
    const arma::vec& mu,
    const arma::vec& offset,
    EnumFamily family,
    EnumLinkFunc link_func,
    double dispersion
) {
  return X.t() * gradient_eta(
      y,
      eta,
      mu,
      offset,
      family,
      link_func,
      dispersion
  );
}
