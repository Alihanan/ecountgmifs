#pragma once

#include <RcppArmadillo.h>
#include <algorithm>
#include <cmath>

#include "enums.h"
#include "numerical_internal_constants.h"

inline double clamp_scalar(double x, double lo, double hi)
{
  return std::max(lo, std::min(x, hi));
}

inline double softplus_scalar(double x)
{
  return std::max(x, 0.0) + std::log1p(std::exp(-std::abs(x)));
}

inline arma::vec softplus_vec(const arma::vec& x)
{
  return arma::clamp(x, 0.0, arma::datum::inf) +
    arma::log1p(arma::exp(-arma::abs(x)));
}

inline arma::vec inverse_link(
    const arma::vec& eta,
    const arma::vec& offset,
    EnumLinkFunc link_func
) {
  switch (link_func) {
  case LOG_LINK: {
    arma::vec eta_safe = arma::clamp(eta, ETA_MIN_CAP, ETA_MAX_CAP);
    arma::vec mu = arma::exp(eta_safe);
    return arma::clamp(mu, MU_MIN_CAP, MU_MAX_CAP);
  }

  case SOFTPLUS_LINK: {
    arma::vec mu = arma::exp(offset) % softplus_vec(eta);
    return arma::clamp(mu, MU_MIN_CAP, MU_MAX_CAP);
  }
  }

  Rcpp::stop("unknown link function");
}
