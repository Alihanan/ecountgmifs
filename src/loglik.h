#pragma once

#include <RcppArmadillo.h>
#include <cmath>
#include <limits>

#include "context.h"
#include "../inst/include/ecountgmifs/api.h"
#include "numerical_internal_constants.h"

struct NegloglikWorkspace
{
  arma::vec mu_safe;
  arma::vec a_mu;
  arma::vec log_a_mu;
  arma::vec log_one_plus_a_mu;
  arma::vec gamma_work;
  arma::vec logp;

  explicit NegloglikWorkspace(
      arma::uword n = 0
  ) :
    mu_safe(n),
    a_mu(n),
    log_a_mu(n),
    log_one_plus_a_mu(n),
    gamma_work(n),
    logp(n)
  {}

  void ensure_size(
      arma::uword n
  ) {
    if (mu_safe.n_elem != n) {
      mu_safe.set_size(n);
    }

    if (a_mu.n_elem != n) {
      a_mu.set_size(n);
    }

    if (log_a_mu.n_elem != n) {
      log_a_mu.set_size(n);
    }

    if (log_one_plus_a_mu.n_elem != n) {
      log_one_plus_a_mu.set_size(n);
    }

    if (gamma_work.n_elem != n) {
      gamma_work.set_size(n);
    }

    if (logp.n_elem != n) {
      logp.set_size(n);
    }
  }
};

inline double poisson_negloglik_inplace(
    const arma::vec& mu,
    const arma::vec& y,
    const arma::vec& y_one_lgamma,
    NegloglikWorkspace& workspace
) {
  workspace.ensure_size(mu.n_elem);

  workspace.mu_safe =
    mu;

  workspace.mu_safe.clamp(
    MU_MIN_CAP,
    MU_MAX_CAP
  );

  workspace.logp =
    arma::log(workspace.mu_safe);

  workspace.logp %=
    y;

  workspace.logp -=
    workspace.mu_safe;

  workspace.logp +=
    y_one_lgamma;

  return -arma::accu(workspace.logp);
}

inline double poisson_negloglik(
    const arma::vec& mu,
    const arma::vec& y,
    const arma::vec& y_one_lgamma
) {
  NegloglikWorkspace workspace(mu.n_elem);

  return poisson_negloglik_inplace(
    mu,
    y,
    y_one_lgamma,
    workspace
  );
}

inline double nb_negloglik_inplace(
    const arma::vec& mu,
    const arma::vec& y,
    double dispersion,
    const arma::vec& y_one_lgamma,
    double poisson_fallback_eps,
    NegloglikWorkspace& workspace
) {
  workspace.ensure_size(mu.n_elem);

  workspace.mu_safe =
    mu;

  workspace.mu_safe.clamp(
    MU_MIN_CAP,
    MU_MAX_CAP
  );

  if (dispersion <= poisson_fallback_eps) {
    return poisson_negloglik_inplace(
      workspace.mu_safe,
      y,
      y_one_lgamma,
      workspace
    );
  }

  if (dispersion <= 0.0) {
    return std::numeric_limits<double>::infinity();
  }

  const double a = dispersion;
  const double one_over_a = 1.0 / a;

  workspace.a_mu =
    workspace.mu_safe;

  workspace.a_mu *=
    a;

  workspace.log_a_mu =
    arma::log(workspace.a_mu);

  workspace.log_one_plus_a_mu =
    arma::log1p(workspace.a_mu);

  workspace.logp =
    workspace.log_a_mu;

  workspace.logp -=
    workspace.log_one_plus_a_mu;

  workspace.logp %=
    y;

  workspace.log_one_plus_a_mu *=
    one_over_a;

  workspace.logp -=
    workspace.log_one_plus_a_mu;

  workspace.gamma_work =
    arma::lgamma(y + one_over_a);

  workspace.gamma_work +=
    y_one_lgamma;

  workspace.gamma_work -=
    std::lgamma(one_over_a);

  workspace.logp +=
    workspace.gamma_work;

  return -arma::accu(workspace.logp);
}

inline double nb_negloglik(
    const arma::vec& mu,
    const arma::vec& y,
    double dispersion,
    const arma::vec& y_one_lgamma,
    double poisson_fallback_eps
) {
  NegloglikWorkspace workspace(mu.n_elem);

  return nb_negloglik_inplace(
    mu,
    y,
    dispersion,
    y_one_lgamma,
    poisson_fallback_eps,
    workspace
  );
}

struct FixedDispersionNegloglikData
{
  const arma::vec* y;
  const arma::vec* y_one_lgamma;
  EnumFamily family;
  double dispersion;
  double one_over_dispersion;
  double lgamma_one_over_dispersion;
  bool use_poisson;
  arma::vec gamma_constant;

  FixedDispersionNegloglikData(
      const arma::vec& y_,
      const arma::vec& y_one_lgamma_,
      EnumFamily family_,
      double dispersion_,
      double poisson_fallback_eps
  ) :
    y(&y_),
    y_one_lgamma(&y_one_lgamma_),
    family(family_),
    dispersion(dispersion_),
    one_over_dispersion(0.0),
    lgamma_one_over_dispersion(0.0),
    use_poisson(
      family_ == POISSON ||
      dispersion_ <= poisson_fallback_eps
    ),
    gamma_constant()
  {
    refresh(
      family_,
      dispersion_,
      poisson_fallback_eps
    );
  }

  void refresh(
      EnumFamily family_new,
      double dispersion_new,
      double poisson_fallback_eps
  ) {
    family =
      family_new;

    dispersion =
      dispersion_new;

    one_over_dispersion =
      0.0;

    lgamma_one_over_dispersion =
      0.0;

    use_poisson =
      family_new == POISSON ||
      dispersion_new <= poisson_fallback_eps;

    if (family == NEGATIVE_BINOMIAL && !use_poisson) {
      if (dispersion <= 0.0) {
        return;
      }

      one_over_dispersion =
        1.0 / dispersion;

      lgamma_one_over_dispersion =
        std::lgamma(one_over_dispersion);

      if (gamma_constant.n_elem != y->n_elem) {
        gamma_constant.set_size(y->n_elem);
      }

      for (arma::uword i = 0; i < y->n_elem; ++i) {
        gamma_constant[i] =
          std::lgamma((*y)[i] + one_over_dispersion) +
          (*y_one_lgamma)[i] -
          lgamma_one_over_dispersion;
      }
    }
  }
};

inline double fixed_dispersion_negloglik_inplace(
    const arma::vec& mu,
    const FixedDispersionNegloglikData& data,
    NegloglikWorkspace& workspace
) {
  if (data.use_poisson) {
    return poisson_negloglik_inplace(
      mu,
      *(data.y),
      *(data.y_one_lgamma),
      workspace
    );
  }

  if (data.dispersion <= 0.0) {
    return std::numeric_limits<double>::infinity();
  }

  const double a =
    data.dispersion;

  workspace.ensure_size(mu.n_elem);

  workspace.mu_safe =
    mu;

  workspace.mu_safe.clamp(
    MU_MIN_CAP,
    MU_MAX_CAP
  );

  workspace.a_mu =
    workspace.mu_safe;

  workspace.a_mu *=
    a;

  workspace.log_a_mu =
    arma::log(workspace.a_mu);

  workspace.log_one_plus_a_mu =
    arma::log1p(workspace.a_mu);

  workspace.logp =
    workspace.log_a_mu;

  workspace.logp -=
    workspace.log_one_plus_a_mu;

  workspace.logp %=
    *(data.y);

  workspace.log_one_plus_a_mu *=
    data.one_over_dispersion;

  workspace.logp -=
    workspace.log_one_plus_a_mu;

  workspace.logp +=
    data.gamma_constant;

  return -arma::accu(workspace.logp);
}

inline double fixed_dispersion_negloglik(
    const arma::vec& mu,
    const FixedDispersionNegloglikData& data
) {
  NegloglikWorkspace workspace(mu.n_elem);

  return fixed_dispersion_negloglik_inplace(
    mu,
    data,
    workspace
  );
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
      ctx.input.train_y_one_lgamma,
      ctx.control.nb_poisson_fallback_eps
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

inline void gradient_beta(
    arma::vec& grad_beta,
    const arma::mat& X,
    const arma::vec& y,
    const arma::vec& eta,
    const arma::vec& mu,
    const arma::vec& offset,
    EnumFamily family,
    EnumLinkFunc link_func,
    double dispersion
) {
  grad_beta = X.t() * gradient_eta(
      y,
      eta,
      mu,
      offset,
      family,
      link_func,
      dispersion
  );
}
