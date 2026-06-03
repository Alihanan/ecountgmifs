#pragma once

#include <RcppArmadillo.h>
#include <nloptrAPI.h>

#include <cmath>
#include <limits>

#include "context.h"
#include "link_function.h"
#include "loglik.h"
#include "numerical_internal_constants.h"

inline arma::vec linear_predictor_nonpen(
    const EcountgmifsContext& ctx,
    const double* theta_ptr
) {
  const arma::uword q = ctx.input.w.n_cols;

  arma::vec theta_candidate(
      const_cast<double*>(theta_ptr),
      q,
      false,
      true
  );

  return ctx.input.offset +
    ctx.input.w * theta_candidate +
    ctx.input.X * ctx.state.beta;
}

inline arma::vec current_linear_predictor_nonpen(
    const EcountgmifsContext& ctx
) {
  return ctx.input.offset +
    ctx.input.w * ctx.state.theta +
    ctx.input.X * ctx.state.beta;
}

inline arma::vec current_mu_nonpen(
    const EcountgmifsContext& ctx
) {
  arma::vec eta = current_linear_predictor_nonpen(ctx);

  return inverse_link(
    eta,
    ctx.input.offset,
    ctx.input.link_func
  );
}

inline double nonpen_theta_objective(
    unsigned n_theta,
    const double* theta,
    double* grad,
    void* data
) {
  if (grad != nullptr) {
    for (unsigned j = 0; j < n_theta; ++j) {
      grad[j] = 0.0;
    }
  }

  EcountgmifsContext* ctx =
    reinterpret_cast<EcountgmifsContext*>(data);

  arma::vec eta = linear_predictor_nonpen(*ctx, theta);

  arma::vec mu = inverse_link(
    eta,
    ctx->input.offset,
    ctx->input.link_func
  );

  return negloglik(*ctx, mu, ctx->state.dispersion);
}

inline double dispersion_objective_current_mu(
    unsigned n_disp,
    const double* dispersion_ptr,
    double* grad,
    void* data
) {
  if (grad != nullptr) {
    for (unsigned j = 0; j < n_disp; ++j) {
      grad[j] = 0.0;
    }
  }

  EcountgmifsContext* ctx =
    reinterpret_cast<EcountgmifsContext*>(data);

  const double dispersion = dispersion_ptr[0];

  arma::vec mu = current_mu_nonpen(*ctx);

  return nb_negloglik(
    mu,
    ctx->input.y,
    dispersion,
    ctx->input.train_y_one_lgamma
  );
}

inline double dispersion_objective_saturated_mu(
    unsigned n_disp,
    const double* dispersion_ptr,
    double* grad,
    void* data
) {
  if (grad != nullptr) {
    for (unsigned j = 0; j < n_disp; ++j) {
      grad[j] = 0.0;
    }
  }

  EcountgmifsContext* ctx =
    reinterpret_cast<EcountgmifsContext*>(data);

  const double dispersion = dispersion_ptr[0];

  arma::vec mu = arma::clamp(
    ctx->input.y,
    MU_MIN_CAP,
    MU_MAX_CAP
  );

  return nb_negloglik(
    mu,
    ctx->input.y,
    dispersion,
    ctx->input.train_y_one_lgamma
  );
}

inline void optim_nonpen_theta_nlopt(EcountgmifsContext& ctx)
{
  const arma::uword q = ctx.state.theta.n_elem;

  if (q == 0) {
    Rcpp::stop("nonpenalized design matrix 'w' must have at least one column");
  }

  nlopt_opt opt = nlopt_create(NLOPT_LN_NELDERMEAD, q);

  if (opt == nullptr) {
    Rcpp::stop("failed to create NLopt optimizer for nonpenalized parameters");
  }

  std::vector<double> lower(q, -HUGE_VAL);

  nlopt_set_lower_bounds(opt, lower.data());
  nlopt_set_min_objective(opt, nonpen_theta_objective, &ctx);
  nlopt_set_xtol_rel(opt, ctx.control.nlopt_optim_reltol);
  nlopt_set_maxeval(opt, static_cast<int>((q + 1) * 100));

  double minf = 0.0;

  nlopt_result result =
    nlopt_optimize(opt, ctx.state.theta.memptr(), &minf);

  nlopt_destroy(opt);

  if (result < 0) {
    Rcpp::warning(
      "NLopt failed while optimizing nonpenalized parameters: %d",
      static_cast<int>(result)
    );
  }

  ctx.state.loglik = minf;
}

inline void optim_dispersion_current_mu_nlopt(EcountgmifsContext& ctx)
{
  if (ctx.input.family == POISSON) {
    ctx.state.dispersion = 0.0;

    arma::vec mu = current_mu_nonpen(ctx);
    ctx.state.loglik = poisson_negloglik(
      mu,
      ctx.input.y,
      ctx.input.train_y_one_lgamma
    );

    return;
  }

  if (ctx.control.fixed_dispersion) {
    ctx.state.dispersion = ctx.control.fixed_dispersion_value;

    arma::vec mu = current_mu_nonpen(ctx);
    ctx.state.loglik = nb_negloglik(
      mu,
      ctx.input.y,
      ctx.state.dispersion,
      ctx.input.train_y_one_lgamma
    );

    return;
  }

  if (ctx.state.dispersion <= 0.0) {
    ctx.state.dispersion = 1e-4;
  }

  nlopt_opt opt = nlopt_create(NLOPT_LN_NELDERMEAD, 1);

  if (opt == nullptr) {
    Rcpp::stop("failed to create NLopt optimizer for dispersion");
  }

  nlopt_set_lower_bounds1(opt, DISP_MIN_CAP);
  nlopt_set_min_objective(opt, dispersion_objective_current_mu, &ctx);
  nlopt_set_xtol_rel(opt, ctx.control.nlopt_optim_reltol);
  nlopt_set_maxeval(opt, 200);

  double dispersion = ctx.state.dispersion;
  double minf = 0.0;

  nlopt_result result =
    nlopt_optimize(opt, &dispersion, &minf);

  nlopt_destroy(opt);

  if (result < 0) {
    Rcpp::warning(
      "NLopt failed while optimizing dispersion: %d",
      static_cast<int>(result)
    );
  }

  if (dispersion < DISP_MIN_CAP) {
    dispersion = DISP_MIN_CAP;
  }

  ctx.state.dispersion = dispersion;
  ctx.state.loglik = minf;
}

inline void fit_unpenalized_model_nlopt(
    EcountgmifsContext& ctx,
    uint32_t max_outer = 1000
) {
  ctx.state.beta.zeros();

  if (ctx.input.family == NEGATIVE_BINOMIAL &&
      ctx.state.dispersion <= 0.0) {
    ctx.state.dispersion =
      ctx.control.fixed_dispersion ?
      ctx.control.fixed_dispersion_value :
    1e-4;
  }

  if (ctx.input.family == POISSON) {
    ctx.state.dispersion = 0.0;
  }

  for (uint32_t iter = 0; iter < max_outer; ++iter) {
    arma::vec theta_old = ctx.state.theta;
    double dispersion_old = ctx.state.dispersion;
    double loglik_old = ctx.state.loglik;

    optim_nonpen_theta_nlopt(ctx);
    optim_dispersion_current_mu_nlopt(ctx);

    const double theta_diff =
      arma::norm(ctx.state.theta - theta_old, 2);

    const double dispersion_diff =
      std::abs(ctx.state.dispersion - dispersion_old);

    const double loglik_diff =
      std::abs(ctx.state.loglik - loglik_old);

    if (theta_diff < ctx.control.tol &&
        dispersion_diff < ctx.control.tol &&
        loglik_diff < ctx.control.tol) {
      ctx.state.initialized = true;
      return;
    }
  }

  ctx.state.initialized = true;
}

inline double fit_saturated_dispersion_nlopt(
    EcountgmifsContext& ctx,
    double& saturated_dispersion
) {
  if (ctx.input.family == POISSON) {
    saturated_dispersion = 0.0;

    arma::vec mu = arma::clamp(
      ctx.input.y,
      MU_MIN_CAP,
      MU_MAX_CAP
    );

    return poisson_negloglik(
      mu,
      ctx.input.y,
      ctx.input.train_y_one_lgamma
    );
  }

  if (ctx.control.fixed_dispersion) {
    saturated_dispersion = ctx.control.fixed_dispersion_value;

    arma::vec mu = arma::clamp(
      ctx.input.y,
      MU_MIN_CAP,
      MU_MAX_CAP
    );

    return nb_negloglik(
      mu,
      ctx.input.y,
      saturated_dispersion,
      ctx.input.train_y_one_lgamma
    );
  }

  saturated_dispersion = ctx.state.dispersion;

  if (saturated_dispersion <= 0.0) {
    saturated_dispersion = 1e-4;
  }

  nlopt_opt opt = nlopt_create(NLOPT_LN_NELDERMEAD, 1);

  if (opt == nullptr) {
    Rcpp::stop("failed to create NLopt optimizer for saturated dispersion");
  }

  nlopt_set_lower_bounds1(opt, DISP_MIN_CAP);
  nlopt_set_min_objective(opt, dispersion_objective_saturated_mu, &ctx);
  nlopt_set_xtol_rel(opt, ctx.control.nlopt_optim_reltol);
  nlopt_set_maxeval(opt, 200);

  double minf = 0.0;

  nlopt_result result =
    nlopt_optimize(opt, &saturated_dispersion, &minf);

  nlopt_destroy(opt);

  if (result < 0) {
    Rcpp::warning(
      "NLopt failed while optimizing saturated dispersion: %d",
      static_cast<int>(result)
    );
  }

  if (saturated_dispersion < DISP_MIN_CAP) {
    saturated_dispersion = DISP_MIN_CAP;
  }

  return minf;
}
