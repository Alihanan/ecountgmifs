#pragma once

#include <RcppArmadillo.h>
#include <nloptrAPI.h>

#include <algorithm>
#include <cmath>
#include <vector>

#include "context.h"
#include "numerical_internal_constants.h"

/*
 * NonpenNlopters owns NLopt objects.
 *
 * ContextInternal owns model state and model-level procedures:
 *   - fit_intercept()
 *   - fit_saturated()
 *
 * NLopt callbacks are static inline member functions.
 * They evaluate trial objectives in local workspaces and do not mutate the
 * accepted context state. The optimizer result is committed once after NLopt
 * returns.
 */

struct NonpenNlopters
{
  nlopt_opt theta_opt;
  nlopt_opt dispersion_opt;

  struct ThetaObjectiveData
  {
    const EcountgmifsContextInternal* ctx;
    arma::vec fixed_xbeta;
    arma::vec eta_work;
    arma::vec mu_work;

    explicit ThetaObjectiveData(
        const EcountgmifsContextInternal& ctx_
    ) :
      ctx(&ctx_),
      fixed_xbeta(ctx_.input.api.X * ctx_.state.api.beta),
      eta_work(ctx_.input.api.X.n_rows),
      mu_work(ctx_.input.api.X.n_rows)
    {}
  };

  struct DispersionObjectiveData
  {
    const EcountgmifsContextInternal* ctx;
    const arma::vec* mu;

    explicit DispersionObjectiveData(
        const EcountgmifsContextInternal& ctx_
    ) :
      ctx(&ctx_),
      mu(&(ctx_.state.api.mu))
    {}
  };

  struct SaturatedDispersionObjectiveData
  {
    const EcountgmifsContextInternal* ctx;

    explicit SaturatedDispersionObjectiveData(
        const EcountgmifsContextInternal& ctx_
    ) :
      ctx(&ctx_)
    {}
  };

  explicit NonpenNlopters(
      EcountgmifsContextInternal& ctx
  ) :
    theta_opt(nullptr),
    dispersion_opt(nullptr)
  {
    const unsigned q =
      static_cast<unsigned>(ctx.state.api.theta.n_elem);

    if (q == 0) {
      Rcpp::stop(
        "NonpenNlopters(): nonpenalized design matrix 'w' "
        "must have at least one column"
      );
    }

    theta_opt = nlopt_create(NLOPT_LN_NELDERMEAD, q);

    if (theta_opt == nullptr) {
      Rcpp::stop("NonpenNlopters(): failed to create theta optimizer");
    }

    std::vector<double> theta_lower(q, -HUGE_VAL);

    nlopt_set_lower_bounds(theta_opt, theta_lower.data());
    nlopt_set_xtol_rel(theta_opt, ctx.control.api.nlopt_optim_reltol);
    nlopt_set_maxeval(theta_opt, static_cast<int>((q + 1) * 100));

    dispersion_opt = nlopt_create(NLOPT_LN_NELDERMEAD, 1);

    if (dispersion_opt == nullptr) {
      nlopt_destroy(theta_opt);
      theta_opt = nullptr;
      Rcpp::stop("NonpenNlopters(): failed to create dispersion optimizer");
    }

    nlopt_set_lower_bounds1(dispersion_opt, DISP_MIN_CAP);
    nlopt_set_xtol_rel(
      dispersion_opt,
      ctx.control.api.nlopt_optim_reltol
    );
    nlopt_set_maxeval(dispersion_opt, 200);
  }

  NonpenNlopters(const NonpenNlopters&) = delete;

  NonpenNlopters& operator=(const NonpenNlopters&) = delete;

  ~NonpenNlopters()
  {
    if (theta_opt != nullptr) {
      nlopt_destroy(theta_opt);
      theta_opt = nullptr;
    }

    if (dispersion_opt != nullptr) {
      nlopt_destroy(dispersion_opt);
      dispersion_opt = nullptr;
    }
  }

  static inline double theta_objective_local(
      unsigned n_theta,
      const double* theta,
      double* grad,
      void* data
  ) {
    if (grad != nullptr) {
      std::fill(grad, grad + n_theta, 0.0);
    }

    ThetaObjectiveData* objective_data =
      reinterpret_cast<ThetaObjectiveData*>(data);

    if (objective_data == nullptr ||
        objective_data->ctx == nullptr) {
      Rcpp::stop("theta_objective_local(): callback data is null");
    }

    const EcountgmifsContextInternal& ctx =
      *(objective_data->ctx);

    arma::vec theta_trial(
        const_cast<double*>(theta),
        static_cast<arma::uword>(n_theta),
        false,
        true
    );

    objective_data->eta_work =
      ctx.input.api.w * theta_trial;

    objective_data->eta_work +=
      objective_data->fixed_xbeta;

    ctx.mu_mean_from_eta_inplace(
      objective_data->eta_work,
      objective_data->mu_work
    );

    return ctx.negloglik_from_mu_dispersion(
      objective_data->mu_work,
      ctx.state.api.dispersion
    );
  }

  static inline double dispersion_objective(
      unsigned n_disp,
      const double* dispersion,
      double* grad,
      void* data
  ) {
    if (grad != nullptr) {
      std::fill(grad, grad + n_disp, 0.0);
    }

    if (n_disp != 1) {
      Rcpp::stop("dispersion_objective(): expected one parameter");
    }

    DispersionObjectiveData* objective_data =
      reinterpret_cast<DispersionObjectiveData*>(data);

    if (objective_data == nullptr ||
        objective_data->ctx == nullptr ||
        objective_data->mu == nullptr) {
      Rcpp::stop("dispersion_objective(): callback data is null");
    }

    return objective_data->ctx->negloglik_from_mu_dispersion(
      *(objective_data->mu),
      dispersion[0]
    );
  }

  static inline double saturated_dispersion_objective(
      unsigned n_disp,
      const double* dispersion,
      double* grad,
      void* data
  ) {
    if (grad != nullptr) {
      std::fill(grad, grad + n_disp, 0.0);
    }

    if (n_disp != 1) {
      Rcpp::stop(
        "saturated_dispersion_objective(): expected one parameter"
      );
    }

    SaturatedDispersionObjectiveData* objective_data =
      reinterpret_cast<SaturatedDispersionObjectiveData*>(data);

    if (objective_data == nullptr ||
        objective_data->ctx == nullptr) {
      Rcpp::stop(
        "saturated_dispersion_objective(): callback data is null"
      );
    }

    return objective_data->ctx->negloglik_saturated_from_dispersion(
      dispersion[0]
    );
  }

  arma::vec optimize_theta(
      EcountgmifsContextInternal& ctx
  ) {
    arma::vec theta_best = ctx.state.api.theta;
    ThetaObjectiveData objective_data(ctx);

    nlopt_set_min_objective(
      theta_opt,
      theta_objective_local,
      &objective_data
    );

    double minf = 0.0;

    nlopt_result result =
      nlopt_optimize(theta_opt, theta_best.memptr(), &minf);

    if (result < 0) {
      Rcpp::warning(
        "NLopt failed while optimizing theta: %d",
        static_cast<int>(result)
      );
    }

    /*
     * The callback may leave ctx at the last trial theta.
     * Commit NLopt's returned best theta.
     */
    ctx.set_theta(theta_best);

    return theta_best;
  }

  double optimize_dispersion(
      EcountgmifsContextInternal& ctx
  ) {
    if (ctx.input.api.family == POISSON) {
      ctx.set_dispersion(0.0);
      return 0.0;
    }

    if (ctx.control.api.fixed_dispersion) {
      ctx.set_dispersion(ctx.control.api.fixed_dispersion_value);
      return ctx.control.api.fixed_dispersion_value;
    }

    DispersionObjectiveData objective_data(ctx);

    nlopt_set_min_objective(
      dispersion_opt,
      dispersion_objective,
      &objective_data
    );

    double dispersion_best = ctx.state.api.dispersion;

    if (dispersion_best <= 0.0 ||
        !std::isfinite(dispersion_best)) {
        dispersion_best = 1e-4;
    }

    double minf = 0.0;

    nlopt_result result =
      nlopt_optimize(dispersion_opt, &dispersion_best, &minf);

    if (result < 0) {
      Rcpp::warning(
        "NLopt failed while optimizing dispersion: %d",
        static_cast<int>(result)
      );
    }

    if (dispersion_best < DISP_MIN_CAP ||
        !std::isfinite(dispersion_best)) {
        dispersion_best = DISP_MIN_CAP;
    }

    /*
     * The callback may leave ctx at the last trial dispersion.
     * Commit NLopt's returned best dispersion.
     */
    ctx.set_dispersion(dispersion_best);

    return dispersion_best;
  }

  double optimize_saturated_dispersion(
      EcountgmifsContextInternal& ctx
  ) {
    if (ctx.input.api.family == POISSON) {
      ctx.set_dispersion_saturated(0.0);
      return 0.0;
    }

    if (ctx.control.api.fixed_dispersion) {
      ctx.set_dispersion_saturated(
        ctx.control.api.fixed_dispersion_value
      );
      return ctx.control.api.fixed_dispersion_value;
    }

    SaturatedDispersionObjectiveData objective_data(ctx);

    nlopt_set_min_objective(
      dispersion_opt,
      saturated_dispersion_objective,
      &objective_data
    );

    double dispersion_best = ctx.state.api.saturated_dispersion;

    if (dispersion_best <= 0.0 ||
        !std::isfinite(dispersion_best)) {
        dispersion_best = ctx.state.api.dispersion;
    }

    if (dispersion_best <= 0.0 ||
        !std::isfinite(dispersion_best)) {
        dispersion_best = 1e-4;
    }

    double minf = 0.0;

    nlopt_result result =
      nlopt_optimize(dispersion_opt, &dispersion_best, &minf);

    if (result < 0) {
      Rcpp::warning(
        "NLopt failed while optimizing saturated dispersion: %d",
        static_cast<int>(result)
      );
    }

    if (dispersion_best < DISP_MIN_CAP ||
        !std::isfinite(dispersion_best)) {
        dispersion_best = DISP_MIN_CAP;
    }

    /*
     * The callback may leave saturated state at the last trial dispersion.
     * Commit NLopt's returned best saturated dispersion.
     */
    ctx.set_dispersion_saturated(dispersion_best);

    return dispersion_best;
  }
};
