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
 * They call context setters, so eta/mu/negloglik recomputation stays hidden
 * behind the context/state transition mechanism.
 */

struct NonpenNlopters
{
  nlopt_opt theta_opt;
  nlopt_opt dispersion_opt;

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
    nlopt_set_min_objective(theta_opt, theta_objective, &ctx);
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

  static inline double theta_objective(
      unsigned n_theta,
      const double* theta,
      double* grad,
      void* data
  ) {
    if (grad != nullptr) {
      std::fill(grad, grad + n_theta, 0.0);
    }

    EcountgmifsContextInternal* ctx =
      reinterpret_cast<EcountgmifsContextInternal*>(data);

    arma::vec theta_trial(
        const_cast<double*>(theta),
        static_cast<arma::uword>(n_theta),
        false,
        true
    );

    ctx->set_theta(theta_trial);

    return ctx->state.api.negloglik;
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

    EcountgmifsContextInternal* ctx =
      reinterpret_cast<EcountgmifsContextInternal*>(data);

    ctx->set_dispersion(dispersion[0]);

    return ctx->state.api.negloglik;
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

    EcountgmifsContextInternal* ctx =
      reinterpret_cast<EcountgmifsContextInternal*>(data);

    ctx->set_dispersion_saturated(dispersion[0]);

    return ctx->state.api.saturated_negloglik;
  }

  arma::vec optimize_theta(
      EcountgmifsContextInternal& ctx
  ) {
    arma::vec theta_best = ctx.state.api.theta;

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

    nlopt_set_min_objective(
      dispersion_opt,
      dispersion_objective,
      &ctx
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

    nlopt_set_min_objective(
      dispersion_opt,
      saturated_dispersion_objective,
      &ctx
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


