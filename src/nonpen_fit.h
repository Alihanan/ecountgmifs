#pragma once

#include <RcppArmadillo.h>

#include <cmath>

#include "context.h"
#include "nlopt_optim_nonpen.h"
#include "debug.h"

/*
 * Nonpenalized and saturated fitting orchestration.
 *
 * This header intentionally does not define optimizer objects and does not
 * define model state. It only coordinates:
 *
 *   ContextInternal:
 *     - state storage
 *     - set_beta()
 *     - set_theta()
 *     - set_dispersion()
 *     - set_dispersion_saturated()
 *     - initialize_nonpen_start()
 *
 *   NonpenNlopters:
 *     - optimize_theta()
 *     - optimize_dispersion()
 *     - optimize_saturated_dispersion()
 */

inline void fit_nonpen(
    EcountgmifsContextInternal& ctx,
    NonpenNlopters& opt,
    uint32_t max_outer = 1000
) {
  /*
   * Fit the non-penalized model:
   *
   *   beta is fixed at zero
   *   theta is optimized
   *   dispersion is optimized unless family is Poisson or dispersion is fixed
   *
   * This is only intercept-only when w contains only the intercept column.
   */

  ECOUNTGMIFS_VERBOSE(
    ctx.control.api.verbose,
    "nonpen: start"
    << ", max_outer=" << max_outer
    << ", theta_n=" << ctx.state.api.theta.n_elem
    << ", beta_n=" << ctx.state.api.beta.n_elem
    << ", family=" << static_cast<int>(ctx.input.api.family)
    << ", fixed_dispersion=" << ctx.control.api.fixed_dispersion
  );

  ctx.initialize_nonpen_start();

  ECOUNTGMIFS_VERBOSE(
    ctx.control.api.verbose,
    "nonpen: after initialize_nonpen_start"
    << ", negloglik=" << ctx.state.api.negloglik
    << ", dispersion=" << ctx.state.api.dispersion
    << ", theta=" << ctx.state.api.theta.t()
  );

  for (uint32_t iter = 0; iter < max_outer; ++iter) {
    const arma::vec theta_old = ctx.state.api.theta;
    const double dispersion_old = ctx.state.api.dispersion;
    const double negloglik_old = ctx.state.api.negloglik;

    ECOUNTGMIFS_VERBOSE(
      ctx.control.api.verbose,
      "nonpen: iter=" << iter + 1
                      << " begin"
                      << ", negloglik=" << negloglik_old
                      << ", dispersion=" << dispersion_old
                      << ", theta=" << theta_old.t()
    );

    ECOUNTGMIFS_VERBOSE(
      ctx.control.api.verbose,
      "nonpen: iter=" << iter + 1 << " optimize_theta start"
    );

    opt.optimize_theta(ctx);

    ECOUNTGMIFS_VERBOSE(
      ctx.control.api.verbose,
      "nonpen: iter=" << iter + 1
                      << " optimize_theta done"
                      << ", negloglik=" << ctx.state.api.negloglik
                      << ", theta=" << ctx.state.api.theta.t()
    );

    ECOUNTGMIFS_VERBOSE(
      ctx.control.api.verbose,
      "nonpen: iter=" << iter + 1 << " optimize_dispersion start"
    );

    opt.optimize_dispersion(ctx);

    ECOUNTGMIFS_VERBOSE(
      ctx.control.api.verbose,
      "nonpen: iter=" << iter + 1
                      << " optimize_dispersion done"
                      << ", negloglik=" << ctx.state.api.negloglik
                      << ", dispersion=" << ctx.state.api.dispersion
    );

    const double theta_diff =
      arma::norm(ctx.state.api.theta - theta_old, 2);

    const double dispersion_diff =
      std::abs(ctx.state.api.dispersion - dispersion_old);

    const double negloglik_diff =
      std::abs(ctx.state.api.negloglik - negloglik_old);

    ECOUNTGMIFS_VERBOSE(
      ctx.control.api.verbose,
      "nonpen: iter=" << iter + 1
                      << " convergence"
                      << ", theta_diff=" << theta_diff
                      << ", dispersion_diff=" << dispersion_diff
                      << ", negloglik_diff=" << negloglik_diff
                      << ", tol=" << ctx.control.api.tol
    );

    if (theta_diff < ctx.control.api.tol &&
        dispersion_diff < ctx.control.api.tol &&
        negloglik_diff < ctx.control.api.tol) {
      ctx.state.api.initialized = true;

      ECOUNTGMIFS_VERBOSE(
        ctx.control.api.verbose,
        "nonpen: converged at iter=" << iter + 1
      );

      return;
    }
  }

  ctx.state.api.initialized = true;

  ECOUNTGMIFS_VERBOSE(
    ctx.control.api.verbose,
    "nonpen: stopped after max_outer"
    << ", final_negloglik=" << ctx.state.api.negloglik
    << ", final_dispersion=" << ctx.state.api.dispersion
    << ", final_theta=" << ctx.state.api.theta.t()
  );
}

inline void fit_saturated(
    EcountgmifsContextInternal& ctx,
    NonpenNlopters& opt
) {
  /*
   * Fit saturated objective:
   *
   *   mean is y directly
   *   only saturated dispersion is optimized
   *
   * This does not touch the regular model state:
   *
   *   beta
   *   theta
   *   eta
   *   mu
   *   negloglik
   */

  ECOUNTGMIFS_VERBOSE(
    ctx.control.api.verbose,
    "saturated: start"
    << ", family=" << static_cast<int>(ctx.input.api.family)
    << ", fixed_dispersion=" << ctx.control.api.fixed_dispersion
  );

  if (ctx.input.api.family == POISSON) {
    ctx.set_dispersion_saturated(0.0);

    ECOUNTGMIFS_VERBOSE(
      ctx.control.api.verbose,
      "saturated: poisson done"
      << ", saturated_dispersion="
      << ctx.state.api.saturated_dispersion
      << ", saturated_negloglik="
      << ctx.state.api.saturated_negloglik
    );

    return;
  }

  if (ctx.control.api.fixed_dispersion) {
    ctx.set_dispersion_saturated(
      ctx.control.api.fixed_dispersion_value
    );

    ECOUNTGMIFS_VERBOSE(
      ctx.control.api.verbose,
      "saturated: fixed dispersion done"
      << ", saturated_dispersion="
      << ctx.state.api.saturated_dispersion
      << ", saturated_negloglik="
      << ctx.state.api.saturated_negloglik
    );

    return;
  }

  ECOUNTGMIFS_VERBOSE(
    ctx.control.api.verbose,
    "saturated: optimize_saturated_dispersion start"
    << ", start_saturated_dispersion="
    << ctx.state.api.saturated_dispersion
  );

  opt.optimize_saturated_dispersion(ctx);

  ECOUNTGMIFS_VERBOSE(
    ctx.control.api.verbose,
    "saturated: optimize_saturated_dispersion done"
    << ", saturated_dispersion="
    << ctx.state.api.saturated_dispersion
    << ", saturated_negloglik="
    << ctx.state.api.saturated_negloglik
  );
}
