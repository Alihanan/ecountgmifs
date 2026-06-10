#pragma once

#include <RcppArmadillo.h>

#include <cmath>
#include <limits>

#include "context.h"
#include "nlopt_optim_nonpen.h"
#include "enet.h"
#include "numerical_internal_constants.h"

/*
 * Main stagewise path loop.
 *
 * This intentionally does not handle criteria yet.
 *
 * Per iteration:
 *
 *   1. compute gradient wrt eta
 *   2. compute gradient wrt beta
 *   3. use enet.h to solve local constrained update
 *   4. beta <- beta + step
 *   5. refit theta
 *   6. refit dispersion
 *
 * All state updates go through:
 *
 *   ctx.set_beta()
 *   ctx.set_theta()      indirectly through opt.optimize_theta()
 *   ctx.set_dispersion() indirectly through opt.optimize_dispersion()
 */

inline bool step_is_zero(
    const arma::vec& step,
    double tol
) {
  if (step.n_elem == 0) {
    return true;
  }

  if (!step.is_finite()) {
    Rcpp::stop("stagewise update contains non-finite values");
  }

  return arma::norm(step, 2) <= tol;
}

inline void fit_stagewise_path(
    EcountgmifsContextInternal& ctx,
    NonpenNlopters& opt
) {
  /*
   * The nonpenalized model should already be fitted before this function:
   *
   *   fit_nonpen(ctx, opt, ...)
   *   fit_saturated(ctx, opt)
   *
   * At entry, beta is usually zero and theta/dispersion are initialized.
   */

  for (uint64_t iter = 0;
       iter < ctx.control.api.iteration_max;
       ++iter) {

    ctx.state.api.iteration = iter + 1;

    const arma::vec beta_old = ctx.state.api.beta;
    const arma::vec theta_old = ctx.state.api.theta;
    const double dispersion_old = ctx.state.api.dispersion;
    const double negloglik_old = ctx.state.api.negloglik;

    arma::vec grad_beta =
      gradient_beta(
        ctx.input.api.X,
        ctx.input.api.y,
        ctx.state.api.eta,
        ctx.state.api.mu,
        ctx.input.api.offset,
        ctx.input.api.family,
        ctx.input.api.link_func,
        ctx.state.api.dispersion
      );

    arma::vec beta_step =
      solve_elastic_net_1D_weight(
        grad_beta,
        ctx.input.api.weight_vec,
        ctx.input.api.enet_alpha,
        ctx.state.api.epsilon,
        ctx.control.api.tol,
        1e-6,
        99,
        ctx.control.api.verbose
      );

    if (step_is_zero(beta_step, ctx.control.api.epsilon_min)) {
      if (ctx.control.api.verbose) {
        Rcpp::Rcout
        << "stagewise path stopped: beta step below epsilon_min at iteration "
        << ctx.state.api.iteration
        << "\n";
      }

      return;
    }

    arma::vec beta_new = beta_old + beta_step;

    ctx.set_beta(beta_new);

    /*
     * Refit nonpenalized part conditional on new beta.
     * These optimizers commit their returned best values via ctx.set_theta()
     * and ctx.set_dispersion().
     */
    opt.optimize_theta(ctx);
    opt.optimize_dispersion(ctx);

    const double beta_diff =
      arma::norm(ctx.state.api.beta - beta_old, 2);

    const double theta_diff =
      arma::norm(ctx.state.api.theta - theta_old, 2);

    const double dispersion_diff =
      std::abs(ctx.state.api.dispersion - dispersion_old);

    const double negloglik_diff =
      std::abs(ctx.state.api.negloglik - negloglik_old);

    const double improvement =
      negloglik_old - ctx.state.api.negloglik;

    ctx.state.api.iteration = iter + 1;

    if (ctx.control.api.verbose) {
      Rcpp::Rcout
      << "iter = " << ctx.state.api.iteration
      << ", negloglik = " << ctx.state.api.negloglik
      << ", improvement = " << improvement
      << ", beta_diff = " << beta_diff
      << ", theta_diff = " << theta_diff
      << ", dispersion_diff = " << dispersion_diff
      << ", negloglik_diff = " << negloglik_diff
      << ", epsilon = " << ctx.state.api.epsilon
      << "\n";
    }

    /*
     * Stop only when all main state changes are small.
     *
     * This requires both:
     *   - objective movement below tol
     *   - L2 movement in beta and theta below tol
     *   - scalar dispersion movement below tol
     */

    /* TURNED OFF FOR NOW
     * beta_diff < ctx.control.api.tol &&
     theta_diff < ctx.control.api.tol &&
     dispersion_diff < ctx.control.api.tol &&
     */
    const bool beta_stalled =
      beta_diff <= ctx.control.api.epsilon_min;

    const bool nonpen_stalled =
      theta_diff < ctx.control.api.tol &&
      dispersion_diff < ctx.control.api.tol;

    const bool objective_stalled =
      negloglik_diff < ctx.control.api.tol;

    if (ctx.control.api.verbose) {
      Rcpp::Rcout
      << "stop flags: beta_stalled=" << beta_stalled
      << ", nonpen_stalled=" << nonpen_stalled
      << ", objective_stalled=" << objective_stalled
      << ", beta_diff=" << beta_diff
      << ", epsilon_min=" << ctx.control.api.epsilon_min
      << ", tol=" << ctx.control.api.tol
      << "\n";
    }

    if (beta_stalled || objective_stalled) {
      if (ctx.control.api.verbose) {
        Rcpp::Rcout
        << "stagewise path stopped: beta step, nonpen parameters, "
        << "and negloglik all stalled\n";
      }

      ctx.update_tracking();
      return;
    }

    /*
     * Optional pseudo-R2 style cutoff using saturated objective.
     *
     * Keep it conservative for now. If saturated_negloglik is available,
     * this stops when the remaining gap to saturated fit is small enough
     * relative to the initial/nonpenalized gap.
     */
    if (ctx.control.api.loglik_reltol_cutoff > 0.0 &&
        std::isfinite(ctx.state.api.saturated_negloglik)) {

      const double gap =
        ctx.state.api.negloglik - ctx.state.api.saturated_negloglik;

      if (gap <= ctx.control.api.loglik_reltol_cutoff) {
        if (ctx.control.api.verbose) {
          Rcpp::Rcout
          << "stagewise path stopped: saturated negloglik gap cutoff\n";
        }

        return;
      }
    }

    // Save state if not break
    ctx.update_tracking();
  }
}
