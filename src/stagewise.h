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

inline arma::vec compute_beta_step(
    EcountgmifsContextInternal& ctx,
    const arma::vec& grad_beta,
    double epsilon
) {
  return solve_elastic_net_1D_weight(
    grad_beta,
    ctx.input.api.weight_vec,
    ctx.input.api.enet_alpha,
    epsilon,
    ctx.control.api.tol,
    1e-6,
    99,
    false
  );
}

inline bool try_beta_step_with_halving(
    EcountgmifsContextInternal& ctx,
    const arma::vec& beta_old,
    const arma::vec& grad_beta,
    double negloglik_old,
    arma::vec& beta_step_out
) {
  /*
   * glmSS-style:
   *
   *   1. try to increase epsilon
   *   2. compute beta step
   *   3. if beta-only objective worsens, halve epsilon and retry
   *   4. if epsilon becomes too small, declare beta converged
   *
   * This only commits beta. Theta/dispersion refit happens after this helper.
   */

  ctx.state.api.epsilon =
    std::min(
      ctx.control.api.epsilon_max,
      ctx.state.api.epsilon * 2.0
    );

  while (true) {
    if (ctx.state.api.epsilon <= ctx.control.api.epsilon_min) {
      ctx.set_beta(beta_old);
      beta_step_out.zeros(beta_old.n_elem);
      return false;
    }

    arma::vec beta_step =
      compute_beta_step(
        ctx,
        grad_beta,
        ctx.state.api.epsilon
      );

    if (step_is_zero(beta_step, ctx.control.api.epsilon_min)) {
      ctx.set_beta(beta_old);
      beta_step_out = beta_step;
      return false;
    }

    ctx.set_beta(beta_old + beta_step);

    /*
     * After ctx.set_beta(), ctx.state.api.negloglik is beta-only updated,
     * with theta and dispersion still fixed.
     *
     * Accept if the beta step does not worsen the minimized objective.
     */
    if (ctx.state.api.negloglik <= negloglik_old) {
      beta_step_out = beta_step;
      return true;
    }

    ctx.state.api.epsilon *= 0.5;

    if (ctx.control.api.verbose) {
      Rcpp::Rcout
      << "stagewise halving: beta step worsened negloglik"
      << ", new epsilon = " << ctx.state.api.epsilon
      << ", old negloglik = " << negloglik_old
      << ", trial negloglik = " << ctx.state.api.negloglik
      << "\n";
    }

    ctx.set_beta(beta_old);
  }
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

    arma::vec beta_step(
        beta_old.n_elem,
        arma::fill::zeros
    );

    const bool beta_step_accepted =
      try_beta_step_with_halving(
        ctx,
        beta_old,
        grad_beta,
        negloglik_old,
        beta_step
      );

    if (!beta_step_accepted) {
      if (ctx.control.api.verbose) {
        Rcpp::Rcout
        << "stagewise path stopped: beta step below epsilon_min "
        << "or no acceptable beta step at iteration "
        << ctx.state.api.iteration
        << "\n";
      }

      return;
    }
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

    if (beta_stalled) {
      ctx.set_message("beta_converged");
      if (ctx.control.api.verbose) {
        Rcpp::Rcout
        << "stagewise path stopped: beta stalled\n";
      }

      return;
    }

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
      ctx.set_message("objective_stalled");

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
        std::isfinite(ctx.state.api.pseudo_r2)) {

      const double target_pseudo_r2 =
        1.0 - ctx.control.api.loglik_reltol_cutoff;

      if (ctx.state.api.pseudo_r2 >= target_pseudo_r2) {
        ctx.set_message("pseudo_r2_cutoff");

        if (ctx.control.api.verbose) {
          Rcpp::Rcout
          << "stagewise path stopped: pseudo_r2 cutoff"
          << ", pseudo_r2=" << ctx.state.api.pseudo_r2
          << ", target=" << target_pseudo_r2
          << "\n";
        }

        return;
      }
    }

    // Save state if not break
    ctx.update_tracking();
  }
}
