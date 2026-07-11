#pragma once

#include <RcppArmadillo.h>

#include <cmath>
#include <iomanip>
#include <limits>
#include <string>
#include <utility>

#include "context.h"
#include "nlopt_optim_nonpen.h"
#include "enet.h"
#include "numerical_internal_constants.h"

#include <chrono> // todo delete

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

inline const char* ecountgmifs_delta_status(
    double delta
) {
  if (delta < 0.0) {
    return "improved";
  }

  if (delta > 0.0) {
    return "worsened";
  }

  return "unchanged";
}

inline arma::uword ecountgmifs_active_beta_count(
    const arma::vec& beta,
    double tol
) {
  return arma::accu(arma::abs(beta) > tol);
}

inline void ecountgmifs_print_iteration_header(
    uint64_t iteration
) {
  Rcpp::Rcout
  << "[ecountgmifs] ############################################################\n"
  << "[ecountgmifs] # stagewise iter " << iteration << "\n"
  << "[ecountgmifs] ############################################################\n";
}

inline void ecountgmifs_print_section_header(
    const char* section
) {
  Rcpp::Rcout
  << "[ecountgmifs] ------------------------------------------------------------\n"
  << "[ecountgmifs] " << section << "\n"
  << "[ecountgmifs] ------------------------------------------------------------\n";
}

inline void ecountgmifs_print_stop_header(
    const char* message
) {
  Rcpp::Rcout
  << "[ecountgmifs] ############################################################\n"
  << "[ecountgmifs] # stop: " << message << "\n"
  << "[ecountgmifs] ############################################################\n";
}

inline void ecountgmifs_print_blank_line()
{
  Rcpp::Rcout << "\n";
}

template <typename T>
inline void ecountgmifs_verbose_field(
    const char* name,
    const T& value
) {
  Rcpp::Rcout
  << "[ecountgmifs]   "
  << std::left << std::setw(16) << name
  << "= " << value << std::right << "\n";
}

inline void ecountgmifs_verbose_delta(
    const char* name,
    double old_value,
    double new_value
) {
  const double delta =
    new_value - old_value;

  Rcpp::Rcout
  << "[ecountgmifs]   "
  << std::left << std::setw(16) << name
  << "= " << old_value << " -> " << new_value << std::right << "\n";

  Rcpp::Rcout
  << "[ecountgmifs]   "
  << std::left << std::setw(16) << "delta"
  << "= " << delta << " " << ecountgmifs_delta_status(delta)
  << std::right << "\n";
}

inline void ecountgmifs_verbose_stop(
    const EcountgmifsContextInternal& ctx,
    const char* message
) {
  if (!ctx.control.api.verbose) {
    return;
  }

  ecountgmifs_print_stop_header(message);

  ecountgmifs_verbose_field(
    "iteration",
    ctx.state.api.iteration
  );
  ecountgmifs_verbose_field(
    "negloglik",
    ctx.state.api.negloglik
  );
  ecountgmifs_verbose_field(
    "pseudo_r2",
    ctx.state.api.pseudo_r2
  );
  ecountgmifs_verbose_field(
    "epsilon",
    ctx.state.api.epsilon
  );
  ecountgmifs_verbose_field(
    "active_beta",
    ecountgmifs_active_beta_count(
      ctx.state.api.beta,
      ctx.control.api.tol
    )
  );
}

inline void compute_beta_step_inplace(
    EcountgmifsContextInternal& ctx,
    const arma::vec& grad_beta,
    double epsilon,
    ElasticNetWeightWorkspace& enet_workspace,
    arma::vec& beta_step_out
) {
  solve_elastic_net_1D_weight_inplace(
    grad_beta,
    ctx.input.api.weight_vec,
    ctx.input.api.enet_alpha,
    epsilon,
    ctx.control.api.tol,
    1e-6,
    99,
    false,
    enet_workspace,
    beta_step_out
  );
}

struct BetaTrialWorkspace
{
  const arma::vec* fixed_wtheta;
  arma::vec beta_candidate;
  arma::vec xbeta_work;
  arma::vec eta_work;
  arma::vec mu_work;

  explicit BetaTrialWorkspace(
      const EcountgmifsContextInternal& ctx
  ) :
    fixed_wtheta(&(ctx.wtheta_n)),
    beta_candidate(ctx.state.api.beta.n_elem),
    xbeta_work(ctx.input.api.X.n_rows),
    eta_work(ctx.input.api.X.n_rows),
    mu_work(ctx.input.api.X.n_rows)
  {}
};

inline double evaluate_beta_trial_negloglik(
    EcountgmifsContextInternal& ctx,
    const arma::vec& beta_old,
    const arma::vec& beta_step,
    BetaTrialWorkspace& workspace
) {
  workspace.beta_candidate =
    beta_old;

  workspace.beta_candidate +=
    beta_step;

  workspace.xbeta_work =
    ctx.input.api.X * workspace.beta_candidate;

  workspace.eta_work =
    *(workspace.fixed_wtheta);

  workspace.eta_work +=
    workspace.xbeta_work;

  ctx.mu_mean_from_eta_inplace(
    workspace.eta_work,
    workspace.mu_work
  );

  return ctx.negloglik_from_mu_dispersion(
    workspace.mu_work,
    ctx.state.api.dispersion
  );
}

inline bool try_beta_step_with_halving(
    EcountgmifsContextInternal& ctx,
    const arma::vec& beta_old,
    const arma::vec& grad_beta,
    double negloglik_old,
    ElasticNetWeightWorkspace& enet_workspace,
    arma::vec& beta_step_out,
    uint64_t& halvings_out
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

  halvings_out = 0;
  ctx.ensure_accepted_linear_caches();
  BetaTrialWorkspace workspace(ctx);

  while (true) {
    if (ctx.state.api.epsilon <= ctx.control.api.epsilon_min) {
      beta_step_out.zeros(beta_old.n_elem);
      return false;
    }

    compute_beta_step_inplace(
      ctx,
      grad_beta,
      ctx.state.api.epsilon,
      enet_workspace,
      beta_step_out
    );

    if (step_is_zero(beta_step_out, ctx.control.api.epsilon_min)) {
      return false;
    }

    const double trial_negloglik =
      evaluate_beta_trial_negloglik(
        ctx,
        beta_old,
        beta_step_out,
        workspace
      );

    /*
     * The trial objective is beta-only updated, with theta and dispersion
     * still fixed. The accepted state is not mutated until acceptance.
     *
     * Accept if the beta step does not worsen the minimized objective.
     */
    if (trial_negloglik <= negloglik_old) {
      ctx.commit_beta_trial(
        std::move(workspace.beta_candidate),
        std::move(workspace.xbeta_work),
        std::move(workspace.eta_work),
        std::move(workspace.mu_work),
        trial_negloglik
      );
      return true;
    }

    ctx.state.api.epsilon *= 0.5;
    ++halvings_out;

    if (ctx.control.api.verbose) {
      const double delta =
        trial_negloglik - negloglik_old;

      Rcpp::Rcout
      << "[ecountgmifs]   halving " << halvings_out
      << ": epsilon=" << ctx.state.api.epsilon
      << ", negloglik=" << negloglik_old
      << " -> " << trial_negloglik
      << ", delta=" << delta
      << " " << ecountgmifs_delta_status(delta)
      << "\n";
    }
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

  double t_grad = 0.0;
  double t_theta = 0.0;
  double t_disp = 0.0;
  double t_start = 0.0;
  double t_fs = 0.0;
  double t_aftertrack = 0.0;
  double t_beforetrack = 0.0;
  auto t = std::chrono::steady_clock::now();

  ElasticNetWeightWorkspace enet_workspace(
    ctx.state.api.beta.n_elem
  );

  arma::vec beta_step(
    ctx.state.api.beta.n_elem,
    arma::fill::zeros
  );

  for (uint64_t iter = 0;
       iter < ctx.control.api.iteration_max;
       ++iter) {


    t_start += (std::chrono::duration<double>(
      std::chrono::steady_clock::now() - t
    ).count() - t_start) / (iter+1);
    Rprintf("Duration at start of iteration step: %.6f s\n",t_start);
    t = std::chrono::steady_clock::now();

    ctx.state.api.iteration = iter + 1;

    const arma::vec beta_old = ctx.state.api.beta;
    arma::vec theta_old;
    const double dispersion_old = ctx.state.api.dispersion;
    const double negloglik_old = ctx.state.api.negloglik;
    const double pseudo_r2_old = ctx.state.api.pseudo_r2;

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

    t_grad += (std::chrono::duration<double>(
      std::chrono::steady_clock::now() - t
    ).count() - t_grad) / (iter+1);
    Rprintf("Duration after gradient step: %.6f s\n",t_grad);
    t = std::chrono::steady_clock::now();

    if (ctx.control.api.verbose) {
      theta_old =
        ctx.state.api.theta;

      const arma::uword active_beta_old =
        ecountgmifs_active_beta_count(
          beta_old,
          ctx.control.api.tol
        );

      const double theta_norm_old =
        arma::norm(theta_old, 2);

      ecountgmifs_print_iteration_header(
        ctx.state.api.iteration
      );
      ecountgmifs_print_section_header(
        "start"
      );
      ecountgmifs_verbose_field(
        "negloglik",
        negloglik_old
      );
      ecountgmifs_verbose_field(
        "pseudo_r2",
        pseudo_r2_old
      );
      ecountgmifs_verbose_field(
        "epsilon",
        ctx.state.api.epsilon
      );
      ecountgmifs_verbose_field(
        "active_beta",
        active_beta_old
      );
      ecountgmifs_verbose_field(
        "dispersion",
        dispersion_old
      );
      ecountgmifs_verbose_field(
        "theta_norm",
        theta_norm_old
      );
    }

    uint64_t beta_halvings = 0;

    const bool beta_step_accepted =
      try_beta_step_with_halving(
        ctx,
        beta_old,
        grad_beta,
        negloglik_old,
        enet_workspace,
        beta_step,
        beta_halvings
      );

    if (!beta_step_accepted) {
      ecountgmifs_verbose_stop(
        ctx,
        "beta_converged"
      );

      return;
    }

    t_fs += (std::chrono::duration<double>(
      std::chrono::steady_clock::now() - t
    ).count() - t_fs) / (iter+1);
    Rprintf("Duration after FS step: %.6f s\n",t_fs);
    t = std::chrono::steady_clock::now();

    const double negloglik_after_beta =
      ctx.state.api.negloglik;

    if (ctx.control.api.verbose) {
      arma::vec weighted_grad =
        arma::abs(grad_beta) / ctx.input.api.weight_vec;

      arma::uword max_grad_j = 0;
      double max_grad = 0.0;
      double max_grad_weighted = 0.0;

      if (weighted_grad.n_elem > 0) {
        max_grad_j = weighted_grad.index_max();
        max_grad = grad_beta[max_grad_j];
        max_grad_weighted = weighted_grad[max_grad_j];
      }

      const arma::uword active_beta_old =
        ecountgmifs_active_beta_count(
          beta_old,
          ctx.control.api.tol
        );

      const arma::uword active_beta_after_beta =
        ecountgmifs_active_beta_count(
          ctx.state.api.beta,
          ctx.control.api.tol
        );

      ecountgmifs_print_blank_line();
      ecountgmifs_print_section_header(
        "beta step"
      );
      ecountgmifs_verbose_field(
        "epsilon",
        ctx.state.api.epsilon
      );
      ecountgmifs_verbose_field(
        "halvings",
        beta_halvings
      );
      ecountgmifs_verbose_field(
        "step_l2",
        arma::norm(beta_step, 2)
      );
      ecountgmifs_verbose_field(
        "step_l1",
        arma::norm(beta_step, 1)
      );
      Rcpp::Rcout
      << "[ecountgmifs]   "
      << std::left << std::setw(16) << "active_beta"
      << "= " << active_beta_old
      << " -> " << active_beta_after_beta
      << std::right << "\n";
      ecountgmifs_verbose_delta(
        "negloglik",
        negloglik_old,
        negloglik_after_beta
      );
      ecountgmifs_verbose_field(
        "max_grad_j",
        max_grad_j + 1
      );
      ecountgmifs_verbose_field(
        "max_grad",
        max_grad
      );
      ecountgmifs_verbose_field(
        "max_grad/weight",
        max_grad_weighted
      );
    }
    /*
     * Refit nonpenalized part conditional on new beta.
     * These optimizers commit their returned best values via ctx.set_theta()
     * and ctx.set_dispersion().
     */
    opt.optimize_theta(ctx);

    const double negloglik_after_theta =
      ctx.state.api.negloglik;

    t_theta += (std::chrono::duration<double>(
      std::chrono::steady_clock::now() - t
    ).count() - t_theta) / (iter+1);
    Rprintf("Duration after theta step: %.6f s\n",t_theta);
    t = std::chrono::steady_clock::now();

    if (ctx.control.api.verbose) {
      ecountgmifs_print_blank_line();
      ecountgmifs_print_section_header(
        "theta refit"
      );
      ecountgmifs_verbose_field(
        "theta_diff",
        arma::norm(ctx.state.api.theta - theta_old, 2)
      );
      ecountgmifs_verbose_field(
        "theta_norm",
        arma::norm(ctx.state.api.theta, 2)
      );
      ecountgmifs_verbose_delta(
        "negloglik",
        negloglik_after_beta,
        negloglik_after_theta
      );
    }

    opt.optimize_dispersion(ctx);

    const double negloglik_after_dispersion =
      ctx.state.api.negloglik;

    if (ctx.control.api.verbose) {
      ecountgmifs_print_blank_line();
      ecountgmifs_print_section_header(
        "dispersion refit"
      );
      Rcpp::Rcout
      << "[ecountgmifs]   "
      << std::left << std::setw(16) << "dispersion"
      << "= " << dispersion_old
      << " -> " << ctx.state.api.dispersion
      << std::right << "\n";
      ecountgmifs_verbose_field(
        "dispersion_diff",
        std::abs(ctx.state.api.dispersion - dispersion_old)
      );
      ecountgmifs_verbose_delta(
        "negloglik",
        negloglik_after_theta,
        negloglik_after_dispersion
      );
    }

    t_disp += (std::chrono::duration<double>(
      std::chrono::steady_clock::now() - t
    ).count() - t_disp) / (iter+1);
    Rprintf("Duration after dispersion step: %.6f s\n",t_disp);
    t = std::chrono::steady_clock::now();

    const double beta_diff =
      arma::norm(ctx.state.api.beta - beta_old, 2);

    const double negloglik_diff =
      std::abs(ctx.state.api.negloglik - negloglik_old);

    ctx.state.api.iteration = iter + 1;

    if (ctx.control.api.verbose) {
      ecountgmifs_print_blank_line();
      ecountgmifs_print_section_header(
        "end"
      );
      ecountgmifs_verbose_field(
        "negloglik",
        ctx.state.api.negloglik
      );
      ecountgmifs_verbose_field(
        "pseudo_r2",
        ctx.state.api.pseudo_r2
      );
      ecountgmifs_verbose_field(
        "epsilon",
        ctx.state.api.epsilon
      );
      ecountgmifs_verbose_field(
        "active_beta",
        ecountgmifs_active_beta_count(
          ctx.state.api.beta,
          ctx.control.api.tol
        )
      );
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
        << "[ecountgmifs]   stop checks:\n";
        ecountgmifs_verbose_field(
          "beta_stalled",
          beta_stalled
        );
        ecountgmifs_verbose_field(
          "beta_diff",
          beta_diff
        );
        ecountgmifs_verbose_field(
          "epsilon_min",
          ctx.control.api.epsilon_min
        );
        ecountgmifs_verbose_stop(
          ctx,
          "beta_converged"
        );
      }

      return;
    }

    const bool objective_stalled =
      negloglik_diff < ctx.control.api.tol;

    if (ctx.control.api.verbose) {
      const double theta_diff =
        arma::norm(ctx.state.api.theta - theta_old, 2);

      const double dispersion_diff =
        std::abs(ctx.state.api.dispersion - dispersion_old);

      const bool nonpen_stalled =
        theta_diff < ctx.control.api.tol &&
        dispersion_diff < ctx.control.api.tol;

      Rcpp::Rcout
      << "[ecountgmifs]   stop checks:\n";
      ecountgmifs_verbose_field(
        "beta_stalled",
        beta_stalled
      );
      ecountgmifs_verbose_field(
        "nonpen_stalled",
        nonpen_stalled
      );
      ecountgmifs_verbose_field(
        "objective_stalled",
        objective_stalled
      );
      ecountgmifs_verbose_field(
        "beta_diff",
        beta_diff
      );
      ecountgmifs_verbose_field(
        "theta_diff",
        theta_diff
      );
      ecountgmifs_verbose_field(
        "dispersion_diff",
        dispersion_diff
      );
      ecountgmifs_verbose_field(
        "negloglik_diff",
        negloglik_diff
      );
    }



    if (beta_stalled || objective_stalled) {
      ctx.set_message("objective_stalled");

      if (ctx.control.api.verbose) {
        ecountgmifs_verbose_stop(
          ctx,
          "objective_stalled"
        );
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
          ecountgmifs_verbose_stop(
            ctx,
            "pseudo_r2_cutoff"
          );
          ecountgmifs_verbose_field(
            "target_pseudo_r2",
            target_pseudo_r2
          );
        }

        return;
      }
    }

    t_beforetrack += (std::chrono::duration<double>(
      std::chrono::steady_clock::now() - t
    ).count() - t_beforetrack) / (iter+1);
    Rprintf("Duration before tracking step: %.6f s\n",t_beforetrack);
    t = std::chrono::steady_clock::now();

    // Save state if not break
    ctx.update_tracking();

    t_aftertrack += (std::chrono::duration<double>(
      std::chrono::steady_clock::now() - t
    ).count() - t_aftertrack) / (iter+1);
    Rprintf("Duration after tracking step: %.6f s\n",t_aftertrack);
    t = std::chrono::steady_clock::now();
  }
}
