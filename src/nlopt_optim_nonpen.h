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
    const arma::vec* fixed_xbeta;
    FixedDispersionNegloglikData likelihood_data;
    arma::vec wtheta_work;
    arma::vec eta_work;
    arma::vec mu_work;
    NegloglikWorkspace likelihood_workspace;

    explicit ThetaObjectiveData(
        const EcountgmifsContextInternal& ctx_
    ) :
      ctx(&ctx_),
      fixed_xbeta(&(ctx_.xbeta_n)),
      likelihood_data(
        ctx_.input.api.y,
        ctx_.input.api.train_y_one_lgamma,
        ctx_.input.api.family,
        ctx_.state.api.dispersion,
        ctx_.control.api.nb_poisson_fallback_eps
      ),
      wtheta_work(ctx_.input.api.X.n_rows),
      eta_work(ctx_.input.api.X.n_rows),
      mu_work(ctx_.input.api.X.n_rows),
      likelihood_workspace(ctx_.input.api.X.n_rows)
    {}
  };

  struct DispersionObjectiveData
  {
    const EcountgmifsContextInternal* ctx;
    const arma::vec* mu;
    NegloglikWorkspace likelihood_workspace;

    explicit DispersionObjectiveData(
        const EcountgmifsContextInternal& ctx_
    ) :
      ctx(&ctx_),
      mu(&(ctx_.state.api.mu)),
      likelihood_workspace(ctx_.input.api.X.n_rows)
    {}
  };

  struct SaturatedDispersionObjectiveData
  {
    const EcountgmifsContextInternal* ctx;
    NegloglikWorkspace likelihood_workspace;

    explicit SaturatedDispersionObjectiveData(
        const EcountgmifsContextInternal& ctx_
    ) :
      ctx(&ctx_),
      likelihood_workspace(ctx_.input.api.y.n_elem)
    {}
  };

  explicit NonpenNlopters(
      EcountgmifsContextInternal& ctx
  ) :
    theta_opt(nullptr),
    dispersion_opt(nullptr),
    theta_objective_data(ctx),
    dispersion_objective_data(ctx)
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

  ThetaObjectiveData theta_objective_data;
  DispersionObjectiveData dispersion_objective_data;

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

    if (n_theta == 1) {
      const arma::uword n =
        ctx.input.api.X.n_rows;

      if (objective_data->mu_work.n_elem != n) {
        objective_data->mu_work.set_size(n);
      }

      const double theta0 =
        theta[0];

      switch (ctx.input.api.link_func) {
      case LOG_LINK: {
        for (arma::uword i = 0; i < n; ++i) {
          const double eta_without_offset =
            (*(objective_data->fixed_xbeta))[i] +
            ctx.input.api.w(i, 0) * theta0;

          const double eta_total =
            clamp_scalar(
              ctx.input.api.offset[i] + eta_without_offset,
              ETA_MIN_CAP,
              ETA_MAX_CAP
            );

          objective_data->mu_work[i] =
            clamp_scalar(
              std::exp(eta_total),
              MU_MIN_CAP,
              MU_MAX_CAP
            );
        }

        break;
      }

      case SOFTPLUS_LINK: {
        for (arma::uword i = 0; i < n; ++i) {
          const double eta_without_offset =
            (*(objective_data->fixed_xbeta))[i] +
            ctx.input.api.w(i, 0) * theta0;

          const double mu_i =
            std::exp(ctx.input.api.offset[i]) *
            softplus_scalar(eta_without_offset);

          objective_data->mu_work[i] =
            clamp_scalar(
              mu_i,
              MU_MIN_CAP,
              MU_MAX_CAP
            );
        }

        break;
      }

      default:
        Rcpp::stop("unknown link function");
      }

      return fixed_dispersion_negloglik_inplace(
        objective_data->mu_work,
        objective_data->likelihood_data,
        objective_data->likelihood_workspace
      );
    }

    arma::vec theta_trial(
        const_cast<double*>(theta),
        static_cast<arma::uword>(n_theta),
        false,
        true
    );

    objective_data->wtheta_work =
      ctx.input.api.w * theta_trial;

    objective_data->eta_work =
      objective_data->wtheta_work;

    objective_data->eta_work +=
      *(objective_data->fixed_xbeta);

    ctx.mu_mean_from_eta_inplace(
      objective_data->eta_work,
      objective_data->mu_work
    );

    return fixed_dispersion_negloglik_inplace(
      objective_data->mu_work,
      objective_data->likelihood_data,
      objective_data->likelihood_workspace
    );
  }

  static inline void prepare_theta_selected_state(
      const arma::vec& theta,
      ThetaObjectiveData& objective_data
  ) {
    if (objective_data.ctx == nullptr) {
      Rcpp::stop("prepare_theta_selected_state(): callback data is null");
    }

    const EcountgmifsContextInternal& ctx =
      *(objective_data.ctx);

    objective_data.wtheta_work =
      ctx.input.api.w * theta;

    objective_data.eta_work =
      objective_data.wtheta_work;

    objective_data.eta_work +=
      *(objective_data.fixed_xbeta);

    ctx.mu_mean_from_eta_inplace(
      objective_data.eta_work,
      objective_data.mu_work
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
      dispersion[0],
      objective_data->likelihood_workspace
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
      dispersion[0],
      objective_data->likelihood_workspace
    );
  }

  arma::vec optimize_theta(
      EcountgmifsContextInternal& ctx
  ) {
    ctx.ensure_accepted_linear_caches();

    arma::vec theta_best = ctx.state.api.theta;

    theta_objective_data.ctx =
      &ctx;

    theta_objective_data.fixed_xbeta =
      &(ctx.xbeta_n);

    theta_objective_data.likelihood_data.refresh(
      ctx.input.api.family,
      ctx.state.api.dispersion,
      ctx.control.api.nb_poisson_fallback_eps
    );

    nlopt_set_min_objective(
      theta_opt,
      theta_objective_local,
      &theta_objective_data
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

    double selected_negloglik = minf;

    if (result < 0) {
      /*
       * Keep the conservative failure path: explicitly evaluate the returned
       * theta because NLopt's objective value is not trusted on failure.
       */
      selected_negloglik =
        theta_objective_local(
          static_cast<unsigned>(theta_best.n_elem),
          theta_best.memptr(),
          nullptr,
          &theta_objective_data
        );
    } else {
      /*
       * NLopt's returned theta and minf correspond on success. Prepare the
       * accepted state vectors without another likelihood evaluation.
       */
      prepare_theta_selected_state(
        theta_best,
        theta_objective_data
      );
    }

    ctx.commit_theta_trial_swap(
      theta_best,
      theta_objective_data.wtheta_work,
      theta_objective_data.eta_work,
      theta_objective_data.mu_work,
      selected_negloglik
    );

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

    dispersion_objective_data.ctx =
      &ctx;

    dispersion_objective_data.mu =
      &(ctx.state.api.mu);

    nlopt_set_min_objective(
      dispersion_opt,
      dispersion_objective,
      &dispersion_objective_data
    );

    double dispersion_best = ctx.state.api.dispersion;

    if (dispersion_best <= 0.0 ||
        !std::isfinite(dispersion_best)) {
        dispersion_best = 1e-4;
    }

    double minf = 0.0;

    nlopt_result result =
      nlopt_optimize(dispersion_opt, &dispersion_best, &minf);

    const bool nlopt_failed =
      result < 0;

    if (nlopt_failed) {
      Rcpp::warning(
        "NLopt failed while optimizing dispersion: %d",
        static_cast<int>(result)
      );
    }

    bool dispersion_corrected = false;

    if (dispersion_best < DISP_MIN_CAP ||
        !std::isfinite(dispersion_best)) {
        dispersion_best = DISP_MIN_CAP;
        dispersion_corrected = true;
    }

    if (nlopt_failed || dispersion_corrected) {
      /*
       * Keep the conservative path when NLopt failed or when the returned
       * dispersion was corrected after optimization: explicitly evaluate the
       * objective at the value being committed.
       */
      const double selected_negloglik =
        ctx.negloglik_from_mu_dispersion(
          ctx.state.api.mu,
          dispersion_best,
          dispersion_objective_data.likelihood_workspace
        );

      ctx.commit_dispersion_trial(
        dispersion_best,
        selected_negloglik
      );
    } else {
      ctx.commit_dispersion_trial(
        dispersion_best,
        minf
      );
    }

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
