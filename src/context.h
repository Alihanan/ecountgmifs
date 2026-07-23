#pragma once

#include <RcppArmadillo.h>
#include <cstdint>
#include <cmath>
#include <string>
#include <utility>
#include <limits>
#include <vector>

#include "../inst/include/ecountgmifs/api.h"
#include "enums.h"
#include "util.h"
#include "r_convert.h"
#include "nlopt_optimizer.h"

inline void check_input_dimensions(
    const arma::mat& X,
    const arma::vec& y,
    const arma::mat& w,
    const arma::vec& offset,
    const arma::vec& weight_vec,
    const arma::vec& yorig,
    const arma::mat& Xtest,
    const arma::vec& ytest,
    const arma::mat& wtest,
    const arma::vec& offsettest
) {
  if (X.n_rows == 0 || X.n_cols == 0) {
    Rcpp::stop("matrix 'X' must have positive dimensions");
  }

  check_vector_length(y, X.n_rows, "y");
  check_matrix_rows(w, X.n_rows, "w");
  check_vector_length(offset, X.n_rows, "offset");
  check_vector_length(weight_vec, X.n_cols, "weight_vec");
  check_vector_length(yorig, X.n_rows, "yorig");

  if (Xtest.n_rows > 0 || Xtest.n_cols > 0) {
    if (Xtest.n_cols != X.n_cols) {
      Rcpp::stop("matrix 'Xtest' must have the same number of columns as 'X'");
    }

    check_vector_length(ytest, Xtest.n_rows, "ytest");
    check_matrix_rows(wtest, Xtest.n_rows, "wtest");
    check_vector_length(offsettest, Xtest.n_rows, "offsettest");
  }
}


struct EcountgmifsInputInternal
{
  EcountgmifsInput api;

  EcountgmifsInputInternal(
    const arma::mat& X,
    const arma::vec& y,
    const arma::mat& w,
    const arma::vec& offset,
    const arma::vec& weight_vec,
    const arma::vec& yorig,
    const arma::mat& Xtest,
    const arma::vec& ytest,
    const arma::mat& wtest,
    const arma::vec& offsettest,
    double enet_alpha,
    SEXP family,
    SEXP link_func,
    Rcpp::Nullable<Rcpp::List> criteria
  ) :
    api {
    X,
    y,
    w,
    offset,
    weight_vec,
    weight_vec_has_prior(weight_vec),
    enet_alpha,

    Xtest,
    ytest,
    wtest,
    offsettest,
    yorig,

    -1.0 * arma::lgamma(y + 1.0),
    -1.0 * arma::lgamma(ytest + 1.0),
    -1.0 * arma::lgamma(yorig + 1.0),

    resolve_family_ptr(family),
    resolve_link_ptr(link_func),
    resolve_criteria_ptrs(criteria)
  }
  {
    check_input_dimensions(
      api.X,
      api.y,
      api.w,
      api.offset,
      api.weight_vec,
      api.yorig,
      api.Xtest,
      api.ytest,
      api.wtest,
      api.offsettest
    );

    check_matrix_finite(api.X, "X");
    check_vector_finite(api.y, "y");
    check_matrix_finite(api.w, "w");
    check_vector_finite(api.offset, "offset");
    check_vector_finite(api.weight_vec, "weight_vec");
    check_vector_finite(api.yorig, "yorig");

    if (arma::any(api.weight_vec <= 0.0)) {
      Rcpp::stop("value of 'weight_vec' must contain only positive values");
    }

    check_matrix_finite(api.Xtest, "Xtest");
    check_vector_finite(api.ytest, "ytest");
    check_matrix_finite(api.wtest, "wtest");
    check_vector_finite(api.offsettest, "offsettest");

    check_finite_scalar(api.enet_alpha, "enet_alpha");

    if (api.enet_alpha < 0.0 || api.enet_alpha > 1.0) {
      Rcpp::stop("value of 'enet_alpha' must be in [0, 1]");
    }
  }

  EcountgmifsInputInternal(
    const EcountgmifsInputInternal&
  ) = delete;

  EcountgmifsInputInternal(
    EcountgmifsInputInternal&&
  ) = delete;

  EcountgmifsInputInternal& operator=(
    const EcountgmifsInputInternal&
  ) = delete;

  EcountgmifsInputInternal& operator=(
    EcountgmifsInputInternal&&
  ) = delete;

  Rcpp::List to_list(bool include_data = false) const
  {
    Rcpp::List out = Rcpp::List::create(
      Rcpp::Named("n") = api.X.n_rows,
      Rcpp::Named("p") = api.X.n_cols,
      Rcpp::Named("q") = api.w.n_cols,
      Rcpp::Named("n_test") = api.Xtest.n_rows,
      Rcpp::Named("p_test") = api.Xtest.n_cols,
      Rcpp::Named("q_test") = api.wtest.n_cols,
      Rcpp::Named("family") =
        Rcpp::List::create(
          Rcpp::Named(api.family->name()) =
            Rcpp::List::create(
              Rcpp::Named("parameter_count") =
                api.family->parameter_count()
            )
        ),
      Rcpp::Named("link_func") =
        Rcpp::List::create(
          Rcpp::Named(api.link_func->name()) =
            Rcpp::List::create(
              Rcpp::Named("parameter_count") =
                api.link_func->parameter_count()
            )
        ),
      Rcpp::Named("enet_alpha") = api.enet_alpha,
      Rcpp::Named("has_prior") = api.has_prior
    );

    if (include_data) {
      out["X"] = api.X;
      out["y"] = api.y;
      out["w"] = api.w;
      out["offset"] = api.offset;
      out["weight_vec"] = api.weight_vec;

      out["Xtest"] = api.Xtest;
      out["ytest"] = api.ytest;
      out["wtest"] = api.wtest;
      out["offsettest"] = api.offsettest;
      out["yorig"] = api.yorig;
    }

    return out;
  }


  static const IEcountgmifsFamily* resolve_family_ptr(
      SEXP family
  )
  {
    Rcpp::XPtr<IEcountgmifsFamily> ptr(family);

    if (ptr.get() == nullptr)
      Rcpp::stop("family contains a null external pointer");

    return ptr.get();
  }

  static const IEcountgmifsLinkFunc* resolve_link_ptr(
      SEXP link_func
  )
  {
    Rcpp::XPtr<IEcountgmifsLinkFunc> ptr(link_func);

    if (ptr.get() == nullptr)
      Rcpp::stop("link_func contains a null external pointer");

    return ptr.get();
  }

  static std::vector<const IEcountgmifsCriterion*> resolve_criteria_ptrs(
      Rcpp::Nullable<Rcpp::List> criteria
  )
  {
    std::vector<const IEcountgmifsCriterion*> out;

    if (criteria.isNull()) {
      return out;
    }

    const Rcpp::List criteria_list(criteria);

    out.reserve(
      static_cast<std::size_t>(criteria_list.size())
    );

    for (R_xlen_t i = 0; i < criteria_list.size(); ++i)
    {
      Rcpp::XPtr<IEcountgmifsCriterion> ptr(
          criteria_list[i]
      );

      if (ptr.get() == nullptr)
        Rcpp::stop(
          "criterion at position %d contains a null external pointer",
          static_cast<int>(i + 1)
        );

      out.push_back(ptr.get());
    }

    return out;
  }
};


struct EcountgmifsControlInternal
{
  EcountgmifsControl api;

  EcountgmifsControlInternal(
    const EcountgmifsInput& input,
    uint64_t iteration_max,
    double epsilon_max,
    double epsilon_start,
    double epsilon_min,
    double tol,
    double loglik_reltol_cutoff,
    double enet_abs_tol,
    double enet_rel_tol,
    uint32_t enet_max_iter,
    int state_track_strategy,
    uint64_t state_track_freq,
    bool verbose,
    bool include_data,

    const arma::vec& theta_initial,
    const arma::vec& theta_lower_bounds,
    const arma::vec& theta_upper_bounds,
    int nlopt_algorithm,
    double nlopt_xtol_rel,
    double nlopt_ftol_rel,
    int nlopt_maxeval
  ) :
    api {
    iteration_max,
    epsilon_max,
    epsilon_start,
    epsilon_min,
    tol,
    loglik_reltol_cutoff,
    enet_abs_tol,
    enet_rel_tol,
    enet_max_iter,

    as_track_strategy(state_track_strategy),
    state_track_freq,
    verbose,
    include_data,

    theta_initial,
    theta_lower_bounds,
    theta_upper_bounds,

    nlopt_algorithm,
    nlopt_xtol_rel,
    nlopt_ftol_rel,
    nlopt_maxeval
  }
  {
    check_positive_integer(api.iteration_max, "iteration_max");

    check_positive_scalar(api.tol, "tol");
    check_nonnegative_scalar(api.loglik_reltol_cutoff, "loglik_reltol_cutoff");
    check_positive_scalar(api.enet_abs_tol, "enet_abs_tol");
    check_nonnegative_scalar(api.enet_rel_tol, "enet_rel_tol");
    check_positive_integer(api.enet_max_iter, "enet_max_iter");


    check_positive_scalar(api.epsilon_max, "epsilon_max");
    check_positive_scalar(api.epsilon_start, "epsilon_start");
    check_nonnegative_scalar(api.epsilon_min, "epsilon_min");
    check_initial_bounds(
      api.epsilon_start,
      api.epsilon_min,
      api.epsilon_max,
      "epsilon"
    );


    check_initial_bounds(
      api.theta_initial,
      api.theta_lower_bounds,
      api.theta_upper_bounds,
      input.w.n_cols,
      "theta"
    );
    check_nonnegative_scalar(
      api.nlopt_xtol_rel,
      "nlopt_xtol_rel"
    );

    check_nonnegative_scalar(
      api.nlopt_ftol_rel,
      "nlopt_ftol_rel"
    );

    check_positive_integer(
      api.nlopt_maxeval,
      "nlopt_maxeval"
    );

    if (
        api.state_track_strategy ==
          EnumStateTrackStrategy::EVERY_K_ITERATION
    ) {
      check_positive_integer(
        api.state_track_freq,
        "state_track_freq"
      );
    }

  }

  Rcpp::List to_list() const
  {
    return Rcpp::List::create(
      Rcpp::Named("iteration_max") = api.iteration_max,
      Rcpp::Named("epsilon_max") = api.epsilon_max,
      Rcpp::Named("epsilon_start") = api.epsilon_start,
      Rcpp::Named("epsilon_min") = api.epsilon_min,
      Rcpp::Named("tol") = api.tol,
      Rcpp::Named("loglik_reltol_cutoff") = api.loglik_reltol_cutoff,
      Rcpp::Named("enet_abs_tol") = api.enet_abs_tol,
      Rcpp::Named("enet_rel_tol") = api.enet_rel_tol,
      Rcpp::Named("enet_max_iter") = api.enet_max_iter,
      Rcpp::Named("state_track_strategy") =
        state_track_strategy_name(
          api.state_track_strategy
        ),
      Rcpp::Named("state_track_freq") = api.state_track_freq,
      Rcpp::Named("verbose") = api.verbose,
      Rcpp::Named("include_data") = api.include_data,
      Rcpp::Named("theta_initial") =
        ecountgmifs::output::to_r_vector(
          api.theta_initial
        ),

      Rcpp::Named("theta_lower_bounds") =
        ecountgmifs::output::to_r_vector(
          api.theta_lower_bounds
        ),

      Rcpp::Named("theta_upper_bounds") =
        ecountgmifs::output::to_r_vector(
          api.theta_upper_bounds
        ),

      Rcpp::Named("nlopt_algorithm") = api.nlopt_algorithm,
      Rcpp::Named("nlopt_xtol_rel") = api.nlopt_xtol_rel,
      Rcpp::Named("nlopt_ftol_rel") = api.nlopt_ftol_rel,
      Rcpp::Named("nlopt_maxeval") = api.nlopt_maxeval
    );
  }
};

struct EcountgmifsStateInternal
{
  const EcountgmifsInput& input;

  EcountgmifsState api;

  explicit EcountgmifsStateInternal(
      const EcountgmifsInput& input_,
      const EcountgmifsControl& control
  ) :
    input(input_),
    api {
    { // EcountgmifsPredictors
      { // EcountgmifsParameters
        arma::vec(
          input.X.n_cols,
          arma::fill::zeros
        ), // beta

        control.theta_initial, //theta
        input.family->initial_parameters(),
        input.link_func->initial_parameters()
      },

      arma::vec(
        input.X.n_rows,
        arma::fill::zeros
      ), // xbeta

      arma::vec(
        input.X.n_rows,
        arma::fill::zeros
      ), // wtheta

      arma::vec(
        input.X.n_rows,
        arma::fill::zeros
      ), // eta

      arma::vec(
        input.X.n_rows,
        arma::fill::zeros
      ), // mu

      arma::uvec(
        input.X.n_cols,
        arma::fill::zeros
      ) // active_set
    },

    arma::datum::nan, // negloglik
    arma::datum::nan, // saturated_dispersion
    arma::datum::nan, // saturated_negloglik
    arma::datum::nan, // null_negloglik

    Rcpp::NumericVector(), // criteria

    0,   // iteration
    1.0  // pseudo_r2
  }
  {
    check_initial_bounds(
      api.param.param.family_parameters,
      input.family->parameter_lower_bounds(),
      input.family->parameter_upper_bounds(),
      input.family->parameter_count(),
      "family parameters"
    );

    check_initial_bounds(
      api.param.param.link_parameters,
      input.link_func->parameter_lower_bounds(),
      input.link_func->parameter_upper_bounds(),
      input.link_func->parameter_count(),
      "link parameters"
    );

    refresh_linear_predictor();
  }

  EcountgmifsStateInternal(
    const EcountgmifsStateInternal&
  ) = delete;

  EcountgmifsStateInternal(
    EcountgmifsStateInternal&&
  ) = delete;

  EcountgmifsStateInternal& operator=(
    const EcountgmifsStateInternal&
  ) = delete;

  EcountgmifsStateInternal& operator=(
    EcountgmifsStateInternal&&
  ) = delete;

  Rcpp::List to_list() const
  {
    return to_list(api);
  }

  static Rcpp::List to_list(
      const EcountgmifsParameters& parameters
  )
  {
    return Rcpp::List::create(
      Rcpp::Named("beta") =
        ecountgmifs::output::to_r_vector(
          parameters.beta
        ),

      Rcpp::Named("theta") =
        ecountgmifs::output::to_r_vector(
          parameters.theta
        ),

      Rcpp::Named("family_parameters") =
        ecountgmifs::output::to_r_vector(
          parameters.family_parameters
        ),

      Rcpp::Named("link_parameters") =
        ecountgmifs::output::to_r_vector(
          parameters.link_parameters
        )
    );
  }

  static Rcpp::List to_list(
      const EcountgmifsPredictors& predictors
  )
  {
    return Rcpp::List::create(
      Rcpp::Named("parameters") =
        to_list(predictors.param),

      Rcpp::Named("xbeta") =
        ecountgmifs::output::to_r_vector(
          predictors.xbeta
        ),

      Rcpp::Named("wtheta") =
        ecountgmifs::output::to_r_vector(
          predictors.wtheta
        ),

      Rcpp::Named("eta") =
        ecountgmifs::output::to_r_vector(
          predictors.eta
        ),

      Rcpp::Named("mu") =
        ecountgmifs::output::to_r_vector(
          predictors.mu
        ),

      Rcpp::Named("active_set") =
        ecountgmifs::output::to_r_logical_vector(
          predictors.active_set
        )
    );
  }

  static Rcpp::List to_list(
      const EcountgmifsState& state
  )
  {
    return Rcpp::List::create(
      Rcpp::Named("predictors") =
        to_list(state.param),

        Rcpp::Named("negloglik") =
          state.negloglik,

        Rcpp::Named("saturated_dispersion") =
          state.saturated_dispersion,

        Rcpp::Named("saturated_negloglik") =
          state.saturated_negloglik,

        Rcpp::Named("null_negloglik") =
          state.null_negloglik,

        Rcpp::Named("criteria") =
          Rcpp::clone(state.criteria),

        Rcpp::Named("iteration") =
          ecountgmifs::output::to_r_integer(
            state.iteration,
            "iteration"
          ),

        Rcpp::Named("pseudo_r2") =
          state.pseudo_r2
    );
  }

  void refresh_linear_predictor()
  {
    api.param.xbeta =
      input.X *
      api.param.param.beta;

    api.param.wtheta =
      input.w *
      api.param.param.theta;

    api.param.eta =
      input.offset +
      api.param.xbeta +
      api.param.wtheta;

    this->refresh_mu();
  }

  void refresh_mu()
  {
    input.link_func->inverse(
        api.param.eta,
        api.param.param.link_parameters,
        api.param.mu
    );

    check_vector_finite(
      api.param.mu,
      "mu"
    );
  }
};

struct EcountgmifsGradientsInternal
{
  const EcountgmifsInput& input;
  const EcountgmifsState& state;

  EcountgmifsGradients api;

  EcountgmifsGradientsInternal(
    const EcountgmifsInput& input_,
    const EcountgmifsState& state_
  ) :
    input(input_),
    state(state_),
    api {
    arma::vec(
      input.X.n_rows,
      arma::fill::zeros
    ), // d_negloglik_d_mu

    arma::vec(
      input.X.n_rows,
      arma::fill::zeros
    ), // d_mu_d_eta

    input.X, // d_eta_d_beta
    input.w, // d_eta_d_theta

    arma::mat(
      input.X.n_rows,
      input.link_func->parameter_count(),
      arma::fill::zeros
    ), // d_mu_d_link_parameters

    arma::vec(
      input.family->parameter_count(),
      arma::fill::zeros
    ), // d_negloglik_d_family_parameters

    arma::vec(
      input.link_func->parameter_count(),
      arma::fill::zeros
    ), // d_negloglik_d_link_parameters

    arma::vec(
      input.X.n_cols,
      arma::fill::zeros
    ) // d_negloglik_d_beta
  }
  {}

  EcountgmifsGradientsInternal(
    const EcountgmifsGradientsInternal&
  ) = delete;

  EcountgmifsGradientsInternal(
    EcountgmifsGradientsInternal&&
  ) = delete;

  EcountgmifsGradientsInternal& operator=(
    const EcountgmifsGradientsInternal&
  ) = delete;

  EcountgmifsGradientsInternal& operator=(
    EcountgmifsGradientsInternal&&
  ) = delete;
};


struct EcountgmifsPathInternal
{
  const EcountgmifsState& current_state;
  const EcountgmifsControl& control;
  const EcountgmifsStagewise& stagewise;

  EcountgmifsPath api;

  EcountgmifsPathInternal(
    const EcountgmifsState& state_,
    const EcountgmifsControl& control_,
    const EcountgmifsStagewise& stagewise_
  ) :
    current_state(state_),
    control(control_),
    stagewise(stagewise_),
    api {}
    {
      api.last_saved_active_set.zeros(
        state_.param.active_set.n_elem
      );
    }

  Rcpp::List to_list() const
  {
    Rcpp::List states(
        static_cast<R_xlen_t>(api.states.size())
    );
    Rcpp::CharacterVector state_names(
        static_cast<R_xlen_t>(api.states.size())
    );

    for (std::size_t i = 0; i < api.states.size(); ++i) {
      const EcountgmifsState& state =
        api.states[i];

      states[static_cast<R_xlen_t>(i)] =
        EcountgmifsStateInternal::to_list(
          state
        );

      state_names[static_cast<R_xlen_t>(i)] =
        "state_at_iter_" +
        std::to_string(state.iteration);
    }

    states.attr("names") = state_names;

    return Rcpp::List::create(
      Rcpp::Named("states") =
        states,

        Rcpp::Named("last_saved_active_set") =
          ecountgmifs::output::to_r_logical_vector(
            api.last_saved_active_set
          ),

          Rcpp::Named("message") =
            api.message
    );
  }

  EcountgmifsPathInternal(
    const EcountgmifsPathInternal&
  ) = delete;

  EcountgmifsPathInternal(
    EcountgmifsPathInternal&&
  ) = delete;

  EcountgmifsPathInternal& operator=(
    const EcountgmifsPathInternal&
  ) = delete;

  EcountgmifsPathInternal& operator=(
    EcountgmifsPathInternal&&
  ) = delete;
};

struct EcountgmifsStagewiseInternal
{
  const EcountgmifsInput& input;
  const EcountgmifsControl& control;

  EcountgmifsState& state;
  EcountgmifsGradients& gradient;

  EcountgmifsStagewise api;

  NloptOptimizerInternal nonpen_optimizer;
  NloptOptimizerInternal family_optimizer;
  NloptOptimizerInternal link_optimizer;



  EcountgmifsStagewiseInternal(
    const EcountgmifsInput& input_,
    const EcountgmifsControl& control_,
    EcountgmifsState& state_,
    EcountgmifsGradients& gradient_
  ) :
    input(input_),
    control(control_),
    state(state_),
    gradient(gradient_),
    api {},

    nonpen_optimizer(
      state_.param.param.theta,
      control_.theta_lower_bounds,
      control_.theta_upper_bounds,
      &EcountgmifsStagewiseInternal::nonpen_objective,
      this,
      control_.nlopt_algorithm,
      control_.nlopt_xtol_rel,
      control_.nlopt_ftol_rel,
      control_.nlopt_maxeval
    ),

    family_optimizer(
      state_.param.param.family_parameters,
      input_.family->parameter_lower_bounds(),
      input_.family->parameter_upper_bounds(),
      &EcountgmifsStagewiseInternal::family_objective,
      this,
      control_.nlopt_algorithm,
      control_.nlopt_xtol_rel,
      control_.nlopt_ftol_rel,
      control_.nlopt_maxeval
    ),

    link_optimizer(
      state_.param.param.link_parameters,
      input_.link_func->parameter_lower_bounds(),
      input_.link_func->parameter_upper_bounds(),
      &EcountgmifsStagewiseInternal::link_objective,
      this,
      control_.nlopt_algorithm,
      control_.nlopt_xtol_rel,
      control_.nlopt_ftol_rel,
      control_.nlopt_maxeval
    )
    {
      api.delta_beta.zeros(input.X.n_cols);
      api.delta_xbeta.zeros(input.X.n_rows);
      api.trial_nu_linear.zeros(input.X.n_rows);
      api.trial_mu_mean.zeros(input.X.n_rows);

      api.epsilon =
        control.epsilon_start;
    }


  EcountgmifsStagewiseInternal(
    const EcountgmifsStagewiseInternal&
  ) = delete;

  EcountgmifsStagewiseInternal(
    EcountgmifsStagewiseInternal&&
  ) = delete;

  EcountgmifsStagewiseInternal& operator=(
    const EcountgmifsStagewiseInternal&
  ) = delete;

  EcountgmifsStagewiseInternal& operator=(
    EcountgmifsStagewiseInternal&&
  ) = delete;

  void fit()
  {
    initialize();

    /*
     * Later:
     * fit_nonpen();
     * fit_saturated();
     * fit_stagewise_path();
     */
  }

private:
  void initialize()
  {
    check_vector_finite(
      state.param.mu,
      "initial mu"
    );

    input.family->negloglik(
        input.y,
        state.param.mu,
        state.param.param.family_parameters,
        state.negloglik
    );

    check_finite_scalar(
      state.negloglik,
      "initial negloglik"
    );

    state.param.active_set.zeros();

    state.saturated_dispersion =
      arma::datum::nan;

    state.saturated_negloglik =
      arma::datum::nan;

    state.null_negloglik =
      arma::datum::nan;

    state.iteration = 0;
    state.pseudo_r2 = arma::datum::nan;

    state.criteria =
      Rcpp::NumericVector(
        static_cast<R_xlen_t>(
          input.criteria.size()
        ),
        NA_REAL
      );

    Rcpp::CharacterVector criterion_names(
        static_cast<R_xlen_t>(
          input.criteria.size()
        )
    );

    for (std::size_t i = 0; i < input.criteria.size(); ++i) {
      criterion_names[
      static_cast<R_xlen_t>(i)
      ] =
        input.criteria[i]->name();
    }

    state.criteria.attr("names") =
      criterion_names;

    api.delta_beta.zeros();
    api.delta_xbeta.zeros();
    api.trial_nu_linear.zeros();
    api.trial_mu_mean.zeros();

    api.epsilon =
      control.epsilon_start;

    api.negloglik_trial =
      arma::datum::nan;

    api.halving_count = 0;

    api.phase =
      EnumStagewisePhase::STAGEWISE_NONPEN;

    api.termination_reason =
      EnumStagewiseTerminationReason::STAGEWISE_RUNNING;

    api.termination_detail =
      "Ready for non-penalized fitting.";
  }



  template <typename Module>
  static arma::vec validated_initial_parameters(
      const Module& module,
      const char* module_type
  )
  {
    arma::vec initial = module.initial_parameters();
    const arma::vec lower = module.parameter_lower_bounds();
    const arma::vec upper = module.parameter_upper_bounds();

    const arma::uword n_parameters =
      module.parameter_count();

    if (
        initial.n_elem != n_parameters ||
          lower.n_elem != n_parameters ||
          upper.n_elem != n_parameters
    ) {
      Rcpp::stop(
        std::string(module_type) + " '" +
          module.name() +
          "' returned parameter vectors of incorrect size"
      );
    }

    if (
        arma::any(initial < lower) ||
          arma::any(initial > upper)
    ) {
      Rcpp::stop(
        std::string(module_type) + " '" +
          module.name() +
          "' returned initial parameters outside their bounds"
      );
    }

    return initial;
  }

  static double nonpen_objective(
      unsigned,
      const double*,
      double*,
      void*
  )
  {
    Rcpp::stop(
      "nonpenalized NLopt objective is not implemented"
    );

    return arma::datum::nan;
  }


  static double family_objective(
      unsigned,
      const double*,
      double*,
      void*
  )
  {
    Rcpp::stop(
      "family NLopt objective is not implemented"
    );

    return arma::datum::nan;
  }


  static double link_objective(
      unsigned,
      const double*,
      double*,
      void*
  )
  {
    Rcpp::stop(
      "link NLopt objective is not implemented"
    );

    return arma::datum::nan;
  }
};


struct EcountgmifsRuntimeInternal
{
  EcountgmifsRuntime api;

  EcountgmifsRuntimeInternal(
    const EcountgmifsState& state,
    const EcountgmifsPath& path,
    const EcountgmifsGradients& gradient,
    const EcountgmifsStagewise& stagewise
  ) :
    api {
    state,
    path,
    gradient,
    stagewise
  }
  {}

  EcountgmifsRuntimeInternal(
    const EcountgmifsRuntimeInternal&
  ) = delete;

  EcountgmifsRuntimeInternal(
    EcountgmifsRuntimeInternal&&
  ) = delete;

  EcountgmifsRuntimeInternal& operator=(
    const EcountgmifsRuntimeInternal&
  ) = delete;

  EcountgmifsRuntimeInternal& operator=(
    EcountgmifsRuntimeInternal&&
  ) = delete;
};


struct EcountgmifsContextInternal
{
  EcountgmifsInputInternal input;
  EcountgmifsControlInternal control;
  EcountgmifsStateInternal state;
  EcountgmifsGradientsInternal gradient;
  EcountgmifsStagewiseInternal stagewise;
  EcountgmifsPathInternal path;
  EcountgmifsRuntimeInternal runtime;

  EcountgmifsContext api;


  EcountgmifsContextInternal(
    const arma::mat& X,
    const arma::vec& y,
    const arma::mat& w,
    const arma::vec& offset,

    const arma::vec& yorig,
    const arma::mat& Xtest,
    const arma::vec& ytest,
    const arma::mat& wtest,
    const arma::vec& offsettest,

    const arma::vec& weight_vec,
    double enet_alpha,
    double epsilon_start,
    double epsilon_max,
    double epsilon_min,
    double tol,
    uint64_t iteration_max,

    SEXP family,
    SEXP link_func,
    Rcpp::Nullable<Rcpp::List> criteria,

    double loglik_reltol_cutoff,
    double enet_abs_tol,
    double enet_rel_tol,
    uint32_t enet_max_iter,
    bool verbose,
    int state_track_strategy,
    uint64_t state_track_freq,
    bool include_data,
    const arma::vec& theta_initial,
    const arma::vec& theta_lower_bounds,
    const arma::vec& theta_upper_bounds,

    int nlopt_algorithm,
    double nlopt_xtol_rel,
    double nlopt_ftol_rel,
    int nlopt_maxeval
    ) :
     input(
      X,
      y,
      w,
      offset,
      weight_vec,
      yorig,
      Xtest,
      ytest,
      wtest,
      offsettest,
      enet_alpha,
      family,
      link_func,
      criteria
    ),
    control(
      input.api,
      iteration_max,
      epsilon_max,
      epsilon_start,
      epsilon_min,
      tol,
      loglik_reltol_cutoff,
      enet_abs_tol,
      enet_rel_tol,
      enet_max_iter,

      state_track_strategy,
      state_track_freq,
      verbose,
      include_data,

      theta_initial,
      theta_lower_bounds,
      theta_upper_bounds,

      nlopt_algorithm,
      nlopt_xtol_rel,
      nlopt_ftol_rel,
      nlopt_maxeval
    ),
    state(
      input.api,
      control.api
    ),

    gradient(
      input.api,
      state.api
    ),

    stagewise(
      input.api,
      control.api,
      state.api,
      gradient.api
    ),

    path(
      state.api,
      control.api,
      stagewise.api
    ),

    runtime(
      state.api,
      path.api,
      gradient.api,
      stagewise.api
    ),

    api {
    input.api,
    control.api,
    runtime.api
  }
  {

  }

  EcountgmifsContextInternal(
    const EcountgmifsContextInternal&
  ) = delete;

  EcountgmifsContextInternal(
    EcountgmifsContextInternal&&
  ) = delete;

  EcountgmifsContextInternal& operator=(
    const EcountgmifsContextInternal&
  ) = delete;

  EcountgmifsContextInternal& operator=(
    EcountgmifsContextInternal&&
  ) = delete;
};
