#pragma once

#include <RcppArmadillo.h>
#include <cstdint>
#include <cmath>
#include <string>
#include <utility>
#include <limits>

#include "../inst/include/ecountgmifs/api.h"
#include "enums.h"
#include "link_function.h"
#include "loglik.h"
#include "criteria.h"
#include "numerical_internal_constants.h"
#include "util.h"

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
    uint32_t family,
    uint32_t link_func,
    double enet_alpha
  ) :
    api {
    X,
    y,
    w,
    offset,
    weight_vec,
    weight_vec_has_prior(weight_vec),

    Xtest,
    ytest,
    wtest,
    offsettest,
    yorig,

    -1.0 * arma::lgamma(y + 1.0),
    -1.0 * arma::lgamma(ytest + 1.0),
    -1.0 * arma::lgamma(yorig + 1.0),

    as_family(family),
    as_link_func(link_func),
    enet_alpha
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
    const EcountgmifsInputInternal& other
  ) :
    api {
    other.api.X,
    other.api.y,
    other.api.w,
    other.api.offset,
    other.api.weight_vec,
    other.api.has_prior,

    other.api.Xtest,
    other.api.ytest,
    other.api.wtest,
    other.api.offsettest,
    other.api.yorig,

    other.api.train_y_one_lgamma,
    other.api.test_y_one_lgamma,
    other.api.orig_y_one_lgamma,

    other.api.family,
    other.api.link_func,
    other.api.enet_alpha
  }
  {}

  EcountgmifsInputInternal(
    EcountgmifsInputInternal&& other
  ) :
    api {
    other.api.X,
    other.api.y,
    other.api.w,
    other.api.offset,
    other.api.weight_vec,
    other.api.has_prior,

    other.api.Xtest,
    other.api.ytest,
    other.api.wtest,
    other.api.offsettest,
    other.api.yorig,

    std::move(other.api.train_y_one_lgamma),
    std::move(other.api.test_y_one_lgamma),
    std::move(other.api.orig_y_one_lgamma),

    other.api.family,
    other.api.link_func,
    other.api.enet_alpha
  }
  {}

  EcountgmifsInputInternal& operator=(
    const EcountgmifsInputInternal& other
  ) = delete;

  EcountgmifsInputInternal& operator=(
    EcountgmifsInputInternal&& other
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
      Rcpp::Named("family") = static_cast<int>(api.family),
      Rcpp::Named("link_func") = static_cast<int>(api.link_func),
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
};


struct EcountgmifsControlInternal
{
  EcountgmifsControl api;

  EcountgmifsControlInternal(
    uint64_t iteration_max,
    double epsilon_max,
    double epsilon_start,
    double epsilon_min,
    double tol,
    double nlopt_optim_reltol,
    double loglik_reltol_cutoff,
    double nb_poisson_fallback_eps,
    bool fixed_dispersion,
    double fixed_dispersion_value,
    int state_track_strategy,
    uint64_t state_track_freq,
    bool verbose,
    bool include_data
  ) :
    api {
    iteration_max,
    epsilon_max,
    epsilon_start,
    epsilon_min,
    tol,
    nlopt_optim_reltol,
    loglik_reltol_cutoff,
    nb_poisson_fallback_eps,
    fixed_dispersion,
    fixed_dispersion_value,
    as_track_strategy(state_track_strategy),
    state_track_freq,
    verbose,
    include_data
  }
  {
    if (api.iteration_max == 0) {
      Rcpp::stop("value of 'iteration_max' must be positive");
    }

    check_positive_scalar(api.epsilon_max, "epsilon_max");
    check_positive_scalar(api.epsilon_start, "epsilon_start");
    check_nonnegative_scalar(api.epsilon_min, "epsilon_min");
    check_positive_scalar(api.tol, "tol");
    check_positive_scalar(api.nlopt_optim_reltol, "nlopt_optim_reltol");
    check_nonnegative_scalar(
      api.loglik_reltol_cutoff,
      "loglik_reltol_cutoff"
    );
    check_nonnegative_scalar(
      api.nb_poisson_fallback_eps,
      "nb_poisson_fallback_eps"
    );

    if (api.epsilon_start > api.epsilon_max) {
      Rcpp::stop("value of 'epsilon_start' must be <= 'epsilon_max'");
    }

    if (api.epsilon_min > api.epsilon_start) {
      Rcpp::stop("value of 'epsilon_min' must be <= 'epsilon_start'");
    }

    if (api.fixed_dispersion) {
      check_positive_scalar(
        api.fixed_dispersion_value,
        "fixed_dispersion_value"
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
      Rcpp::Named("nlopt_optim_reltol") = api.nlopt_optim_reltol,
      Rcpp::Named("loglik_reltol_cutoff") = api.loglik_reltol_cutoff,
      Rcpp::Named("nb_poisson_fallback_eps") =
        api.nb_poisson_fallback_eps,
      Rcpp::Named("fixed_dispersion") = api.fixed_dispersion,
      Rcpp::Named("fixed_dispersion_value") = api.fixed_dispersion_value,
      Rcpp::Named("state_track_strategy") =
        static_cast<int>(api.state_track_strategy),
        Rcpp::Named("state_track_freq") = api.state_track_freq,
        Rcpp::Named("verbose") = api.verbose
    );
  }
};


struct EcountgmifsStateInternal
{
  EcountgmifsState api;

  EcountgmifsStateInternal(
    arma::uword p,
    arma::uword q,
    double epsilon_start
  ) :
    api {
    arma::vec(p, arma::fill::zeros), // beta
    arma::vec(q, arma::fill::zeros), // theta
    arma::vec(),  // eta
    arma::vec(),  // mu
    0.0,   // dispersion
    std::numeric_limits<double>::infinity(), // negloglik
    0.0,   // saturated_dispersion
    std::numeric_limits<double>::infinity(), // saturated_negloglik
    std::numeric_limits<double>::infinity(), // null_negloglik
    Rcpp::NumericVector(), // criteria
    0, // iteration
    epsilon_start,
    false, // initialized?
    1.0 // pseudo r^2
  }
  {
    if (p == 0) {
      Rcpp::stop("state cannot be initialized with p = 0");
    }

    check_positive_scalar(epsilon_start, "epsilon_start");
  }

  Rcpp::List to_list() const
  {
    return Rcpp::List::create(
      Rcpp::Named("iteration") = api.iteration,
      Rcpp::Named("beta") = api.beta,
      Rcpp::Named("theta") = api.theta,
      Rcpp::Named("dispersion") = api.dispersion,
      Rcpp::Named("eta") = api.eta,
      Rcpp::Named("mu") = api.mu,
      Rcpp::Named("negloglik") = api.negloglik,
      Rcpp::Named("saturated_dispersion") = api.saturated_dispersion,
      Rcpp::Named("saturated_negloglik") = api.saturated_negloglik,
      Rcpp::Named("criteria") = api.criteria,
      Rcpp::Named("epsilon") = api.epsilon,
      Rcpp::Named("initialized") = api.initialized
    );
  }
};


struct EcountgmifsContextInternal
{
  EcountgmifsInputInternal input;
  EcountgmifsControlInternal control;
  EcountgmifsStateInternal state;


  std::vector<EcountgmifsState> states;
  arma::uvec last_saved_active_set;
  std::string message;
  Rcpp::List criteria;
  arma::vec work_n;

  EcountgmifsContext api;




  EcountgmifsContextInternal(
    const EcountgmifsInputInternal& input_,
    const EcountgmifsControlInternal& control_
  ) :
    input(input_),
    control(control_),
    state(
      input.api.X.n_cols,
      input.api.w.n_cols,
      control.api.epsilon_start
    ),
    states(),
    message("not initialized"),
    criteria(),
    work_n(),
    api {
    input.api,
    control.api,
    state.api
  }
  {}

  EcountgmifsContextInternal(
    EcountgmifsInputInternal&& input_,
    EcountgmifsControlInternal&& control_
  ) :
    input(std::move(input_)),
    control(std::move(control_)),
    state(
      input.api.X.n_cols,
      input.api.w.n_cols,
      control.api.epsilon_start
    ),
    criteria(),
    work_n(),
    api {
    input.api,
    control.api,
    state.api
  }
  {}

  arma::vec nu_linear() const
  {
    return input.api.w * state.api.theta +
      input.api.X * state.api.beta;
  }

  arma::vec mu_mean_from_eta(
      const arma::vec& eta
  ) const {
    switch (input.api.link_func) {
    case LOG_LINK: {
      arma::vec eta_safe =
        arma::clamp(
          input.api.offset + eta,
          ETA_MIN_CAP,
          ETA_MAX_CAP
        );

      return arma::clamp(
        arma::exp(eta_safe),
        MU_MIN_CAP,
        MU_MAX_CAP
      );
    }

    case SOFTPLUS_LINK: {
      return arma::clamp(
        arma::exp(input.api.offset) % softplus_vec(eta),
        MU_MIN_CAP,
        MU_MAX_CAP
      );
    }
    }

    Rcpp::stop("unknown link function");
  }

  void mu_mean_from_eta_inplace(
      const arma::vec& eta,
      arma::vec& mu
  ) const {
    if (mu.n_elem != eta.n_elem) {
      mu.set_size(eta.n_elem);
    }

    switch (input.api.link_func) {
    case LOG_LINK: {
      for (arma::uword i = 0; i < eta.n_elem; ++i) {
        const double eta_total =
          clamp_scalar(
            input.api.offset[i] + eta[i],
            ETA_MIN_CAP,
            ETA_MAX_CAP
          );

        const double mu_i =
          std::exp(eta_total);

        mu[i] =
          clamp_scalar(
            mu_i,
            MU_MIN_CAP,
            MU_MAX_CAP
          );
      }

      return;
    }

    case SOFTPLUS_LINK: {
      for (arma::uword i = 0; i < eta.n_elem; ++i) {
        const double mu_i =
          std::exp(input.api.offset[i]) *
          softplus_scalar(eta[i]);

        mu[i] =
          clamp_scalar(
            mu_i,
            MU_MIN_CAP,
            MU_MAX_CAP
          );
      }

      return;
    }
    }

    Rcpp::stop("unknown link function");
  }


  double negloglik_from_mu_mean(
      const arma::vec& mu
  ) const {
    return negloglik_from_mu_dispersion(
      mu,
      state.api.dispersion
    );
  }

  double negloglik_from_mu_dispersion(
      const arma::vec& mu,
      double dispersion
  ) const {
    switch (input.api.family) {
    case POISSON:
      return poisson_negloglik(
        mu,
        input.api.y,
        input.api.train_y_one_lgamma
      );

    case NEGATIVE_BINOMIAL:
      return nb_negloglik(
        mu,
        input.api.y,
        dispersion,
        input.api.train_y_one_lgamma,
        control.api.nb_poisson_fallback_eps
      );
    }

    Rcpp::stop("unknown family");
  }

  arma::uvec active_set_from_beta(
      const arma::vec& beta
  ) const {
    return arma::find(
      arma::abs(beta) > control.api.tol
    );
  }

  bool active_sets_equal(
      const arma::uvec& a,
      const arma::uvec& b
  ) const {
    if (a.n_elem != b.n_elem) {
      return false;
    }

    if (a.n_elem == 0) {
      return true;
    }

    return arma::all(a == b);
  }

  bool active_set_changed_from_last_saved_state() const
  {
    if (state.api.iteration == 0) {
      return false;
    }

    const arma::uvec current =
      active_set_from_beta(state.api.beta);

    if (last_saved_active_set.n_elem == 0) {
      return true;
    }

    return !active_sets_equal(last_saved_active_set, current);
  }

  bool should_track_state()
  {
    if (state.api.iteration == 0) {
      return false;
    }
    switch (control.api.state_track_strategy) {
    case NO_STATE_TRACKING:
      return false;

    case ALL_ITERATION:
      return true;

    case EVERY_K_ITERATION:
      return (control.api.state_track_freq != 0 &&
          state.api.iteration % control.api.state_track_freq == 0);

    case ACTIVE_SET_CHANGE:
      return active_set_changed_from_last_saved_state();
    }

    Rcpp::stop("unknown state tracking strategy");
  }

  void track_state(){
    states.push_back(state.api);
  }

  void set_criteria(
      const Rcpp::List& criteria_new
  ) {
    criteria = criteria_new;
  }

  void evaluate_criteria() {
    const R_xlen_t m = criteria.size();

    Rcpp::NumericVector values(m);
    Rcpp::CharacterVector value_names(m);

    Rcpp::CharacterVector list_names;
    const bool has_list_names =
      criteria.hasAttribute("names") &&
      Rf_length(criteria.attr("names")) == m;

    if (has_list_names) {
      list_names = Rcpp::CharacterVector(criteria.attr("names"));
    }

    for (R_xlen_t i = 0; i < m; ++i) {
      Rcpp::List criterion_obj(criteria[i]);

      if (!criterion_obj.containsElementNamed("pointer")) {
        Rcpp::stop("criterion object is missing field 'pointer'");
      }

      SEXP criterion_ptr = criterion_obj["pointer"];
      void* address = R_ExternalPtrAddr(criterion_ptr);

      if (address == nullptr) {
        Rcpp::stop("criterion pointer is null");
      }

      criterion_fun_t criterion =
        reinterpret_cast<criterion_fun_t>(address);

      values[i] = criterion(&api);

      if (has_list_names &&
          STRING_ELT(list_names, i) != NA_STRING &&
          CHAR(STRING_ELT(list_names, i))[0] != '\0') {
        value_names[i] = STRING_ELT(list_names, i);
      } else if (criterion_obj.containsElementNamed("name")) {
        value_names[i] = Rcpp::as<std::string>(criterion_obj["name"]);
      }
    }

    values.attr("names") = value_names;
    state.api.criteria = values;
  }

  void update_tracking()
  {
    const bool save_state =
      should_track_state();

    if (save_state) {
      track_state();

      if (control.api.state_track_strategy == ACTIVE_SET_CHANGE) {
        last_saved_active_set =
          active_set_from_beta(state.api.beta);
      }
    }
  }

  void ensure_state_vectors()
  {
    const arma::uword n = input.api.X.n_rows;

    if (state.api.eta.n_elem != n) {
      state.api.eta.set_size(n);
    }

    if (state.api.mu.n_elem != n) {
      state.api.mu.set_size(n);
    }

    if (work_n.n_elem != n) {
      work_n.set_size(n);
    }
  }

  void refresh_eta_inplace()
  {
    ensure_state_vectors();

    state.api.eta = input.api.w * state.api.theta;
    work_n = input.api.X * state.api.beta;
    state.api.eta += work_n;
  }

  void refresh_mu_inplace()
  {
    ensure_state_vectors();
    mu_mean_from_eta_inplace(
      state.api.eta,
      state.api.mu
    );
  }

  void refresh_negloglik_from_mu()
  {
    state.api.negloglik =
      negloglik_from_mu_mean(state.api.mu);
  }

  void refresh_pseudo_r2()
  {
    state.api.pseudo_r2 =
      pseudo_r2_from_negloglik();
  }

  void refresh()
  {
    refresh_eta_inplace();
    refresh_mu_inplace();
    refresh_negloglik_from_mu();
    refresh_pseudo_r2();
  }

  void refresh_after_dispersion_change()
  {
    if (state.api.mu.n_elem != input.api.X.n_rows) {
      refresh();
      return;
    }

    refresh_negloglik_from_mu();
    refresh_pseudo_r2();
  }

  void set_null_negloglik_from_current_state()
  {
    state.api.null_negloglik = state.api.negloglik;
    refresh();
  }

  double pseudo_r2_from_negloglik()
  {
    if (!std::isfinite(state.api.null_negloglik) ||
        !std::isfinite(state.api.negloglik) ||
        !std::isfinite(state.api.saturated_negloglik)) {
        return 0.0;
    }

    const double denominator =
      state.api.null_negloglik - state.api.saturated_negloglik;

    if (denominator <= 0.0 || !std::isfinite(denominator)) {
      return 0.0;
    }

    const double numerator =
      state.api.null_negloglik - state.api.negloglik;

    return std::max(
        0.0,
        std::min(
          1.0,
          numerator / denominator
        )
      );
  }

  void set_theta(const arma::vec& theta_new)
  {
    if (theta_new.n_elem != state.api.theta.n_elem) {
      Rcpp::stop("set_theta(): incompatible theta size");
    }
    state.api.theta = theta_new;
    refresh();
  }
  void set_beta(const arma::vec& beta_new)
  {
    if (beta_new.n_elem != state.api.beta.n_elem) {
      Rcpp::stop("set_beta(): incompatible beta size");
    }
    state.api.beta = beta_new;
    refresh();
  }

  void commit_beta_trial(
      arma::vec&& beta_new,
      arma::vec&& eta_new,
      arma::vec&& mu_new,
      double negloglik_new
  ) {
    if (beta_new.n_elem != state.api.beta.n_elem) {
      Rcpp::stop("commit_beta_trial(): incompatible beta size");
    }

    if (eta_new.n_elem != input.api.X.n_rows ||
        mu_new.n_elem != input.api.X.n_rows) {
      Rcpp::stop("commit_beta_trial(): incompatible eta/mu size");
    }

    state.api.beta =
      std::move(beta_new);

    state.api.eta =
      std::move(eta_new);

    state.api.mu =
      std::move(mu_new);

    state.api.negloglik =
      negloglik_new;

    refresh_pseudo_r2();
  }

  void set_theta_beta(
      const arma::vec& theta_new,
      const arma::vec& beta_new
  ) {
    if (theta_new.n_elem != state.api.theta.n_elem) {
      Rcpp::stop("set_theta_beta(): incompatible theta size");
    }

    if (beta_new.n_elem != state.api.beta.n_elem) {
      Rcpp::stop("set_theta_beta(): incompatible beta size");
    }

    state.api.theta = theta_new;
    state.api.beta = beta_new;
    refresh();
  }
  void set_dispersion(double dispersion_new)
  {
    if (input.api.family == NEGATIVE_BINOMIAL) {
      if (dispersion_new <= 0.0 || !std::isfinite(dispersion_new)) {
        Rcpp::stop("set_dispersion(): dispersion must be positive and finite");
      }
    }

    if (input.api.family == POISSON) {
      dispersion_new = 0.0;
    }

    state.api.dispersion = dispersion_new;
    refresh_after_dispersion_change();
  }


  void set_dispersion_saturated(
      double dispersion_new
  ) {
    if (input.api.family == NEGATIVE_BINOMIAL) {
      if (dispersion_new <= 0.0 || !std::isfinite(dispersion_new)) {
        Rcpp::stop(
          "set_dispersion_saturated(): "
          "dispersion must be positive and finite"
        );
      }
    }

    if (input.api.family == POISSON) {
      dispersion_new = 0.0;
    }

    state.api.saturated_dispersion = dispersion_new;
    state.api.saturated_negloglik =
      negloglik_saturated_from_dispersion(
        state.api.saturated_dispersion
      );
  }

  double negloglik_saturated_from_dispersion(
      double dispersion
  ) const {
    arma::vec mu_saturated = input.api.y;

    switch (input.api.family) {
    case POISSON:
      return poisson_negloglik(
        mu_saturated,
        input.api.y,
        input.api.train_y_one_lgamma
      );

    case NEGATIVE_BINOMIAL:
      return nb_negloglik(
        mu_saturated,
        input.api.y,
        dispersion,
        input.api.train_y_one_lgamma,
        control.api.nb_poisson_fallback_eps
      );
    }

    Rcpp::stop("unknown family");
  }

  void set_message(
      const std::string& message_new
  ) {
    message = message_new;
  }

  void initialize_nonpen_start()
  {
    /*
     * This initializes the non-penalized model start.
     *
     * It is not the full fit.
     * It sets beta = 0, initializes dispersion, and gives theta[0]
     * a stable intercept-like starting value when w[, 0] is an intercept.
     */
    arma::vec beta_zero(
        state.api.beta.n_elem,
        arma::fill::zeros
    );

    set_beta(beta_zero);

    if (input.api.family == POISSON) {
      set_dispersion(0.0);
    } else if (control.api.fixed_dispersion) {
      set_dispersion(control.api.fixed_dispersion_value);
    } else if (state.api.dispersion <= 0.0 ||
      !std::isfinite(state.api.dispersion)) {
      set_dispersion(1e-4);
    }

    if (state.api.theta.n_elem == 0 || input.api.w.n_cols == 0) {
      return;
    }

    if (input.api.w.n_rows == 0) {
      return;
    }

    const arma::vec intercept_col(
        input.api.w.n_rows,
        arma::fill::ones
    );

    const bool first_column_is_intercept =
      arma::approx_equal(
        input.api.w.col(0),
        intercept_col,
        "absdiff",
        1e-12
      );

    if (!first_column_is_intercept) {
      return;
    }

    const double y_sum = arma::accu(input.api.y);
    const double exposure_sum =
      arma::accu(arma::exp(input.api.offset));

    if (y_sum <= 0.0 ||
        exposure_sum <= 0.0 ||
        !std::isfinite(exposure_sum)) {
        return;
    }

    const double mean_rate = y_sum / exposure_sum;

    if (mean_rate <= 0.0 || !std::isfinite(mean_rate)) {
      return;
    }

    arma::vec theta_new = state.api.theta;

    switch (input.api.link_func) {
    case LOG_LINK:
      theta_new[0] = std::log(mean_rate);
      set_theta(theta_new);
      return;

    case SOFTPLUS_LINK:
      theta_new[0] =
        mean_rate > 20.0 ?
        mean_rate :
      std::log(std::expm1(mean_rate));
      set_theta(theta_new);
      return;
    }

    Rcpp::stop("unknown link function");
  }

  EcountgmifsContextInternal(
    EcountgmifsContextInternal&& other
  ) :
    input(std::move(other.input)),
    control(std::move(other.control)),
    state(std::move(other.state)),
    states(std::move(other.states)),
    last_saved_active_set(std::move(other.last_saved_active_set)),
    message(std::move(other.message)),
    criteria(std::move(other.criteria)),
    work_n(std::move(other.work_n)),
    api {
    input.api,
    control.api,
    state.api
  }
  {}


  EcountgmifsContextInternal& operator=(
    const EcountgmifsContextInternal& other
  ) = delete;

  EcountgmifsContextInternal& operator=(
    EcountgmifsContextInternal&& other
  ) = delete;

  EcountgmifsContextInternal(
    const EcountgmifsContextInternal&
  ) = delete;

  Rcpp::List states_to_dynamic_tracking_list() const
  {
    const R_xlen_t m =
      static_cast<R_xlen_t>(states.size());

    Rcpp::NumericVector iteration(m);
    Rcpp::NumericVector epsilon(m);
    Rcpp::NumericVector dispersion(m);
    Rcpp::NumericVector negloglik(m);
    Rcpp::NumericVector saturated_dispersion(m);
    Rcpp::NumericVector saturated_negloglik(m);
    Rcpp::LogicalVector initialized(m);
    Rcpp::NumericVector pseudo_r2(m);

    Rcpp::List beta(m);
    Rcpp::List theta(m);
    Rcpp::List eta(m);
    Rcpp::List mu(m);
    Rcpp::List criteria;

    for (R_xlen_t i = 0; i < m; ++i) {
      const EcountgmifsState& state =
        states[static_cast<std::size_t>(i)];

      iteration[i] =
        static_cast<double>(state.iteration);

      epsilon[i] =
        state.epsilon;

      dispersion[i] =
        state.dispersion;

      negloglik[i] =
        state.negloglik;

      saturated_dispersion[i] =
        state.saturated_dispersion;

      saturated_negloglik[i] =
        state.saturated_negloglik;

      pseudo_r2[i] =
        state.pseudo_r2;

      initialized[i] =
        state.initialized;

      beta[i] =
        state.beta;

      theta[i] =
        state.theta;

      eta[i] =
        state.eta;

      mu[i] =
        state.mu;
    }

    Rcpp::CharacterVector criterion_names;

    for (R_xlen_t i = 0; i < m; ++i) {
      const EcountgmifsState& state =
        states[static_cast<std::size_t>(i)];

      if (state.criteria.size() == 0) {
        continue;
      }

      if (state.criteria.hasAttribute("names")) {
        criterion_names =
          Rcpp::CharacterVector(state.criteria.attr("names"));
      }

      break;
    }

    const R_xlen_t n_criteria =
      criterion_names.size();

    criteria =
      Rcpp::List(n_criteria);

    Rcpp::CharacterVector state_names(m);

    for (R_xlen_t i = 0; i < m; ++i) {
      state_names[i] =
        "state_" + std::to_string(i);
    }

    for (R_xlen_t j = 0; j < n_criteria; ++j) {
      Rcpp::NumericVector criterion_values(m, NA_REAL);

      criterion_values.attr("names") =
        state_names;

      for (R_xlen_t i = 0; i < m; ++i) {
        const EcountgmifsState& state =
          states[static_cast<std::size_t>(i)];

        if (state.criteria.size() <= j) {
          continue;
        }

        criterion_values[i] =
          state.criteria[j];
      }

      criteria[j] =
        criterion_values;
    }

    criteria.attr("names") =
      criterion_names;

    return Rcpp::List::create(
      Rcpp::Named("iteration") = iteration,
      Rcpp::Named("epsilon") = epsilon,
      Rcpp::Named("dispersion") = dispersion,
      Rcpp::Named("negloglik") = negloglik,
      Rcpp::Named("saturated_dispersion") = saturated_dispersion,
      Rcpp::Named("saturated_negloglik") = saturated_negloglik,
      Rcpp::Named("pseudo_r2") = pseudo_r2,
      Rcpp::Named("initialized") = initialized,
      Rcpp::Named("beta") = beta,
      Rcpp::Named("theta") = theta,
      Rcpp::Named("eta") = eta,
      Rcpp::Named("mu") = mu,
      Rcpp::Named("criteria") = criteria
    );
  }

  Rcpp::List to_list() const
  {
    return Rcpp::List::create(
      Rcpp::Named("input") = input.to_list(control.api.include_data),
      Rcpp::Named("control") = control.to_list(),
      Rcpp::Named("message") = message,
      Rcpp::Named("states") = states_to_dynamic_tracking_list()
    );
  }
};


inline EcountgmifsContextInternal make_context(
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
    uint32_t iteration_max,

    uint32_t family,
    uint32_t link_func,

    double nlopt_optim_reltol,
    double loglik_reltol_cutoff,
    double nb_poisson_fallback_eps,
    bool verbose,
    bool fixed_dispersion,
    double fixed_dispersion_value,
    int state_track_strategy,
    uint64_t state_track_freq,
    bool include_data
) {
  return EcountgmifsContextInternal(
    EcountgmifsInputInternal(
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
      family,
      link_func,
      enet_alpha
    ),
    EcountgmifsControlInternal(
      static_cast<uint64_t>(iteration_max),
      epsilon_max,
      epsilon_start,
      epsilon_min,
      tol,
      nlopt_optim_reltol,
      loglik_reltol_cutoff,
      nb_poisson_fallback_eps,
      fixed_dispersion,
      fixed_dispersion_value,
      state_track_strategy,
      state_track_freq,
      verbose,
      include_data
    )
  );
}
