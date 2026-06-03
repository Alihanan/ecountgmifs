#pragma once

#include <RcppArmadillo.h>
#include <cstdint>

#include "enums.h"
#include "criteria.h"

struct EcountgmifsInput {
  const arma::mat& X;
  const arma::vec& y;
  const arma::mat& w;
  const arma::vec& offset;
  const arma::vec& weight_vec;

  const arma::mat& Xtest;
  const arma::vec& ytest;
  const arma::mat& wtest;
  const arma::vec& offsettest;
  const arma::vec& yorig;

  const arma::vec train_y_one_lgamma;
  const arma::vec test_y_one_lgamma;
  const arma::vec orig_y_one_lgamma;

  EnumFamily family;
  EnumLinkFunc link_func;
  double enet_alpha;

  EcountgmifsInput(
    const arma::mat& X_,
    const arma::vec& y_,
    const arma::mat& w_,
    const arma::vec& offset_,
    const arma::vec& weight_vec_,
    const arma::vec& yorig_,
    const arma::mat& Xtest_,
    const arma::vec& ytest_,
    const arma::mat& wtest_,
    const arma::vec& offsettest_,
    uint32_t family_int_,
    uint32_t linkfunc_int_,
    double enet_alpha_
  ) :
    X(X_),
    y(y_),
    w(w_),
    offset(offset_),
    weight_vec(weight_vec_),
    Xtest(Xtest_),
    ytest(ytest_),
    wtest(wtest_),
    offsettest(offsettest_),
    yorig(yorig_),
    train_y_one_lgamma(-1.0 * arma::lgamma(y_ + 1.0)),
    test_y_one_lgamma(-1.0 * arma::lgamma(ytest_ + 1.0)),
    orig_y_one_lgamma(-1.0 * arma::lgamma(yorig_ + 1.0)),
    family(as_family(family_int_)),
    link_func(as_link_func(linkfunc_int_)),
    enet_alpha(enet_alpha_)
  {}


  Rcpp::List to_list(bool include_data = false) const
  {
    Rcpp::List out = Rcpp::List::create(
      Rcpp::Named("n") = X.n_rows,
      Rcpp::Named("p") = X.n_cols,
      Rcpp::Named("q") = w.n_cols,
      Rcpp::Named("n_test") = Xtest.n_rows,
      Rcpp::Named("p_test") = Xtest.n_cols,
      Rcpp::Named("q_test") = wtest.n_cols,
      Rcpp::Named("enet_alpha") = enet_alpha
    );

    if (include_data) {
      out["X"] = X;
      out["y"] = y;
      out["w"] = w;
      out["offset"] = offset;
      out["weight_vec"] = weight_vec;

      out["Xtest"] = Xtest;
      out["ytest"] = ytest;
      out["wtest"] = wtest;
      out["offsettest"] = offsettest;
      out["yorig"] = yorig;
    }

    return out;
  }
};
struct EcountgmifsControl {
  uint64_t iteration_max;
  double epsilon_max;
  double epsilon_start;
  double epsilon_min;
  double tol;
  double nlopt_optim_reltol;
  double loglik_reltol_cutoff;
  bool fixed_dispersion;
  double fixed_dispersion_value;
  EnumStateTrackStrategy state_track_strategy;
  uint64_t state_track_freq;
  bool verbose;

  Rcpp::List to_list() const
  {
    return Rcpp::List::create(
      Rcpp::Named("iteration_max") = iteration_max,
      Rcpp::Named("epsilon_max") = epsilon_max,
      Rcpp::Named("epsilon_start") = epsilon_start,
      Rcpp::Named("epsilon_min") = epsilon_min,
      Rcpp::Named("tol") = tol,
      Rcpp::Named("nlopt_optim_reltol") = nlopt_optim_reltol,
      Rcpp::Named("loglik_reltol_cutoff") = loglik_reltol_cutoff,
      Rcpp::Named("fixed_dispersion") = fixed_dispersion,
      Rcpp::Named("fixed_dispersion_value") = fixed_dispersion_value,
      Rcpp::Named("state_track_freq") = state_track_freq,
      Rcpp::Named("verbose") = verbose
    );
  }
};

struct EcountgmifsState
{
  arma::vec beta;
  arma::vec theta;

  double dispersion;
  double loglik;
  uint64_t iteration;
  double epsilon;

  bool initialized;

  EcountgmifsState(
    arma::uword p,
    arma::uword q,
    double epsilon_start
  ) :
    beta(p, arma::fill::zeros),
    theta(q, arma::fill::zeros),
    dispersion(0.0),
    loglik(0.0),
    iteration(0),
    epsilon(epsilon_start),
    initialized(false)
  {}

  Rcpp::List to_list() const
  {
    return Rcpp::List::create(
      Rcpp::Named("iteration") = iteration,
      Rcpp::Named("beta") = beta,
      Rcpp::Named("theta") = theta,
      Rcpp::Named("dispersion") = dispersion,
      Rcpp::Named("loglik") = loglik,
      Rcpp::Named("epsilon") = epsilon,
      Rcpp::Named("initialized") = initialized
    );
  }
};

struct EcountgmifsContext {
  EcountgmifsInput input;
  EcountgmifsControl control;
  EcountgmifsState state;

  EcountgmifsContext(
    const EcountgmifsInput& input_,
    const EcountgmifsControl& control_
  ) :
    input(input_),
    control(control_),
    state(
      input_.X.n_cols,
      input_.w.n_cols,
      control_.epsilon_start
    )
  {}
};

inline EcountgmifsControl make_control(
    uint64_t iteration_max,
    double epsilon_max,
    double epsilon_start,
    double epsilon_min_tol,
    double tol,
    double nlopt_optim_reltol,
    double loglik_reltol_cutoff,
    bool is_fixed_disp,
    double fixed_disp_value,
    int state_track_strategy,
    uint64_t state_track_freq,
    bool verbose
) {
  return EcountgmifsControl {
    iteration_max,
    epsilon_max,
    epsilon_start,
    epsilon_min_tol,
    tol,
    nlopt_optim_reltol,
    loglik_reltol_cutoff,
    is_fixed_disp,
    fixed_disp_value,
    as_track_strategy(state_track_strategy),
    static_cast<uint64_t>(state_track_freq),
    verbose
  };
}

struct EcountgmifsTrace
{
  std::vector<uint64_t> iteration;
  std::vector<double> loglik;
  std::vector<double> dispersion;
  std::vector<double> epsilon;

  std::vector<double> enet_norm;
  std::vector<double> nnz;
  std::vector<double> df;
  std::vector<double> active_size;

  std::vector<arma::vec> beta;
  std::vector<arma::vec> theta;

  bool save_beta;
  bool save_theta;

  EcountgmifsTrace(
    bool save_beta_ = true,
    bool save_theta_ = true
  ) :
    save_beta(save_beta_),
    save_theta(save_theta_)
  {}

  void reserve(uint64_t n)
  {
    iteration.reserve(n);
    loglik.reserve(n);
    dispersion.reserve(n);
    epsilon.reserve(n);

    enet_norm.reserve(n);
    nnz.reserve(n);
    df.reserve(n);
    active_size.reserve(n);

    if (save_beta) beta.reserve(n);
    if (save_theta) theta.reserve(n);
  }

  void append(
      const EcountgmifsContext& ctx,
      double enet_norm_,
      double nnz_,
      double df_,
      double active_size_
  ) {
    iteration.push_back(ctx.state.iteration);
    loglik.push_back(ctx.state.loglik);
    dispersion.push_back(ctx.state.dispersion);
    epsilon.push_back(ctx.state.epsilon);

    enet_norm.push_back(enet_norm_);
    nnz.push_back(nnz_);
    df.push_back(df_);
    active_size.push_back(active_size_);

    if (save_beta) beta.push_back(ctx.state.beta);
    if (save_theta) theta.push_back(ctx.state.theta);
  }

  Rcpp::List to_list() const
  {
    Rcpp::List out = Rcpp::List::create(
      Rcpp::Named("iteration") = Rcpp::wrap(iteration),
      Rcpp::Named("loglik") = Rcpp::wrap(loglik),
      Rcpp::Named("dispersion") = Rcpp::wrap(dispersion),
      Rcpp::Named("epsilon") = Rcpp::wrap(epsilon),
      Rcpp::Named("enet_norm") = Rcpp::wrap(enet_norm),
      Rcpp::Named("nnz") = Rcpp::wrap(nnz),
      Rcpp::Named("df") = Rcpp::wrap(df),
      Rcpp::Named("active_size") = Rcpp::wrap(active_size)
    );

    if (save_beta) {
      Rcpp::List beta_out(beta.size());

      for (std::size_t i = 0; i < beta.size(); ++i) {
        beta_out[i] = beta[i];
      }

      out["beta"] = beta_out;
    }

    if (save_theta) {
      Rcpp::List theta_out(theta.size());

      for (std::size_t i = 0; i < theta.size(); ++i) {
        theta_out[i] = theta[i];
      }

      out["theta"] = theta_out;
    }

    return out;
  }
};

inline EcountgmifsContext make_context(
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
    double epsilon_min_tol,
    double tol,
    uint32_t iteration_max,

    uint32_t family,
    uint32_t linkfunc_int,

    double nlopt_optim_reltol,
    double loglik_reltol_cutoff,
    bool verbose,
    bool is_fixed_disp,
    double fixed_disp_value,
    int state_track_strategy,
    uint64_t state_track_freq
) {
  EcountgmifsControl control = make_control(
    static_cast<uint64_t>(iteration_max),
    epsilon_max,
    epsilon_start,
    epsilon_min_tol,
    tol,
    nlopt_optim_reltol,
    loglik_reltol_cutoff,
    is_fixed_disp,
    fixed_disp_value,
    state_track_strategy,
    state_track_freq,
    verbose
  );

  EcountgmifsInput input(
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
      linkfunc_int,
      enet_alpha
  );

  return EcountgmifsContext(
    input,
    control
  );
}
