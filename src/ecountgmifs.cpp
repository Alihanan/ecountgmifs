#include <RcppArmadillo.h>
#include <nloptrAPI.h>
// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::depends(nloptr)]]



#include <vector>
#include <string>
#include <cstdint>
#include <cmath>
#include <algorithm>
#include <iomanip>
#include <time.h>
#include <limits>

#include "context.h"
#include "nonpen_fit.h"
#include "stagewise.h"
#include "debug.h"

// [[Rcpp::export]]
Rcpp::List ecountgmifs_cpp(
    arma::mat X,
    arma::vec y,
    arma::mat w,
    arma::vec offset,

    arma::vec yorig,
    arma::mat Xtest,
    arma::vec ytest,
    arma::mat wtest,
    arma::vec offsettest,

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
    bool verbose = false,
    bool is_fixed_disp = false,
    double fixed_disp_value = 0.0,
    bool include_data = false,
    int state_track_strategy = 0,
    uint64_t state_track_freq = 1,

    Rcpp::Nullable<Rcpp::List> criteria = R_NilValue
) {

  ECOUNTGMIFS_VERBOSE(verbose,
    "input dimensions: n=" << X.n_rows
                           << ", p=" << X.n_cols
                           << ", q=" << w.n_cols
                           << ", ntest=" << Xtest.n_rows
  );

  ECOUNTGMIFS_VERBOSE(verbose,
    "controls: iteration_max=" << iteration_max
                               << ", epsilon_start=" << epsilon_start
                               << ", epsilon_max=" << epsilon_max
                               << ", epsilon_min=" << epsilon_min_tol
                               << ", tol=" << tol
                               << ", nlopt_reltol=" << nlopt_optim_reltol
  );

  // #########################################################
  //  Step 1: Save all arguments into a single context struct
  // #########################################################
  EcountgmifsContextInternal ctx = make_context(
    X,
    y,
    w,
    offset,
    yorig,
    Xtest,
    ytest,
    wtest,
    offsettest,
    weight_vec,
    enet_alpha,
    epsilon_start,
    epsilon_max,
    epsilon_min_tol,
    tol,
    iteration_max,
    family,
    linkfunc_int,
    nlopt_optim_reltol,
    loglik_reltol_cutoff,
    verbose,
    is_fixed_disp,
    fixed_disp_value,
    state_track_strategy,
    state_track_freq,
    include_data
  );

  ECOUNTGMIFS_VERBOSE(verbose,
    "context created: beta_n=" << ctx.state.api.beta.n_elem
                               << ", theta_n=" << ctx.state.api.theta.n_elem
  );

  ECOUNTGMIFS_VERBOSE(verbose,
    "initial state: negloglik=" << ctx.state.api.negloglik
                                << ", dispersion=" << ctx.state.api.dispersion
                                << ", epsilon=" << ctx.state.api.epsilon
  );

  // #########################################################
  //  Step 2: Estimate intercept/nonpen-only and saturated model
  // #########################################################
  NonpenNlopters opt(ctx); // non-penalized parameter optimizator

  ECOUNTGMIFS_VERBOSE(verbose,"fitting nonpenalized model");

  fit_nonpen(ctx, opt, ctx.control.api.iteration_max);

  ECOUNTGMIFS_VERBOSE(verbose,
    "nonpenalized fit done: initialized=" << ctx.state.api.initialized
                                          << ", negloglik=" << ctx.state.api.negloglik
                                          << ", dispersion=" << ctx.state.api.dispersion
                                          << ", theta_norm=" << arma::norm(ctx.state.api.theta, 2)
  );

  ECOUNTGMIFS_VERBOSE(verbose,"fitting saturated model");
  fit_saturated(ctx, opt);
  ECOUNTGMIFS_VERBOSE(verbose,
    "saturated fit done: saturated_negloglik="
    << ctx.state.api.saturated_negloglik
    << ", saturated_dispersion="
    << ctx.state.api.saturated_dispersion
  );


  Rcpp::List criteria_list;

  if (criteria.isNull()) {
    criteria_list = Rcpp::List::create();
  } else {
    criteria_list = Rcpp::List(criteria);
  }

  // #########################################################
  //  Step 3: Run the main Forward-Stagewise loop
  // #########################################################
  ctx.track_state(); // save first non-penalized state regardless

  fit_stagewise_path(ctx, opt);

  ctx.track_state(); // save last state regardless

  // #########################################################
  //  Step X: Construct the final output list
  // #########################################################
  /*
  Rcpp::List output =
    Rcpp::List::create(
      Rcpp::Named("model") = 123
      //Rcpp::Named("model") = modelOutputList,
      //Rcpp::Named("iterations") = step,
      //Rcpp::Named("message") = converge_message,
      //Rcpp::Named("tol") = tol,
      //Rcpp::Named("epsilon") = epsilon,
      //Rcpp::Named("max_iter") = max_iter,
      //Rcpp::Named("family") = family,

      // Rcpp::Named("L1_dynamic") = all_norms,
      //
      // //Rcpp::Named("beta_dynamic") = all_beta_list,
      // Rcpp::Named("beta_dynamic") = all_beta01_list,

      //Rcpp::Named("mut2_dynamic") = all_mut2_list,
      //Rcpp::Named("dmu_dy_dynamic") = all_dmu_dy_list,
      //Rcpp::Named("ddl_dy_dynamic") = all_ddl_dy_list,
      //Rcpp::Named("ugrad_dynamic") = all_ugrad_list,
      //Rcpp::Named("dgamma_dy_dynamic") = all_dgamma_dy_list,

      // Rcpp::Named("alpha_dynamic") = all_alpha,
      // Rcpp::Named("gamma_dynamic") = all_gamma,
      // Rcpp::Named("loglik_dynamic") = all_loglik,
      // Rcpp::Named("loglikorig_dynamic") = all_loglikorig,

      //Rcpp::Named("testloglik_dynamic") = all_testloglik,
      //Rcpp::Named("BIC_dynamic") = all_BICorig,
      //Rcpp::Named("AIC_dynamic") = all_AICorig,
      // Rcpp::Named("OLS_loglik") = OLS_loglik,
      // Rcpp::Named("EDF_dynamic") = all_edf,
      // Rcpp::Named("LOO_dynamic") = all_loo

  //Rcpp::Named("X") = X,
  //Rcpp::Named("y") = y//,
  //Rcpp::Named("w") = w,
  //Rcpp::Named("offset") = offset
    );
   */

  Rcpp::List output =
    Rcpp::List::create(
      Rcpp::Named("model") = 123,
      Rcpp::Named("beta") = ctx.state.api.beta,
      Rcpp::Named("theta") = ctx.state.api.theta,
      Rcpp::Named("dispersion") = ctx.state.api.dispersion,
      Rcpp::Named("negloglik") = ctx.state.api.negloglik,
      Rcpp::Named("dispersion.saturated") =
        ctx.state.api.saturated_dispersion,
      Rcpp::Named("negloglik.saturated") =
        ctx.state.api.saturated_negloglik,
      Rcpp::Named("iteration") = ctx.state.api.iteration,
      Rcpp::Named("epsilon") = ctx.state.api.epsilon,
      Rcpp::Named("initialized") = ctx.state.api.initialized,
      Rcpp::Named("context") = ctx.to_list()
    );


  return output;
}


