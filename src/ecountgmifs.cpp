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


#include "nlopt_optim_nonpen.h"
#include "context.h"


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
    int state_track_strategy = 0,
    uint64_t state_track_freq = 1,

    Rcpp::Nullable<Rcpp::List> criteria = R_NilValue
) {

  // #########################################################
  //  Step 1: Save all arguments into a single context struct
  // #########################################################
  EcountgmifsContext ctx = make_context(
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
    state_track_freq
  );

  // #########################################################
  //  Step 2: Estimate intercept/nonpen-only and
  // #########################################################
  fit_unpenalized_model_nlopt(ctx);

  double saturated_dispersion = 0.0;
  double saturated_loglik = fit_saturated_dispersion_nlopt(
    ctx,
    saturated_dispersion
  );


  Rcpp::List criteria_list;

  if (criteria.isNull()) {
    criteria_list = Rcpp::List::create();
  } else {
    criteria_list = Rcpp::List(criteria);
  }


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
      Rcpp::Named("model") = criteria_list,
      Rcpp::Named("theta") = ctx.state.theta,
      Rcpp::Named("dispersion") = ctx.state.dispersion,
      Rcpp::Named("loglik.unpenalized") = ctx.state.loglik,
      Rcpp::Named("dispersion.saturated") = saturated_dispersion,
      Rcpp::Named("loglik.saturated") = saturated_loglik,
      Rcpp::Named("initialized") = ctx.state.initialized
    );


  return output;
}





