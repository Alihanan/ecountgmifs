#include <RcppArmadillo.h>
#include <nloptrAPI.h>
// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::depends(nloptr)]]

// #include <vector>
// #include <string>
// #include <cstdint>
// #include <cmath>
// #include <algorithm>
// #include <iomanip>
// #include <time.h>
// #include <limits>

// #include "context.h"
// #include "nonpen_fit.h"
// #include "stagewise.h"
// #include "debug.h"



#include "example.h" // TODO remove?
#include <cstdint>
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

    SEXP family,
    SEXP link_func,
    Rcpp::Nullable<Rcpp::List> criteria,

    double loglik_reltol_cutoff,
    double enet_abs_tol,
    double enet_rel_tol,
    uint32_t enet_max_iter,
    bool verbose,

    bool include_data,
    int state_track_strategy,
    uint64_t state_track_freq,

    const arma::vec& theta_initial,
    const arma::vec& theta_lower_bounds,
    const arma::vec& theta_upper_bounds,
    int nlopt_algorithm,
    double nlopt_xtol_rel,
    double nlopt_ftol_rel,
    int nlopt_maxeval
) {
  EcountgmifsContextInternal ctx(
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
      link_func,
      criteria,

      loglik_reltol_cutoff,
      enet_abs_tol,
      enet_rel_tol,
      enet_max_iter,

      verbose,
      state_track_strategy,
      state_track_freq,
      include_data,

      theta_initial,
      theta_lower_bounds,
      theta_upper_bounds,

      nlopt_algorithm,
      nlopt_xtol_rel,
      nlopt_ftol_rel,
      nlopt_maxeval
  );


  ctx.stagewise.fit();


  return Rcpp::List::create(
    Rcpp::Named("input") =
      ctx.input.to_list(
        ctx.control.api.include_data
      ),

      Rcpp::Named("control") =
        ctx.control.to_list(),

        Rcpp::Named("state") =
          ctx.path.current_state_to_list(),

          Rcpp::Named("path") =
            ctx.path.to_list(),

            Rcpp::Named("stagewise") =
              ctx.stagewise.to_list()
  );
}

/*
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


 SEXP family = R_NilValue,
 SEXP link_func = R_NilValue,
 Rcpp::Nullable<Rcpp::List> criteria = R_NilValue

 double nlopt_optim_reltol,
 double loglik_reltol_cutoff,
 double nb_poisson_fallback_eps,
 double enet_abs_tol,
 double enet_rel_tol,
 uint32_t enet_max_iter,
 bool verbose = false,
 bool is_fixed_disp = false,
 double fixed_disp_value = 0.0,
 bool include_data = false,
 int state_track_strategy = 0,
 uint64_t state_track_freq = 1,


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
                            << ", enet_abs_tol=" << enet_abs_tol
                            << ", enet_rel_tol=" << enet_rel_tol
                            << ", enet_max_iter=" << enet_max_iter
                            << ", nlopt_reltol=" << nlopt_optim_reltol
 );

 // #########################################################
 //  Step 0: Resolve family, link function, criteria
 // #########################################################
 Rcpp::XPtr<IEcountgmifsFamily> family_ptr(family);
 Rcpp::XPtr<IEcountgmifsLinkFunc> link_ptr(link_func);

 if (family_ptr.get() == nullptr) {
 Rcpp::stop("`family` contains a null external pointer.");
 }

 if (link_ptr.get() == nullptr) {
 Rcpp::stop("`link_func` contains a null external pointer.");
 }

 IEcountgmifsFamily& family_impl = *family_ptr;
 IEcountgmifsLinkFunc& link_impl = *link_ptr;

 ECOUNTGMIFS_VERBOSE(
 verbose,
 "plugins: family_parameter_count="
 << family_impl->parameter_count()
 << ", link_parameter_count="
 << link_impl->parameter_count()
 );

 Rcpp::List criteria_list;

 if (criteria.isNull()) {
 criteria_list = Rcpp::List::create();
 } else {
 criteria_list = Rcpp::List(criteria);
 }

 for (R_xlen_t i = 0; i < criteria_list.size(); ++i) {
 SEXP criterion_sexp = criteria_list[i];

 if (Rf_isNull(criterion_sexp)) {
 Rcpp::stop(
 "Criterion at position %d is NULL.",
 static_cast<int>(i + 1)
 );
 }

 Rcpp::XPtr<IEcountgmifsCriterion> criterion_ptr(
 criterion_sexp
 );

 if (criterion_ptr.get() == nullptr) {
 Rcpp::stop(
 "Criterion at position %d contains a null external pointer.",
 static_cast<int>(i + 1)
 );
 }
 }


 // #########################################################
 //  Step 1: Save all arguments into a single context struct
 // #########################################################
 EcountgmifsContextInternal ctx(
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
 criteria,

 nlopt_optim_reltol,
 loglik_reltol_cutoff,
 nb_poisson_fallback_eps,
 enet_abs_tol,
 enet_rel_tol,
 enet_max_iter,
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
 ctx.set_null_negloglik_from_current_state();
 ctx.set_message("running");

 Rcpp::List criteria_list;

 if (criteria.isNull()) {
 criteria_list = Rcpp::List::create();
 } else {
 criteria_list = Rcpp::List(criteria);
 }

 ctx.set_criteria(criteria_list);
 ctx.evaluate_criteria();

 // #########################################################
 //  Step 3: Run the main Forward-Stagewise loop
 // #########################################################
 ctx.track_state(); // save first non-penalized state regardless

 fit_stagewise_path(ctx, opt);

 ctx.evaluate_criteria();
 ctx.track_state(); // save last state regardless

 // #########################################################
 //  Step 4: Construct the final output list
 // #########################################################
 Rcpp::List output = ctx.to_list();
 output["criteria"] = ctx.state.api.criteria;

 return output;
 }

 */
