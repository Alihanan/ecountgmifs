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
    int nlopt_maxeval,
    SEXP family_link = R_NilValue
) {
  EcountgmifsContextInternal ctx(
      X,
      y,
      w,
      offset,

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
      nlopt_maxeval,
      family_link
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
