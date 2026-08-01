#include <RcppArmadillo.h>
#include <nloptrAPI.h>
// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::depends(nloptr)]]

#include "example.h" // TODO remove once examples are moved out of the fit TU.
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
    double epsilon_min,

    uint32_t null_iteration_max,
    uint32_t stagewise_iteration_max,
    double null_family_parameter_abs_tol,
    double stagewise_objective_rel_tol,
    double stagewise_beta_step_norm_tol,

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

    int null_nonpen_nlopt_algorithm,
    double null_nonpen_nlopt_xtol_rel,
    double null_nonpen_nlopt_ftol_rel,
    int null_nonpen_nlopt_maxeval,

    int null_family_nlopt_algorithm,
    double null_family_nlopt_xtol_rel,
    double null_family_nlopt_ftol_rel,
    int null_family_nlopt_maxeval,

    int null_link_nlopt_algorithm,
    double null_link_nlopt_xtol_rel,
    double null_link_nlopt_ftol_rel,
    int null_link_nlopt_maxeval,

    int saturated_family_nlopt_algorithm,
    double saturated_family_nlopt_xtol_rel,
    double saturated_family_nlopt_ftol_rel,
    int saturated_family_nlopt_maxeval,

    int stagewise_nonpen_nlopt_algorithm,
    double stagewise_nonpen_nlopt_xtol_rel,
    double stagewise_nonpen_nlopt_ftol_rel,
    int stagewise_nonpen_nlopt_maxeval,

    int stagewise_family_nlopt_algorithm,
    double stagewise_family_nlopt_xtol_rel,
    double stagewise_family_nlopt_ftol_rel,
    int stagewise_family_nlopt_maxeval,

    int stagewise_link_nlopt_algorithm,
    double stagewise_link_nlopt_xtol_rel,
    double stagewise_link_nlopt_ftol_rel,
    int stagewise_link_nlopt_maxeval,

    SEXP family_link = R_NilValue
) {

  const auto same_nlopt_settings = [](
      int algorithm_a,
      double xtol_rel_a,
      double ftol_rel_a,
      int maxeval_a,
      int algorithm_b,
      double xtol_rel_b,
      double ftol_rel_b,
      int maxeval_b
  ) noexcept
  {
    return
      algorithm_a == algorithm_b &&
      xtol_rel_a == xtol_rel_b &&
      ftol_rel_a == ftol_rel_b &&
      maxeval_a == maxeval_b;
  };

  if (
      !same_nlopt_settings(
        null_nonpen_nlopt_algorithm,
        null_nonpen_nlopt_xtol_rel,
        null_nonpen_nlopt_ftol_rel,
        null_nonpen_nlopt_maxeval,
        stagewise_nonpen_nlopt_algorithm,
        stagewise_nonpen_nlopt_xtol_rel,
        stagewise_nonpen_nlopt_ftol_rel,
        stagewise_nonpen_nlopt_maxeval
      )
  ) {
    Rcpp::stop(
      "nonpen NLopt settings must be identical across fitting phases"
    );
  }

  if (
      !same_nlopt_settings(
        null_family_nlopt_algorithm,
        null_family_nlopt_xtol_rel,
        null_family_nlopt_ftol_rel,
        null_family_nlopt_maxeval,
        saturated_family_nlopt_algorithm,
        saturated_family_nlopt_xtol_rel,
        saturated_family_nlopt_ftol_rel,
        saturated_family_nlopt_maxeval
      ) ||
      !same_nlopt_settings(
        null_family_nlopt_algorithm,
        null_family_nlopt_xtol_rel,
        null_family_nlopt_ftol_rel,
        null_family_nlopt_maxeval,
        stagewise_family_nlopt_algorithm,
        stagewise_family_nlopt_xtol_rel,
        stagewise_family_nlopt_ftol_rel,
        stagewise_family_nlopt_maxeval
      )
  ) {
    Rcpp::stop(
      "family NLopt settings must be identical across fitting phases"
    );
  }

  if (
      !same_nlopt_settings(
        null_link_nlopt_algorithm,
        null_link_nlopt_xtol_rel,
        null_link_nlopt_ftol_rel,
        null_link_nlopt_maxeval,
        stagewise_link_nlopt_algorithm,
        stagewise_link_nlopt_xtol_rel,
        stagewise_link_nlopt_ftol_rel,
        stagewise_link_nlopt_maxeval
      )
  ) {
    Rcpp::stop(
      "link NLopt settings must be identical across fitting phases"
    );
  }

  const auto as_nlopt_algorithm =
    [](
        int algorithm,
        const char* name
    )
    {
      if (
          algorithm < 0 ||
          algorithm >=
            static_cast<int>(
              NLOPT_NUM_ALGORITHMS
            )
      ) {
        Rcpp::stop(
          "%s is not a valid NLopt algorithm",
          name
        );
      }

      return static_cast<nlopt_algorithm>(
        algorithm
      );
    };

  const EcountgmifsNloptControl nonpen_nlopt {
    as_nlopt_algorithm(
      null_nonpen_nlopt_algorithm,
      "nonpen_nlopt_algorithm"
    ),
    null_nonpen_nlopt_xtol_rel,
    null_nonpen_nlopt_ftol_rel,
    null_nonpen_nlopt_maxeval
  };

  const EcountgmifsNloptControl family_nlopt {
    as_nlopt_algorithm(
      null_family_nlopt_algorithm,
      "family_nlopt_algorithm"
    ),
    null_family_nlopt_xtol_rel,
    null_family_nlopt_ftol_rel,
    null_family_nlopt_maxeval
  };

  const EcountgmifsNloptControl link_nlopt {
    as_nlopt_algorithm(
      null_link_nlopt_algorithm,
      "link_nlopt_algorithm"
    ),
    null_link_nlopt_xtol_rel,
    null_link_nlopt_ftol_rel,
    null_link_nlopt_maxeval
  };

  EcountgmifsContextInternal ctx(
      X,
      y,
      w,
      offset,

      weight_vec,
      enet_alpha,
      epsilon_start,
      epsilon_max,
      epsilon_min,
      null_iteration_max,
      stagewise_iteration_max,
      null_family_parameter_abs_tol,
      stagewise_objective_rel_tol,
      stagewise_beta_step_norm_tol,

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

      nonpen_nlopt,
      family_nlopt,
      link_nlopt,
      family_link
  );

  ctx.stagewise.fit();

  Rcpp::List output = ctx.to_list();

  output.attr("class") =
    Rcpp::CharacterVector::create(
      "ecountgmifs",
      "list"
    );

  return output;
}
