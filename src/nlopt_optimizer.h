#pragma once

#include <RcppArmadillo.h>
#include <nloptrAPI.h>

#include <limits>
#include <string>


struct NloptOptimizerInternal
{
  NloptOptimizerInternal(
    arma::vec& parameters_,
    const arma::vec& lower,
    const arma::vec& upper,
    nlopt_func objective,
    void* objective_data,
    int algorithm,
    double xtol_rel,
    double ftol_rel,
    int maxeval
  ) :
  parameters(parameters_)
  {
    if (
        lower.n_elem != parameters.n_elem ||
          upper.n_elem != parameters.n_elem
    ) {
      Rcpp::stop(
        "NLopt parameters and bounds have incompatible sizes"
      );
    }

    if (parameters.n_elem == 0) {
      return;
    }

    if (
        parameters.n_elem >
      std::numeric_limits<unsigned>::max()
    ) {
      Rcpp::stop("NLopt parameter count is too large");
    }

    opt = nlopt_create(
      static_cast<nlopt_algorithm>(algorithm),
      static_cast<unsigned>(parameters.n_elem)
    );

    if (opt == nullptr) {
      Rcpp::stop("failed to create NLopt optimizer");
    }

    check_result(
      nlopt_set_lower_bounds(
        opt,
        lower.memptr()
      ),
      "set lower bounds"
    );

    check_result(
      nlopt_set_upper_bounds(
        opt,
        upper.memptr()
      ),
      "set upper bounds"
    );

    check_result(
      nlopt_set_min_objective(
        opt,
        objective,
        objective_data
      ),
      "set objective"
    );

    if (xtol_rel > 0.0) {
      check_result(
        nlopt_set_xtol_rel(
          opt,
          xtol_rel
        ),
        "set relative parameter tolerance"
      );
    }

    if (ftol_rel > 0.0) {
      check_result(
        nlopt_set_ftol_rel(
          opt,
          ftol_rel
        ),
        "set relative objective tolerance"
      );
    }

    check_result(
      nlopt_set_maxeval(
        opt,
        maxeval
      ),
      "set maximum evaluations"
    );
  }


  ~NloptOptimizerInternal()
  {
    if (opt != nullptr) {
      nlopt_destroy(opt);
    }
  }


  NloptOptimizerInternal(
    const NloptOptimizerInternal&
  ) = delete;

  NloptOptimizerInternal(
    NloptOptimizerInternal&&
  ) = delete;

  NloptOptimizerInternal& operator=(
    const NloptOptimizerInternal&
  ) = delete;

  NloptOptimizerInternal& operator=(
    NloptOptimizerInternal&&
  ) = delete;


  void optimize()
  {
    if (opt == nullptr) {
      return;
    }

    double objective_value =
      arma::datum::nan;

    const nlopt_result result =
      nlopt_optimize(
        opt,
        parameters.memptr(),
        &objective_value
      );

    if (result < 0) {
      Rcpp::stop(
        "NLopt optimization failed with code %d",
        static_cast<int>(result)
      );
    }
  }


private:
  arma::vec& parameters;
  nlopt_opt opt = nullptr;


  static void check_result(
      nlopt_result result,
      const char* operation
  )
  {
    if (result < 0) {
      Rcpp::stop(
        std::string("failed to ") +
          operation +
          " in NLopt"
      );
    }
  }
};
