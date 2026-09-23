#include <RcppArmadillo.h>
#include <nloptrAPI.h>
// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::depends(nloptr)]]

#include "example.h" // TODO remove once examples are moved out of the fit TU.
#include <cstdint>
#include "context.h"

#include <string>
#include <unordered_set>

// Standalone access to the very same NB2 objective used by the fitting core.
// Keep this bridge in the existing translation unit: example.h also defines
// factory functions and must not be included in a second translation unit.
// [[Rcpp::export]]
double nb2_loglik_cpp(
    const arma::vec& y,
    const arma::vec& mu,
    const double dispersion,
    const double mu_min_cap,
    const double mu_max_cap,
    const double poisson_fallback_eps
)
{
  if (y.n_elem == 0 || y.n_elem != mu.n_elem ||
      !y.is_finite() || !mu.is_finite() ||
      arma::any(y < 0.0) || arma::any(mu < 0.0)) {
    Rcpp::stop("y and mu must be non-empty, non-negative finite vectors of equal length");
  }
  if (!std::isfinite(dispersion) || dispersion < 0.0) {
    Rcpp::stop("dispersion must be finite and non-negative");
  }

  ecountgmifs_examples::NB2Family family(
    mu_min_cap,
    mu_max_cap,
    poisson_fallback_eps,
    dispersion,
    0.0,
    std::numeric_limits<double>::infinity()
  );
  family.prepare_response(y);

  const arma::vec family_parameters = {dispersion};
  double negative_loglik = 0.0;
  family.negloglik(y, mu, family_parameters, negative_loglik);
  return -negative_loglik;
}

namespace
{

Rcpp::CharacterVector matrix_column_names(
    SEXP matrix_sexp,
    const char* fallback_prefix
)
{
  if (!Rf_isMatrix(matrix_sexp)) {
    Rcpp::stop("%s must be a matrix", fallback_prefix);
  }

  const Rcpp::RObject matrix(matrix_sexp);
  const Rcpp::IntegerVector dimensions =
    matrix.attr("dim");

  const R_xlen_t column_count =
    static_cast<R_xlen_t>(dimensions[1]);

  Rcpp::CharacterVector raw_names;

  SEXP dimnames_sexp = matrix.attr("dimnames");

  if (!Rf_isNull(dimnames_sexp)) {
    const Rcpp::List dimnames(dimnames_sexp);

    if (
        dimnames.size() >= 2 &&
        !Rf_isNull(dimnames[1])
    ) {
      raw_names = Rcpp::CharacterVector(dimnames[1]);
    }
  }

  Rcpp::CharacterVector result(column_count);
  std::unordered_set<std::string> used_names;

  for (R_xlen_t i = 0; i < column_count; ++i) {
    std::string base_name;

    if (i < raw_names.size()) {
      SEXP raw_name = raw_names[i];

      if (raw_name != NA_STRING) {
        base_name = Rcpp::as<std::string>(raw_name);
      }
    }

    if (base_name.empty()) {
      base_name =
        std::string(fallback_prefix) +
        std::to_string(i + 1);
    }

    std::string unique_name = base_name;
    std::size_t suffix = 1;

    while (used_names.find(unique_name) != used_names.end()) {
      unique_name =
        base_name + "." +
        std::to_string(suffix);
      ++suffix;
    }

    used_names.insert(unique_name);
    result[i] = unique_name;
  }

  return result;
}


void set_vector_names(
    Rcpp::List& owner,
    const char* field,
    const Rcpp::CharacterVector& names
)
{
  if (
      !owner.containsElementNamed(field) ||
      Rf_isNull(owner[field])
  ) {
    return;
  }

  Rcpp::RObject value = owner[field];

  if (Rf_xlength(value) != names.size()) {
    return;
  }

  value.attr("names") = Rcpp::clone(names);
  owner[field] = value;
}


Rcpp::List name_state_parameters(
    Rcpp::List state,
    const Rcpp::CharacterVector& predictor_names,
    const Rcpp::CharacterVector& unpenalized_names
)
{
  if (
      !state.containsElementNamed("predictors") ||
      Rf_isNull(state["predictors"])
  ) {
    return state;
  }

  Rcpp::List predictors = state["predictors"];

  if (
      predictors.containsElementNamed("parameters") &&
      !Rf_isNull(predictors["parameters"])
  ) {
    Rcpp::List parameters = predictors["parameters"];

    set_vector_names(
      parameters,
      "beta",
      predictor_names
    );

    set_vector_names(
      parameters,
      "theta",
      unpenalized_names
    );

    predictors["parameters"] = parameters;
  }

  set_vector_names(
    predictors,
    "active_set",
    predictor_names
  );

  state["predictors"] = predictors;
  return state;
}


void name_vector_list(
    Rcpp::List& owner,
    const char* field,
    const Rcpp::CharacterVector& names
)
{
  if (
      !owner.containsElementNamed(field) ||
      Rf_isNull(owner[field])
  ) {
    return;
  }

  Rcpp::List values = owner[field];

  for (R_xlen_t i = 0; i < values.size(); ++i) {
    if (Rf_isNull(values[i])) {
      continue;
    }

    Rcpp::RObject value = values[i];

    if (Rf_xlength(value) == names.size()) {
      value.attr("names") = Rcpp::clone(names);
      values[i] = value;
    }
  }

  owner[field] = values;
}


void decorate_output_parameter_names(
    Rcpp::List& output,
    const Rcpp::CharacterVector& predictor_names,
    const Rcpp::CharacterVector& unpenalized_names
)
{
  if (output.containsElementNamed("input")) {
    Rcpp::List input = output["input"];

    input["predictor_names"] =
      Rcpp::clone(predictor_names);

    input["unpenalized_names"] =
      Rcpp::clone(unpenalized_names);

    set_vector_names(
      input,
      "weight_vec",
      predictor_names
    );

    if (
        input.containsElementNamed("X") &&
        !Rf_isNull(input["X"])
    ) {
      Rcpp::NumericMatrix X = input["X"];
      Rcpp::colnames(X) = predictor_names;
      input["X"] = X;
    }

    if (
        input.containsElementNamed("w") &&
        !Rf_isNull(input["w"])
    ) {
      Rcpp::NumericMatrix w = input["w"];
      Rcpp::colnames(w) = unpenalized_names;
      input["w"] = w;
    }

    output["input"] = input;
  }

  if (
      output.containsElementNamed("terminal_state") &&
      !Rf_isNull(output["terminal_state"])
  ) {
    output["terminal_state"] =
      name_state_parameters(
        Rcpp::List(output["terminal_state"]),
        predictor_names,
        unpenalized_names
      );
  }

  if (
      !output.containsElementNamed("path") ||
      Rf_isNull(output["path"])
  ) {
    return;
  }

  Rcpp::List path = output["path"];

  set_vector_names(
    path,
    "null_theta",
    unpenalized_names
  );

  set_vector_names(
    path,
    "last_saved_active_set",
    predictor_names
  );

  if (
      path.containsElementNamed("states") &&
      !Rf_isNull(path["states"])
  ) {
    Rcpp::List states = path["states"];

    name_vector_list(
      states,
      "beta",
      predictor_names
    );

    name_vector_list(
      states,
      "theta",
      unpenalized_names
    );

    name_vector_list(
      states,
      "active_set",
      predictor_names
    );

    path["states"] = states;
  }

  if (
      path.containsElementNamed("best_criteria") &&
      !Rf_isNull(path["best_criteria"])
  ) {
    Rcpp::List best_criteria = path["best_criteria"];

    for (R_xlen_t i = 0; i < best_criteria.size(); ++i) {
      if (Rf_isNull(best_criteria[i])) {
        continue;
      }

      Rcpp::List best = best_criteria[i];

      if (
          best.containsElementNamed("state") &&
          !Rf_isNull(best["state"])
      ) {
        best["state"] =
          name_state_parameters(
            Rcpp::List(best["state"]),
            predictor_names,
            unpenalized_names
          );
      }

      best_criteria[i] = best;
    }

    path["best_criteria"] = best_criteria;
  }

  output["path"] = path;
}

} // anonymous namespace


// [[Rcpp::export]]
Rcpp::List ecountgmifs_cpp(
    SEXP X,
    arma::vec y,
    SEXP w,
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

  const Rcpp::CharacterVector predictor_names =
    matrix_column_names(
      X,
      "X"
    );

  const Rcpp::CharacterVector unpenalized_names =
    matrix_column_names(
      w,
      "theta"
    );

  const arma::mat X_arma =
    Rcpp::as<arma::mat>(X);

  const arma::mat w_arma =
    Rcpp::as<arma::mat>(w);

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
      X_arma,
      y,
      w_arma,
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

  decorate_output_parameter_names(
    output,
    predictor_names,
    unpenalized_names
  );

  output.attr("class") =
    Rcpp::CharacterVector::create(
      "ecountgmifs",
      "list"
    );

  return output;
}
