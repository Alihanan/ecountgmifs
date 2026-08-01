#include <RcppArmadillo.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

#include "../inst/include/ecountgmifs/api.h"

namespace
{

void validate_parameter_metadata(
    const std::string& plugin_name,
    const arma::vec& initial,
    const arma::vec& lower,
    const arma::vec& upper
)
{
  if (
      initial.n_elem != lower.n_elem ||
      initial.n_elem != upper.n_elem
  ) {
    Rcpp::stop(
      "R plugin '%s': initial values and bounds must have equal lengths",
      plugin_name.c_str()
    );
  }

  for (arma::uword i = 0; i < initial.n_elem; ++i) {
    if (!std::isfinite(initial[i])) {
      Rcpp::stop(
        "R plugin '%s': initial parameter %d must be finite",
        plugin_name.c_str(),
        static_cast<int>(i + 1)
      );
    }

    if (std::isnan(lower[i]) || std::isnan(upper[i])) {
      Rcpp::stop(
        "R plugin '%s': parameter bounds must not be NaN",
        plugin_name.c_str()
      );
    }

    if (lower[i] > upper[i]) {
      Rcpp::stop(
        "R plugin '%s': lower bound exceeds upper bound at parameter %d",
        plugin_name.c_str(),
        static_cast<int>(i + 1)
      );
    }

    if (initial[i] < lower[i] || initial[i] > upper[i]) {
      Rcpp::stop(
        "R plugin '%s': initial parameter %d lies outside its bounds",
        plugin_name.c_str(),
        static_cast<int>(i + 1)
      );
    }
  }
}

void validate_parameter_count(
    const std::string& plugin_name,
    const char* parameter_type,
    const arma::vec& parameters,
    const arma::uword expected
)
{
  if (parameters.n_elem != expected) {
    Rcpp::stop(
      "R plugin '%s': expected %d %s parameters; received %d",
      plugin_name.c_str(),
      static_cast<int>(expected),
      parameter_type,
      static_cast<int>(parameters.n_elem)
    );
  }
}

Rcpp::List require_list_result(
    const SEXP result,
    const std::string& plugin_name,
    const char* callback_name
)
{
  if (TYPEOF(result) != VECSXP) {
    Rcpp::stop(
      "R plugin '%s': %s() must return a list",
      plugin_name.c_str(),
      callback_name
    );
  }

  return Rcpp::List(result);
}

SEXP require_named_element(
    const Rcpp::List& result,
    const std::string& plugin_name,
    const char* callback_name,
    const char* element_name
)
{
  if (!result.containsElementNamed(element_name)) {
    Rcpp::stop(
      "R plugin '%s': %s() did not return '%s'",
      plugin_name.c_str(),
      callback_name,
      element_name
    );
  }

  return result[element_name];
}

double require_scalar_result(
    const SEXP result,
    const std::string& plugin_name,
    const char* callback_name
)
{
  if (!Rf_isNumeric(result) || Rf_xlength(result) != 1) {
    Rcpp::stop(
      "R plugin '%s': %s() must return one numeric value",
      plugin_name.c_str(),
      callback_name
    );
  }

  const double value = Rcpp::as<double>(result);

  if (!std::isfinite(value)) {
    Rcpp::stop(
      "R plugin '%s': %s() returned a non-finite value",
      plugin_name.c_str(),
      callback_name
    );
  }

  return value;
}

void copy_vector_result(
    const SEXP result,
    arma::vec& destination,
    const arma::uword expected_length,
    const std::string& plugin_name,
    const char* callback_name,
    const char* result_name,
    const bool require_finite = true
)
{
  if (!Rf_isNumeric(result)) {
    Rcpp::stop(
      "R plugin '%s': %s() result '%s' must be numeric",
      plugin_name.c_str(),
      callback_name,
      result_name
    );
  }

  const Rcpp::NumericVector values(result);

  if (
      static_cast<arma::uword>(values.size()) !=
      expected_length
  ) {
    Rcpp::stop(
      "R plugin '%s': %s() result '%s' has length %d; expected %d",
      plugin_name.c_str(),
      callback_name,
      result_name,
      static_cast<int>(values.size()),
      static_cast<int>(expected_length)
    );
  }

  destination.set_size(expected_length);

  for (arma::uword i = 0; i < expected_length; ++i) {
    const double value = values[static_cast<R_xlen_t>(i)];

    if (
        require_finite &&
        !std::isfinite(value)
    ) {
      Rcpp::stop(
        "R plugin '%s': %s() result '%s' contains a non-finite value",
        plugin_name.c_str(),
        callback_name,
        result_name
      );
    }

    destination[i] = value;
  }
}

void copy_matrix_result(
    const SEXP result,
    arma::mat& destination,
    const arma::uword expected_rows,
    const arma::uword expected_columns,
    const std::string& plugin_name,
    const char* callback_name,
    const char* result_name
)
{
  if (
      expected_columns == 0 &&
      Rf_isNull(result)
  ) {
    destination.set_size(expected_rows, 0);
    return;
  }

  if (!Rf_isMatrix(result) || !Rf_isNumeric(result)) {
    Rcpp::stop(
      "R plugin '%s': %s() result '%s' must be a numeric matrix",
      plugin_name.c_str(),
      callback_name,
      result_name
    );
  }

  const Rcpp::NumericMatrix values(result);

  if (
      static_cast<arma::uword>(values.nrow()) !=
        expected_rows ||
      static_cast<arma::uword>(values.ncol()) !=
        expected_columns
  ) {
    Rcpp::stop(
      "R plugin '%s': %s() result '%s' has dimensions %d x %d; expected %d x %d",
      plugin_name.c_str(),
      callback_name,
      result_name,
      values.nrow(),
      values.ncol(),
      static_cast<int>(expected_rows),
      static_cast<int>(expected_columns)
    );
  }

  destination.set_size(
    expected_rows,
    expected_columns
  );

  for (
      arma::uword column = 0;
      column < expected_columns;
      ++column
  ) {
    for (
        arma::uword row = 0;
        row < expected_rows;
        ++row
    ) {
      const double value = values(
        static_cast<int>(row),
        static_cast<int>(column)
      );

      if (!std::isfinite(value)) {
        Rcpp::stop(
          "R plugin '%s': %s() result '%s' contains a non-finite value",
          plugin_name.c_str(),
          callback_name,
          result_name
        );
      }

      destination(row, column) = value;
    }
  }
}

Rcpp::List input_to_list(
    const EcountgmifsInput& input
)
{
  return Rcpp::List::create(
    Rcpp::Named("X") = Rcpp::wrap(input.X),
    Rcpp::Named("y") = Rcpp::wrap(input.y),
    Rcpp::Named("w") = Rcpp::wrap(input.w),
    Rcpp::Named("offset") = Rcpp::wrap(input.offset),
    Rcpp::Named("weight_vec") = Rcpp::wrap(input.weight_vec),
    Rcpp::Named("n") = input.X.n_rows,
    Rcpp::Named("p") = input.X.n_cols,
    Rcpp::Named("q") = input.w.n_cols,
    Rcpp::Named("has_prior") = input.has_prior,
    Rcpp::Named("enet_alpha") = input.enet_alpha,
    Rcpp::Named("family_name") =
      input.family_link->family_name(),
    Rcpp::Named("link_name") =
      input.family_link->link_name(),
    Rcpp::Named("family_parameter_count") =
      input.family_link->family_parameter_count(),
    Rcpp::Named("link_parameter_count") =
      input.family_link->link_parameter_count()
  );
}

Rcpp::List nlopt_control_to_list(
    const EcountgmifsNloptControl& control
)
{
  return Rcpp::List::create(
    Rcpp::Named("algorithm") =
      static_cast<int>(control.algorithm),
    Rcpp::Named("xtol_rel") =
      control.xtol_rel,
    Rcpp::Named("ftol_rel") =
      control.ftol_rel,
    Rcpp::Named("maxeval") =
      control.maxeval
  );
}

Rcpp::List control_to_list(
    const EcountgmifsControl& control
)
{
  Rcpp::List result = Rcpp::List::create(
    Rcpp::Named("null_iteration_max") =
      control.null_iteration_max,
    Rcpp::Named("stagewise_iteration_max") =
      control.stagewise_iteration_max,
    Rcpp::Named("null_family_parameter_abs_tol") =
      control.null_family_parameter_abs_tol,
    Rcpp::Named("stagewise_objective_rel_tol") =
      control.stagewise_objective_rel_tol,
    Rcpp::Named("stagewise_beta_step_norm_tol") =
      control.stagewise_beta_step_norm_tol,
    Rcpp::Named("epsilon_max") =
      control.epsilon_max,
    Rcpp::Named("epsilon_start") =
      control.epsilon_start,
    Rcpp::Named("epsilon_min") =
      control.epsilon_min,
    Rcpp::Named("loglik_reltol_cutoff") =
      control.loglik_reltol_cutoff,
    Rcpp::Named("enet_abs_tol") =
      control.enet_abs_tol,
    Rcpp::Named("enet_rel_tol") =
      control.enet_rel_tol,
    Rcpp::Named("enet_max_iter") =
      control.enet_max_iter,
    Rcpp::Named("state_track_strategy") =
      static_cast<int>(control.state_track_strategy),
    Rcpp::Named("state_track_freq") =
      control.state_track_freq,
    Rcpp::Named("verbose") =
      control.verbose,
    Rcpp::Named("include_data") =
      control.include_data,
    Rcpp::Named("theta_initial") =
      Rcpp::wrap(control.theta_initial),
    Rcpp::Named("theta_lower_bounds") =
      Rcpp::wrap(control.theta_lower_bounds),
    Rcpp::Named("theta_upper_bounds") =
      Rcpp::wrap(control.theta_upper_bounds)
  );

  result.push_back(
    nlopt_control_to_list(control.nonpen_nlopt),
    "nonpen_nlopt"
  );

  result.push_back(
    nlopt_control_to_list(control.family_nlopt),
    "family_nlopt"
  );

  result.push_back(
    nlopt_control_to_list(control.link_nlopt),
    "link_nlopt"
  );

  return result;
}

Rcpp::List state_to_list(
    const EcountgmifsState& state
)
{
  const EcountgmifsPredictors& predictors =
    state.param;

  const EcountgmifsParameters& parameters =
    predictors.param;

  return Rcpp::List::create(
    Rcpp::Named("beta") =
      Rcpp::wrap(parameters.beta),
    Rcpp::Named("theta") =
      Rcpp::wrap(parameters.theta),
    Rcpp::Named("family_parameters") =
      Rcpp::wrap(parameters.family_parameters),
    Rcpp::Named("link_parameters") =
      Rcpp::wrap(parameters.link_parameters),
    Rcpp::Named("xbeta") =
      Rcpp::wrap(predictors.xbeta),
    Rcpp::Named("wtheta") =
      Rcpp::wrap(predictors.wtheta),
    Rcpp::Named("eta") =
      Rcpp::wrap(predictors.eta),
    Rcpp::Named("mu") =
      Rcpp::wrap(predictors.mu),
    Rcpp::Named("active_set") =
      Rcpp::wrap(predictors.active_set),
    Rcpp::Named("negloglik") =
      state.negloglik,
    Rcpp::Named("criteria") =
      Rcpp::clone(state.criteria),
    Rcpp::Named("iteration") =
      state.iteration,
    Rcpp::Named("pseudo_r2") =
      state.pseudo_r2,
    Rcpp::Named("elapsed_time") =
      state.elapsed_time
  );
}

class RFunctionLink final :
  public IEcountgmifsLinkFunc
{
private:
  std::string name_;
  Rcpp::Function prepare_;
  Rcpp::Function inverse_;
  Rcpp::Function grad_;
  arma::vec initial_parameters_;
  arma::vec lower_bounds_;
  arma::vec upper_bounds_;
  Rcpp::Environment environment_;

public:
  RFunctionLink(
      std::string name,
      Rcpp::Function prepare,
      Rcpp::Function inverse,
      Rcpp::Function grad,
      arma::vec initial_parameters,
      arma::vec lower_bounds,
      arma::vec upper_bounds,
      Rcpp::Environment environment
  ) :
    name_(std::move(name)),
    prepare_(prepare),
    inverse_(inverse),
    grad_(grad),
    initial_parameters_(
      std::move(initial_parameters)
    ),
    lower_bounds_(std::move(lower_bounds)),
    upper_bounds_(std::move(upper_bounds)),
    environment_(environment)
  {
    validate_parameter_metadata(
      name_,
      initial_parameters_,
      lower_bounds_,
      upper_bounds_
    );
  }

  std::string name() const override
  {
    return name_;
  }

  arma::uword parameter_count()
    const noexcept override
  {
    return initial_parameters_.n_elem;
  }

  arma::vec initial_parameters() const override
  {
    return initial_parameters_;
  }

  arma::vec parameter_lower_bounds()
    const override
  {
    return lower_bounds_;
  }

  arma::vec parameter_upper_bounds()
    const override
  {
    return upper_bounds_;
  }

  void prepare(
      const EcountgmifsInput& input,
      const EcountgmifsControl& control
  ) const override
  {
    prepare_(
      input_to_list(input),
      control_to_list(control),
      environment_
    );
  }

  void inverse(
      const arma::vec& eta,
      const arma::vec& link_parameters,
      arma::vec& mu
  ) const override
  {
    validate_parameter_count(
      name_,
      "link",
      link_parameters,
      parameter_count()
    );

    copy_vector_result(
      inverse_(
        Rcpp::wrap(eta),
        Rcpp::wrap(link_parameters),
        environment_
      ),
      mu,
      eta.n_elem,
      name_,
      "inverse",
      "mu"
    );
  }

  void grad(
      const arma::vec& eta,
      const arma::vec& link_parameters,
      arma::vec& d_mu_d_eta,
      arma::mat& d_mu_d_link_parameters
  ) const override
  {
    validate_parameter_count(
      name_,
      "link",
      link_parameters,
      parameter_count()
    );

    const Rcpp::List result =
      require_list_result(
        grad_(
          Rcpp::wrap(eta),
          Rcpp::wrap(link_parameters),
          environment_
        ),
        name_,
        "grad"
      );

    copy_vector_result(
      require_named_element(
        result,
        name_,
        "grad",
        "d_mu_d_eta"
      ),
      d_mu_d_eta,
      eta.n_elem,
      name_,
      "grad",
      "d_mu_d_eta"
    );

    copy_matrix_result(
      require_named_element(
        result,
        name_,
        "grad",
        "d_mu_d_link_parameters"
      ),
      d_mu_d_link_parameters,
      eta.n_elem,
      parameter_count(),
      name_,
      "grad",
      "d_mu_d_link_parameters"
    );
  }
};

class RFunctionFamily final :
  public IEcountgmifsFamily
{
private:
  std::string name_;
  Rcpp::Function prepare_;
  Rcpp::Function negloglik_;
  Rcpp::Function grad_;
  arma::vec initial_parameters_;
  arma::vec lower_bounds_;
  arma::vec upper_bounds_;
  Rcpp::Environment environment_;

public:
  RFunctionFamily(
      std::string name,
      Rcpp::Function prepare,
      Rcpp::Function negloglik,
      Rcpp::Function grad,
      arma::vec initial_parameters,
      arma::vec lower_bounds,
      arma::vec upper_bounds,
      Rcpp::Environment environment
  ) :
    name_(std::move(name)),
    prepare_(prepare),
    negloglik_(negloglik),
    grad_(grad),
    initial_parameters_(
      std::move(initial_parameters)
    ),
    lower_bounds_(std::move(lower_bounds)),
    upper_bounds_(std::move(upper_bounds)),
    environment_(environment)
  {
    validate_parameter_metadata(
      name_,
      initial_parameters_,
      lower_bounds_,
      upper_bounds_
    );
  }

  std::string name() const override
  {
    return name_;
  }

  arma::uword parameter_count()
    const noexcept override
  {
    return initial_parameters_.n_elem;
  }

  arma::vec initial_parameters() const override
  {
    return initial_parameters_;
  }

  arma::vec parameter_lower_bounds()
    const override
  {
    return lower_bounds_;
  }

  arma::vec parameter_upper_bounds()
    const override
  {
    return upper_bounds_;
  }

  void prepare(
      const EcountgmifsInput& input,
      const EcountgmifsControl& control
  ) const override
  {
    prepare_(
      input_to_list(input),
      control_to_list(control),
      environment_
    );
  }

  void negloglik(
      const arma::vec& y,
      const arma::vec& mu,
      const arma::vec& family_parameters,
      double& negloglik
  ) const override
  {
    if (y.n_elem != mu.n_elem) {
      Rcpp::stop(
        "R family '%s': y and mu lengths differ",
        name_.c_str()
      );
    }

    validate_parameter_count(
      name_,
      "family",
      family_parameters,
      parameter_count()
    );

    negloglik = require_scalar_result(
      negloglik_(
        Rcpp::wrap(y),
        Rcpp::wrap(mu),
        Rcpp::wrap(family_parameters),
        environment_
      ),
      name_,
      "negloglik"
    );
  }

  void grad(
      const arma::vec& y,
      const arma::vec& mu,
      const arma::vec& family_parameters,
      arma::vec& d_negloglik_d_mu,
      arma::vec& d_negloglik_d_family_parameters
  ) const override
  {
    if (y.n_elem != mu.n_elem) {
      Rcpp::stop(
        "R family '%s': y and mu lengths differ",
        name_.c_str()
      );
    }

    validate_parameter_count(
      name_,
      "family",
      family_parameters,
      parameter_count()
    );

    const Rcpp::List result =
      require_list_result(
        grad_(
          Rcpp::wrap(y),
          Rcpp::wrap(mu),
          Rcpp::wrap(family_parameters),
          environment_
        ),
        name_,
        "grad"
      );

    copy_vector_result(
      require_named_element(
        result,
        name_,
        "grad",
        "d_negloglik_d_mu"
      ),
      d_negloglik_d_mu,
      y.n_elem,
      name_,
      "grad",
      "d_negloglik_d_mu"
    );

    copy_vector_result(
      require_named_element(
        result,
        name_,
        "grad",
        "d_negloglik_d_family_parameters"
      ),
      d_negloglik_d_family_parameters,
      parameter_count(),
      name_,
      "grad",
      "d_negloglik_d_family_parameters"
    );
  }
};

class RFunctionFamilyLink final :
  public IEcountgmifsFamilyLink
{
private:
  std::string family_name_;
  std::string link_name_;
  Rcpp::Function prepare_;
  Rcpp::Function inverse_;
  Rcpp::Function negloglik_;
  Rcpp::Function grad_;
  arma::vec family_initial_parameters_;
  arma::vec family_lower_bounds_;
  arma::vec family_upper_bounds_;
  arma::vec link_initial_parameters_;
  arma::vec link_lower_bounds_;
  arma::vec link_upper_bounds_;
  Rcpp::Environment environment_;

public:
  RFunctionFamilyLink(
      std::string family_name,
      std::string link_name,
      Rcpp::Function prepare,
      Rcpp::Function inverse,
      Rcpp::Function negloglik,
      Rcpp::Function grad,
      arma::vec family_initial_parameters,
      arma::vec family_lower_bounds,
      arma::vec family_upper_bounds,
      arma::vec link_initial_parameters,
      arma::vec link_lower_bounds,
      arma::vec link_upper_bounds,
      Rcpp::Environment environment
  ) :
    family_name_(std::move(family_name)),
    link_name_(std::move(link_name)),
    prepare_(prepare),
    inverse_(inverse),
    negloglik_(negloglik),
    grad_(grad),
    family_initial_parameters_(
      std::move(family_initial_parameters)
    ),
    family_lower_bounds_(
      std::move(family_lower_bounds)
    ),
    family_upper_bounds_(
      std::move(family_upper_bounds)
    ),
    link_initial_parameters_(
      std::move(link_initial_parameters)
    ),
    link_lower_bounds_(
      std::move(link_lower_bounds)
    ),
    link_upper_bounds_(
      std::move(link_upper_bounds)
    ),
    environment_(environment)
  {
    validate_parameter_metadata(
      family_name_,
      family_initial_parameters_,
      family_lower_bounds_,
      family_upper_bounds_
    );

    validate_parameter_metadata(
      link_name_,
      link_initial_parameters_,
      link_lower_bounds_,
      link_upper_bounds_
    );
  }

  std::string family_name() const override
  {
    return family_name_;
  }

  std::string link_name() const override
  {
    return link_name_;
  }

  arma::uword family_parameter_count()
    const noexcept override
  {
    return family_initial_parameters_.n_elem;
  }

  arma::uword link_parameter_count()
    const noexcept override
  {
    return link_initial_parameters_.n_elem;
  }

  arma::vec family_initial_parameters()
    const override
  {
    return family_initial_parameters_;
  }

  arma::vec family_parameter_lower_bounds()
    const override
  {
    return family_lower_bounds_;
  }

  arma::vec family_parameter_upper_bounds()
    const override
  {
    return family_upper_bounds_;
  }

  arma::vec link_initial_parameters()
    const override
  {
    return link_initial_parameters_;
  }

  arma::vec link_parameter_lower_bounds()
    const override
  {
    return link_lower_bounds_;
  }

  arma::vec link_parameter_upper_bounds()
    const override
  {
    return link_upper_bounds_;
  }

  void prepare(
      const EcountgmifsInput& input,
      const EcountgmifsControl& control
  ) const override
  {
    prepare_(
      input_to_list(input),
      control_to_list(control),
      environment_
    );
  }

  void inverse(
      const arma::vec& eta,
      const arma::vec& link_parameters,
      arma::vec& mu
  ) const override
  {
    validate_parameter_count(
      link_name_,
      "link",
      link_parameters,
      link_parameter_count()
    );

    copy_vector_result(
      inverse_(
        Rcpp::wrap(eta),
        Rcpp::wrap(link_parameters),
        environment_
      ),
      mu,
      eta.n_elem,
      family_name_ + "/" + link_name_,
      "inverse",
      "mu"
    );
  }

  void negloglik(
      const arma::vec& y,
      const arma::vec& mu,
      const arma::vec& family_parameters,
      double& negloglik
  ) const override
  {
    if (y.n_elem != mu.n_elem) {
      Rcpp::stop(
        "R family-link '%s/%s': y and mu lengths differ",
        family_name_.c_str(),
        link_name_.c_str()
      );
    }

    validate_parameter_count(
      family_name_,
      "family",
      family_parameters,
      family_parameter_count()
    );

    negloglik = require_scalar_result(
      negloglik_(
        Rcpp::wrap(y),
        Rcpp::wrap(mu),
        Rcpp::wrap(family_parameters),
        environment_
      ),
      family_name_ + "/" + link_name_,
      "negloglik"
    );
  }

  void grad(
      const arma::vec& y,
      const arma::vec& eta,
      const arma::vec& mu,
      const arma::vec& family_parameters,
      const arma::vec& link_parameters,
      arma::vec& d_negloglik_d_mu,
      arma::vec& d_mu_d_eta,
      arma::mat& d_mu_d_link_parameters,
      arma::vec& d_negloglik_d_eta,
      arma::vec& d_negloglik_d_family_parameters,
      arma::vec& d_negloglik_d_link_parameters
  ) const override
  {
    if (
        y.n_elem != mu.n_elem ||
        eta.n_elem != mu.n_elem
    ) {
      Rcpp::stop(
        "R family-link '%s/%s': y, eta, and mu lengths must match",
        family_name_.c_str(),
        link_name_.c_str()
      );
    }

    validate_parameter_count(
      family_name_,
      "family",
      family_parameters,
      family_parameter_count()
    );

    validate_parameter_count(
      link_name_,
      "link",
      link_parameters,
      link_parameter_count()
    );

    const std::string plugin_name =
      family_name_ + "/" + link_name_;

    const Rcpp::List result =
      require_list_result(
        grad_(
          Rcpp::wrap(y),
          Rcpp::wrap(eta),
          Rcpp::wrap(mu),
          Rcpp::wrap(family_parameters),
          Rcpp::wrap(link_parameters),
          environment_
        ),
        plugin_name,
        "grad"
      );

    copy_vector_result(
      require_named_element(
        result,
        plugin_name,
        "grad",
        "d_negloglik_d_mu"
      ),
      d_negloglik_d_mu,
      y.n_elem,
      plugin_name,
      "grad",
      "d_negloglik_d_mu"
    );

    copy_vector_result(
      require_named_element(
        result,
        plugin_name,
        "grad",
        "d_mu_d_eta"
      ),
      d_mu_d_eta,
      y.n_elem,
      plugin_name,
      "grad",
      "d_mu_d_eta"
    );

    copy_matrix_result(
      require_named_element(
        result,
        plugin_name,
        "grad",
        "d_mu_d_link_parameters"
      ),
      d_mu_d_link_parameters,
      y.n_elem,
      link_parameter_count(),
      plugin_name,
      "grad",
      "d_mu_d_link_parameters"
    );

    copy_vector_result(
      require_named_element(
        result,
        plugin_name,
        "grad",
        "d_negloglik_d_eta"
      ),
      d_negloglik_d_eta,
      y.n_elem,
      plugin_name,
      "grad",
      "d_negloglik_d_eta"
    );

    copy_vector_result(
      require_named_element(
        result,
        plugin_name,
        "grad",
        "d_negloglik_d_family_parameters"
      ),
      d_negloglik_d_family_parameters,
      family_parameter_count(),
      plugin_name,
      "grad",
      "d_negloglik_d_family_parameters"
    );

    copy_vector_result(
      require_named_element(
        result,
        plugin_name,
        "grad",
        "d_negloglik_d_link_parameters"
      ),
      d_negloglik_d_link_parameters,
      link_parameter_count(),
      plugin_name,
      "grad",
      "d_negloglik_d_link_parameters"
    );
  }
};

class RFunctionCriterion final :
  public IEcountgmifsCriterion
{
private:
  std::string name_;
  Rcpp::Function prepare_;
  Rcpp::Function evaluate_;
  Rcpp::Environment environment_;

public:
  RFunctionCriterion(
      std::string name,
      Rcpp::Function prepare,
      Rcpp::Function evaluate,
      Rcpp::Environment environment
  ) :
    name_(std::move(name)),
    prepare_(prepare),
    evaluate_(evaluate),
    environment_(environment)
  {}

  std::string name() const override
  {
    return name_;
  }

  void prepare(
      const EcountgmifsInput& input,
      const EcountgmifsControl& control
  ) const override
  {
    prepare_(
      input_to_list(input),
      control_to_list(control),
      environment_
    );
  }

  double evaluate(
      const EcountgmifsInput& input,
      const EcountgmifsControl& control,
      const EcountgmifsState& state
  ) const override
  {
    return require_scalar_result(
      evaluate_(
        input_to_list(input),
        control_to_list(control),
        state_to_list(state),
        environment_
      ),
      name_,
      "evaluate"
    );
  }
};

} // anonymous namespace

// [[Rcpp::export]]
SEXP create_r_link(
    const std::string& name,
    Rcpp::Function prepare,
    Rcpp::Function inverse,
    Rcpp::Function grad,
    arma::vec initial_parameters,
    arma::vec lower_bounds,
    arma::vec upper_bounds,
    Rcpp::Environment environment
)
{
  return Rcpp::XPtr<IEcountgmifsLinkFunc>(
    new RFunctionLink(
      name,
      prepare,
      inverse,
      grad,
      std::move(initial_parameters),
      std::move(lower_bounds),
      std::move(upper_bounds),
      environment
    ),
    true
  );
}

// [[Rcpp::export]]
SEXP create_r_family(
    const std::string& name,
    Rcpp::Function prepare,
    Rcpp::Function negloglik,
    Rcpp::Function grad,
    arma::vec initial_parameters,
    arma::vec lower_bounds,
    arma::vec upper_bounds,
    Rcpp::Environment environment
)
{
  return Rcpp::XPtr<IEcountgmifsFamily>(
    new RFunctionFamily(
      name,
      prepare,
      negloglik,
      grad,
      std::move(initial_parameters),
      std::move(lower_bounds),
      std::move(upper_bounds),
      environment
    ),
    true
  );
}

// [[Rcpp::export]]
SEXP create_r_family_link(
    const std::string& family_name,
    const std::string& link_name,
    Rcpp::Function prepare,
    Rcpp::Function inverse,
    Rcpp::Function negloglik,
    Rcpp::Function grad,
    arma::vec family_initial_parameters,
    arma::vec family_lower_bounds,
    arma::vec family_upper_bounds,
    arma::vec link_initial_parameters,
    arma::vec link_lower_bounds,
    arma::vec link_upper_bounds,
    Rcpp::Environment environment
)
{
  return Rcpp::XPtr<IEcountgmifsFamilyLink>(
    new RFunctionFamilyLink(
      family_name,
      link_name,
      prepare,
      inverse,
      negloglik,
      grad,
      std::move(family_initial_parameters),
      std::move(family_lower_bounds),
      std::move(family_upper_bounds),
      std::move(link_initial_parameters),
      std::move(link_lower_bounds),
      std::move(link_upper_bounds),
      environment
    ),
    true
  );
}

// [[Rcpp::export]]
SEXP create_r_criterion(
    const std::string& name,
    Rcpp::Function prepare,
    Rcpp::Function evaluate,
    Rcpp::Environment environment
)
{
  return Rcpp::XPtr<IEcountgmifsCriterion>(
    new RFunctionCriterion(
      name,
      prepare,
      evaluate,
      environment
    ),
    true
  );
}

// [[Rcpp::export]]
Rcpp::List inspect_link_plugin(SEXP pointer)
{
  if (TYPEOF(pointer) != EXTPTRSXP) {
    Rcpp::stop("link plugin must be an external pointer");
  }

  Rcpp::XPtr<IEcountgmifsLinkFunc> plugin(pointer);
  if (plugin.get() == nullptr) {
    Rcpp::stop("link plugin contains a null external pointer");
  }

  const std::string plugin_name = plugin->name();
  if (plugin_name.empty()) {
    Rcpp::stop("link plugin name must not be empty");
  }

  const arma::uword count = plugin->parameter_count();
  const arma::vec initial = plugin->initial_parameters();
  const arma::vec lower = plugin->parameter_lower_bounds();
  const arma::vec upper = plugin->parameter_upper_bounds();

  if (initial.n_elem != count) {
    Rcpp::stop("link plugin parameter_count() disagrees with initial_parameters()");
  }
  validate_parameter_metadata(plugin_name, initial, lower, upper);

  return Rcpp::List::create(
    Rcpp::Named("name") = plugin_name,
    Rcpp::Named("parameter_count") = count,
    Rcpp::Named("initial_parameters") = initial,
    Rcpp::Named("lower_bounds") = lower,
    Rcpp::Named("upper_bounds") = upper
  );
}

// [[Rcpp::export]]
Rcpp::List inspect_family_plugin(SEXP pointer)
{
  if (TYPEOF(pointer) != EXTPTRSXP) {
    Rcpp::stop("family plugin must be an external pointer");
  }

  Rcpp::XPtr<IEcountgmifsFamily> plugin(pointer);
  if (plugin.get() == nullptr) {
    Rcpp::stop("family plugin contains a null external pointer");
  }

  const std::string plugin_name = plugin->name();
  if (plugin_name.empty()) {
    Rcpp::stop("family plugin name must not be empty");
  }

  const arma::uword count = plugin->parameter_count();
  const arma::vec initial = plugin->initial_parameters();
  const arma::vec lower = plugin->parameter_lower_bounds();
  const arma::vec upper = plugin->parameter_upper_bounds();

  if (initial.n_elem != count) {
    Rcpp::stop("family plugin parameter_count() disagrees with initial_parameters()");
  }
  validate_parameter_metadata(plugin_name, initial, lower, upper);

  return Rcpp::List::create(
    Rcpp::Named("name") = plugin_name,
    Rcpp::Named("parameter_count") = count,
    Rcpp::Named("initial_parameters") = initial,
    Rcpp::Named("lower_bounds") = lower,
    Rcpp::Named("upper_bounds") = upper
  );
}

// [[Rcpp::export]]
Rcpp::List inspect_family_link_plugin(SEXP pointer)
{
  if (TYPEOF(pointer) != EXTPTRSXP) {
    Rcpp::stop("family-link plugin must be an external pointer");
  }

  Rcpp::XPtr<IEcountgmifsFamilyLink> plugin(pointer);
  if (plugin.get() == nullptr) {
    Rcpp::stop("family-link plugin contains a null external pointer");
  }

  const std::string family_name = plugin->family_name();
  const std::string link_name = plugin->link_name();
  if (family_name.empty() || link_name.empty()) {
    Rcpp::stop("family-link plugin names must not be empty");
  }

  const arma::uword family_count = plugin->family_parameter_count();
  const arma::vec family_initial = plugin->family_initial_parameters();
  const arma::vec family_lower = plugin->family_parameter_lower_bounds();
  const arma::vec family_upper = plugin->family_parameter_upper_bounds();

  if (family_initial.n_elem != family_count) {
    Rcpp::stop(
      "family-link family_parameter_count() disagrees with family_initial_parameters()"
    );
  }
  validate_parameter_metadata(
    family_name,
    family_initial,
    family_lower,
    family_upper
  );

  const arma::uword link_count = plugin->link_parameter_count();
  const arma::vec link_initial = plugin->link_initial_parameters();
  const arma::vec link_lower = plugin->link_parameter_lower_bounds();
  const arma::vec link_upper = plugin->link_parameter_upper_bounds();

  if (link_initial.n_elem != link_count) {
    Rcpp::stop(
      "family-link link_parameter_count() disagrees with link_initial_parameters()"
    );
  }
  validate_parameter_metadata(
    link_name,
    link_initial,
    link_lower,
    link_upper
  );

  return Rcpp::List::create(
    Rcpp::Named("family_name") = family_name,
    Rcpp::Named("link_name") = link_name,
    Rcpp::Named("family_parameter_count") = family_count,
    Rcpp::Named("family_initial_parameters") = family_initial,
    Rcpp::Named("family_lower_bounds") = family_lower,
    Rcpp::Named("family_upper_bounds") = family_upper,
    Rcpp::Named("link_parameter_count") = link_count,
    Rcpp::Named("link_initial_parameters") = link_initial,
    Rcpp::Named("link_lower_bounds") = link_lower,
    Rcpp::Named("link_upper_bounds") = link_upper
  );
}

// [[Rcpp::export]]
Rcpp::List inspect_criterion_plugin(SEXP pointer)
{
  if (TYPEOF(pointer) != EXTPTRSXP) {
    Rcpp::stop("criterion plugin must be an external pointer");
  }

  Rcpp::XPtr<IEcountgmifsCriterion> plugin(pointer);
  if (plugin.get() == nullptr) {
    Rcpp::stop("criterion plugin contains a null external pointer");
  }

  const std::string plugin_name = plugin->name();
  if (plugin_name.empty()) {
    Rcpp::stop("criterion plugin name must not be empty");
  }

  return Rcpp::List::create(
    Rcpp::Named("name") = plugin_name
  );
}

