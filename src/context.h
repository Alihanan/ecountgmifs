#pragma once

#include <RcppArmadillo.h>
#include <cstdint>
#include <cmath>
#include <string>
#include <utility>
#include <limits>
#include <vector>
#include <algorithm>
#include <memory>
#include <chrono>
#include <unordered_map>

#include "../inst/include/ecountgmifs/api.h"
#include "enums.h"
#include "util.h"
#include "r_convert.h"
#include "nlopt_optimizer.h"
#include "enet.h"
#include "debug.h"

inline void check_input_dimensions(
    const arma::mat& X,
    const arma::vec& y,
    const arma::mat& w,
    const arma::vec& offset,
    const arma::vec& weight_vec
) {
  if (X.n_rows == 0 || X.n_cols == 0) {
    Rcpp::stop("matrix 'X' must have positive dimensions");
  }

  check_vector_length(y, X.n_rows, "y");
  check_matrix_rows(w, X.n_rows, "w");
  check_vector_length(offset, X.n_rows, "offset");
  check_vector_length(weight_vec, X.n_cols, "weight_vec");
}


struct EcountgmifsDefaultFamilyLink final : public IEcountgmifsFamilyLink
{
  const IEcountgmifsFamily& family;
  const IEcountgmifsLinkFunc& link_func;

  EcountgmifsDefaultFamilyLink(
    const IEcountgmifsFamily& family_,
    const IEcountgmifsLinkFunc& link_func_
  ) :
    family(family_),
    link_func(link_func_)
  {}

  std::string family_name() const override
  {
    return family.name();
  }

  std::string link_name() const override
  {
    return link_func.name();
  }

  arma::uword family_parameter_count() const noexcept override
  {
    return family.parameter_count();
  }

  arma::uword link_parameter_count() const noexcept override
  {
    return link_func.parameter_count();
  }

  void prepare(
      const EcountgmifsInput& input,
      const EcountgmifsControl& control
  ) const override
  {
    family.prepare(
      input,
      control
    );

    link_func.prepare(
      input,
      control
    );
  }

  arma::vec family_initial_parameters() const override
  {
    return family.initial_parameters();
  }

  arma::vec family_parameter_lower_bounds() const override
  {
    return family.parameter_lower_bounds();
  }

  arma::vec family_parameter_upper_bounds() const override
  {
    return family.parameter_upper_bounds();
  }

  arma::vec link_initial_parameters() const override
  {
    return link_func.initial_parameters();
  }

  arma::vec link_parameter_lower_bounds() const override
  {
    return link_func.parameter_lower_bounds();
  }

  arma::vec link_parameter_upper_bounds() const override
  {
    return link_func.parameter_upper_bounds();
  }

  void inverse(
      const arma::vec& eta,
      const arma::vec& link_parameters,
      arma::vec& mu
  ) const override
  {
    link_func.inverse(
      eta,
      link_parameters,
      mu
    );
  }

  void negloglik(
      const arma::vec& y,
      const arma::vec& mu,
      const arma::vec& family_parameters,
      double& negloglik_value
  ) const override
  {
    family.negloglik(
      y,
      mu,
      family_parameters,
      negloglik_value
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
    family.grad(
      y,
      mu,
      family_parameters,
      d_negloglik_d_mu,
      d_negloglik_d_family_parameters
    );

    link_func.grad(
      eta,
      link_parameters,
      d_mu_d_eta,
      d_mu_d_link_parameters
    );

    d_negloglik_d_eta =
      d_negloglik_d_mu %
      d_mu_d_eta;

    if (link_parameter_count() == 0) {
      d_negloglik_d_link_parameters.reset();
    } else {
      d_negloglik_d_link_parameters =
        d_mu_d_link_parameters.t() *
        d_negloglik_d_mu;
    }
  }
};


struct EcountgmifsInputInternal
{
private:
  const IEcountgmifsFamilyLink* supplied_family_link_;
  const IEcountgmifsFamily* family_;
  const IEcountgmifsLinkFunc* link_func_;

  std::unique_ptr<EcountgmifsDefaultFamilyLink>
    default_family_link_;

public:
  EcountgmifsInput api;

  EcountgmifsInputInternal(
    const arma::mat& X,
    const arma::vec& y,
    const arma::mat& w,
    const arma::vec& offset,
    const arma::vec& weight_vec,
    double enet_alpha,
    SEXP family,
    SEXP link_func,
    Rcpp::Nullable<Rcpp::List> criteria,
    SEXP family_link
  ) :
    supplied_family_link_(
      resolve_optional_family_link_ptr(
        family_link
      )
    ),

    family_(
      resolve_optional_family_ptr(
        family,
        supplied_family_link_ == nullptr
      )
    ),

    link_func_(
      resolve_optional_link_ptr(
        link_func,
        supplied_family_link_ == nullptr
      )
    ),

    default_family_link_(
      supplied_family_link_ == nullptr
    ? std::make_unique<
      EcountgmifsDefaultFamilyLink
    >(
      *family_,
      *link_func_
    )
    : nullptr
    ),

    api {
    X,
    y,
    w,
    offset,
    weight_vec,
    weight_vec_has_prior(weight_vec),
    enet_alpha,

    family_,
    link_func_,
    supplied_family_link_ != nullptr
    ? supplied_family_link_
    : default_family_link_.get(),
      resolve_criteria_ptrs(criteria)
  }
  {
    check_input_dimensions(
      api.X,
      api.y,
      api.w,
      api.offset,
      api.weight_vec
    );

    check_matrix_finite(api.X, "X");
    check_vector_finite(api.y, "y");
    check_matrix_finite(api.w, "w");
    check_vector_finite(api.offset, "offset");
    check_vector_finite(api.weight_vec, "weight_vec");

    if (arma::any(api.weight_vec <= 0.0)) {
      Rcpp::stop("value of 'weight_vec' must contain only positive values");
    }


    check_finite_scalar(api.enet_alpha, "enet_alpha");

    if (api.enet_alpha < 0.0 || api.enet_alpha > 1.0) {
      Rcpp::stop("value of 'enet_alpha' must be in [0, 1]");
    }
  }

  EcountgmifsInputInternal(
    const EcountgmifsInputInternal&
  ) = delete;

  EcountgmifsInputInternal(
    EcountgmifsInputInternal&&
  ) = delete;

  EcountgmifsInputInternal& operator=(
    const EcountgmifsInputInternal&
  ) = delete;

  EcountgmifsInputInternal& operator=(
    EcountgmifsInputInternal&&
  ) = delete;

  Rcpp::List to_list(bool include_data = false) const
  {
    Rcpp::CharacterVector criterion_names(
        static_cast<R_xlen_t>(
          api.criteria.size()
        )
    );

    for (
        std::size_t i = 0;
        i < api.criteria.size();
        ++i
    ) {
      criterion_names[
      static_cast<R_xlen_t>(i)
      ] =
        api.criteria[i]->name();
    }

    Rcpp::List out = Rcpp::List::create(
      Rcpp::Named("n") = api.X.n_rows,
      Rcpp::Named("p") = api.X.n_cols,
      Rcpp::Named("q") = api.w.n_cols,
      Rcpp::Named("family") =
        Rcpp::List::create(
          Rcpp::Named(
            api.family_link->family_name()
          ) =
            Rcpp::List::create(
              Rcpp::Named("parameter_count") =
                api.family_link->
                family_parameter_count()
            )
        ),
        Rcpp::Named("link_func") =
          Rcpp::List::create(
            Rcpp::Named(
              api.family_link->link_name()
            ) =
              Rcpp::List::create(
                Rcpp::Named("parameter_count") =
                  api.family_link->
                  link_parameter_count()
              )
          ),
          Rcpp::Named("family_link_supplied") =
            supplied_family_link_ != nullptr,
              Rcpp::Named("criteria") =
                criterion_names,
                Rcpp::Named("enet_alpha") = api.enet_alpha,
                Rcpp::Named("has_prior") = api.has_prior,
                Rcpp::Named("weight_vec") =
                  ecountgmifs::output::to_r_vector(api.weight_vec)
    );

    if (include_data) {
      out["X"] = api.X;
      out["y"] = api.y;
      out["w"] = api.w;
      out["offset"] = api.offset;
      out["weight_vec"] = api.weight_vec;
    }

    return out;
  }

private:
  static const IEcountgmifsFamilyLink*
    resolve_optional_family_link_ptr(
      SEXP family_link
    )
    {
      if (Rf_isNull(family_link)) {
        return nullptr;
      }

      Rcpp::XPtr<IEcountgmifsFamilyLink> ptr(
          family_link
      );

      if (ptr.get() == nullptr) {
        Rcpp::stop(
          "family_link contains a null external pointer"
        );
      }

      return ptr.get();
    }

  static const IEcountgmifsFamily*
    resolve_optional_family_ptr(
      SEXP family,
      bool required
    )
    {
      if (Rf_isNull(family)) {
        if (required) {
          Rcpp::stop(
            "family must be supplied when family_link is NULL"
          );
        }

        return nullptr;
      }

      Rcpp::XPtr<IEcountgmifsFamily> ptr(family);

      if (ptr.get() == nullptr) {
        Rcpp::stop(
          "family contains a null external pointer"
        );
      }

      return ptr.get();
    }

  static const IEcountgmifsLinkFunc*
    resolve_optional_link_ptr(
      SEXP link_func,
      bool required
    )
    {
      if (Rf_isNull(link_func)) {
        if (required) {
          Rcpp::stop(
            "link_func must be supplied when family_link is NULL"
          );
        }

        return nullptr;
      }

      Rcpp::XPtr<IEcountgmifsLinkFunc> ptr(
          link_func
      );

      if (ptr.get() == nullptr) {
        Rcpp::stop(
          "link_func contains a null external pointer"
        );
      }

      return ptr.get();
    }

  static std::vector<const IEcountgmifsCriterion*> resolve_criteria_ptrs(
      Rcpp::Nullable<Rcpp::List> criteria
  )
  {
    std::vector<const IEcountgmifsCriterion*> out;

    if (criteria.isNull()) {
      return out;
    }

    const Rcpp::List criteria_list(criteria);

    out.reserve(
      static_cast<std::size_t>(criteria_list.size())
    );

    for (R_xlen_t i = 0; i < criteria_list.size(); ++i)
    {
      Rcpp::XPtr<IEcountgmifsCriterion> ptr(
          criteria_list[i]
      );

      if (ptr.get() == nullptr)
        Rcpp::stop(
          "criterion at position %d contains a null external pointer",
          static_cast<int>(i + 1)
        );

      out.push_back(ptr.get());
    }

    return out;
  }
};


struct EcountgmifsControlInternal
{
  EcountgmifsControl api;

  EcountgmifsControlInternal(
    const EcountgmifsInput& input,
    uint64_t null_iteration_max,
    uint64_t stagewise_iteration_max,
    double null_family_parameter_abs_tol,
    double stagewise_objective_rel_tol,
    double stagewise_beta_step_norm_tol,
    double epsilon_max,
    double epsilon_start,
    double epsilon_min,
    double loglik_reltol_cutoff,
    double enet_abs_tol,
    double enet_rel_tol,
    uint32_t enet_max_iter,
    int state_track_strategy,
    uint64_t state_track_freq,
    bool verbose,
    bool include_data,

    const arma::vec& theta_initial,
    const arma::vec& theta_lower_bounds,
    const arma::vec& theta_upper_bounds,

    const EcountgmifsNloptControl& nonpen_nlopt,
    const EcountgmifsNloptControl& family_nlopt,
    const EcountgmifsNloptControl& link_nlopt
  ) :
    api {
    null_iteration_max,
    stagewise_iteration_max,
    null_family_parameter_abs_tol,
    stagewise_objective_rel_tol,
    stagewise_beta_step_norm_tol,
    epsilon_max,
    epsilon_start,
    epsilon_min,
    loglik_reltol_cutoff,
    enet_abs_tol,
    enet_rel_tol,
    enet_max_iter,

    as_track_strategy(state_track_strategy),
    state_track_freq,
    verbose,
    include_data,

    theta_initial,
    theta_lower_bounds,
    theta_upper_bounds,

    nonpen_nlopt,
    family_nlopt,
    link_nlopt
  }
  {
    check_positive_integer(
      api.null_iteration_max,
      "null_iteration_max"
    );

    check_positive_integer(
      api.stagewise_iteration_max,
      "stagewise_iteration_max"
    );

    check_positive_scalar(
      api.null_family_parameter_abs_tol,
      "null_family_parameter_abs_tol"
    );

    check_nonnegative_scalar(
      api.stagewise_objective_rel_tol,
      "stagewise_objective_rel_tol"
    );

    check_nonnegative_scalar(
      api.stagewise_beta_step_norm_tol,
      "stagewise_beta_step_norm_tol"
    );

    check_nonnegative_scalar(
      api.loglik_reltol_cutoff,
      "loglik_reltol_cutoff"
    );

    check_positive_scalar(
      api.enet_abs_tol,
      "enet_abs_tol"
    );

    check_nonnegative_scalar(
      api.enet_rel_tol,
      "enet_rel_tol"
    );

    check_positive_integer(
      api.enet_max_iter,
      "enet_max_iter"
    );

    check_positive_scalar(
      api.epsilon_max,
      "epsilon_max"
    );

    check_positive_scalar(
      api.epsilon_start,
      "epsilon_start"
    );

    check_nonnegative_scalar(
      api.epsilon_min,
      "epsilon_min"
    );

    check_initial_bounds(
      api.epsilon_start,
      api.epsilon_min,
      api.epsilon_max,
      "epsilon"
    );

    check_initial_bounds(
      api.theta_initial,
      api.theta_lower_bounds,
      api.theta_upper_bounds,
      input.w.n_cols,
      "theta"
    );

    check_nlopt_control(
      api.nonpen_nlopt,
      "nonpen_nlopt"
    );

    check_nlopt_control(
      api.family_nlopt,
      "family_nlopt"
    );

    check_nlopt_control(
      api.link_nlopt,
      "link_nlopt"
    );

    if (
        api.state_track_strategy ==
          EnumStateTrackStrategy::EVERY_K_ITERATION
    ) {
      check_positive_integer(
        api.state_track_freq,
        "state_track_freq"
      );
    }
  }

  Rcpp::List to_list() const
  {
    Rcpp::List result;

    result["null_iteration_max"] =
      api.null_iteration_max;

    result["stagewise_iteration_max"] =
      api.stagewise_iteration_max;

    result["null_family_parameter_abs_tol"] =
      api.null_family_parameter_abs_tol;

    result["stagewise_objective_rel_tol"] =
      api.stagewise_objective_rel_tol;

    result["stagewise_beta_step_norm_tol"] =
      api.stagewise_beta_step_norm_tol;

    result["epsilon_max"] =
      api.epsilon_max;

    result["epsilon_start"] =
      api.epsilon_start;

    result["epsilon_min"] =
      api.epsilon_min;

    result["loglik_reltol_cutoff"] =
      api.loglik_reltol_cutoff;

    result["enet_abs_tol"] =
      api.enet_abs_tol;

    result["enet_rel_tol"] =
      api.enet_rel_tol;

    result["enet_max_iter"] =
      api.enet_max_iter;

    result["state_track_strategy"] =
      state_track_strategy_name(
        api.state_track_strategy
      );

    result["state_track_freq"] =
      api.state_track_freq;

    result["verbose"] =
      api.verbose;

    result["include_data"] =
      api.include_data;

    result["theta_initial"] =
      ecountgmifs::output::to_r_vector(
        api.theta_initial
      );

    result["theta_lower_bounds"] =
      ecountgmifs::output::to_r_vector(
        api.theta_lower_bounds
      );

    result["theta_upper_bounds"] =
      ecountgmifs::output::to_r_vector(
        api.theta_upper_bounds
      );

    result["nonpen_nlopt"] =
      nlopt_control_to_list(
        api.nonpen_nlopt
      );

    result["family_nlopt"] =
      nlopt_control_to_list(
        api.family_nlopt
      );

    result["link_nlopt"] =
      nlopt_control_to_list(
        api.link_nlopt
      );

    return result;
  }

private:
  static void check_nlopt_control(
      const EcountgmifsNloptControl& nlopt_control,
      const char* name
  )
  {
    const int algorithm =
      static_cast<int>(
        nlopt_control.algorithm
      );

    if (
        algorithm < 0 ||
          algorithm >=
          static_cast<int>(
            NLOPT_NUM_ALGORITHMS
          )
    ) {
      Rcpp::stop(
        "%s.algorithm is not a valid NLopt algorithm",
        name
      );
    }

    check_nonnegative_scalar(
      nlopt_control.xtol_rel,
      (std::string(name) + ".xtol_rel").c_str()
    );

    check_nonnegative_scalar(
      nlopt_control.ftol_rel,
      (std::string(name) + ".ftol_rel").c_str()
    );

    check_positive_integer(
      nlopt_control.maxeval,
      (std::string(name) + ".maxeval").c_str()
    );
  }

  static Rcpp::List nlopt_control_to_list(
      const EcountgmifsNloptControl& nlopt_control
  )
  {
    return Rcpp::List::create(
      Rcpp::Named("algorithm") =
        static_cast<int>(
          nlopt_control.algorithm
        ),

        Rcpp::Named("xtol_rel") =
          nlopt_control.xtol_rel,

          Rcpp::Named("ftol_rel") =
            nlopt_control.ftol_rel,

            Rcpp::Named("maxeval") =
              nlopt_control.maxeval
    );
  }
};

struct EcountgmifsStateInternal
{
private:
  const EcountgmifsInput& input;
  const EcountgmifsControl& control;
  EcountgmifsState api;

public:
  explicit EcountgmifsStateInternal(
      const EcountgmifsInput& input_,
      const EcountgmifsControl& control_
  ) :
    input(input_),
    control(control_),
    api {
    { // EcountgmifsPredictors
      { // EcountgmifsParameters
        arma::vec(
          input.X.n_cols,
          arma::fill::zeros
        ), // beta

        control.theta_initial, // theta
        input.family_link->family_initial_parameters(),
        input.family_link->link_initial_parameters()
      },

      arma::vec(
        input.X.n_rows,
        arma::fill::zeros
      ), // xbeta

      arma::vec(
        input.X.n_rows,
        arma::fill::zeros
      ), // wtheta

      arma::vec(
        input.X.n_rows,
        arma::fill::zeros
      ), // eta

      arma::vec(
        input.X.n_rows,
        arma::fill::zeros
      ), // mu

      arma::uvec(
        input.X.n_cols,
        arma::fill::zeros
      ) // active_set
    },

    arma::datum::nan, // negloglik
    Rcpp::NumericVector(), // criteria
    0, // iteration
    arma::datum::nan, // pseudo_r2; snapshots only
    0.0 // elapsed_time; stagewise snapshots only
  }
  {
    /*
     * Prepare all model modules for this fit before the first
     * inverse-link or negative-log-likelihood evaluation.
     *
     * For the default family-link adapter, this prepares the
     * separate family and link modules. For a supplied fused
     * family-link, it prepares that module directly.
     */
    input.family_link->prepare(
        input,
        control
    );

    /*
     * Prepare every information criterion for this fit.
     */
    for (
        const IEcountgmifsCriterion* criterion :
      input.criteria
    ) {
      criterion->prepare(
          input,
          control
      );
    }

    check_initial_bounds(
      api.param.param.family_parameters,
      input.family_link->family_parameter_lower_bounds(),
      input.family_link->family_parameter_upper_bounds(),
      input.family_link->family_parameter_count(),
      "family parameters"
    );

    check_initial_bounds(
      api.param.param.link_parameters,
      input.family_link->link_parameter_lower_bounds(),
      input.family_link->link_parameter_upper_bounds(),
      input.family_link->link_parameter_count(),
      "link parameters"
    );
    initialize_criteria();
    refresh_after_constructor();
  }

  EcountgmifsStateInternal(
    const EcountgmifsStateInternal&
  ) = delete;

  EcountgmifsStateInternal(
    EcountgmifsStateInternal&&
  ) = delete;

  EcountgmifsStateInternal& operator=(
    const EcountgmifsStateInternal&
  ) = delete;

  EcountgmifsStateInternal& operator=(
    EcountgmifsStateInternal&&
  ) = delete;

  const EcountgmifsState& view() const noexcept
  {
    return api;
  }

  const arma::vec& beta() const noexcept
  {
    return api.param.param.beta;
  }

  const arma::vec& theta() const noexcept
  {
    return api.param.param.theta;
  }

  const arma::vec& family_parameters() const noexcept
  {
    return api.param.param.family_parameters;
  }

  const arma::vec& link_parameters() const noexcept
  {
    return api.param.param.link_parameters;
  }

  const arma::vec& eta() const noexcept
  {
    return api.param.eta;
  }

  const arma::vec& mu() const noexcept
  {
    return api.param.mu;
  }

  double negloglik() const noexcept
  {
    return api.negloglik;
  }

  uint64_t iteration() const noexcept
  {
    return api.iteration;
  }

  void begin_stagewise_iteration(
      uint64_t iteration
  ) noexcept
  {
    api.iteration = iteration;
  }

  void set_elapsed_time(
      double elapsed_time
  )
  {
    check_nonnegative_scalar(
      elapsed_time,
      "elapsed_time"
    );

    api.elapsed_time =
      elapsed_time;
  }

  void set_beta(
      const arma::vec& beta
  )
  {
    copy_parameter_vector(
      beta,
      api.param.param.beta,
      "beta"
    );

    refresh_after_beta();
  }

  void set_theta(
      const arma::vec& theta
  )
  {
    set_theta(
      static_cast<unsigned>(theta.n_elem),
      theta.memptr()
    );
  }

  void set_family_parameters(
      const arma::vec& family_parameters
  )
  {
    set_family_parameters(
      static_cast<unsigned>(family_parameters.n_elem),
      family_parameters.memptr()
    );
  }

  void set_link_parameters(
      const arma::vec& link_parameters
  )
  {
    set_link_parameters(
      static_cast<unsigned>(link_parameters.n_elem),
      link_parameters.memptr()
    );
  }

  void set_theta(
      unsigned n,
      const double* values
  )
  {
    copy_parameter_data(
      n,
      values,
      api.param.param.theta,
      "theta"
    );

    refresh_after_theta();
  }

  void set_family_parameters(
      unsigned n,
      const double* values
  )
  {
    copy_parameter_data(
      n,
      values,
      api.param.param.family_parameters,
      "family"
    );

    refresh_after_family_parameters();
  }

  void set_link_parameters(
      unsigned n,
      const double* values
  )
  {
    copy_parameter_data(
      n,
      values,
      api.param.param.link_parameters,
      "link"
    );

    refresh_after_link_parameters();
  }

  void refresh_after_beta()
  {
    update_xbeta();
  }

  void refresh_after_theta()
  {
    update_wtheta();
  }

  void refresh_after_link_parameters()
  {
    update_mu();
  }

  void refresh_after_family_parameters()
  {
    update_negloglik();
  }
  void evaluate_criteria()
  {
    if (
        api.criteria.size() !=
          static_cast<R_xlen_t>(
            input.criteria.size()
          )
    ) {
      Rcpp::stop(
        "criterion storage size mismatch"
      );
    }

    for (
        std::size_t i = 0;
        i < input.criteria.size();
        ++i
    ) {
      const double value =
        input.criteria[i]->evaluate(
            input,
            control,
            api
        );

      check_finite_scalar(
        value,
        input.criteria[i]->name().c_str()
      );

      api.criteria[
      static_cast<R_xlen_t>(i)
      ] =
        value;
    }
  }

  Rcpp::List to_list() const
  {
    return to_list(api);
  }

  static Rcpp::List to_list(
      const EcountgmifsParameters& parameters
  )
  {
    return Rcpp::List::create(
      Rcpp::Named("beta") =
        ecountgmifs::output::to_r_vector(
          parameters.beta
        ),

        Rcpp::Named("theta") =
          ecountgmifs::output::to_r_vector(
            parameters.theta
          ),

          Rcpp::Named("family_parameters") =
            ecountgmifs::output::to_r_vector(
              parameters.family_parameters
            ),

            Rcpp::Named("link_parameters") =
              ecountgmifs::output::to_r_vector(
                parameters.link_parameters
              )
    );
  }

  static Rcpp::List to_list(
      const EcountgmifsPredictors& predictors
  )
  {
    return Rcpp::List::create(
      Rcpp::Named("parameters") =
        to_list(predictors.param),

        Rcpp::Named("xbeta") =
          ecountgmifs::output::to_r_vector(
            predictors.xbeta
          ),

          Rcpp::Named("wtheta") =
            ecountgmifs::output::to_r_vector(
              predictors.wtheta
            ),

            Rcpp::Named("eta") =
              ecountgmifs::output::to_r_vector(
                predictors.eta
              ),

              Rcpp::Named("mu") =
                ecountgmifs::output::to_r_vector(
                  predictors.mu
                ),

                Rcpp::Named("active_set") =
                  ecountgmifs::output::to_r_logical_vector(
                    predictors.active_set
                  )
    );
  }

  static Rcpp::List to_list(
      const EcountgmifsState& state
  )
  {
    return Rcpp::List::create(
      Rcpp::Named("predictors") =
        to_list(state.param),

        Rcpp::Named("negloglik") =
          state.negloglik,

          Rcpp::Named("criteria") =
            Rcpp::clone(state.criteria),

            Rcpp::Named("iteration") =
              ecountgmifs::output::to_r_integer(
                state.iteration,
                "iteration"
              ),

              Rcpp::Named("pseudo_r2") =
                state.pseudo_r2,

                Rcpp::Named("elapsed_time") =
                  state.elapsed_time
    );
  }

private:
  void initialize_criteria()
  {
    const R_xlen_t criterion_count =
      static_cast<R_xlen_t>(
        input.criteria.size()
      );

    api.criteria =
      Rcpp::NumericVector(
        criterion_count,
        NA_REAL
      );

    Rcpp::CharacterVector criterion_names(
        criterion_count
    );

    for (
        std::size_t i = 0;
        i < input.criteria.size();
        ++i
    ) {
      criterion_names[
      static_cast<R_xlen_t>(i)
      ] =
        input.criteria[i]->name();
    }

    api.criteria.attr("names") =
      criterion_names;
  }
  static void copy_parameter_vector(
      const arma::vec& source,
      arma::vec& destination,
      const char* name
  )
  {
    if (source.n_elem != destination.n_elem) {
      Rcpp::stop(
        "incorrect %s parameter count",
        name
      );
    }

    if (source.memptr() != destination.memptr()) {
      std::copy_n(
        source.memptr(),
        source.n_elem,
        destination.memptr()
      );
    }
  }

  static void copy_parameter_data(
      unsigned n,
      const double* values,
      arma::vec& destination,
      const char* name
  )
  {
    if (n != destination.n_elem) {
      Rcpp::stop(
        "incorrect %s parameter count",
        name
      );
    }

    if (values != destination.memptr()) {
      std::copy_n(
        values,
        n,
        destination.memptr()
      );
    }
  }

  void refresh_after_constructor()
  {
    api.param.xbeta =
      input.X *
      api.param.param.beta;

    api.param.wtheta =
      input.w *
      api.param.param.theta;

    update_active_set();
    update_eta();
  }

  void update_xbeta()
  {
    api.param.xbeta =
      input.X * api.param.param.beta;

    update_active_set();
    update_eta();
  }

  void update_wtheta()
  {
    api.param.wtheta =
      input.w * api.param.param.theta;

    update_eta();
  }

  void update_eta()
  {
    /*
     * Reuse the already allocated eta buffer. Writing the three terms
     * sequentially avoids constructing a temporary n-vector for the
     * chained Armadillo addition on every parameter trial.
     */
    api.param.eta =
      input.offset;

    api.param.eta +=
      api.param.xbeta;

    api.param.eta +=
      api.param.wtheta;

    update_mu();
  }

  void update_mu()
  {
    input.family_link->inverse(
        api.param.eta,
        api.param.param.link_parameters,
        api.param.mu
    );

    check_vector_finite(
      api.param.mu,
      "mu"
    );

    update_negloglik();
  }

  void update_negloglik()
  {
    input.family_link->negloglik(
        input.y,
        api.param.mu,
        api.param.param.family_parameters,
        api.negloglik
    );

    check_finite_scalar(
      api.negloglik,
      "negloglik"
    );
  }

  void update_active_set() noexcept
  {
    /*
     * Fill the persistent buffer directly. The former conv_to expression
     * created a temporary p-vector for every accepted and rejected beta
     * trial.
     */
    for (
        arma::uword i = 0;
        i < api.param.param.beta.n_elem;
        ++i
    ) {
      api.param.active_set[i] =
        api.param.param.beta[i] != 0.0;
    }
  }
};


struct EcountgmifsGradientsInternal
{
private:
  const EcountgmifsInput& input;
  const EcountgmifsStateInternal& state;
  EcountgmifsGradients api;

public:
  EcountgmifsGradientsInternal(
    const EcountgmifsInput& input_,
    const EcountgmifsStateInternal& state_
  ) :
  input(input_),
  state(state_),
  api {
    arma::vec(
      input.X.n_rows,
      arma::fill::zeros
    ), // d_negloglik_d_mu

    arma::vec(
      input.X.n_rows,
      arma::fill::zeros
    ), // d_mu_d_eta

    arma::vec(
      input.X.n_rows,
      arma::fill::zeros
    ), // d_negloglik_d_eta

    input.X, // d_eta_d_beta
    input.w, // d_eta_d_theta

    arma::mat(
      input.X.n_rows,
      input.family_link->link_parameter_count(),
      arma::fill::zeros
    ), // d_mu_d_link_parameters

    arma::vec(
      input.family_link->family_parameter_count(),
      arma::fill::zeros
    ), // d_negloglik_d_family_parameters

    arma::vec(
      input.family_link->link_parameter_count(),
      arma::fill::zeros
    ), // d_negloglik_d_link_parameters

    arma::vec(
      input.X.n_cols,
      arma::fill::zeros
    ) // d_negloglik_d_beta
  }
  {}

  EcountgmifsGradientsInternal(
    const EcountgmifsGradientsInternal&
  ) = delete;

  EcountgmifsGradientsInternal(
    EcountgmifsGradientsInternal&&
  ) = delete;

  EcountgmifsGradientsInternal& operator=(
    const EcountgmifsGradientsInternal&
  ) = delete;

  EcountgmifsGradientsInternal& operator=(
    EcountgmifsGradientsInternal&&
  ) = delete;

  const EcountgmifsGradients& view() const noexcept
  {
    return api;
  }

  const arma::vec& update_beta_gradient()
  {
    update_derivatives();

    api.d_negloglik_d_beta =
      api.d_eta_d_beta.t() *
      api.d_negloglik_d_eta;

    check_vector_finite(
      api.d_negloglik_d_beta,
      "beta gradient"
    );

    return api.d_negloglik_d_beta;
  }

  void write_theta_gradient(
      unsigned n,
      double* out
  )
  {
    check_gradient_size(
      n,
      input.w.n_cols,
      "theta"
    );

    update_derivatives();

    arma::vec gradient_view(
        out,
        static_cast<arma::uword>(n),
        false,
        true
    );

    gradient_view =
      api.d_eta_d_theta.t() *
      api.d_negloglik_d_eta;
  }

  void write_family_gradient(
      unsigned n,
      double* out
  )
  {
    check_gradient_size(
      n,
      input.family_link->family_parameter_count(),
      "family"
    );

    update_derivatives();

    std::copy_n(
      api.d_negloglik_d_family_parameters.memptr(),
      n,
      out
    );
  }

  void write_link_gradient(
      unsigned n,
      double* out
  )
  {
    check_gradient_size(
      n,
      input.family_link->link_parameter_count(),
      "link"
    );

    update_derivatives();

    std::copy_n(
      api.d_negloglik_d_link_parameters.memptr(),
      n,
      out
    );
  }

private:
  static void check_gradient_size(
      unsigned n,
      arma::uword expected,
      const char* name
  )
  {
    if (n != expected) {
      Rcpp::stop(
        "incorrect %s gradient size",
        name
      );
    }
  }

  void update_derivatives()
  {
    input.family_link->grad(
        input.y,
        state.eta(),
        state.mu(),
        state.family_parameters(),
        state.link_parameters(),
        api.d_negloglik_d_mu,
        api.d_mu_d_eta,
        api.d_mu_d_link_parameters,
        api.d_negloglik_d_eta,
        api.d_negloglik_d_family_parameters,
        api.d_negloglik_d_link_parameters
    );
  }
};


struct EcountgmifsPathInternal
{
private:
  const EcountgmifsInput& input;
  const EcountgmifsStateInternal& current_state;
  const EcountgmifsControl& control;
  EcountgmifsPath api;

  /*
   * A best state is materialized only after the first subsequent iteration
   * that does not improve that criterion. During a consecutive improvement
   * streak, only these small flags and scalar values are updated.
   */
  std::vector<uint8_t> best_pending_;
  std::vector<uint8_t> criterion_improved_;

  bool previous_state_valid_ = false;
  uint64_t previous_iteration_ = 0;
  double previous_negloglik_ = arma::datum::nan;
  double previous_pseudo_r2_ = arma::datum::nan;
  double previous_elapsed_time_ = 0.0;
  Rcpp::NumericVector previous_criteria_;

  std::unordered_map<uint64_t, R_xlen_t>
    saved_state_index_by_iteration_;

public:
  EcountgmifsPathInternal(
    const EcountgmifsInput& input_,
    const EcountgmifsStateInternal& state_,
    const EcountgmifsControl& control_
  ) :
  input(input_),
  current_state(state_),
  control(control_),
  api {},
  best_pending_(
    input_.criteria.size(),
    static_cast<uint8_t>(0)
  ),
  criterion_improved_(
    input_.criteria.size(),
    static_cast<uint8_t>(0)
  ),
  previous_criteria_(
    static_cast<R_xlen_t>(
      input_.criteria.size()
    )
  )
  {
    api.best_criteria.resize(
      input.criteria.size()
    );

    constexpr R_xlen_t initial_capacity = 16;

    api.saved_states.state_names =
      Rcpp::CharacterVector(initial_capacity);

    api.saved_states.iterations =
      Rcpp::NumericVector(initial_capacity);

    api.saved_states.negloglik =
      Rcpp::NumericVector(initial_capacity);

    api.saved_states.pseudo_r2 =
      Rcpp::NumericVector(initial_capacity);

    api.saved_states.elapsed_time =
      Rcpp::NumericVector(initial_capacity);

    api.saved_states.criteria =
      Rcpp::List(
        static_cast<R_xlen_t>(
          input.criteria.size()
        )
      );

    for (
        std::size_t i = 0;
        i < input.criteria.size();
        ++i
    ) {
      api.best_criteria[i].name =
        input.criteria[i]->name();

      api.saved_states.criteria[
      static_cast<R_xlen_t>(i)
      ] =
        Rcpp::NumericVector(initial_capacity);
    }

    api.saved_states.beta = Rcpp::List(initial_capacity);
    api.saved_states.theta = Rcpp::List(initial_capacity);
    api.saved_states.family_parameters = Rcpp::List(initial_capacity);
    api.saved_states.link_parameters = Rcpp::List(initial_capacity);
    api.saved_states.xbeta = Rcpp::List(initial_capacity);
    api.saved_states.wtheta = Rcpp::List(initial_capacity);
    api.saved_states.eta = Rcpp::List(initial_capacity);
    api.saved_states.mu = Rcpp::List(initial_capacity);
    api.saved_states.active_set = Rcpp::List(initial_capacity);
    api.saved_states.count = 0;

    api.state_indices =
      Rcpp::IntegerVector(initial_capacity);

    api.state_count = 0;

    api.last_saved_active_set.zeros(
      current_state.view().param.active_set.n_elem
    );
  }

  EcountgmifsPathInternal(
    const EcountgmifsPathInternal&
  ) = delete;

  EcountgmifsPathInternal(
    EcountgmifsPathInternal&&
  ) = delete;

  EcountgmifsPathInternal& operator=(
    const EcountgmifsPathInternal&
  ) = delete;

  EcountgmifsPathInternal& operator=(
    EcountgmifsPathInternal&&
  ) = delete;

  const EcountgmifsPath& view() const noexcept
  {
    return api;
  }

  void set_null_time(
      double elapsed_time
  )
  {
    check_nonnegative_scalar(
      elapsed_time,
      "null_time"
    );

    api.null_time =
      elapsed_time;
  }

  void set_saturated_time(
      double elapsed_time
  )
  {
    check_nonnegative_scalar(
      elapsed_time,
      "saturated_time"
    );

    api.saturated_time =
      elapsed_time;
  }

  void set_total_time(
      double elapsed_time
  )
  {
    check_nonnegative_scalar(
      elapsed_time,
      "total_time"
    );

    api.total_time =
      elapsed_time;
  }

  void store_null_model()
  {
    api.null_negloglik =
      current_state.negloglik();

    api.null_theta =
      current_state.theta();

    api.null_family_parameters =
      current_state.family_parameters();

    api.null_link_parameters =
      current_state.link_parameters();
  }

  void store_saturated_model(
      const arma::vec& family_parameters,
      double negloglik
  )
  {
    check_finite_scalar(
      negloglik,
      "saturated negloglik"
    );

    api.saturated_family_parameters =
      family_parameters;

    api.saturated_negloglik =
      negloglik;
  }

  double pseudo_r2(
      double negloglik
  ) const noexcept
  {
    const double denominator =
      api.null_negloglik -
      api.saturated_negloglik;

    if (
        !std::isfinite(negloglik) ||
          !std::isfinite(denominator) ||
          denominator <= 0.0
    ) {
      return arma::datum::nan;
    }

    return
    (
        api.null_negloglik -
          negloglik
    ) /
      denominator;
  }

  /*
   * Process criterion values for the initial null state. No preceding state
   * exists yet, so this call can only start pending improvement streaks.
   */
  void update_best_criteria()
  {
    update_best_criteria_impl(
      nullptr,
      nullptr,
      nullptr,
      nullptr
    );
  }

  /*
   * Process criterion values for a completed stagewise iteration. The four
   * supplied parameter vectors represent the preceding completed state and
   * can be materialized only if an improvement streak ends here.
   */
  void update_best_criteria(
      const arma::vec& previous_beta,
      const arma::vec& previous_theta,
      const arma::vec& previous_family_parameters,
      const arma::vec& previous_link_parameters
  )
  {
    update_best_criteria_impl(
      &previous_beta,
      &previous_theta,
      &previous_family_parameters,
      &previous_link_parameters
    );
  }

  /*
   * Save the initial/current state according to the configured strategy.
   * This overload is used when no preceding parameter backup is needed.
   */
  void save_current_state(
      bool force = false
  )
  {
    save_current_state_impl(
      force,
      nullptr,
      nullptr,
      nullptr,
      nullptr
    );
  }

  /*
   * For ACTIVE_SET_CHANGE, a support change at the current iteration proves
   * that the preceding active-set segment has ended. Save that preceding
   * state once, using the already available parameter backups.
   */
  void save_current_state(
      const arma::vec& previous_beta,
      const arma::vec& previous_theta,
      const arma::vec& previous_family_parameters,
      const arma::vec& previous_link_parameters,
      bool force = false
  )
  {
    save_current_state_impl(
      force,
      &previous_beta,
      &previous_theta,
      &previous_family_parameters,
      &previous_link_parameters
    );
  }

  /*
   * Advance the cheap preceding-state metadata only after both best-criterion
   * and ordinary path tracking have had the opportunity to consume it.
   */
  void remember_current_state()
  {
    const EcountgmifsState& current =
      current_state.view();

    if (
        current.criteria.size() !=
          previous_criteria_.size()
    ) {
      Rcpp::stop(
        "criterion tracking size mismatch"
      );
    }

    previous_state_valid_ =
      true;

    previous_iteration_ =
      current.iteration;

    previous_negloglik_ =
      current.negloglik;

    previous_pseudo_r2_ =
      pseudo_r2(
        current.negloglik
      );

    previous_elapsed_time_ =
      current.elapsed_time;

    std::copy(
      current.criteria.begin(),
      current.criteria.end(),
      previous_criteria_.begin()
    );
  }

  /*
   * At termination there is no following non-improving iteration to close a
   * final improvement streak. Materialize the terminal state once and point
   * every still-pending criterion to it.
   */
  void flush_pending_best_criteria()
  {
    bool has_pending = false;

    for (const uint8_t pending : best_pending_) {
      if (pending != 0) {
        has_pending = true;
        break;
      }
    }

    if (!has_pending) {
      return;
    }

    const R_xlen_t state_index =
      ensure_current_state_saved();

    for (
        std::size_t i = 0;
        i < best_pending_.size();
        ++i
    ) {
      if (best_pending_[i] == 0) {
        continue;
      }

      api.best_criteria[i].state_index =
        state_index;

      best_pending_[i] =
        0;
    }
  }

  void finalize_message(
      EnumStagewiseTerminationReason reason,
      const std::string& detail
  )
  {
    api.message =
      "[" +
      std::string(
        stagewise_termination_reason_label(reason)
      ) +
        "] " +
        detail;
  }

  Rcpp::List current_state_to_list() const
  {
    EcountgmifsState snapshot =
      current_state.view();

    snapshot.pseudo_r2 =
      pseudo_r2(snapshot.negloglik);

    return EcountgmifsStateInternal::to_list(
      snapshot
    );
  }

  Rcpp::List to_list() const
  {
    const R_xlen_t criterion_count =
      static_cast<R_xlen_t>(
        api.best_criteria.size()
      );

    Rcpp::List best_criteria(
        criterion_count
    );

    Rcpp::CharacterVector best_criterion_names(
        criterion_count
    );

    for (
        std::size_t i = 0;
        i < api.best_criteria.size();
        ++i
    ) {
      const R_xlen_t r_index =
        static_cast<R_xlen_t>(i);

      const EcountgmifsBestCriterion& best =
        api.best_criteria[i];

      if (best.state_index < 0) {
        Rcpp::stop(
          "best criterion '%s' has no finalized state",
          best.name.c_str()
        );
      }

      best_criterion_names[r_index] =
        best.name;

      best_criteria[r_index] =
        Rcpp::List::create(
          Rcpp::Named("value") =
            best.value,

            Rcpp::Named("state") =
              saved_state_to_list(
                best.state_index
              )
        );
    }

    best_criteria.attr("names") =
      best_criterion_names;

    const R_xlen_t state_count =
      api.state_count;

    Rcpp::CharacterVector state_names(
        state_count
    );

    Rcpp::NumericVector iterations(
        state_count
    );

    Rcpp::NumericVector negloglik(
        state_count
    );

    Rcpp::NumericVector pseudo_r2_values(
        state_count
    );

    Rcpp::NumericVector elapsed_time(
        state_count
    );

    Rcpp::List beta(state_count);
    Rcpp::List theta(state_count);
    Rcpp::List family_parameters(state_count);
    Rcpp::List link_parameters(state_count);
    Rcpp::List xbeta(state_count);
    Rcpp::List wtheta(state_count);
    Rcpp::List eta(state_count);
    Rcpp::List mu(state_count);
    Rcpp::List active_set(state_count);

    for (R_xlen_t i = 0; i < state_count; ++i)
    {
      const R_xlen_t saved_index =
        checked_saved_index(
          api.state_indices[i]
        );

      state_names[i] =
        api.saved_states.state_names[saved_index];

      iterations[i] =
        api.saved_states.iterations[saved_index];

      negloglik[i] =
        api.saved_states.negloglik[saved_index];

      pseudo_r2_values[i] =
        api.saved_states.pseudo_r2[saved_index];

      elapsed_time[i] =
        api.saved_states.elapsed_time[saved_index];

      beta[i] =
        api.saved_states.beta[saved_index];

      theta[i] =
        api.saved_states.theta[saved_index];

      family_parameters[i] =
        api.saved_states.family_parameters[saved_index];

      link_parameters[i] =
        api.saved_states.link_parameters[saved_index];

      xbeta[i] =
        api.saved_states.xbeta[saved_index];

      wtheta[i] =
        api.saved_states.wtheta[saved_index];

      eta[i] =
        api.saved_states.eta[saved_index];

      mu[i] =
        api.saved_states.mu[saved_index];

      active_set[i] =
        api.saved_states.active_set[saved_index];
    }

    Rcpp::List criteria(
        criterion_count
    );

    Rcpp::CharacterVector criterion_names(
        criterion_count
    );

    for (
        std::size_t criterion_index = 0;
        criterion_index < api.best_criteria.size();
        ++criterion_index
    ) {
      const R_xlen_t r_criterion_index =
        static_cast<R_xlen_t>(
          criterion_index
        );

      const Rcpp::NumericVector saved_values =
        api.saved_states.criteria[
      r_criterion_index
        ];

      Rcpp::NumericVector criterion_values(
          state_count
      );

      for (R_xlen_t state_index = 0;
           state_index < state_count;
           ++state_index)
      {
        const R_xlen_t saved_index =
          checked_saved_index(
            api.state_indices[state_index]
          );

        criterion_values[state_index] =
          saved_values[saved_index];
      }

      criterion_values.attr("names") =
        state_names;

      criterion_names[r_criterion_index] =
        api.best_criteria[
      criterion_index
        ].name;

      criteria[r_criterion_index] =
        criterion_values;
    }

    criteria.attr("names") =
      criterion_names;

    iterations.attr("names") = state_names;
    negloglik.attr("names") = state_names;
    pseudo_r2_values.attr("names") = state_names;
    elapsed_time.attr("names") = state_names;
    beta.attr("names") = state_names;
    theta.attr("names") = state_names;
    family_parameters.attr("names") = state_names;
    link_parameters.attr("names") = state_names;
    xbeta.attr("names") = state_names;
    wtheta.attr("names") = state_names;
    eta.attr("names") = state_names;
    mu.attr("names") = state_names;
    active_set.attr("names") = state_names;

    Rcpp::List states =
      Rcpp::List::create(
        Rcpp::Named("iteration") =
          iterations,

          Rcpp::Named("negloglik") =
            negloglik,

            Rcpp::Named("criteria") =
              criteria,

              Rcpp::Named("pseudo_r2") =
                pseudo_r2_values,

                Rcpp::Named("elapsed_time") =
                  elapsed_time,

                  Rcpp::Named("beta") =
                    beta,

                    Rcpp::Named("theta") =
                      theta,

                      Rcpp::Named("family_parameters") =
                        family_parameters,

                        Rcpp::Named("link_parameters") =
                          link_parameters,

                          Rcpp::Named("xbeta") =
                            xbeta,

                            Rcpp::Named("wtheta") =
                              wtheta,

                              Rcpp::Named("eta") =
                                eta,

                                Rcpp::Named("mu") =
                                  mu,

                                  Rcpp::Named("active_set") =
                                    active_set
      );

    return Rcpp::List::create(
      Rcpp::Named("null_negloglik") =
        api.null_negloglik,

        Rcpp::Named("null_theta") =
          ecountgmifs::output::to_r_vector(
            api.null_theta
          ),

          Rcpp::Named("null_family_parameters") =
            ecountgmifs::output::to_r_vector(
              api.null_family_parameters
            ),

            Rcpp::Named("null_link_parameters") =
              ecountgmifs::output::to_r_vector(
                api.null_link_parameters
              ),

              Rcpp::Named("saturated_negloglik") =
                api.saturated_negloglik,

                Rcpp::Named("saturated_family_parameters") =
                  ecountgmifs::output::to_r_vector(
                    api.saturated_family_parameters
                  ),

                  Rcpp::Named("null_time") =
                    api.null_time,

                    Rcpp::Named("saturated_time") =
                      api.saturated_time,

                      Rcpp::Named("total_time") =
                        api.total_time,

                        Rcpp::Named("best_criteria") =
                          best_criteria,

                          Rcpp::Named("states") =
                            states,

                            Rcpp::Named("last_saved_active_set") =
                              ecountgmifs::output::to_r_logical_vector(
                                api.last_saved_active_set
                              ),

                              Rcpp::Named("active_set_changed") =
                                api.active_set_changed,

                                Rcpp::Named("message") =
                                  api.message
    );
  }

private:
  static bool active_set_differs(
      const arma::uvec& current_active_set,
      const arma::vec& previous_beta
  ) noexcept
  {
    if (
        current_active_set.n_elem !=
          previous_beta.n_elem
    ) {
      return true;
    }

    for (
        arma::uword i = 0;
        i < previous_beta.n_elem;
        ++i
    ) {
      const bool previous_active =
        previous_beta[i] != 0.0;

      const bool current_active =
        current_active_set[i] != 0;

      if (previous_active != current_active) {
        return true;
      }
    }

    return false;
  }

  static arma::uvec active_set_from_beta(
      const arma::vec& beta
  )
  {
    arma::uvec active_set(
        beta.n_elem
    );

    for (
        arma::uword i = 0;
        i < beta.n_elem;
        ++i
    ) {
      active_set[i] =
        beta[i] != 0.0;
    }

    return active_set;
  }

  R_xlen_t checked_saved_index(
      int index
  ) const
  {
    if (
        index < 0 ||
          static_cast<R_xlen_t>(index) >=
          api.saved_states.count
    ) {
      Rcpp::stop(
        "saved-state index is out of range"
      );
    }

    return static_cast<R_xlen_t>(
      index
    );
  }

  R_xlen_t checked_saved_index(
      R_xlen_t index
  ) const
  {
    if (
        index < 0 ||
          index >= api.saved_states.count
    ) {
      Rcpp::stop(
        "saved-state index is out of range"
      );
    }

    return index;
  }

  static R_xlen_t grown_capacity(
      R_xlen_t current_capacity
  )
  {
    constexpr R_xlen_t minimum_capacity = 16;

    if (current_capacity < minimum_capacity) {
      return minimum_capacity;
    }

    if (
        current_capacity >
      std::numeric_limits<R_xlen_t>::max() / 2
    ) {
      Rcpp::stop(
        "saved-state storage capacity overflow"
      );
    }

    return current_capacity * 2;
  }

  static void grow_numeric_vector(
      Rcpp::NumericVector& values,
      R_xlen_t initialized_count,
      R_xlen_t new_capacity
  )
  {
    Rcpp::NumericVector grown(
        new_capacity
    );

    std::copy_n(
      values.begin(),
      initialized_count,
      grown.begin()
    );

    values =
      grown;
  }

  static void grow_integer_vector(
      Rcpp::IntegerVector& values,
      R_xlen_t initialized_count,
      R_xlen_t new_capacity
  )
  {
    Rcpp::IntegerVector grown(
        new_capacity
    );

    std::copy_n(
      values.begin(),
      initialized_count,
      grown.begin()
    );

    values =
      grown;
  }

  static void grow_character_vector(
      Rcpp::CharacterVector& values,
      R_xlen_t initialized_count,
      R_xlen_t new_capacity
  )
  {
    Rcpp::CharacterVector grown(
        new_capacity
    );

    for (
        R_xlen_t i = 0;
        i < initialized_count;
        ++i
    ) {
      grown[i] =
        values[i];
    }

    values =
      grown;
  }

  static void grow_list(
      Rcpp::List& values,
      R_xlen_t initialized_count,
      R_xlen_t new_capacity
  )
  {
    Rcpp::List grown(
        new_capacity
    );

    for (
        R_xlen_t i = 0;
        i < initialized_count;
        ++i
    ) {
      grown[i] =
        values[i];
    }

    values =
      grown;
  }

  void ensure_saved_state_capacity()
  {
    const R_xlen_t current_capacity =
      api.saved_states.iterations.size();

    if (
        api.saved_states.count <
          current_capacity
    ) {
      return;
    }

    const R_xlen_t new_capacity =
      grown_capacity(
        current_capacity
      );

    grow_character_vector(
      api.saved_states.state_names,
      api.saved_states.count,
      new_capacity
    );

    grow_numeric_vector(
      api.saved_states.iterations,
      api.saved_states.count,
      new_capacity
    );

    grow_numeric_vector(
      api.saved_states.negloglik,
      api.saved_states.count,
      new_capacity
    );

    grow_numeric_vector(
      api.saved_states.pseudo_r2,
      api.saved_states.count,
      new_capacity
    );

    grow_numeric_vector(
      api.saved_states.elapsed_time,
      api.saved_states.count,
      new_capacity
    );

    for (
        R_xlen_t i = 0;
        i < api.saved_states.criteria.size();
        ++i
    ) {
      Rcpp::NumericVector values =
        api.saved_states.criteria[i];

      grow_numeric_vector(
        values,
        api.saved_states.count,
        new_capacity
      );

      api.saved_states.criteria[i] =
        values;
    }

    grow_list(
      api.saved_states.beta,
      api.saved_states.count,
      new_capacity
    );

    grow_list(
      api.saved_states.theta,
      api.saved_states.count,
      new_capacity
    );

    grow_list(
      api.saved_states.family_parameters,
      api.saved_states.count,
      new_capacity
    );

    grow_list(
      api.saved_states.link_parameters,
      api.saved_states.count,
      new_capacity
    );

    grow_list(
      api.saved_states.xbeta,
      api.saved_states.count,
      new_capacity
    );

    grow_list(
      api.saved_states.wtheta,
      api.saved_states.count,
      new_capacity
    );

    grow_list(
      api.saved_states.eta,
      api.saved_states.count,
      new_capacity
    );

    grow_list(
      api.saved_states.mu,
      api.saved_states.count,
      new_capacity
    );

    grow_list(
      api.saved_states.active_set,
      api.saved_states.count,
      new_capacity
    );
  }

  void ensure_path_index_capacity()
  {
    const R_xlen_t current_capacity =
      api.state_indices.size();

    if (
        api.state_count <
          current_capacity
    ) {
      return;
    }

    grow_integer_vector(
      api.state_indices,
      api.state_count,
      grown_capacity(
        current_capacity
      )
    );
  }

  void append_path_state_index(
      R_xlen_t state_index,
      const arma::uvec& active_set
  )
  {
    if (
        state_index >
      static_cast<R_xlen_t>(
        std::numeric_limits<int>::max()
      )
    ) {
      Rcpp::stop(
        "too many saved states for R integer indexing"
      );
    }

    const int r_index =
      static_cast<int>(state_index);

    if (
        api.state_count > 0 &&
          api.state_indices[
    api.state_count - 1
          ] == r_index
    ) {
      api.last_saved_active_set =
        active_set;

      return;
    }

    ensure_path_index_capacity();

    api.state_indices[
    api.state_count
    ] =
      r_index;

    ++api.state_count;

    api.last_saved_active_set =
    active_set;
  }


  R_xlen_t append_saved_state(
      uint64_t iteration,
      double negloglik,
      double pseudo_r2_value,
      double elapsed_time,
      const Rcpp::NumericVector& criteria,
      const arma::vec& beta,
      const arma::vec& theta,
      const arma::vec& family_parameters,
      const arma::vec& link_parameters,
      const arma::vec& xbeta,
      const arma::vec& wtheta,
      const arma::vec& eta,
      const arma::vec& mu,
      const arma::uvec& active_set
  )
  {
    const auto existing =
      saved_state_index_by_iteration_.find(
        iteration
      );

    if (
        existing !=
          saved_state_index_by_iteration_.end()
    ) {
      return existing->second;
    }

    if (
        criteria.size() !=
          static_cast<R_xlen_t>(
            input.criteria.size()
          )
    ) {
      Rcpp::stop(
        "saved-state criterion size mismatch"
      );
    }

    ensure_saved_state_capacity();

    const R_xlen_t state_index =
      api.saved_states.count;

    api.saved_states.state_names[state_index] =
      "iter_" +
      std::to_string(iteration);

    api.saved_states.iterations[state_index] =
      static_cast<double>(iteration);

    api.saved_states.negloglik[state_index] =
      negloglik;

    api.saved_states.pseudo_r2[state_index] =
      pseudo_r2_value;

    api.saved_states.elapsed_time[state_index] =
      elapsed_time;

    for (
        std::size_t i = 0;
        i < input.criteria.size();
        ++i
    ) {
      const R_xlen_t r_index =
        static_cast<R_xlen_t>(i);

      Rcpp::NumericVector values =
        api.saved_states.criteria[r_index];

      values[state_index] =
        criteria[r_index];
    }

    api.saved_states.beta[state_index] =
      ecountgmifs::output::to_r_vector(
        beta
      );

    api.saved_states.theta[state_index] =
      ecountgmifs::output::to_r_vector(
        theta
      );

    api.saved_states.family_parameters[state_index] =
      ecountgmifs::output::to_r_vector(
        family_parameters
      );

    api.saved_states.link_parameters[state_index] =
      ecountgmifs::output::to_r_vector(
        link_parameters
      );

    api.saved_states.xbeta[state_index] =
      ecountgmifs::output::to_r_vector(
        xbeta
      );

    api.saved_states.wtheta[state_index] =
      ecountgmifs::output::to_r_vector(
        wtheta
      );

    api.saved_states.eta[state_index] =
      ecountgmifs::output::to_r_vector(
        eta
      );

    api.saved_states.mu[state_index] =
      ecountgmifs::output::to_r_vector(
        mu
      );

    api.saved_states.active_set[state_index] =
      ecountgmifs::output::to_r_logical_vector(
        active_set
      );

    ++api.saved_states.count;

    saved_state_index_by_iteration_.emplace(
      iteration,
      state_index
    );

    return state_index;
  }

  R_xlen_t ensure_current_state_saved()
  {
    const EcountgmifsState& current =
      current_state.view();

    return append_saved_state(
      current.iteration,
      current.negloglik,
      pseudo_r2(current.negloglik),
      current.elapsed_time,
      current.criteria,
      current.param.param.beta,
      current.param.param.theta,
      current.param.param.family_parameters,
      current.param.param.link_parameters,
      current.param.xbeta,
      current.param.wtheta,
      current.param.eta,
      current.param.mu,
      current.param.active_set
    );
  }

  R_xlen_t ensure_previous_state_saved(
      const arma::vec& previous_beta,
      const arma::vec& previous_theta,
      const arma::vec& previous_family_parameters,
      const arma::vec& previous_link_parameters
  )
  {
    if (!previous_state_valid_) {
      Rcpp::stop(
        "no preceding state is available for deferred saving"
      );
    }

    const auto existing =
      saved_state_index_by_iteration_.find(
        previous_iteration_
      );

    if (
        existing !=
          saved_state_index_by_iteration_.end()
    ) {
      return existing->second;
    }

    arma::vec xbeta =
      input.X *
      previous_beta;

    arma::vec wtheta =
      input.w *
      previous_theta;

    arma::vec eta =
      input.offset;

    eta += xbeta;
    eta += wtheta;

    arma::vec mu(
        input.X.n_rows
    );

    input.family_link->inverse(
        eta,
        previous_link_parameters,
        mu
    );

    check_vector_finite(
      mu,
      "saved mu"
    );

    const arma::uvec active_set =
      active_set_from_beta(
        previous_beta
      );

    return append_saved_state(
      previous_iteration_,
      previous_negloglik_,
      previous_pseudo_r2_,
      previous_elapsed_time_,
      previous_criteria_,
      previous_beta,
      previous_theta,
      previous_family_parameters,
      previous_link_parameters,
      xbeta,
      wtheta,
      eta,
      mu,
      active_set
    );
  }

  void update_best_criteria_impl(
      const arma::vec* previous_beta,
      const arma::vec* previous_theta,
      const arma::vec* previous_family_parameters,
      const arma::vec* previous_link_parameters
  )
  {
    const EcountgmifsState& current =
      current_state.view();

    const std::size_t criterion_count =
      api.best_criteria.size();

    if (
        current.criteria.size() !=
          static_cast<R_xlen_t>(
            criterion_count
          )
    ) {
      Rcpp::stop(
        "criterion tracking size mismatch"
      );
    }

    bool save_previous = false;

    for (
        std::size_t i = 0;
        i < criterion_count;
        ++i
    ) {
      const double candidate =
        current.criteria[
      static_cast<R_xlen_t>(i)
        ];

      const bool improved =
        candidate <
          api.best_criteria[i].value;

      criterion_improved_[i] =
        static_cast<uint8_t>(
          improved
        );

      if (
          best_pending_[i] != 0 &&
            !improved
      ) {
        save_previous =
          true;
      }
    }

    R_xlen_t previous_state_index = -1;

    if (save_previous) {
      if (
          previous_beta == nullptr ||
            previous_theta == nullptr ||
            previous_family_parameters == nullptr ||
            previous_link_parameters == nullptr
      ) {
        Rcpp::stop(
          "preceding parameters are required to finalize a best state"
        );
      }

      previous_state_index =
        ensure_previous_state_saved(
          *previous_beta,
          *previous_theta,
          *previous_family_parameters,
          *previous_link_parameters
        );
    }

    for (
        std::size_t i = 0;
        i < criterion_count;
        ++i
    ) {
      EcountgmifsBestCriterion& best =
        api.best_criteria[i];

      if (criterion_improved_[i] != 0) {
        best.value =
          current.criteria[
        static_cast<R_xlen_t>(i)
          ];

        best_pending_[i] =
          1;

        continue;
      }

      if (best_pending_[i] != 0) {
        best.state_index =
          previous_state_index;

        best_pending_[i] =
          0;
      }
    }
  }

  void save_current_state_impl(
      bool force,
      const arma::vec* previous_beta,
      const arma::vec* previous_theta,
      const arma::vec* previous_family_parameters,
      const arma::vec* previous_link_parameters
  )
  {
    if (
        control.state_track_strategy ==
          EnumStateTrackStrategy::NO_STATE_TRACKING
    ) {
      return;
    }

    const EcountgmifsState& current =
      current_state.view();

    if (force) {
      const R_xlen_t state_index =
        ensure_current_state_saved();

      append_path_state_index(
        state_index,
        current.param.active_set
      );

      return;
    }

    switch (control.state_track_strategy)
    {
    case EnumStateTrackStrategy::ACTIVE_SET_CHANGE:
    {
      if (
          previous_beta == nullptr ||
            previous_theta == nullptr ||
            previous_family_parameters == nullptr ||
            previous_link_parameters == nullptr ||
            !previous_state_valid_
      ) {
      return;
    }

      api.active_set_changed =
        active_set_differs(
          current.param.active_set,
          *previous_beta
        );

      if (!api.active_set_changed) {
        return;
      }

      const R_xlen_t state_index =
        ensure_previous_state_saved(
          *previous_beta,
          *previous_theta,
          *previous_family_parameters,
          *previous_link_parameters
        );

      append_path_state_index(
        state_index,
        active_set_from_beta(
          *previous_beta
        )
      );

      return;
    }

    case EnumStateTrackStrategy::ALL_ITERATION:
    {
      const R_xlen_t state_index =
        ensure_current_state_saved();

      append_path_state_index(
        state_index,
        current.param.active_set
      );

      return;
    }

    case EnumStateTrackStrategy::EVERY_K_ITERATION:
    {
      if (
          current.iteration %
            control.state_track_freq != 0
      ) {
      return;
    }

      const R_xlen_t state_index =
        ensure_current_state_saved();

      append_path_state_index(
        state_index,
        current.param.active_set
      );

      return;
    }

    case EnumStateTrackStrategy::NO_STATE_TRACKING:
      return;
    }
  }

  Rcpp::NumericVector saved_criteria_at(
      R_xlen_t state_index
  ) const
  {
    state_index =
      checked_saved_index(
        state_index
      );

    const R_xlen_t criterion_count =
      static_cast<R_xlen_t>(
        api.best_criteria.size()
      );

    Rcpp::NumericVector values(
        criterion_count
    );

    Rcpp::CharacterVector names(
        criterion_count
    );

    for (
        R_xlen_t i = 0;
        i < criterion_count;
        ++i
    ) {
      const Rcpp::NumericVector saved_values =
        api.saved_states.criteria[i];

      values[i] =
        saved_values[state_index];

      names[i] =
        api.best_criteria[
      static_cast<std::size_t>(i)
        ].name;
    }

    values.attr("names") =
      names;

    return values;
  }

  Rcpp::List saved_state_to_list(
      R_xlen_t state_index
  ) const
  {
    state_index =
      checked_saved_index(
        state_index
      );

    Rcpp::List parameters =
      Rcpp::List::create(
        Rcpp::Named("beta") =
          api.saved_states.beta[state_index],

                               Rcpp::Named("theta") =
                                 api.saved_states.theta[state_index],

                                                       Rcpp::Named("family_parameters") =
                                                         api.saved_states.family_parameters[state_index],

                                                                                           Rcpp::Named("link_parameters") =
                                                                                             api.saved_states.link_parameters[state_index]
      );

    Rcpp::List predictors =
      Rcpp::List::create(
        Rcpp::Named("parameters") =
          parameters,

          Rcpp::Named("xbeta") =
            api.saved_states.xbeta[state_index],

                                  Rcpp::Named("wtheta") =
                                    api.saved_states.wtheta[state_index],

                                                           Rcpp::Named("eta") =
                                                             api.saved_states.eta[state_index],

                                                                                 Rcpp::Named("mu") =
                                                                                   api.saved_states.mu[state_index],

                                                                                                      Rcpp::Named("active_set") =
                                                                                                        api.saved_states.active_set[state_index]
      );

    return Rcpp::List::create(
      Rcpp::Named("predictors") =
        predictors,

        Rcpp::Named("negloglik") =
          api.saved_states.negloglik[state_index],

                                    Rcpp::Named("criteria") =
                                      saved_criteria_at(
                                        state_index
                                      ),

                                      Rcpp::Named("iteration") =
                                        api.saved_states.iterations[state_index],

                                                                   Rcpp::Named("pseudo_r2") =
                                                                     api.saved_states.pseudo_r2[state_index],

                                                                                               Rcpp::Named("elapsed_time") =
                                                                                                 api.saved_states.elapsed_time[state_index]
    );
  }
};


struct EcountgmifsStagewiseInternal
{
private:
  const EcountgmifsInput& input;
  const EcountgmifsControl& control;

  EcountgmifsStateInternal& state;
  EcountgmifsGradientsInternal& gradient;
  EcountgmifsPathInternal& path;

  EcountgmifsStagewise api;

  NloptOptimizerInternal null_nonpen_optimizer;
  NloptOptimizerInternal null_family_optimizer;
  NloptOptimizerInternal null_link_optimizer;

  NloptOptimizerInternal stagewise_nonpen_optimizer;
  NloptOptimizerInternal stagewise_family_optimizer;
  NloptOptimizerInternal stagewise_link_optimizer;

  uint64_t nonpen_evaluation_count = 0;
  uint64_t family_evaluation_count = 0;
  uint64_t link_evaluation_count = 0;
  uint64_t saturated_family_evaluation_count = 0;

public:
  EcountgmifsStagewiseInternal(
    const EcountgmifsInput& input_,
    const EcountgmifsControl& control_,
    EcountgmifsStateInternal& state_,
    EcountgmifsGradientsInternal& gradient_,
    EcountgmifsPathInternal& path_
  ) :
  input(input_),
  control(control_),
  state(state_),
  gradient(gradient_),
  path(path_),
  api {},

  null_nonpen_optimizer(
    state_.theta(),
    control_.theta_lower_bounds,
    control_.theta_upper_bounds,
    &EcountgmifsStagewiseInternal::nonpen_objective,
    this,
    control_.nonpen_nlopt.algorithm,
    control_.nonpen_nlopt.xtol_rel,
    control_.nonpen_nlopt.ftol_rel,
    control_.nonpen_nlopt.maxeval
  ),

  null_family_optimizer(
    state_.family_parameters(),
    input_.family_link->family_parameter_lower_bounds(),
    input_.family_link->family_parameter_upper_bounds(),
    &EcountgmifsStagewiseInternal::family_objective,
    this,
    control_.family_nlopt.algorithm,
    control_.family_nlopt.xtol_rel,
    control_.family_nlopt.ftol_rel,
    control_.family_nlopt.maxeval
  ),

  null_link_optimizer(
    state_.link_parameters(),
    input_.family_link->link_parameter_lower_bounds(),
    input_.family_link->link_parameter_upper_bounds(),
    &EcountgmifsStagewiseInternal::link_objective,
    this,
    control_.link_nlopt.algorithm,
    control_.link_nlopt.xtol_rel,
    control_.link_nlopt.ftol_rel,
    control_.link_nlopt.maxeval
  ),

  stagewise_nonpen_optimizer(
    state_.theta(),
    control_.theta_lower_bounds,
    control_.theta_upper_bounds,
    &EcountgmifsStagewiseInternal::nonpen_objective,
    this,
    control_.nonpen_nlopt.algorithm,
    control_.nonpen_nlopt.xtol_rel,
    control_.nonpen_nlopt.ftol_rel,
    control_.nonpen_nlopt.maxeval
  ),

  stagewise_family_optimizer(
    state_.family_parameters(),
    input_.family_link->family_parameter_lower_bounds(),
    input_.family_link->family_parameter_upper_bounds(),
    &EcountgmifsStagewiseInternal::family_objective,
    this,
    control_.family_nlopt.algorithm,
    control_.family_nlopt.xtol_rel,
    control_.family_nlopt.ftol_rel,
    control_.family_nlopt.maxeval
  ),

  stagewise_link_optimizer(
    state_.link_parameters(),
    input_.family_link->link_parameter_lower_bounds(),
    input_.family_link->link_parameter_upper_bounds(),
    &EcountgmifsStagewiseInternal::link_objective,
    this,
    control_.link_nlopt.algorithm,
    control_.link_nlopt.xtol_rel,
    control_.link_nlopt.ftol_rel,
    control_.link_nlopt.maxeval
  )
  {
    api.saturated_family_parameters =
      input.family_link->family_initial_parameters();

    api.theta_before_optimize.set_size(
      state.theta().n_elem
    );

    api.family_parameters_before_optimize.set_size(
      state.family_parameters().n_elem
    );

    api.link_parameters_before_optimize.set_size(
      state.link_parameters().n_elem
    );

    api.saturated_family_parameters_before_optimize.set_size(
      api.saturated_family_parameters.n_elem
    );

    api.beta_start.zeros(
      input.X.n_cols
    );

    api.beta_trial.zeros(
      input.X.n_cols
    );

    api.delta_beta.zeros(
      input.X.n_cols
    );

    api.epsilon =
      control.epsilon_start;
  }

  EcountgmifsStagewiseInternal(
    const EcountgmifsStagewiseInternal&
  ) = delete;

  EcountgmifsStagewiseInternal(
    EcountgmifsStagewiseInternal&&
  ) = delete;

  EcountgmifsStagewiseInternal& operator=(
    const EcountgmifsStagewiseInternal&
  ) = delete;

  EcountgmifsStagewiseInternal& operator=(
    EcountgmifsStagewiseInternal&&
  ) = delete;

  const EcountgmifsStagewise& view() const noexcept
  {
    return api;
  }

  Rcpp::List to_list() const
  {
    return Rcpp::List::create(
      Rcpp::Named("phase") =
        static_cast<int>(api.phase),

        Rcpp::Named("termination_reason") =
          static_cast<int>(api.termination_reason),

          Rcpp::Named("termination_detail") =
            api.termination_detail,

            Rcpp::Named("epsilon") =
              api.epsilon,

              Rcpp::Named("halving_count") =
                api.halving_count
    );
  }

  void fit()
  {
    using Clock =
      std::chrono::steady_clock;

    const auto total_start =
      Clock::now();

    begin_fit();

    const auto null_start =
      Clock::now();

    fit_null_model();

    path.set_null_time(
      elapsed_seconds(null_start)
    );

    const auto saturated_start =
      Clock::now();

    fit_saturated_model();

    path.set_saturated_time(
      elapsed_seconds(saturated_start)
    );

    /*
     * The current State is still the fitted null model because saturated
     * fitting uses separate Stagewise workspace. Evaluate its criteria only
     * after the completed null fit, then save it once both baselines exist.
     */
    state.set_elapsed_time(0.0);
    state.evaluate_criteria();
    path.update_best_criteria();
    path.save_current_state(true);
    path.remember_current_state();

    fit_stagewise_model();

    /*
     * Wall-clock total includes null, saturated, every stagewise iteration,
     * criteria, state tracking, stopping checks, and other fitting overhead.
     * Result serialization occurs after fit() and is intentionally excluded.
     */
    path.set_total_time(
      elapsed_seconds(total_start)
    );
  }

private:
  static double elapsed_seconds(
      const std::chrono::steady_clock::time_point& start
  ) noexcept
  {
    return std::chrono::duration<double>(
      std::chrono::steady_clock::now() - start
    ).count();
  }

  void begin_fit()
  {
    if (
        api.phase !=
          EnumStagewisePhase::STAGEWISE_NOT_STARTED
    ) {
      Rcpp::stop(
        "EcountgmifsStagewiseInternal::fit() "
        "may only be called once"
      );
    }

    api.phase =
      EnumStagewisePhase::STAGEWISE_NONPEN;

    api.termination_reason =
      EnumStagewiseTerminationReason::
        STAGEWISE_RUNNING;

    api.termination_detail =
      "Ready for non-penalized fitting.";
  }

  void fit_null_model()
  {
    api.phase =
      EnumStagewisePhase::STAGEWISE_NONPEN;

    ECOUNTGMIFS_VERBOSE(
      control.verbose,
      "null: start"
      << ", max_outer=" << control.null_iteration_max
      << ", family_tolerance=" << control.null_family_parameter_abs_tol
      << ", initial_negloglik=" << state.negloglik()
    );

    /*
     * Allocate these buffers once. They are reused for every outer
     * theta/link/family alternation.
     */
    arma::vec theta_previous(
        state.theta().n_elem
    );

    arma::vec link_parameters_previous(
        state.link_parameters().n_elem
    );

    arma::vec family_parameters_previous(
        state.family_parameters().n_elem
    );

    const auto vectors_exactly_equal =
      [](
          const arma::vec& current,
          const arma::vec& previous
      ) noexcept
      {
        if (
            current.n_elem !=
              previous.n_elem
        ) {
          return false;
        }

        if (current.n_elem == 0) {
          return true;
        }

        return arma::all(
          current == previous
        );
      };

      const auto vectors_within_absolute_tolerance =
        [&](
            const arma::vec& current,
            const arma::vec& previous
        ) noexcept
        {
          if (
              current.n_elem !=
                previous.n_elem
          ) {
            return false;
          }

          if (current.n_elem == 0) {
            return true;
          }

          for (
              arma::uword i = 0;
              i < current.n_elem;
              ++i
          ) {
            if (
                std::abs(
                  current[i] - previous[i]
                ) >=
                  control.null_family_parameter_abs_tol
            ) {
              return false;
            }
          }

          return true;
        };

        for (
            uint64_t outer_iteration = 0;
            outer_iteration < control.null_iteration_max;
            ++outer_iteration
        ) {
          if ((outer_iteration & 255U) == 0U) {
            Rcpp::checkUserInterrupt();
          }

          /*
           * Save the complete null-model state before the alternating update.
           */
          const double negloglik_previous =
            state.negloglik();

          theta_previous =
            state.theta();

          link_parameters_previous =
            state.link_parameters();

          family_parameters_previous =
            state.family_parameters();

          ECOUNTGMIFS_VERBOSE(
            control.verbose,
            "null: iter=" << outer_iteration + 1
                          << " begin"
                          << ", negloglik="
                          << negloglik_previous
          );

          /*
           * Optimize nonpenalized regression parameters.
           */
          nonpen_evaluation_count = 0;

          ECOUNTGMIFS_VERBOSE(
            control.verbose,
            "null: iter=" << outer_iteration + 1
                          << " optimize_theta start"
          );

          optimize_theta_safely(
            null_nonpen_optimizer,
            "null"
          );

          ECOUNTGMIFS_VERBOSE(
            control.verbose,
            "null: iter=" << outer_iteration + 1
                          << " optimize_theta done"
                          << ", evaluations="
                          << nonpen_evaluation_count
                          << ", negloglik="
                          << state.negloglik()
          );

          /*
           * Optimize link parameters, when the link has any.
           */
          link_evaluation_count = 0;

          ECOUNTGMIFS_VERBOSE(
            control.verbose,
            "null: iter=" << outer_iteration + 1
                          << " optimize_link start"
          );

          optimize_link_safely(
            null_link_optimizer,
            "null"
          );

          ECOUNTGMIFS_VERBOSE(
            control.verbose,
            "null: iter=" << outer_iteration + 1
                          << " optimize_link done"
                          << ", evaluations="
                          << link_evaluation_count
                          << ", negloglik="
                          << state.negloglik()
          );

          /*
           * Optimize family parameters.
           */
          family_evaluation_count = 0;

          ECOUNTGMIFS_VERBOSE(
            control.verbose,
            "null: iter=" << outer_iteration + 1
                          << " optimize_family start"
          );

          optimize_family_safely(
            null_family_optimizer,
            "null"
          );

          const double negloglik_current =
            state.negloglik();

          ECOUNTGMIFS_VERBOSE(
            control.verbose,
            "null: iter=" << outer_iteration + 1
                          << " optimize_family done"
                          << ", evaluations="
                          << family_evaluation_count
                          << ", negloglik="
                          << negloglik_current
          );

          check_finite_scalar(
            negloglik_current,
            "null-model negloglik"
          );

          /*
           * Match the original glmSS stopping rule:
           *
           *   - objective unchanged exactly;
           *   - theta unchanged exactly;
           *   - link parameters unchanged exactly;
           *   - family parameters changed by less than control.null_family_parameter_abs_tol.
           *
           * glmSS used exact comparison for the objective and intercept, and an
           * absolute 1e-18 comparison for dispersion.
           */
          const bool negloglik_same =
            negloglik_current ==
            negloglik_previous;

          const bool theta_same =
            vectors_exactly_equal(
              state.theta(),
              theta_previous
            );

          const bool link_parameters_same =
            vectors_exactly_equal(
              state.link_parameters(),
              link_parameters_previous
            );

          const bool family_parameters_same =
            vectors_within_absolute_tolerance(
              state.family_parameters(),
              family_parameters_previous
            );

          const double objective_absolute_change =
            std::abs(
              negloglik_current -
                negloglik_previous
            );

          double theta_max_absolute_change = 0.0;

          if (state.theta().n_elem > 0) {
            theta_max_absolute_change =
              arma::abs(
                state.theta() -
                  theta_previous
              ).max();
          }

          double link_max_absolute_change = 0.0;

          if (state.link_parameters().n_elem > 0) {
            link_max_absolute_change =
              arma::abs(
                state.link_parameters() -
                  link_parameters_previous
              ).max();
          }

          double family_max_absolute_change = 0.0;

          if (state.family_parameters().n_elem > 0) {
            family_max_absolute_change =
              arma::abs(
                state.family_parameters() -
                  family_parameters_previous
              ).max();
          }

          ECOUNTGMIFS_VERBOSE(
            control.verbose,
            "null: iter=" << outer_iteration + 1
                          << " summary"
                          << ", objective_absolute_change="
                          << objective_absolute_change
                          << ", negloglik_same="
                          << negloglik_same
                          << ", theta_max_absolute_change="
                          << theta_max_absolute_change
                          << ", theta_same="
                          << theta_same
                          << ", link_max_absolute_change="
                          << link_max_absolute_change
                          << ", link_same="
                          << link_parameters_same
                          << ", family_max_absolute_change="
                          << family_max_absolute_change
                          << ", family_same="
                          << family_parameters_same
                          << ", family_tolerance="
                          << control.null_family_parameter_abs_tol
          );

          if (
              negloglik_same &&
                theta_same &&
                link_parameters_same &&
                family_parameters_same
          ) {
            path.store_null_model();

            api.phase =
              EnumStagewisePhase::STAGEWISE_SATURATED;

            api.termination_detail =
              "Null model fitted; ready for saturated fitting.";

            ECOUNTGMIFS_VERBOSE(
              control.verbose,
              "null: converged"
              << ", outer_iterations="
              << outer_iteration + 1
              << ", negloglik="
              << state.negloglik()
            );

            return;
          }
        }

        /*
         * Preserve the current behavior when the outer limit is reached:
         * retain the last valid fitted state and continue to saturated fitting.
         */
        path.store_null_model();

        api.phase =
          EnumStagewisePhase::STAGEWISE_SATURATED;

        api.termination_detail =
          "Null-model outer iteration limit reached; "
          "ready for saturated fitting.";

        ECOUNTGMIFS_VERBOSE(
          control.verbose,
          "null: outer iteration limit reached"
          << ", max_outer="
          << control.null_iteration_max
          << ", negloglik="
          << state.negloglik()
        );
  }

  void fit_saturated_model()
  {
    api.phase =
      EnumStagewisePhase::STAGEWISE_SATURATED;

    Rcpp::checkUserInterrupt();

    /*
     * Use the fitted null family parameters as the saturated optimizer's
     * starting point, but keep the workspace separate from current State.
     */
    api.saturated_family_parameters =
      state.family_parameters();

    refresh_saturated_negloglik();

    saturated_family_evaluation_count = 0;

    ECOUNTGMIFS_VERBOSE(
      control.verbose,
      "saturated: start"
      << ", family_parameters="
      << api.saturated_family_parameters.t()
      << ", initial_negloglik="
      << api.saturated_negloglik
    );

    NloptOptimizerInternal saturated_family_optimizer(
        api.saturated_family_parameters,
        input.family_link->family_parameter_lower_bounds(),
        input.family_link->family_parameter_upper_bounds(),
        &EcountgmifsStagewiseInternal::saturated_family_objective,
        this,
        control.family_nlopt.algorithm,
        control.family_nlopt.xtol_rel,
        control.family_nlopt.ftol_rel,
        control.family_nlopt.maxeval
    );

    optimize_saturated_family_safely(
      saturated_family_optimizer
    );

    path.store_saturated_model(
      api.saturated_family_parameters,
      api.saturated_negloglik
    );

    ECOUNTGMIFS_VERBOSE(
      control.verbose,
      "saturated: done"
      << ", evaluations="
      << saturated_family_evaluation_count
      << ", family_parameters="
      << api.saturated_family_parameters.t()
      << ", negloglik="
      << api.saturated_negloglik
    );

    api.phase =
      EnumStagewisePhase::STAGEWISE_ITERATION;

    api.termination_detail =
      "Saturated model fitted; ready for stagewise fitting.";
  }

  void fit_stagewise_model()
  {
    api.phase =
      EnumStagewisePhase::STAGEWISE_ITERATION;

    ElasticNetWeightWorkspace enet_workspace(
        input.X.n_cols
    );

    ECOUNTGMIFS_VERBOSE(
      control.verbose,
      "stagewise: start"
      << ", max_iterations="
      << control.stagewise_iteration_max
      << ", epsilon_start="
      << api.epsilon
      << ", negloglik="
      << state.negloglik()
    );

    for (
        uint64_t iteration = 1;
        iteration <= control.stagewise_iteration_max;
        ++iteration
    ) {
      const auto iteration_start =
        std::chrono::steady_clock::now();

      if ((iteration & 255U) == 0U) {
        Rcpp::checkUserInterrupt();
      }

      const double negloglik_previous =
        state.negloglik();

      if (iteration > 1) {
        api.epsilon =
          std::min(
            control.epsilon_max,
            api.epsilon * 2.0
          );
      }

      bool converged = beta_optimize(
        enet_workspace,
        iteration,
        negloglik_previous
      );

      if (
          converged == true
      ) {
        return;
      }

      optimize_theta_safely(
        stagewise_nonpen_optimizer,
        "stagewise"
      );

      optimize_link_safely(
        stagewise_link_optimizer,
        "stagewise"
      );

      optimize_family_safely(
        stagewise_family_optimizer,
        "stagewise"
      );

      /*
       * State iteration means completed stagewise updates, not attempted
       * iterations. Set it only after the whole accepted/refitted update.
       */
      state.begin_stagewise_iteration(
        iteration
      );

      state.evaluate_criteria();

      state.set_elapsed_time(
        elapsed_seconds(iteration_start)
      );

      path.update_best_criteria(
        api.beta_start,
        api.theta_before_optimize,
        api.family_parameters_before_optimize,
        api.link_parameters_before_optimize
      );

      const double negloglik_current =
        state.negloglik();

      const double pseudo_r2_current =
        path.pseudo_r2(
          negloglik_current
        );

      path.save_current_state(
        api.beta_start,
        api.theta_before_optimize,
        api.family_parameters_before_optimize,
        api.link_parameters_before_optimize
      );

      path.remember_current_state();

      const double objective_scale =
        std::max(
          1.0,
          std::max(
            std::abs(negloglik_previous),
            std::abs(negloglik_current)
          )
        );

      const double objective_relative_change =
        std::abs(
          negloglik_current -
            negloglik_previous
        ) /
          objective_scale;

      const double beta_step_norm =
        arma::norm(
          api.delta_beta,
          2
        );

      ECOUNTGMIFS_VERBOSE(
        control.verbose,
        "stagewise: iter="
        << iteration
        << " accepted"
        << ", epsilon="
        << api.epsilon
        << ", halvings="
        << api.halving_count
        << ", beta_step_l2="
        << beta_step_norm
        << ", negloglik="
        << negloglik_current
        << ", relative_change="
        << objective_relative_change
        << ", pseudo_r2="
        << pseudo_r2_current
      );

      if (
          beta_step_norm <=
            control.stagewise_beta_step_norm_tol
      ) {
        finish_stagewise(
          EnumStagewiseTerminationReason::
            STAGEWISE_BETA_STALLED,
            "The accepted beta-step norm is within stagewise_beta_step_norm_tol."
        );

        return;
      }

      if (
          objective_relative_change <=
            control.stagewise_objective_rel_tol
      ) {
        finish_stagewise(
          EnumStagewiseTerminationReason::
            STAGEWISE_OBJECTIVE_STALLED,
            "The relative negative log-likelihood change is within tolerance."
        );

        return;
      }

      if (
          control.loglik_reltol_cutoff > 0.0 &&
            std::isfinite(pseudo_r2_current) &&
            pseudo_r2_current >=
            1.0 - control.loglik_reltol_cutoff
      ) {
        finish_stagewise(
          EnumStagewiseTerminationReason::
            STAGEWISE_PSEUDO_R2_CUTOFF_REACHED,
            "The pseudo-R2 cutoff was reached."
        );

        return;
      }
    }

    finish_stagewise(
      EnumStagewiseTerminationReason::
        STAGEWISE_ITERATION_LIMIT_REACHED,
        "The stagewise iteration limit was reached."
    );
  }

  bool beta_optimize(
      ElasticNetWeightWorkspace& enet_workspace,
      uint64_t iteration,
      double negloglik_previous
  )
  {
    /*
     * Immutable beta for every trial in this halving sequence.
     */
    api.beta_start =
      state.beta();

    const arma::vec& beta_gradient =
      gradient.update_beta_gradient();

    prepare_elastic_net_gradient(
      beta_gradient,
      enet_workspace
    );

    api.halving_count = 0;

    while (
        api.epsilon >=
          control.epsilon_min
    ) {
      solve_elastic_net_1D_weight_prepared_inplace(
        input.weight_vec,
        input.enet_alpha,
        api.epsilon,
        control.enet_abs_tol,
        control.enet_rel_tol,
        control.enet_max_iter,
        false,
        enet_workspace,
        api.delta_beta
      );

      if (api.delta_beta.is_zero()) {
        /*
         * State may contain the preceding rejected trial.
         */
        state.set_beta(
          api.beta_start
        );

        finish_stagewise(
          EnumStagewiseTerminationReason::
            STAGEWISE_BETA_STEP_ZERO,
            "The stagewise beta step is zero."
        );

        return true;
      }

      /*
       * Every trial is formed independently from the unchanged
       * iteration-start beta.
       */
      api.beta_trial =
        api.beta_start;

      api.beta_trial +=
        api.delta_beta;

      /*
       * Directly install and fully evaluate the candidate in State.
       */
      state.set_beta(
        api.beta_trial
      );

      api.negloglik_trial =
        state.negloglik();

      if (
          api.negloglik_trial <=
            negloglik_previous
      ) {
        /*
         * The accepted candidate is already the current State.
         */
        return false;
      }

      /*
       * Log the epsilon that produced the rejected candidate
       * before halving it for the next attempt.
       */
      ++api.halving_count;

      ECOUNTGMIFS_VERBOSE(
        control.verbose,
        "stagewise: iter="
        << iteration
        << " reject_beta"
        << ", halving="
        << api.halving_count
        << ", epsilon="
        << api.epsilon
        << ", current_negloglik="
        << negloglik_previous
        << ", trial_negloglik="
        << api.negloglik_trial
      );

      api.epsilon *=
        0.5;
    }

    /*
     * The final rejected candidate is still stored in State.
     */
    state.set_beta(
      api.beta_start
    );

    finish_stagewise(
      EnumStagewiseTerminationReason::
        STAGEWISE_EPSILON_MIN_REACHED,
        "Epsilon fell below epsilon_min before a beta step was accepted."
    );

    return true;
  }

  void finish_stagewise(
      EnumStagewiseTerminationReason reason,
      const char* detail
  )
  {
    api.phase =
      EnumStagewisePhase::STAGEWISE_FINISHED;

    api.termination_reason =
      reason;

    api.termination_detail =
      detail;

    /*
     * Path decides whether/how to store snapshots. A forced save still
     * respects NO_STATE_TRACKING and replaces an existing same-iteration
     * snapshot instead of duplicating it.
     */
    path.save_current_state(true);
    path.flush_pending_best_criteria();
    path.finalize_message(
      reason,
      api.termination_detail
    );

    ECOUNTGMIFS_VERBOSE(
      control.verbose,
      "stagewise: stop"
      << ", reason="
      << stagewise_termination_reason_label(reason)
      << ", iteration="
      << state.iteration()
      << ", negloglik="
      << state.negloglik()
      << ", pseudo_r2="
      << path.pseudo_r2(state.negloglik())
      << ", epsilon="
      << api.epsilon
    );
  }

  bool optimize_theta_safely(
      NloptOptimizerInternal& optimizer,
      const char* phase
  )
  {
    if (state.theta().is_empty()) {
      return true;
    }
    api.theta_before_optimize =
      state.theta();

    const double negloglik_before =
      state.negloglik();

    optimizer.set_parameters(
      api.theta_before_optimize
    );

    optimizer.optimize();

    state.set_theta(
      optimizer.parameters()
    );

    if (
        state.negloglik() <=
          negloglik_before
    ) {
      return true;
    }

    const double rejected_negloglik =
      state.negloglik();

    state.set_theta(
      api.theta_before_optimize
    );

    Rcpp::warning(
      "%s theta optimization increased negative log-likelihood "
      "from %.17g to %.17g; previous theta was restored",
      phase,
      negloglik_before,
      rejected_negloglik
    );

    return false;
  }

  bool optimize_family_safely(
      NloptOptimizerInternal& optimizer,
      const char* phase
  )
  {
    if (state.family_parameters().is_empty()) {
      return true;
    }
    api.family_parameters_before_optimize =
      state.family_parameters();

    const double negloglik_before =
      state.negloglik();

    optimizer.set_parameters(
      api.family_parameters_before_optimize
    );

    optimizer.optimize();

    state.set_family_parameters(
      optimizer.parameters()
    );

    if (
        state.negloglik() <=
          negloglik_before
    ) {
      return true;
    }

    const double rejected_negloglik =
      state.negloglik();

    state.set_family_parameters(
      api.family_parameters_before_optimize
    );

    Rcpp::warning(
      "%s family-parameter optimization increased negative "
      "log-likelihood from %.17g to %.17g; previous family "
      "parameters were restored",
      phase,
      negloglik_before,
      rejected_negloglik
    );

    return false;
  }

  bool optimize_link_safely(
      NloptOptimizerInternal& optimizer,
      const char* phase
  )
  {
    if (state.link_parameters().is_empty()) {
      return true;
    }

    api.link_parameters_before_optimize =
      state.link_parameters();

    const double negloglik_before =
      state.negloglik();

    optimizer.set_parameters(
      api.link_parameters_before_optimize
    );

    optimizer.optimize();

    state.set_link_parameters(
      optimizer.parameters()
    );

    if (
        state.negloglik() <=
          negloglik_before
    ) {
      return true;
    }

    const double rejected_negloglik =
      state.negloglik();

    state.set_link_parameters(
      api.link_parameters_before_optimize
    );

    Rcpp::warning(
      "%s link-parameter optimization increased negative "
      "log-likelihood from %.17g to %.17g; previous link "
      "parameters were restored",
      phase,
      negloglik_before,
      rejected_negloglik
    );

    return false;
  }

  bool optimize_saturated_family_safely(
      NloptOptimizerInternal& optimizer
  )
  {
    api.saturated_family_parameters_before_optimize =
      api.saturated_family_parameters;

    const double negloglik_before =
      api.saturated_negloglik;

    optimizer.set_parameters(
      api.saturated_family_parameters_before_optimize
    );

    optimizer.optimize();

    set_saturated_family_parameters(
      static_cast<unsigned>(
        optimizer.parameters().n_elem
      ),
      optimizer.parameters().memptr()
    );

    if (
        api.saturated_negloglik <=
          negloglik_before
    ) {
      return true;
    }

    const double rejected_negloglik =
      api.saturated_negloglik;

    set_saturated_family_parameters(
      static_cast<unsigned>(
        api.saturated_family_parameters_before_optimize.n_elem
      ),
      api.saturated_family_parameters_before_optimize.memptr()
    );

    Rcpp::warning(
      "saturated family-parameter optimization increased negative "
      "log-likelihood from %.17g to %.17g; previous family "
      "parameters were restored",
      negloglik_before,
      rejected_negloglik
    );

    return false;
  }

  void set_saturated_family_parameters(
      unsigned n,
      const double* values
  )
  {
    if (n != api.saturated_family_parameters.n_elem) {
      Rcpp::stop(
        "incorrect saturated family parameter count"
      );
    }

    if (values != api.saturated_family_parameters.memptr()) {
      std::copy_n(
        values,
        n,
        api.saturated_family_parameters.memptr()
      );
    }

    refresh_saturated_negloglik();
  }

  void refresh_saturated_negloglik()
  {
    input.family_link->negloglik(
        input.y,
        input.y,
        api.saturated_family_parameters,
        api.saturated_negloglik
    );

    check_finite_scalar(
      api.saturated_negloglik,
      "saturated negloglik"
    );
  }

  static double saturated_family_objective(
      unsigned n,
      const double* values,
      double*,
      void* data
  )
  {
    auto& stagewise =
      *static_cast<EcountgmifsStagewiseInternal*>(data);

      ++stagewise.saturated_family_evaluation_count;

      stagewise.set_saturated_family_parameters(
        n,
        values
      );

      return stagewise.api.saturated_negloglik;
  }

  static double nonpen_objective(
      unsigned n,
      const double* values,
      double* grad,
      void* data
  )
  {
    auto& stagewise =
      *static_cast<EcountgmifsStagewiseInternal*>(data);

      ++stagewise.nonpen_evaluation_count;

      stagewise.state.set_theta(
        n,
        values
      );

      if (grad != nullptr) {
        stagewise.gradient.write_theta_gradient(
          n,
          grad
        );
      }

      return stagewise.state.negloglik();
  }

  static double family_objective(
      unsigned n,
      const double* values,
      double* grad,
      void* data
  )
  {
    auto& stagewise =
      *static_cast<EcountgmifsStagewiseInternal*>(data);

      ++stagewise.family_evaluation_count;

      stagewise.state.set_family_parameters(
        n,
        values
      );

      if (grad != nullptr) {
        stagewise.gradient.write_family_gradient(
          n,
          grad
        );
      }

      return stagewise.state.negloglik();
  }

  static double link_objective(
      unsigned n,
      const double* values,
      double* grad,
      void* data
  )
  {
    auto& stagewise =
      *static_cast<EcountgmifsStagewiseInternal*>(data);

      ++stagewise.link_evaluation_count;

      stagewise.state.set_link_parameters(
        n,
        values
      );

      if (grad != nullptr) {
        stagewise.gradient.write_link_gradient(
          n,
          grad
        );
      }

      return stagewise.state.negloglik();
  }
};


struct EcountgmifsRuntimeInternal
{
  EcountgmifsRuntime api;

  EcountgmifsRuntimeInternal(
    const EcountgmifsState& state,
    const EcountgmifsPath& path,
    const EcountgmifsGradients& gradient,
    const EcountgmifsStagewise& stagewise
  ) :
    api {
    state,
    path,
    gradient,
    stagewise
  }
  {}

  const EcountgmifsRuntime& view() const noexcept
  {
    return api;
  }

  EcountgmifsRuntimeInternal(
    const EcountgmifsRuntimeInternal&
  ) = delete;

  EcountgmifsRuntimeInternal(
    EcountgmifsRuntimeInternal&&
  ) = delete;

  EcountgmifsRuntimeInternal& operator=(
    const EcountgmifsRuntimeInternal&
  ) = delete;

  EcountgmifsRuntimeInternal& operator=(
    EcountgmifsRuntimeInternal&&
  ) = delete;
};


struct EcountgmifsContextInternal
{
  EcountgmifsInputInternal input;
  EcountgmifsControlInternal control;
  EcountgmifsStateInternal state;
  EcountgmifsGradientsInternal gradient;
  EcountgmifsPathInternal path;
  EcountgmifsStagewiseInternal stagewise;
  EcountgmifsRuntimeInternal runtime;

  EcountgmifsContext api;


  EcountgmifsContextInternal(
    const arma::mat& X,
    const arma::vec& y,
    const arma::mat& w,
    const arma::vec& offset,

    const arma::vec& weight_vec,
    double enet_alpha,
    double epsilon_start,
    double epsilon_max,
    double epsilon_min,
    uint64_t null_iteration_max,
    uint64_t stagewise_iteration_max,
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
    int state_track_strategy,
    uint64_t state_track_freq,
    bool include_data,
    const arma::vec& theta_initial,
    const arma::vec& theta_lower_bounds,
    const arma::vec& theta_upper_bounds,

    const EcountgmifsNloptControl& nonpen_nlopt,
    const EcountgmifsNloptControl& family_nlopt,
    const EcountgmifsNloptControl& link_nlopt,
    SEXP family_link
  ) :
    input(
      X,
      y,
      w,
      offset,
      weight_vec,
      enet_alpha,
      family,
      link_func,
      criteria,
      family_link
    ),
    control(
      input.api,
      null_iteration_max,
      stagewise_iteration_max,
      null_family_parameter_abs_tol,
      stagewise_objective_rel_tol,
      stagewise_beta_step_norm_tol,
      epsilon_max,
      epsilon_start,
      epsilon_min,
      loglik_reltol_cutoff,
      enet_abs_tol,
      enet_rel_tol,
      enet_max_iter,

      state_track_strategy,
      state_track_freq,
      verbose,
      include_data,

      theta_initial,
      theta_lower_bounds,
      theta_upper_bounds,

      nonpen_nlopt,
      family_nlopt,
      link_nlopt
    ),
    state(
      input.api,
      control.api
    ),

    gradient(
      input.api,
      state
    ),

    path(
      input.api,
      state,
      control.api
    ),

    stagewise(
      input.api,
      control.api,
      state,
      gradient,
      path
    ),

    runtime(
      state.view(),
      path.view(),
      gradient.view(),
      stagewise.view()
    ),

    api {
    input.api,
    control.api,
    runtime.view()
  }
  {

  }

  EcountgmifsContextInternal(
    const EcountgmifsContextInternal&
  ) = delete;

  EcountgmifsContextInternal(
    EcountgmifsContextInternal&&
  ) = delete;

  EcountgmifsContextInternal& operator=(
    const EcountgmifsContextInternal&
  ) = delete;

  EcountgmifsContextInternal& operator=(
    EcountgmifsContextInternal&&
  ) = delete;

  Rcpp::List to_list() const
  {
    return Rcpp::List::create(
      Rcpp::Named("input") =
        input.to_list(
          control.api.include_data
        ),

        Rcpp::Named("control") =
          control.to_list(),

          Rcpp::Named("terminal_state") =
            path.current_state_to_list(),

            Rcpp::Named("path") =
              path.to_list(),

              Rcpp::Named("stagewise") =
                stagewise.to_list()
    );
  }
};
