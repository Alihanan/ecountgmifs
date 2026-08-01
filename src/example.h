#pragma once

// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::plugins(cpp17)]]
#include <RcppArmadillo.h>

#include <cmath>
#include <stdexcept>
#include <algorithm>
#include <limits>

#include "../inst/include/ecountgmifs/api.h"

namespace ecountgmifs_examples
{

inline double clamp_mu(
    const double mu,
    const double mu_min_cap,
    const double mu_max_cap
)
{
  return std::max(
    mu_min_cap,
    std::min(
      mu,
      mu_max_cap
    )
  );
}

inline double log1pexp(const double x)
{
  if (x > 0.0)
    return x + std::log1p(std::exp(-x));

  return std::log1p(std::exp(x));
}

inline double logistic(const double x)
{
  if (x >= 0.0)
  {
    const double z = std::exp(-x);
    return 1.0 / (1.0 + z);
  }

  const double z = std::exp(x);
  return z / (1.0 + z);
}

struct LogLink : public IEcountgmifsLinkFunc
{
  arma::uword parameter_count() const noexcept override { return 0; }

  std::string name() const override
  {
    return "Log";
  }

  void prepare(
      const EcountgmifsInput& input,
      const EcountgmifsControl& control
  ) const override
  {
    (void) input;
    (void) control;
  }

  void inverse(
      const arma::vec& eta,
      const arma::vec& link_parameters,
      arma::vec& mu
  ) const override
  {
    if (!link_parameters.empty())
      throw std::invalid_argument("LogLink expects no link parameters.");

    mu = arma::exp(eta);
  }

  void grad(
      const arma::vec& eta,
      const arma::vec& link_parameters,
      arma::vec& d_mu_d_eta,
      arma::mat& d_mu_d_link_parameters
  ) const override
  {
    if (!link_parameters.empty())
      throw std::invalid_argument("LogLink expects no link parameters.");

    d_mu_d_eta = arma::exp(eta);
    d_mu_d_link_parameters.set_size(eta.n_elem, 0);
  }
};

struct SoftplusLink : public IEcountgmifsLinkFunc
{
  arma::uword parameter_count() const noexcept override { return 1; }

  arma::vec initial_parameters() const override
  {
    return arma::vec({1.0});
  }

  arma::vec parameter_lower_bounds() const override
  {
    return arma::vec({1e-8});
  }

  std::string name() const override
  {
    return "Softplus";
  }

  void prepare(
      const EcountgmifsInput& input,
      const EcountgmifsControl& control
  ) const override
  {
    (void) input;
    (void) control;
  }

  void inverse(
      const arma::vec& eta,
      const arma::vec& link_parameters,
      arma::vec& mu
  ) const override
  {
    validate(link_parameters);
    const double a = link_parameters[0];

    mu.set_size(eta.n_elem);
    for (arma::uword i = 0; i < eta.n_elem; ++i)
      mu[i] = log1pexp(a * eta[i]) / a;
  }

  void grad(
      const arma::vec& eta,
      const arma::vec& link_parameters,
      arma::vec& d_mu_d_eta,
      arma::mat& d_mu_d_link_parameters
  ) const override
  {
    validate(link_parameters);
    const double a = link_parameters[0];

    d_mu_d_eta.set_size(eta.n_elem);
    d_mu_d_link_parameters.set_size(eta.n_elem, 1);

    for (arma::uword i = 0; i < eta.n_elem; ++i)
    {
      const double a_eta = a * eta[i];
      const double s = logistic(a_eta);
      const double mu_i = log1pexp(a_eta) / a;

      d_mu_d_eta[i] = s;
      d_mu_d_link_parameters(i, 0) = (eta[i] * s - mu_i) / a;
    }
  }

private:
  static void validate(const arma::vec& link_parameters)
  {
    if (link_parameters.n_elem != 1)
      throw std::invalid_argument("SoftplusLink expects one parameter: a.");

    if (!std::isfinite(link_parameters[0]) || link_parameters[0] <= 0.0)
      throw std::invalid_argument("SoftplusLink parameter a must be > 0.");
  }
};

struct PoissonFamily final : public IEcountgmifsFamily
{
  PoissonFamily(
    const double mu_min_cap,
    const double mu_max_cap
  ) :
  mu_min_cap_(mu_min_cap),
  mu_max_cap_(mu_max_cap)
  {
    validate_mu_caps();
  }

  std::string name() const override
  {
    return "Poisson";
  }

  arma::uword parameter_count() const noexcept override
  {
    return 0;
  }

  void prepare(
      const EcountgmifsInput& input,
      const EcountgmifsControl& control
  ) const override
  {
    (void) input;
    (void) control;
  }

  void negloglik(
      const arma::vec& y,
      const arma::vec& mu,
      const arma::vec& family_parameters,
      double& negloglik_value
  ) const override
  {
    validate_dimensions(y, mu);

    if (!family_parameters.empty())
      throw std::invalid_argument(
          "PoissonFamily expects no family parameters."
      );

    negloglik_value = 0.0;

    for (arma::uword i = 0; i < y.n_elem; ++i)
    {
      const double mu_safe = clamp_mu(
        mu[i],
          mu_min_cap_,
          mu_max_cap_
      );

      negloglik_value +=
        mu_safe -
        y[i] * std::log(mu_safe) +
        std::lgamma(y[i] + 1.0);
    }
  }

  void grad(
      const arma::vec& y,
      const arma::vec& mu,
      const arma::vec& family_parameters,
      arma::vec& d_negloglik_d_mu,
      arma::vec& d_negloglik_d_family_parameters
  ) const override
  {
    validate_dimensions(y, mu);

    if (!family_parameters.empty())
      throw std::invalid_argument(
          "PoissonFamily expects no family parameters."
      );

    d_negloglik_d_mu =
      1.0 - y / mu;

    d_negloglik_d_family_parameters.reset();
  }

private:
  double mu_min_cap_;
  double mu_max_cap_;

  void validate_mu_caps() const
  {
    if (
        !std::isfinite(mu_min_cap_) ||
          !std::isfinite(mu_max_cap_) ||
          mu_min_cap_ <= 0.0 ||
          mu_min_cap_ >= mu_max_cap_
    )
      throw std::invalid_argument(
          "PoissonFamily: invalid mu caps."
      );
  }

  static void validate_dimensions(
      const arma::vec& y,
      const arma::vec& mu
  )
  {
    if (y.n_elem != mu.n_elem)
      throw std::invalid_argument(
          "PoissonFamily: y and mu lengths differ."
      );
  }
};

struct NB2Family final : public IEcountgmifsFamily
{
private:
  double mu_min_cap_;
  double mu_max_cap_;
  double poisson_fallback_eps_;
  double dispersion_initial_;
  double dispersion_lower_bound_;
  double dispersion_upper_bound_;

  mutable arma::vec log_factorial_;

public:
  NB2Family(
    double mu_min_cap,
    double mu_max_cap,
    double poisson_fallback_eps,
    double dispersion_initial,
    double dispersion_lower_bound,
    double dispersion_upper_bound
  ) :
  mu_min_cap_(mu_min_cap),
  mu_max_cap_(mu_max_cap),
  poisson_fallback_eps_(poisson_fallback_eps),
  dispersion_initial_(dispersion_initial),
  dispersion_lower_bound_(dispersion_lower_bound),
  dispersion_upper_bound_(dispersion_upper_bound)
  {
    if (!std::isfinite(dispersion_initial_)) {
      Rcpp::stop("NB2 initial dispersion must be finite");
    }

    if (
        std::isnan(dispersion_lower_bound_) ||
          std::isnan(dispersion_upper_bound_)
    ) {
      Rcpp::stop("NB2 dispersion bounds must not be NaN");
    }

    if (dispersion_lower_bound_ < 0.0) {
      Rcpp::stop(
        "NB2 dispersion lower bound must be non-negative"
      );
    }

    if (
        dispersion_lower_bound_ >=
          dispersion_upper_bound_
    ) {
      Rcpp::stop(
        "NB2 dispersion lower bound must be smaller "
        "than the upper bound"
      );
    }

    if (
        dispersion_initial_ < dispersion_lower_bound_ ||
          dispersion_initial_ > dispersion_upper_bound_
    ) {
      Rcpp::stop(
        "NB2 initial dispersion must lie within "
        "the dispersion bounds"
      );
    }

    validate_settings();
  }


  std::string name() const override
  {
    return "NB2";
  }

  arma::uword parameter_count() const noexcept override
  {
    return 1;
  }

  void prepare(
      const EcountgmifsInput& input,
      const EcountgmifsControl& control
  ) const override
  {
    (void) control;

    log_factorial_ =
      arma::lgamma(input.y + 1.0);
  }

  void negloglik(
      const arma::vec& y,
      const arma::vec& mu,
      const arma::vec& family_parameters,
      double& negloglik_value
  ) const override
  {
    validate_dimensions(y, mu);
    validate_prepared_cache(y);

    const double dispersion =
      get_finite_dispersion(family_parameters);

    if (dispersion <= poisson_fallback_eps_)
    {
      poisson_negloglik(
        y,
        mu,
        negloglik_value
      );

      return;
    }

    if (dispersion <= 0.0)
    {
      negloglik_value =
        std::numeric_limits<double>::infinity();

      return;
    }

    const double one_over_dispersion =
      1.0 / dispersion;

    const double lgamma_one_over_dispersion =
      std::lgamma(one_over_dispersion);

    negloglik_value = 0.0;

    for (arma::uword i = 0; i < y.n_elem; ++i)
    {
      const double mu_safe = clamp_mu(
        mu[i],
          mu_min_cap_,
          mu_max_cap_
      );

      const double dispersion_mu =
        dispersion * mu_safe;

      const double log_one_plus_dispersion_mu =
        std::log1p(dispersion_mu);

      negloglik_value -=
        y[i] *
        (
            std::log(dispersion_mu) -
              log_one_plus_dispersion_mu
        ) -
          one_over_dispersion *
          log_one_plus_dispersion_mu +
          std::lgamma(y[i] + one_over_dispersion) -
          log_factorial_[i] -
          lgamma_one_over_dispersion;
    }
  }

  void grad(
      const arma::vec& y,
      const arma::vec& mu,
      const arma::vec& family_parameters,
      arma::vec& d_negloglik_d_mu,
      arma::vec& d_negloglik_d_family_parameters
  ) const override
  {
    validate_dimensions(y, mu);

    const double dispersion =
      get_finite_dispersion(family_parameters);

    if (dispersion <= poisson_fallback_eps_)
    {
      d_negloglik_d_mu =
        1.0 - y / mu;

      // Inside the fallback region, the likelihood no longer
      // depends on dispersion.
      d_negloglik_d_family_parameters.zeros(1);

      return;
    }

    const double r =
      1.0 / dispersion;

    d_negloglik_d_mu =
      -y / mu +
      (1.0 + dispersion * y) /
        (1.0 + dispersion * mu);

    double d_negloglik_d_r = 0.0;

    for (arma::uword i = 0; i < y.n_elem; ++i)
    {
      const double r_plus_mu =
        r + mu[i];

      d_negloglik_d_r +=
        R::digamma(r) -
        R::digamma(y[i] + r) -
        std::log(r / r_plus_mu) -
        1.0 +
        (r + y[i]) / r_plus_mu;
    }

    d_negloglik_d_family_parameters.set_size(1);

    d_negloglik_d_family_parameters[0] =
      -d_negloglik_d_r /
      (dispersion * dispersion);
  }

  arma::vec initial_parameters() const override
  {
    return arma::vec({dispersion_initial_});
  }

  arma::vec parameter_lower_bounds() const override
  {
    return arma::vec({dispersion_lower_bound_});
  }

  arma::vec parameter_upper_bounds() const override
  {
    return arma::vec({dispersion_upper_bound_});
  }


private:
  void poisson_negloglik(
      const arma::vec& y,
      const arma::vec& mu,
      double& negloglik_value
  ) const
  {
    negloglik_value = 0.0;

    for (arma::uword i = 0; i < y.n_elem; ++i)
    {
      const double mu_safe = clamp_mu(
        mu[i],
          mu_min_cap_,
          mu_max_cap_
      );

      negloglik_value +=
        mu_safe -
        y[i] * std::log(mu_safe) +
        log_factorial_[i];
    }
  }

  void validate_settings() const
  {
    if (
        !std::isfinite(mu_min_cap_) ||
          !std::isfinite(mu_max_cap_) ||
          mu_min_cap_ <= 0.0 ||
          mu_min_cap_ >= mu_max_cap_
    )
      throw std::invalid_argument(
          "NB2Family: invalid mu caps."
      );

    if (
        !std::isfinite(poisson_fallback_eps_) ||
          poisson_fallback_eps_ < 0.0
    )
      throw std::invalid_argument(
          "NB2Family: poisson fallback epsilon must be non-negative."
      );
  }

  static double get_finite_dispersion(
      const arma::vec& family_parameters
  )
  {
    if (family_parameters.n_elem != 1)
      throw std::invalid_argument(
          "NB2Family expects one dispersion parameter."
      );

    if (!std::isfinite(family_parameters[0]))
      throw std::invalid_argument(
          "NB2Family dispersion must be finite."
      );

    return family_parameters[0];
  }

  void validate_prepared_cache(
      const arma::vec& y
  ) const
  {
    if (log_factorial_.n_elem != y.n_elem)
      throw std::runtime_error(
          "NB2Family::prepare() must be called before negloglik()."
      );
  }

  static void validate_dimensions(
      const arma::vec& y,
      const arma::vec& mu
  )
  {
    if (y.n_elem != mu.n_elem)
      throw std::invalid_argument(
          "NB2Family: y and mu lengths differ."
      );
  }
};

struct NB2LogFamilyLink final : public IEcountgmifsFamilyLink
{
private:
  NB2Family family_;
  LogLink link_func_;
  double poisson_fallback_eps_;

public:
  NB2LogFamilyLink(
    double mu_min_cap,
    double mu_max_cap,
    double poisson_fallback_eps,
    double dispersion_initial,
    double dispersion_lower_bound,
    double dispersion_upper_bound
  ) :
  family_(
    mu_min_cap,
    mu_max_cap,
    poisson_fallback_eps,
    dispersion_initial,
    dispersion_lower_bound,
    dispersion_upper_bound
  ),
  poisson_fallback_eps_(poisson_fallback_eps)
  {}

  std::string family_name() const override
  {
    return family_.name();
  }

  std::string link_name() const override
  {
    return link_func_.name();
  }

  arma::uword family_parameter_count() const noexcept override
  {
    return family_.parameter_count();
  }

  arma::uword link_parameter_count() const noexcept override
  {
    return link_func_.parameter_count();
  }

  void prepare(
      const EcountgmifsInput& input,
      const EcountgmifsControl& control
  ) const override
  {
    family_.prepare(input, control);
    link_func_.prepare(input, control);
  }

  arma::vec family_initial_parameters() const override
  {
    return family_.initial_parameters();
  }

  arma::vec family_parameter_lower_bounds() const override
  {
    return family_.parameter_lower_bounds();
  }

  arma::vec family_parameter_upper_bounds() const override
  {
    return family_.parameter_upper_bounds();
  }

  arma::vec link_initial_parameters() const override
  {
    return link_func_.initial_parameters();
  }

  arma::vec link_parameter_lower_bounds() const override
  {
    return link_func_.parameter_lower_bounds();
  }

  arma::vec link_parameter_upper_bounds() const override
  {
    return link_func_.parameter_upper_bounds();
  }

  void inverse(
      const arma::vec& eta,
      const arma::vec& link_parameters,
      arma::vec& mu
  ) const override
  {
    link_func_.inverse(
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
    family_.negloglik(
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
    /*
    family_.grad(
      y,
      mu,
      family_parameters,
      d_negloglik_d_mu,
      d_negloglik_d_family_parameters
    );

    link_func_.grad(
      eta,
      link_parameters,
      d_mu_d_eta,
      d_mu_d_link_parameters
    );
    */

    const double dispersion =
      family_parameters[0];

    if (dispersion <= poisson_fallback_eps_) {
      d_negloglik_d_eta =
        mu - y;
    } else {
      d_negloglik_d_eta =
        (mu - y) /
          (1.0 + dispersion * mu);
    }

    d_negloglik_d_link_parameters.reset();
  }
};

inline double effective_parameter_count(
    const EcountgmifsState& state
)
{
  const EcountgmifsParameters& parameters =
    state.param.param;

  const arma::uword beta_df =
    arma::accu(parameters.beta != 0.0);

  return static_cast<double>(
    beta_df +
      parameters.theta.n_elem +
      parameters.family_parameters.n_elem +
      parameters.link_parameters.n_elem
  );
}

struct AICCriterion : public IEcountgmifsCriterion
{
  std::string name() const override
  {
    return "AIC";
  }

  void prepare(
      const EcountgmifsInput& input,
      const EcountgmifsControl& control
  ) const override
  {
    (void) input;
    (void) control;
  }

  double evaluate(
      const EcountgmifsInput&,
      const EcountgmifsControl&,
      const EcountgmifsState& state
  ) const override
  {
    const double k =
      effective_parameter_count(state);

    return
    2.0 * state.negloglik +
      2.0 * k;
  }
};

struct BICCriterion : public IEcountgmifsCriterion
{
  std::string name() const override
  {
    return "BIC";
  }

  void prepare(
      const EcountgmifsInput& input,
      const EcountgmifsControl& control
  ) const override
  {
    (void) input;
    (void) control;
  }

  double evaluate(
      const EcountgmifsInput& input,
      const EcountgmifsControl&,
      const EcountgmifsState& state
  ) const override
  {
    const double k =
      effective_parameter_count(state);

    const arma::uword n =
      input.y.n_elem;

    if (n == 0)
      throw std::runtime_error(
          "BICCriterion: zero observations."
      );

    return
    2.0 * state.negloglik +
      std::log(static_cast<double>(n)) * k;
  }
};


enum class InformationPenaltyType
{
  AIC = 0,
    BIC = 1,
    SABIC = 2,
    EBIC = 3
};

enum class InformationDfType
{
  FULL = 0,
    NNZ = 1,
    HEDF = 2
};

struct InformationCriterion final : public IEcountgmifsCriterion
{
private:
  std::string name_;
  InformationPenaltyType penalty_type_;
  InformationDfType df_type_;
  double gamma_;

  static double nnz(
      const EcountgmifsState& state
  )
  {
    return static_cast<double>(
      arma::accu(
        state.param.param.beta != 0.0
      )
    );
  }

  double degrees_of_freedom(
      const EcountgmifsInput& input,
      const EcountgmifsState& state
  ) const
  {
    const double beta_df = nnz(state);

    switch (df_type_)
    {
    case InformationDfType::FULL:
      return effective_parameter_count(state);

    case InformationDfType::NNZ:
      return beta_df;

    case InformationDfType::HEDF:
      if (input.X.n_cols == 0)
        throw std::runtime_error(
            "InformationCriterion: HEDF requires at least one predictor."
        );

      return
      beta_df *
        static_cast<double>(input.X.n_rows) /
          static_cast<double>(input.X.n_cols);
    }

    throw std::runtime_error(
        "InformationCriterion: unknown degrees-of-freedom type."
    );
  }

public:
  InformationCriterion(
    std::string name,
    const int penalty_type,
    const int df_type,
    const double gamma
  ) :
  name_(std::move(name)),
  penalty_type_(
    static_cast<InformationPenaltyType>(
      penalty_type
    )
  ),
  df_type_(
    static_cast<InformationDfType>(
      df_type
    )
  ),
  gamma_(gamma)
  {
    if (penalty_type < 0 || penalty_type > 3)
      throw std::invalid_argument(
          "InformationCriterion: invalid penalty type."
      );

    if (df_type < 0 || df_type > 2)
      throw std::invalid_argument(
          "InformationCriterion: invalid degrees-of-freedom type."
      );

    if (!std::isfinite(gamma_) || gamma_ < 0.0)
      throw std::invalid_argument(
          "InformationCriterion: gamma must be finite and non-negative."
      );
  }

  std::string name() const override
  {
    return name_;
  }

  void prepare(
      const EcountgmifsInput& input,
      const EcountgmifsControl& control
  ) const override
  {
    (void) input;
    (void) control;
  }

  double evaluate(
      const EcountgmifsInput& input,
      const EcountgmifsControl&,
      const EcountgmifsState& state
  ) const override
  {
    const double n =
      static_cast<double>(input.y.n_elem);

    if (n <= 0.0)
      throw std::runtime_error(
          "InformationCriterion: zero observations."
      );

    const double df =
      degrees_of_freedom(input, state);

    switch (penalty_type_)
    {
    case InformationPenaltyType::AIC:
      return
      2.0 * state.negloglik +
        2.0 * df;

    case InformationPenaltyType::BIC:
      return
      2.0 * state.negloglik +
        std::log(n) * df;

    case InformationPenaltyType::SABIC:
      return
      2.0 * state.negloglik +
        std::log((n + 2.0) / 24.0) * df;

    case InformationPenaltyType::EBIC:
    {
      const double p =
        static_cast<double>(input.X.n_cols);

      const double beta_df =
        nnz(state);

      const double log_choose =
        std::lgamma(p + 1.0) -
        std::lgamma(beta_df + 1.0) -
        std::lgamma(p - beta_df + 1.0);

      return
      2.0 * state.negloglik +
        std::log(n) * df +
        2.0 * gamma_ * log_choose;
    }
    }

    throw std::runtime_error(
        "InformationCriterion: unknown penalty type."
    );
  }
};

} // namespace ecountgmifs_examples

// [[Rcpp::export]]
SEXP example_create_log_link()
{
  return Rcpp::XPtr<IEcountgmifsLinkFunc>(
    new ecountgmifs_examples::LogLink(), true
  );
}

// [[Rcpp::export]]
SEXP example_create_softplus_link()
{
  return Rcpp::XPtr<IEcountgmifsLinkFunc>(
    new ecountgmifs_examples::SoftplusLink(), true
  );
}

// [[Rcpp::export]]
SEXP example_create_poisson_family(
    const double mu_min_cap = 1e-12,
    const double mu_max_cap = 1e12
)
{
  return Rcpp::XPtr<IEcountgmifsFamily>(
    new ecountgmifs_examples::PoissonFamily(
        mu_min_cap,
        mu_max_cap
    ),
    true
  );
}

// [[Rcpp::export]]
SEXP example_create_nb2_family(
    const double mu_min_cap = 1e-12,
    const double mu_max_cap = 1e12,
    const double poisson_fallback_eps = 1e-8,
    const double dispersion_initial = 1e-4,
    const double dispersion_lower_bound = 1e-12,
    const double dispersion_upper_bound = 1e12
)
{
  return Rcpp::XPtr<IEcountgmifsFamily>(
    new ecountgmifs_examples::NB2Family(
        mu_min_cap,
        mu_max_cap,
        poisson_fallback_eps,
        dispersion_initial,
        dispersion_lower_bound,
        dispersion_upper_bound
    ),
    true
  );
}

// [[Rcpp::export]]
SEXP example_create_nb2_log_family_link(
    const double mu_min_cap = 1e-12,
    const double mu_max_cap = 1e12,
    const double poisson_fallback_eps = 1e-8,
    const double dispersion_initial = 1e-4,
    const double dispersion_lower_bound = 1e-12,
    const double dispersion_upper_bound = 1e12
)
{
  return Rcpp::XPtr<IEcountgmifsFamilyLink>(
    new ecountgmifs_examples::NB2LogFamilyLink(
        mu_min_cap,
        mu_max_cap,
        poisson_fallback_eps,
        dispersion_initial,
        dispersion_lower_bound,
        dispersion_upper_bound
    ),
    true
  );
}

// [[Rcpp::export]]
SEXP example_create_aic_criterion()
{
  return Rcpp::XPtr<IEcountgmifsCriterion>(
    new ecountgmifs_examples::AICCriterion(), true
  );
}

// [[Rcpp::export]]
SEXP example_create_bic_criterion()
{
  return Rcpp::XPtr<IEcountgmifsCriterion>(
    new ecountgmifs_examples::BICCriterion(), true
  );
}


// [[Rcpp::export]]
SEXP example_create_information_criterion(
    const std::string& name,
    const int penalty_type,
    const int df_type,
    const double gamma = 0.5
)
{
  return Rcpp::XPtr<IEcountgmifsCriterion>(
    new ecountgmifs_examples::InformationCriterion(
        name,
        penalty_type,
        df_type,
        gamma
    ),
    true
  );
}
