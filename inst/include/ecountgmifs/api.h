#pragma once

#include <RcppArmadillo.h>
#include <cstdint>
#include <string>
#include <vector>
#include <limits>

enum EnumStateTrackStrategy
{
  ACTIVE_SET_CHANGE = 0,
  ALL_ITERATION = 1,
  EVERY_K_ITERATION = 2,
  NO_STATE_TRACKING = 3
};

enum EnumStagewisePhase
{
  STAGEWISE_NOT_STARTED = 0,
  STAGEWISE_NONPEN = 1,
  STAGEWISE_SATURATED = 2,
  STAGEWISE_ITERATION = 3,
  STAGEWISE_FINISHED = 4
};
enum EnumStagewiseTerminationReason
{
  STAGEWISE_NOT_INITIALIZED = 0,
  STAGEWISE_RUNNING = 1,

  STAGEWISE_BETA_STEP_ZERO = 2,
  STAGEWISE_BETA_STALLED = 3,
  STAGEWISE_OBJECTIVE_STALLED = 4,
  STAGEWISE_PSEUDO_R2_CUTOFF_REACHED = 5,

  STAGEWISE_EPSILON_MIN_REACHED = 6,
  STAGEWISE_ITERATION_LIMIT_REACHED = 7
};

struct IEcountgmifsCriterion;


struct IEcountgmifsLinkFunc
{
  virtual ~IEcountgmifsLinkFunc() = default;

  virtual std::string name() const = 0;

  virtual arma::uword parameter_count() const noexcept = 0;

  virtual void inverse(
      const arma::vec& eta,
      const arma::vec& link_parameters,
      arma::vec& mu
  ) const = 0;

  virtual void grad(
      const arma::vec& eta,
      const arma::vec& link_parameters,
      arma::vec& d_mu_d_eta,
      arma::mat& d_mu_d_link_parameters
  ) const = 0;

  virtual arma::vec initial_parameters() const
  {
    return arma::vec(parameter_count(), arma::fill::zeros);
  }

  virtual arma::vec parameter_lower_bounds() const
  {
    arma::vec lower(parameter_count());
    lower.fill(-arma::datum::inf);
    return lower;
  }

  virtual arma::vec parameter_upper_bounds() const
  {
    arma::vec upper(parameter_count());
    upper.fill(arma::datum::inf);
    return upper;
  }
};
struct IEcountgmifsFamily
{
  virtual ~IEcountgmifsFamily() = default;

  virtual std::string name() const = 0;

  virtual arma::uword parameter_count() const noexcept = 0;

  virtual void negloglik(
      const arma::vec& y,
      const arma::vec& mu,
      const arma::vec& family_parameters,
      double& negloglik
  ) const = 0;

  virtual void grad(
      const arma::vec& y,
      const arma::vec& mu,
      const arma::vec& family_parameters,
      arma::vec& d_negloglik_d_mu,
      arma::vec& d_negloglik_d_family_parameters
  ) const = 0;

  virtual arma::vec initial_parameters() const
  {
    return arma::vec(parameter_count(), arma::fill::zeros);
  }

  virtual arma::vec parameter_lower_bounds() const
  {
    arma::vec lower(parameter_count());
    lower.fill(-arma::datum::inf);
    return lower;
  }

  virtual arma::vec parameter_upper_bounds() const
  {
    arma::vec upper(parameter_count());
    upper.fill(arma::datum::inf);
    return upper;
  }
};


/*
 * Combined family-link interface.
 *
 * A custom implementation may evaluate d(negative log-likelihood)/d(eta)
 * directly using a numerically stable fused formula. When no combined
 * implementation is supplied, the package constructs an internal adapter
 * that delegates to IEcountgmifsFamily and IEcountgmifsLinkFunc and applies
 * the ordinary chain rule.
 */
struct IEcountgmifsFamilyLink
{
  virtual ~IEcountgmifsFamilyLink() = default;

  virtual std::string family_name() const = 0;
  virtual std::string link_name() const = 0;

  virtual arma::uword family_parameter_count() const noexcept = 0;
  virtual arma::uword link_parameter_count() const noexcept = 0;

  virtual arma::vec family_initial_parameters() const
  {
    return arma::vec(
      family_parameter_count(),
      arma::fill::zeros
    );
  }

  virtual arma::vec family_parameter_lower_bounds() const
  {
    arma::vec lower(family_parameter_count());
    lower.fill(-arma::datum::inf);
    return lower;
  }

  virtual arma::vec family_parameter_upper_bounds() const
  {
    arma::vec upper(family_parameter_count());
    upper.fill(arma::datum::inf);
    return upper;
  }

  virtual arma::vec link_initial_parameters() const
  {
    return arma::vec(
      link_parameter_count(),
      arma::fill::zeros
    );
  }

  virtual arma::vec link_parameter_lower_bounds() const
  {
    arma::vec lower(link_parameter_count());
    lower.fill(-arma::datum::inf);
    return lower;
  }

  virtual arma::vec link_parameter_upper_bounds() const
  {
    arma::vec upper(link_parameter_count());
    upper.fill(arma::datum::inf);
    return upper;
  }

  virtual void inverse(
      const arma::vec& eta,
      const arma::vec& link_parameters,
      arma::vec& mu
  ) const = 0;

  virtual void negloglik(
      const arma::vec& y,
      const arma::vec& mu,
      const arma::vec& family_parameters,
      double& negloglik
  ) const = 0;

  /*
   * Fill all derivatives needed by the fitting code.
   *
   * d_negloglik_d_eta is the derivative used for beta and theta. A custom
   * combined implementation may calculate it directly rather than as
   * d_negloglik_d_mu % d_mu_d_eta.
   */
  virtual void grad(
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
  ) const = 0;
};



/*
 * Public read-only input struct.
 *
 * This exposes model data to user-defined C++ criteria.
 * The matrix/vector fields are const references, so plugins can read but should
 * not mutate package-owned data.
 */
struct EcountgmifsInput {
  const arma::mat& X;
  const arma::vec& y;
  const arma::mat& w;
  const arma::vec& offset;
  const arma::vec& weight_vec;
  bool has_prior;
  double enet_alpha;

  const arma::vec train_y_one_lgamma;

  /*
   * family_link is always non-null internally.
   *
   * If the user supplies a combined implementation, family and link_func may
   * be null. Otherwise family/link_func are retained and family_link points to
   * an internal delegating adapter.
   */
  const IEcountgmifsFamily* family;
  const IEcountgmifsLinkFunc* link_func;
  const IEcountgmifsFamilyLink* family_link;
  std::vector<const IEcountgmifsCriterion*> criteria;
};


/*
 * Public read-only control view.
 *
 * This exposes model data to user-defined C++ criteria.
 * The matrix/vector fields are const references, so plugins can read but should
 * not mutate package-owned data.
 */
struct EcountgmifsControl {
  uint64_t iteration_max;
  double epsilon_max;
  double epsilon_start;
  double epsilon_min;
  double tol;
  double loglik_reltol_cutoff;
  double enet_abs_tol;
  double enet_rel_tol;
  uint32_t enet_max_iter;

  EnumStateTrackStrategy state_track_strategy;
  uint64_t state_track_freq;
  bool verbose;
  bool include_data;

  const arma::vec& theta_initial;
  const arma::vec& theta_lower_bounds;
  const arma::vec& theta_upper_bounds;

  int nlopt_algorithm;
  double nlopt_xtol_rel;
  double nlopt_ftol_rel;
  int nlopt_maxeval;
};

struct EcountgmifsParameters
{
  arma::vec beta;
  arma::vec theta;

  // Future module-specific parameters passed from loglik/other parts
  arma::vec family_parameters;
  arma::vec link_parameters;
};

struct EcountgmifsGradients
{
  arma::vec d_negloglik_d_mu;
  arma::vec d_mu_d_eta;

  const arma::mat& d_eta_d_beta;
  const arma::mat& d_eta_d_theta;

  // Future gradients wrt other parameters such as dispersion
  arma::mat d_mu_d_link_parameters;
  arma::vec d_negloglik_d_family_parameters;
  arma::vec d_negloglik_d_link_parameters;

  // main gradient for FS
  arma::vec d_negloglik_d_beta;
};

struct EcountgmifsPredictors
{
  EcountgmifsParameters param;

  arma::vec xbeta;
  arma::vec wtheta;
  arma::vec eta;
  arma::vec mu;

  arma::uvec active_set;
};

struct EcountgmifsStagewise
{
  EnumStagewisePhase phase = EnumStagewisePhase::STAGEWISE_NOT_STARTED;
  EnumStagewiseTerminationReason termination_reason =
    EnumStagewiseTerminationReason::STAGEWISE_NOT_INITIALIZED;

  /*
   * Immutable beta at the beginning of the current
   * stagewise iteration.
   */
  arma::vec beta_start;

  /*
   * Candidate beta for the current epsilon:
   *
   *   beta_trial = beta_start + delta_beta
   */
  arma::vec beta_trial;

  /*
   * Elastic-net step for the current epsilon.
   */
  arma::vec delta_beta;

  /*
   * Numerical explanation produced by stagewise.
   *
   * Examples:
   *   "Relative negative log-likelihood difference = ..."
   *   "Epsilon = ..., epsilon_min = ..."
   */
  std::string termination_detail = "Not initialized";

  double epsilon = 0.0;

  double negloglik_trial = arma::datum::nan;

  uint32_t halving_count = 0;
};

struct EcountgmifsState
{
  EcountgmifsPredictors param;

  double negloglik = arma::datum::nan;

  Rcpp::NumericVector criteria;

  uint64_t iteration = 0;

  /*
   * This value is populated when a State snapshot is saved to Path.
   * The live mutable State does not own the null/saturated baselines.
   */
  double pseudo_r2 = arma::datum::nan;
};

struct EcountgmifsBestCriterion
{
  std::string name;

  double value =
    std::numeric_limits<double>::infinity();

  EcountgmifsState state;
};

struct EcountgmifsPath
{
  /*
   * Fit-wide null-model result. Beta is omitted because the null model
   * always has beta = 0 and storing a p-length zero vector is unnecessary.
   */
  double null_negloglik = arma::datum::nan;
  arma::vec null_theta;
  arma::vec null_family_parameters;
  arma::vec null_link_parameters;

  /*
   * Fit-wide saturated-model result. Family parameters are kept generic:
   * an empty vector for parameter-free families, one value for NB2, and
   * any other size required by another family implementation.
   */
  double saturated_negloglik = arma::datum::nan;
  arma::vec saturated_family_parameters;

  std::vector<EcountgmifsBestCriterion>
    best_criteria;

  std::vector<EcountgmifsState> states;
  arma::uvec last_saved_active_set;
  bool active_set_changed = false;
  std::string message;
};

struct EcountgmifsRuntime {
  const EcountgmifsState& state;
  const EcountgmifsPath& path;
  const EcountgmifsGradients& gradient;
  const EcountgmifsStagewise& stagewise;
};

struct EcountgmifsContext {
  const EcountgmifsInput& input;
  const EcountgmifsControl& control;
  const EcountgmifsRuntime& runtime;
};

struct IEcountgmifsCriterion
{
  virtual ~IEcountgmifsCriterion() = default;

  virtual std::string name() const = 0;

  virtual double evaluate(
      const EcountgmifsInput& input,
      const EcountgmifsControl& control,
      const EcountgmifsState& state
  ) const = 0;
};



