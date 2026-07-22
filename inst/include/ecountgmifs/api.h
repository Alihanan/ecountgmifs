#pragma once

#include <RcppArmadillo.h>
#include <cstdint>
#include <string>
#include <vector>

enum EnumStateTrackStrategy
{
  ACTIVE_SET_CHANGE = 0,
  ALL_ITERATION = 1,
  EVERY_K_ITERATION = 2,
  NO_STATE_TRACKING = 3
};
enum EnumTerminationStatus
{
  RUNNING = 0,
  CONVERGED = 1,
  ITERATION_LIMIT_REACHED = 2,
  EPSILON_MIN_REACHED = 3,
  FAILED = 4
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

  const arma::mat& Xtest;
  const arma::vec& ytest;
  const arma::mat& wtest;
  const arma::vec& offsettest;
  const arma::vec& yorig;

  const arma::vec train_y_one_lgamma;
  const arma::vec test_y_one_lgamma;
  const arma::vec orig_y_one_lgamma;

  const IEcountgmifsFamily* family;
  const IEcountgmifsLinkFunc* link_func;
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
  double nlopt_optim_reltol;
  double loglik_reltol_cutoff;
  double nb_poisson_fallback_eps;
  double enet_abs_tol;
  double enet_rel_tol;
  uint32_t enet_max_iter;
  bool fixed_dispersion;
  double fixed_dispersion_value;
  EnumStateTrackStrategy state_track_strategy;
  uint64_t state_track_freq;
  bool verbose;
  bool include_data;
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

struct EcountgmifsStagewiseState
{
  arma::vec delta_beta;
  arma::vec delta_xbeta;
  arma::vec trial_nu_linear;
  arma::vec trial_mu_mean;

  double epsilon = 0.0;

  double negloglik_trial = arma::datum::nan;

  uint32_t halving_count = 0;
};

struct EcountgmifsState
{
  EcountgmifsPredictors param;

  double negloglik;
  double saturated_dispersion;
  double saturated_negloglik;
  double null_negloglik;

  Rcpp::NumericVector criteria;

  uint64_t iteration;

  double pseudo_r2 = 1.0;
};

struct EcountgmifsPath
{
  std::vector<EcountgmifsState> states;
  arma::uvec last_saved_active_set;
  bool active_set_changed;

  EnumTerminationStatus termination_status;

  std::string message;
};

struct EcountgmifsRuntime {
  EcountgmifsState& state;
  EcountgmifsPath& path;
  EcountgmifsGradients& gradient;
  EcountgmifsStagewiseState& stagewise;
};

struct EcountgmifsContext {
  EcountgmifsInput& input;
  EcountgmifsControl& control;
  EcountgmifsRuntime& runtime;
};

struct IEcountgmifsCriterion
{
  virtual ~IEcountgmifsCriterion() = default;

  virtual std::string name() const = 0;

  virtual double evaluate(
      const EcountgmifsContext& context
  ) const = 0;
};




