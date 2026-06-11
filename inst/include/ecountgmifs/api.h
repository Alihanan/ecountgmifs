#pragma once

#include <RcppArmadillo.h>
#include <cstdint>

enum EnumStateTrackStrategy
{
  ACTIVE_SET_CHANGE = 0,
  ALL_ITERATION = 1,
  EVERY_K_ITERATION = 2,
  NO_STATE_TRACKING = 3
};

enum EnumLinkFunc
{
  LOG_LINK = 0,
  SOFTPLUS_LINK = 1
};

enum EnumFamily
{
  NEGATIVE_BINOMIAL = 0,
  POISSON = 1
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

  const arma::mat& Xtest;
  const arma::vec& ytest;
  const arma::mat& wtest;
  const arma::vec& offsettest;
  const arma::vec& yorig;

  const arma::vec train_y_one_lgamma;
  const arma::vec test_y_one_lgamma;
  const arma::vec orig_y_one_lgamma;

  EnumFamily family;
  EnumLinkFunc link_func;
  double enet_alpha;
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
  bool fixed_dispersion;
  double fixed_dispersion_value;
  EnumStateTrackStrategy state_track_strategy;
  uint64_t state_track_freq;
  bool verbose;
  bool include_data;
};

struct EcountgmifsState
{
  arma::vec beta;
  arma::vec theta;

  arma::vec eta;
  arma::vec mu;

  double dispersion;
  double negloglik;

  double saturated_dispersion;
  double saturated_negloglik;

  double null_negloglik;

  Rcpp::NumericVector criteria;

  uint64_t iteration;
  double epsilon;
  bool initialized;

  double pseudo_r2 = 1.0;
};

struct EcountgmifsContext {
  EcountgmifsInput& input;
  EcountgmifsControl& control;
  EcountgmifsState& state;
};
