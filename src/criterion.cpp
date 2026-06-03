#include <RcppArmadillo.h>
#include "../inst/include/ecountgmifs/score.h"

// [[Rcpp::depends(RcppArmadillo)]]

// [[Rcpp::export]]
double ecountgmifs_test_call_score_criterion(
    SEXP criterion,
    uint64_t iteration = 10,

    double negloglik = 100.0,
    double unpenalized_negloglik = 150.0,
    double saturated_negloglik = 80.0,

    double n = 50.0,
    double p = 12.0,

    double nnz = 3.0,
    double df = 4.0,
    double active_size = 3.0,

    double dispersion = 0.25,
    double enet_norm = 2.5,
    double epsilon = 1e-4
) {
  typedef double (*criterion_fun_t)(const Score*);

  void* address = R_ExternalPtrAddr(criterion);

  if (address == nullptr) {
    Rcpp::stop("criterion pointer is null");
  }

  criterion_fun_t criterion_fun =
    reinterpret_cast<criterion_fun_t>(address);

  Score score;

  score.iteration = iteration;

  score.negloglik = negloglik;
  score.unpenalized_negloglik = unpenalized_negloglik;
  score.saturated_negloglik = saturated_negloglik;

  score.n = n;
  score.p = p;

  score.nnz = nnz;
  score.df = df;
  score.active_size = active_size;

  score.dispersion = dispersion;
  score.enet_norm = enet_norm;
  score.epsilon = epsilon;

  return criterion_fun(&score);
}
