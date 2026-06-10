#include <RcppArmadillo.h>
#include "../inst/include/ecountgmifs/api.h"
#include "../src/numerical_internal_constants.h"
// [[Rcpp::depends(RcppArmadillo)]]

// [[Rcpp::export]]
double ecountgmifs_call_context_criterion(
    SEXP criterion_ptr,

    double negloglik = 100.0,
    uint64_t iteration = 10,
    double dispersion = 0.25,
    double epsilon = 1e-4,
    bool initialized = true,

    double n = 50.0,
    double p = 12.0,
    double q = 1.0,

    double beta_value = 0.0,
    double beta_nonzero_value = 1.0,
    uint64_t nnz = 4,

    double enet_alpha = 0.75,
    bool include_data = false
) {
  typedef double (*criterion_fun_t)(const EcountgmifsContext*);

  void* address = R_ExternalPtrAddr(criterion_ptr);

  if (address == nullptr) {
    Rcpp::stop("criterion pointer is null");
  }

  criterion_fun_t criterion =
    reinterpret_cast<criterion_fun_t>(address);

  const arma::uword n_arma = static_cast<arma::uword>(n);
  const arma::uword p_arma = static_cast<arma::uword>(p);
  const arma::uword q_arma = static_cast<arma::uword>(q);

  arma::mat X(
      n_arma,
      p_arma,
      arma::fill::ones
  );

  arma::vec y(
      n_arma,
      arma::fill::ones
  );

  arma::mat w(
      n_arma,
      q_arma,
      arma::fill::ones
  );

  arma::vec offset(
      n_arma,
      arma::fill::zeros
  );

  arma::vec weight_vec(
      p_arma,
      arma::fill::ones
  );

  const arma::uword n_test = 0;

  arma::mat Xtest(
      n_test,
      p_arma,
      arma::fill::zeros
  );

  arma::vec ytest(n_test);

  arma::mat wtest(
      n_test,
      q_arma,
      arma::fill::zeros
  );

  arma::vec offsettest(n_test);
  arma::vec yorig = y;

  const arma::vec train_y_one_lgamma =
    -1.0 * arma::lgamma(y + 1.0);

    const arma::vec test_y_one_lgamma(n_test);

    const arma::vec orig_y_one_lgamma =
      -1.0 * arma::lgamma(yorig + 1.0);

      EcountgmifsInput input {
        X,
        y,
        w,
        offset,
        weight_vec,

        Xtest,
        ytest,
        wtest,
        offsettest,
        yorig,

        train_y_one_lgamma,
        test_y_one_lgamma,
        orig_y_one_lgamma,

        NEGATIVE_BINOMIAL,
        LOG_LINK,
        enet_alpha
      };

      EcountgmifsControl control {
        1000,                // iteration_max
        1.0,                 // epsilon_max
        epsilon,             // epsilon_start
        1e-8,                // epsilon_min
        1e-8,                // tol
        1e-8,                // nlopt_optim_reltol
        0.0,                 // loglik_reltol_cutoff / negloglik cutoff if not renamed yet
        false,               // fixed_dispersion
        0.0,                 // fixed_dispersion_value
        NO_STATE_TRACKING,   // state_track_strategy
        1,                   // state_track_freq
        false,               // verbose
        include_data         // include_data
      };

      arma::vec beta(
          p_arma,
          arma::fill::value(beta_value)
      );

      for (arma::uword j = 0;
           j < beta.n_elem && j < static_cast<arma::uword>(nnz);
           ++j) {
        beta[j] = beta_nonzero_value;
      }

      arma::vec theta(
          q_arma,
          arma::fill::zeros
      );

      arma::vec eta =
        w * theta + X * beta;

      arma::vec mu =
        arma::clamp(
          arma::exp(offset + eta),
          MU_MIN_CAP,
          MU_MAX_CAP
        );

      EcountgmifsState state {
        beta,
        theta,
        eta,
        mu,
        dispersion,
        negloglik,
        dispersion,
        negloglik,
        Rcpp::NumericVector(),
        iteration,
        epsilon,
        initialized
      };

      EcountgmifsContext ctx {
        input,
        control,
        state
      };

      return criterion(&ctx);
}
