#pragma once

#include <RcppArmadillo.h>

inline void check_finite_scalar(
    double x,
    const std::string& name
) {
  if (!std::isfinite(x)) {
    Rcpp::stop("value of '" + name + "' must be finite");
  }
}

inline void check_matrix_rows(
    const arma::mat& x,
    arma::uword n,
    const std::string& name
) {
  if (x.n_rows != n) {
    Rcpp::stop("matrix '" + name + "' has incompatible number of rows");
  }
}

inline void check_vector_finite(
    const arma::vec& x,
    const std::string& name
) {
  if (!x.is_finite()) {
    Rcpp::stop("vector '" + name + "' must contain only finite values");
  }
}

inline void check_matrix_finite(
    const arma::mat& x,
    const std::string& name
) {
  if (!x.is_finite()) {
    Rcpp::stop("matrix '" + name + "' must contain only finite values");
  }
}

inline void check_positive_scalar(
    double x,
    const std::string& name
) {
  check_finite_scalar(x, name);

  if (x <= 0.0) {
    Rcpp::stop("value of '" + name + "' must be positive");
  }
}

inline void check_nonnegative_scalar(
    double x,
    const std::string& name
) {
  check_finite_scalar(x, name);

  if (x < 0.0) {
    Rcpp::stop("value of '" + name + "' must be non-negative");
  }
}



inline void check_vector_length(
    const arma::vec& x,
    arma::uword n,
    const std::string& name
) {
  if (x.n_elem != n) {
    Rcpp::stop("vector '" + name + "' has incompatible length");
  }
}

inline bool weight_vec_has_prior(
    const arma::vec& weight_vec
) {
  if (weight_vec.n_elem == 0) {
    return false;
  }

  return !arma::approx_equal(
      weight_vec,
      arma::vec(weight_vec.n_elem, arma::fill::ones),
      "absdiff",
      1e-12
  );
}


