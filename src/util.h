#pragma once

#include <RcppArmadillo.h>
#include <type_traits>

inline void check_finite_scalar(
    double x,
    const std::string& name
) {
  if (!std::isfinite(x)) {
    Rcpp::stop("value of '" + name + "' must be finite");
  }
}

inline void check_initial_bounds(
    double initial,
    double lower,
    double upper,
    const char* name
)
{
  if (initial < lower || initial > upper) {
    Rcpp::stop(
      "initial value of '%s' must lie within its bounds",
      name
    );
  }
}


inline void check_initial_bounds(
    const arma::vec& initial,
    const arma::vec& lower,
    const arma::vec& upper,
    arma::uword expected_size,
    const char* name
)
{
  if (
      initial.n_elem != expected_size ||
        lower.n_elem != expected_size ||
        upper.n_elem != expected_size
  ) {
    Rcpp::stop(
      "'%s' initial values and bounds must each contain %llu elements",
      name,
      static_cast<unsigned long long>(expected_size)
    );
  }

  if (
      arma::any(initial < lower) ||
        arma::any(initial > upper)
  ) {
    Rcpp::stop(
      "initial values of '%s' must lie within their bounds",
      name
    );
  }
}

template <typename Integer>
inline void check_positive_integer(
    Integer value,
    const char* name
)
{
  static_assert(
    std::is_integral<Integer>::value,
    "check_positive_integer requires an integer type"
  );

  if constexpr (std::is_signed<Integer>::value) {
    if (value <= 0) {
      Rcpp::stop(
        "value of '%s' must be positive",
        name
      );
    }
  } else {
    if (value == 0) {
      Rcpp::stop(
        "value of '%s' must be positive",
        name
      );
    }
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


