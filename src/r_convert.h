#pragma once

#include <RcppArmadillo.h>
#include <algorithm>
#include <cstdint>
#include <limits>
#include <type_traits>

namespace ecountgmifs::output
{

template <
  typename Integer,
  std::enable_if_t<
    std::is_integral_v<Integer> &&
    !std::is_same_v<Integer, bool>,
    int
  > = 0
>
inline Rcpp::RObject to_r_integer(
    Integer value,
    const char* value_name = "integer value"
)
{
  bool fits_r_integer = false;
  bool fits_exact_r_numeric = false;

  if constexpr (std::is_signed_v<Integer>) {
    const std::intmax_t wide_value =
      static_cast<std::intmax_t>(value);

    fits_r_integer =
      wide_value >=
      static_cast<std::intmax_t>(
        std::numeric_limits<int>::min()
      ) &&
        wide_value <=
        static_cast<std::intmax_t>(
          std::numeric_limits<int>::max()
        );

    constexpr std::intmax_t max_exact_numeric =
      static_cast<std::intmax_t>(1) << 53;

    fits_exact_r_numeric =
      wide_value >= -max_exact_numeric &&
      wide_value <= max_exact_numeric;
  } else {
    const std::uintmax_t wide_value =
      static_cast<std::uintmax_t>(value);

    fits_r_integer =
      wide_value <=
      static_cast<std::uintmax_t>(
        std::numeric_limits<int>::max()
      );

    constexpr std::uintmax_t max_exact_numeric =
      static_cast<std::uintmax_t>(1) << 53;

    fits_exact_r_numeric =
      wide_value <= max_exact_numeric;
  }

  if (fits_r_integer) {
    return Rcpp::wrap(
      static_cast<int>(value)
    );
  }

  if (fits_exact_r_numeric) {
    Rcpp::warning(
      "%s exceeds the R integer range; returning it as numeric",
      value_name
    );
  } else {
    Rcpp::warning(
      "%s exceeds both the R integer range and the exact numeric "
      "integer range; precision may be lost",
      value_name
    );
  }

  return Rcpp::wrap(
    static_cast<double>(value)
  );
}

inline Rcpp::NumericVector to_r_vector(
    const arma::vec& value
)
{
  Rcpp::NumericVector out(value.n_elem);

  if (value.n_elem > 0) {
    std::copy(
      value.begin(),
      value.end(),
      out.begin()
    );
  }

  return out;
}

inline Rcpp::NumericMatrix to_r_matrix(
    const arma::mat& value
)
{
  Rcpp::NumericMatrix out(
      value.n_rows,
      value.n_cols
  );

  if (value.n_elem > 0) {
    std::copy(
      value.begin(),
      value.end(),
      out.begin()
    );
  }

  return out;
}

inline Rcpp::LogicalVector to_r_logical_vector(
    const arma::uvec& value
)
{
  Rcpp::LogicalVector out(value.n_elem);

  for (arma::uword i = 0; i < value.n_elem; ++i) {
    out[i] =
      value[i] != 0;
  }

  return out;
}

} // namespace ecountgmifs::output
