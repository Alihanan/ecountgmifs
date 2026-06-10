#pragma once
#ifndef _ENET_H_
#define _ENET_H_

#include <RcppArmadillo.h>

template <typename T>
inline int signum(T x) {
  return (T(0) < x) - (x < T(0));
}

// Soft-thresholding: soft(v, tau) = sign(v) % max(|v|-tau, 0)
static inline arma::vec soft_threshold_weight(const arma::vec& c, const arma::vec& lambdaAlphaWeight) {
  return arma::sign(c) % arma::clamp(arma::abs(c) - lambdaAlphaWeight, 0.0, arma::datum::inf);
}

// g(x) = alpha * ||x||_1 + (1-alpha) * ||x||_2^2
static inline double elastic_net_g_weight(const arma::vec& x, const arma::vec& weight_vec, double alpha) {
  return alpha * arma::accu(weight_vec % arma::abs(x)) + (1.0 - alpha) * arma::accu(weight_vec % arma::square(x));
}


static inline arma::vec x_of_lambda_weight(const arma::vec& c, double alpha, double lambda, const arma::vec& weight_vec) {
  return -1 * soft_threshold_weight(c, lambda * alpha * weight_vec) / (2.0 * lambda * (1-alpha) * weight_vec);
}

static inline double g_at_weight(const arma::vec& c, double alpha, double lambda, const arma::vec& weight_vec)
{
  arma::vec xl = x_of_lambda_weight(c, alpha, lambda, weight_vec);
  return elastic_net_g_weight(xl, weight_vec, alpha);
}

// Solves:
//
//     min_x c^T x
//     s.t.  sum_j weight_j * |x_j| <= epsilon
//
// Analytical LASSO solution:
// all mass goes to the coordinate with largest |c_j| / weight_j.
//
inline arma::vec solve_lasso_1D_weight(
    const arma::vec& c,
    const arma::vec& weight_vec,
    double epsilon
) {
  if (c.n_elem != weight_vec.n_elem) {
    Rcpp::stop("c and weight_vec must have the same length.");
  }

  if (epsilon < 0.0) {
    Rcpp::stop("epsilon must be non-negative.");
  }

  if (arma::any(weight_vec <= 0.0)) {
    Rcpp::stop("All weights must be positive.");
  }

  arma::vec x(c.n_elem, arma::fill::zeros);

  if (epsilon == 0.0 || arma::norm(c, 2) == 0.0) {
    return x;
  }

  arma::vec score = arma::abs(c) / weight_vec;
  arma::uword j = score.index_max();

  if (score[j] == 0.0) {
    return x;
  }

  x[j] = -epsilon * signum(c[j]) / weight_vec[j];

  return x;
}
// -----------------------------------------------------------------------------
// Weighted ridge analytical solution
//
// Solves:
//
//     min_x c^T x
//     s.t.  sum_j weight_j * x_j^2 <= eps
//
// Solution:
//
//     x_j = -sqrt(eps) * (c_j / weight_j)
//           / sqrt(sum_k c_k^2 / weight_k)
// -----------------------------------------------------------------------------

inline arma::vec solve_ridge_1D_weight(
    const arma::vec& c,
    const arma::vec& weight_vec,
    double eps
) {
  if (c.n_elem != weight_vec.n_elem) {
    Rcpp::stop("c and weight_vec must have the same length.");
  }

  if (eps < 0.0) {
    Rcpp::stop("eps must be non-negative.");
  }

  if (arma::any(weight_vec <= 0.0)) {
    Rcpp::stop("All weights must be positive.");
  }

  arma::vec x(c.n_elem, arma::fill::zeros);

  if (eps == 0.0 || arma::norm(c, 2) == 0.0) {
    return x;
  }

  arma::vec inv_weight_c = c / weight_vec;

  double denom_sq = arma::dot(c, inv_weight_c);

  if (denom_sq <= 0.0) {
    return x;
  }

  double denom = std::sqrt(denom_sq);

  x = -std::sqrt(eps) * inv_weight_c / denom;

  return x;
}

// Solve min c^T x s.t. alpha||x||_1 + (1-alpha)||x||_2^2 <= eps
inline arma::vec solve_elastic_net_1D_weight(
    const arma::vec& c,
    const arma::vec& weight_vec,
    double alpha,
    double eps,
    double abs_tol = 1e-10,
    double rel_tol = 1e-6,
    uint32_t max_iter = 99,
    bool verbose = false
)
{
  if (c.n_elem != weight_vec.n_elem) {
    Rcpp::stop("c and weight_vec must have the same length.");
  }

  if (eps < 0.0) {
    Rcpp::stop("eps must be non-negative.");
  }

  if (alpha < 0.0 || alpha > 1.0) {
    Rcpp::stop("alpha must be in [0, 1].");
  }

  if (arma::any(weight_vec <= 0.0)) {
    Rcpp::stop("All weights must be positive.");
  }

  if (!c.is_finite()) {
    Rcpp::stop("value of 'c' must contain only finite values");
  }

  if (!weight_vec.is_finite()) {
    Rcpp::stop("value of 'weight_vec' must contain only finite values");
  }

  arma::vec x(c.n_elem, arma::fill::zeros);

  if (eps == 0.0) {
    return x;
  }

  if (arma::norm(c, 2) == 0.0) {
    return x;
  }

  if (alpha == 1.0) {
    return solve_lasso_1D_weight(
      c,
      weight_vec,
      eps
    );
  }

  if (alpha == 0.0) {
    return solve_ridge_1D_weight(
      c,
      weight_vec,
      eps
    );
  }

  const double min_lambda = 1e-16;

  double lambda_hi = 1.0;
  double ghi = g_at_weight(
    c,
    alpha,
    lambda_hi,
    weight_vec
  );

  double lambda_lo = lambda_hi;
  double glo = ghi;
  bool has_lo = false;

  if (verbose) {
    Rcpp::Rcout << "elastic-net lambda_hi search\n";
  }

  while (ghi > eps) {
    lambda_lo = lambda_hi;
    glo = ghi;
    has_lo = true;

    lambda_hi *= 2.0;

    if (!std::isfinite(lambda_hi)) {
      Rcpp::stop("lambda_hi became non-finite during elastic-net search.");
    }

    ghi = g_at_weight(
      c,
      alpha,
      lambda_hi,
      weight_vec
    );

    if (verbose) {
      Rcpp::Rcout
      << "  lambda_hi = " << lambda_hi
      << ", g = " << ghi
      << ", eps = " << eps
      << "\n";
    }
  }

  if (verbose) {
    Rcpp::Rcout << "elastic-net lambda_lo search\n";
  }

  if (!has_lo) {
    lambda_lo = lambda_hi;
    glo = ghi;

    while (glo <= eps && lambda_lo > min_lambda) {
      lambda_lo *= 0.5;

      if (lambda_lo <= min_lambda) {
        break;
      }

      glo = g_at_weight(
        c,
        alpha,
        lambda_lo,
        weight_vec
      );

      if (verbose) {
        Rcpp::Rcout
        << "  lambda_lo = " << lambda_lo
        << ", g = " << glo
        << ", eps = " << eps
        << "\n";
      }
    }
  }

  if ((lambda_lo <= min_lambda) || (glo <= eps)) {
    Rcpp::stop("Could not bracket elastic-net lambda solution.");
  }

  arma::vec xmid;
  uint32_t iter = 0;

  while (true) {
    ++iter;

    if (iter > max_iter) {
      Rcpp::stop("More than max_iter iterations during elastic-net bisection.");
    }

    double lambda_mid = 0.5 * (lambda_lo + lambda_hi);

    if (lambda_mid <= min_lambda) {
      Rcpp::stop("lambda_mid became too small during elastic-net bisection.");
    }

    xmid = x_of_lambda_weight(
      c,
      alpha,
      lambda_mid,
      weight_vec
    );

    double gmid = elastic_net_g_weight(
      xmid,
      weight_vec,
      alpha
    );

    if (verbose) {
      Rcpp::Rcout
      << "  iter = " << iter
      << ", lambda_mid = " << lambda_mid
      << ", g = " << gmid
      << ", eps = " << eps
      << "\n";
    }

    if (std::abs(gmid - eps) <= std::max(abs_tol, eps * rel_tol)) {
      if (verbose) {
        Rcpp::Rcout << "elastic-net bisection converged\n";
      }

      return xmid;
    }

    if (gmid > eps) {
      lambda_lo = lambda_mid;
    } else {
      lambda_hi = lambda_mid;
    }
  }

  return x_of_lambda_weight(
    c,
    alpha,
    lambda_hi,
    weight_vec
  );
}


#endif
