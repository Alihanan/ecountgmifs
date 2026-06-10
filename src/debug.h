#pragma once

#define ECOUNTGMIFS_VERBOSE(VERBOSE, MSG)                          \
do {                                                               \
  if (VERBOSE) {                                                   \
    Rcpp::Rcout << "[ecountgmifs] " << MSG << "\n";                \
  }                                                                \
} while (false)
