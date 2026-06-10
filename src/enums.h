#pragma once

#include "../inst/include/ecountgmifs/api.h"

inline EnumStateTrackStrategy as_track_strategy(int x)
{
  switch (x) {
  case 0: return ACTIVE_SET_CHANGE;
  case 1: return ALL_ITERATION;
  case 2: return EVERY_K_ITERATION;
  case 3: return NO_STATE_TRACKING;
  default:
    Rcpp::stop("invalid coefficient save strategy");
  }
}

inline EnumLinkFunc as_link_func(uint32_t x)
{
  switch (x) {
  case 0: return LOG_LINK;
  case 1: return SOFTPLUS_LINK;
  default:
    Rcpp::stop("invalid link function");
  }
}

inline EnumFamily as_family(uint32_t x)
{
  switch (x) {
  case 0: return NEGATIVE_BINOMIAL;
  case 1: return POISSON;
  default:
    Rcpp::stop("invalid family");
  }
}
