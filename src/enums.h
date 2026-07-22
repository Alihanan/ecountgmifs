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

inline const char* termination_status_name(
    EnumTerminationStatus status
) noexcept
{
  switch (status) {
  case RUNNING:
    return "RUNNING";

  case CONVERGED:
    return "CONVERGED";

  case ITERATION_LIMIT_REACHED:
    return "ITERATION_LIMIT_REACHED";

  case EPSILON_MIN_REACHED:
    return "EPSILON_MIN_REACHED";

  case FAILED:
    return "FAILED";
  }

  return "UNKNOWN_TERMINATION_STATUS";
}
