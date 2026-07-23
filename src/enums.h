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

inline const char* state_track_strategy_name(
    EnumStateTrackStrategy strategy
) noexcept
{
  switch (strategy) {
  case EnumStateTrackStrategy::ACTIVE_SET_CHANGE:
    return "Active set change";

  case EnumStateTrackStrategy::ALL_ITERATION:
    return "All iterations";

  case EnumStateTrackStrategy::EVERY_K_ITERATION:
    return "Every k iterations";

  case EnumStateTrackStrategy::NO_STATE_TRACKING:
    return "No state tracking";
  }

  return "Unknown";
}

inline const char* stagewise_termination_reason_label(
    EnumStagewiseTerminationReason reason
) noexcept
{
  switch (reason) {
  case EnumStagewiseTerminationReason::STAGEWISE_NOT_INITIALIZED:
    return "Not initialized";

  case EnumStagewiseTerminationReason::STAGEWISE_RUNNING:
    return "Running";

  case EnumStagewiseTerminationReason::STAGEWISE_BETA_STEP_ZERO:
    return "Beta step zero";

  case EnumStagewiseTerminationReason::STAGEWISE_BETA_STALLED:
    return "Beta stalled";

  case EnumStagewiseTerminationReason::STAGEWISE_OBJECTIVE_STALLED:
    return "Objective stalled";

  case EnumStagewiseTerminationReason::
    STAGEWISE_PSEUDO_R2_CUTOFF_REACHED:
    return "Pseudo-R2 cutoff reached";

  case EnumStagewiseTerminationReason::
    STAGEWISE_EPSILON_MIN_REACHED:
    return "Minimum epsilon reached";

  case EnumStagewiseTerminationReason::
    STAGEWISE_ITERATION_LIMIT_REACHED:
    return "Iteration limit reached";
  }

  return "Unknown termination reason";
}
