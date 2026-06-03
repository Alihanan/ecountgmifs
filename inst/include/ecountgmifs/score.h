#pragma once

#include <cstdint>

struct Score
{
  uint64_t iteration;

  double negloglik;
  double unpenalized_negloglik;
  double saturated_negloglik;

  double n;
  double p;

  double nnz;
  double df;
  double active_size;

  double dispersion;
  double enet_norm;
  double epsilon;
};
