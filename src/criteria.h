#pragma once
#include "../inst/include/ecountgmifs/score.h"
/*
 * It includes:
 *
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
*/

typedef double (*criterion_fun_t)(const Score*);

struct CriterionSpec
{
  std::string name;
  criterion_fun_t fn;

  CriterionSpec(
    const std::string& name_,
    criterion_fun_t fn_
  ) :
    name(name_),
    fn(fn_)
  {}
};
