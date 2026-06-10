#pragma once

#include <RcppArmadillo.h>

#include <limits>
#include <string>
#include <utility>
#include <vector>

#include "../inst/include/ecountgmifs/api.h"

typedef double (*criterion_fun_t)(const EcountgmifsContext*);

struct CriterionValue
{
  std::string name;
  criterion_fun_t function;

  double value;

  CriterionValue(
    std::string name_,
    criterion_fun_t function_
  ) :
    name(std::move(name_)),
    function(function_),
    value(std::numeric_limits<double>::infinity())
  {}
};
