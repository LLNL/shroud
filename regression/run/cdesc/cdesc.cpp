// Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and
// other Shroud Project Developers.
// See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//

#include "cdesc.hpp"

void Rank2In(int *arg)
{
}

void GetScalar1(std::string & name, void *value)
{
}


// Specialize template to emulate some sort of database which
// returns different values based on type.
template<>
int getData<int>()
{
  return 1;
}
template<>
long getData<long>()
{
  return 2;
}
template<>
float getData<float>()
{
  return 3.0;
}
template<>
double getData<double>()
{
  return 4.0;
}
