// Copyright (c) 2018-2019, Lawrence Livermore National Security, LLC. 
//
// Produced at the Lawrence Livermore National Laboratory 
//
// LLNL-CODE-738041.
//
// All rights reserved. 
//
// This file is part of Shroud.
//
// For details about use and distribution, please read LICENSE.
//
// #######################################################################
//
// Tests for templates.yaml
//
// Wrap std::vector

#include <templates.hpp>

#include <cassert>
#include <vector>

// explicitly instantiate template
//template class std::vector<float>;

void test_vector_int()
{
  std::vector<int> vec;

  vec.push_back(1);
  vec.push_back(2);
  assert(vec.size() == 2);

}

void test_pairs()
{
  mypair<int> myobject (100, 75);
  int big = myobject.getmax();
  assert(big == 100);
}

int main(int argc, char *argv[])
{
  test_vector_int();

  test_pairs();
}
