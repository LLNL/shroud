// Copyright (c) 2017-2021, Lawrence Livermore National Security, LLC and
// other Shroud Project Developers.
// See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
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

void test_struct1()
{
    userStruct<double> mystruct;

    struct {
        int npts;
        double value;
    } explicit_struct;
       

    assert(sizeof(mystruct) == sizeof(explicit_struct));
}

int main(int argc, char *argv[])
{
  test_vector_int();

  test_pairs();

  test_struct1();
  
}
