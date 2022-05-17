// Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and
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

void test_structAsClass()
{
    structAsClass<double> mystruct = {2, 3.5};

    assert(mystruct.get_npts() == 2);
    assert(mystruct.get_value() == 3.5);

    mystruct.set_npts(5);
    mystruct.set_value(2.5);

    assert(mystruct.npts == 5);
    assert(mystruct.value == 2.5);

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

  test_structAsClass();
  
}
