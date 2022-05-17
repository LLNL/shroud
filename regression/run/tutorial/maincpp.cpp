// Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and
// other Shroud Project Developers.
// See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
// #######################################################################
//
// Tests for tutorial.cpp
//

#include "tutorial.hpp"

int tester1()
{
  return tutorial::Class1::DIRECTION::UP;
}
int tester2()
{
  return tutorial::Class1::UP;
}



int main(int argc, char *argv[])
{
    tutorial::Class1 * obj = new tutorial::Class1;

    obj->Method1();

    delete obj;
}
