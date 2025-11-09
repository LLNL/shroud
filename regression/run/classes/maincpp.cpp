// Copyright Shroud Project Developers. See LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
// #######################################################################
//
// Tests for classes.cpp
//

#include "classes.hpp"

int tester1()
{
  return classes::Class1::DIRECTION::UP;
}
int tester2()
{
  return classes::Class1::UP;
}

int main(int argc, char *argv[])
{
#if 0
    classes::Class1 * obj = new classes::Class1;

    obj->Method1();

    delete obj;
#else
    classes::Class1 obj;

    obj.Method1();
#endif
}
