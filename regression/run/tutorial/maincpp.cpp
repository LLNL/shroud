// Copyright Shroud Project Developers. See LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
// #######################################################################
//
// Tests for tutorial.cpp
//

#include "tutorial.hpp"


int main(int argc, char *argv[])
{
    tutorial::Class1 * obj = new tutorial::Class1;

    obj->Method1();

    delete obj;
}
