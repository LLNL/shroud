// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
// other Shroud Project Developers.
// See the top-level COPYRIGHT file for details.
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
