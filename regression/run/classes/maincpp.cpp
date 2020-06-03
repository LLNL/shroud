// Copyright (c) 2017-2020, Lawrence Livermore National Security, LLC and
// other Shroud Project Developers.
// See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
// #######################################################################
//
// Tests for classes.cpp
//

#include "classes.hpp"

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
