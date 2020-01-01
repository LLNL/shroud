// wrapuser_int.cpp
// This is generated code, do not edit
// Copyright (c) 2017-2020, Lawrence Livermore National Security, LLC and
// other Shroud Project Developers.
// See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//
#include "wrapuser_int.h"
#include "templates.hpp"

// splicer begin class.user.CXX_definitions
// splicer end class.user.CXX_definitions

extern "C" {

// splicer begin class.user.C_definitions
// splicer end class.user.C_definitions

void TEM_user_int_nested_double(TEM_user_int * self, int arg1,
    double arg2)
{
    user<int> *SH_this = static_cast<user<int> *>(self->addr);
    // splicer begin class.user.method.nested_double
    SH_this->nested<double>(arg1, arg2);
    return;
    // splicer end class.user.method.nested_double
}

}  // extern "C"
