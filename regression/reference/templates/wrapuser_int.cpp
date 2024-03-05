// wrapuser_int.cpp
// This file is generated by Shroud nowrite-version. Do not edit.
// Copyright (c) 2017-2024, Lawrence Livermore National Security, LLC and
// other Shroud Project Developers.
// See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//

// cxx_header
#include "templates.hpp"
// shroud
#include "wrapuser_int.h"

// splicer begin class.user.CXX_definitions
// splicer end class.user.CXX_definitions

extern "C" {

// splicer begin class.user.C_definitions
// splicer end class.user.C_definitions

// Generated by cxx_template
// ----------------------------------------
// Function:  void nested
// Statement: c_subroutine
// ----------------------------------------
// Argument:  int arg1
// Statement: c_in_native_scalar
// ----------------------------------------
// Argument:  double arg2
// Statement: c_in_native_scalar
void TEM_user_int_nested_double(TEM_user_int * self, int arg1,
    double arg2)
{
    user<int> *SH_this = static_cast<user<int> *>(self->addr);
    // splicer begin class.user.method.nested_double
    SH_this->nested<double>(arg1, arg2);
    // splicer end class.user.method.nested_double
}

}  // extern "C"
