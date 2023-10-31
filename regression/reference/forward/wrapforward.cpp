// wrapforward.cpp
// This file is generated by Shroud nowrite-version. Do not edit.
// Copyright (c) 2017-2023, Lawrence Livermore National Security, LLC and
// other Shroud Project Developers.
// See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//

// cxx_header
#include "forward.hpp"
// shroud
#include "wrapforward.h"

// splicer begin CXX_definitions
// splicer end CXX_definitions

extern "C" {

// splicer begin C_definitions
// splicer end C_definitions

// ----------------------------------------
// Function:  int passStruct1
// Attrs:     +intent(function)
// Statement: c_function_native_scalar
// ----------------------------------------
// Argument:  const Cstruct1 * arg
// Attrs:     +intent(in)
// Statement: c_in_struct_*
int FOR_passStruct1(const Cstruct1 * arg)
{
    // splicer begin function.passStruct1
    int SHC_rv = forward::passStruct1(arg);
    return SHC_rv;
    // splicer end function.passStruct1
}

}  // extern "C"
