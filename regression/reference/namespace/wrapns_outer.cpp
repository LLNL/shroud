// wrapns_outer.cpp
// This file is generated by Shroud nowrite-version. Do not edit.
// Copyright (c) 2017-2021, Lawrence Livermore National Security, LLC and
// other Shroud Project Developers.
// See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//

// cxx_header
#include "namespace.hpp"
// shroud
#include "wrapns_outer.h"

// splicer begin namespace.outer.CXX_definitions
// splicer end namespace.outer.CXX_definitions

extern "C" {

// splicer begin namespace.outer.C_definitions
// splicer end namespace.outer.C_definitions

// ----------------------------------------
// Function:  void One
// Attrs:     +intent(subroutine)
// Exact:     c_subroutine
void NS_outer_one(void)
{
    // splicer begin namespace.outer.function.one
    outer::One();
    // splicer end namespace.outer.function.one
}

}  // extern "C"
