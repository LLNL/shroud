// wraperror.cpp
// This file is generated by Shroud nowrite-version. Do not edit.
// Copyright (c) 2017-2023, Lawrence Livermore National Security, LLC and
// other Shroud Project Developers.
// See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//

// cxx_header
#include "error.hpp"
// shroud
#include "wraperror.h"

// splicer begin CXX_definitions
// splicer end CXX_definitions

extern "C" {

// splicer begin C_definitions
// splicer end C_definitions

// ----------------------------------------
// Function:  void BadFstatements
// Statement: c_subroutine
void ERR_BadFstatements(void)
{
    // splicer begin function.BadFstatements
    BadFstatements();
    ===>{no_c_var} = 11;<===
    ===>{bad_format = 12;<===
    // splicer end function.BadFstatements
}

// ----------------------------------------
// Function:  void BadFstatements
// Statement: f_subroutine
no-such-type ERR_BadFstatements_bufferify(void)
{
    // splicer begin function.BadFstatements_bufferify
    BadFstatements();
    // splicer end function.BadFstatements_bufferify
}

// ----------------------------------------
// Function:  void AssumedRank
// Statement: c_subroutine
// ----------------------------------------
// Argument:  int * data
// Statement: c_inout_native_*
void ERR_AssumedRank(int * data)
{
    // splicer begin function.AssumedRank
    AssumedRank(data);
    // splicer end function.AssumedRank
}

// Generated by fortran_generic
// ----------------------------------------
// Function:  void AssumedRank
// Statement: f_subroutine
// ----------------------------------------
// Argument:  int * data +rank(0)
// Statement: f_inout_native_*
void ERR_AssumedRank_0d_bufferify(int * data)
{
    // splicer begin function.AssumedRank_0d_bufferify
    AssumedRank(data);
    // splicer end function.AssumedRank_0d_bufferify
}

// Generated by fortran_generic
// ----------------------------------------
// Function:  void AssumedRank
// Statement: f_subroutine
// ----------------------------------------
// Argument:  int * data +rank(1)
// Statement: f_inout_native_*
void ERR_AssumedRank_1d_bufferify(int * data)
{
    // splicer begin function.AssumedRank_1d_bufferify
    AssumedRank(data);
    // splicer end function.AssumedRank_1d_bufferify
}

// Generated by fortran_generic
// ----------------------------------------
// Function:  void AssumedRank
// Statement: f_subroutine
// ----------------------------------------
// Argument:  int * data +rank(2)
// Statement: f_inout_native_*
void ERR_AssumedRank_2d_bufferify(int * data)
{
    // splicer begin function.AssumedRank_2d_bufferify
    AssumedRank(data);
    // splicer end function.AssumedRank_2d_bufferify
}

}  // extern "C"
