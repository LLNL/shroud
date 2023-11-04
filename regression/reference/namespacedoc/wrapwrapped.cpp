// wrapwrapped.cpp
// This file is generated by Shroud nowrite-version. Do not edit.
// Copyright (c) 2017-2023, Lawrence Livermore National Security, LLC and
// other Shroud Project Developers.
// See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//

// shroud
#include "wrapwrapped.h"

// splicer begin CXX_definitions
// splicer end CXX_definitions

extern "C" {

// splicer begin C_definitions
// splicer end C_definitions

// ----------------------------------------
// Function:  void worker3
// Statement: c_subroutine
void WWW_inner3_worker3(void)
{
    // splicer begin function.worker3
    outer::inner3::worker3();
    // splicer end function.worker3
}

// ----------------------------------------
// Function:  void worker
// Statement: c_subroutine
void WWW_worker(void)
{
    // splicer begin function.worker
    outer::worker();
    // splicer end function.worker
}

}  // extern "C"
