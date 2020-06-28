// wrapwrapped.cpp
// This file is generated by Shroud 0.12.1. Do not edit.
// Copyright (c) 2017-2019, Lawrence Livermore National Security, LLC and
// other Shroud Project Developers.
// See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//
#include "wrapwrapped.h"
#include <cstdlib>
#include "typeswrapped.h"

// splicer begin CXX_definitions
// splicer end CXX_definitions

extern "C" {

// splicer begin C_definitions
// splicer end C_definitions

// ----------------------------------------
// Function:  void worker3
// Requested: c
// Match:     c_default
void WWW_inner3_worker3(void)
{
    // splicer begin function.worker3
    outer::inner3::worker3();
    // splicer end function.worker3
}

// ----------------------------------------
// Function:  void worker
// Requested: c
// Match:     c_default
void WWW_worker(void)
{
    // splicer begin function.worker
    outer::worker();
    // splicer end function.worker
}

// Release library allocated memory.
void WWW_SHROUD_memory_destructor(WWW_SHROUD_capsule_data *cap)
{
    cap->addr = nullptr;
    cap->idtor = 0;  // avoid deleting again
}

}  // extern "C"
