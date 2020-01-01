// wrapwrapped.cpp
// This is generated code, do not edit
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

// void worker3()
void WWW_inner3_worker3()
{
    // splicer begin function.worker3
    outer::inner3::worker3();
    return;
    // splicer end function.worker3
}

// void worker()
void WWW_worker()
{
    // splicer begin function.worker
    outer::worker();
    return;
    // splicer end function.worker
}

// Release library allocated memory.
void WWW_SHROUD_memory_destructor(WWW_SHROUD_capsule_data *cap)
{
    cap->addr = NULL;
    cap->idtor = 0;  // avoid deleting again
}

}  // extern "C"
