// wrapNames.cpp
// This file is generated by Shroud 0.11.0. Do not edit.
// Copyright (c) 2017-2020, Lawrence Livermore National Security, LLC and
// other Shroud Project Developers.
// See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//
#include "wrapNames.h"
#include <cstdlib>
#include "typesNames.h"

// splicer begin CXX_definitions
// splicer end CXX_definitions

extern "C" {

// splicer begin C_definitions
// splicer end C_definitions

// ----------------------------------------
// Function:  void AFunction
// Requested: c
// Match:     c_default
void NAM_afunction()
{
    // splicer begin function.afunction
    ignore1::ignore2::AFunction();
    // splicer end function.afunction
}

// Release library allocated memory.
void NAM_SHROUD_memory_destructor(NAM_SHROUD_capsule_data *cap)
{
    cap->addr.base = nullptr;
    cap->idtor = 0;  // avoid deleting again
}

}  // extern "C"
