// wrapreferences.cpp
// This file is generated by Shroud 0.11.0. Do not edit.
// Copyright (c) 2017-2020, Lawrence Livermore National Security, LLC and
// other Shroud Project Developers.
// See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//
#include <cstdlib>
#include "references.hpp"
#include "typesreferences.h"

// splicer begin CXX_definitions
// splicer end CXX_definitions

extern "C" {

// splicer begin C_definitions
// splicer end C_definitions

// Release library allocated memory.
void REF_SHROUD_memory_destructor(REF_SHROUD_capsule_data *cap)
{
    cap->addr.base = nullptr;
    cap->idtor = 0;  // avoid deleting again
}

}  // extern "C"
