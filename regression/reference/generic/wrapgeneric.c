// wrapgeneric.c
// This is generated code, do not edit
// Copyright (c) 2017-2019, Lawrence Livermore National Security, LLC and
// other Shroud Project Developers.
// See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//
#include <stdlib.h>
#include "generic.h"
#include "typesgeneric.h"

// splicer begin C_definitions
// splicer end C_definitions

// Release C++ allocated memory.
void GEN_SHROUD_memory_destructor(GEN_SHROUD_capsule_data *cap)
{
    cap->addr = NULL;
    cap->idtor = 0;  // avoid deleting again
}
