// wrapInterface.c
// This file is generated by Shroud 0.12.0. Do not edit.
// Copyright (c) 2017-2020, Lawrence Livermore National Security, LLC and
// other Shroud Project Developers.
// See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//
#include <stdlib.h>
#include "interface.h"
#include "typesInterface.h"

// splicer begin C_definitions
// splicer end C_definitions

// Release library allocated memory.
void INT_SHROUD_memory_destructor(INT_SHROUD_capsule_data *cap)
{
    cap->addr = NULL;
    cap->idtor = 0;  // avoid deleting again
}
