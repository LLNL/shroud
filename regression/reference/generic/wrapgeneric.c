// wrapgeneric.c
// This is generated code, do not edit
// Copyright (c) 2017-2020, Lawrence Livermore National Security, LLC and
// other Shroud Project Developers.
// See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//
#include "wrapgeneric.h"
#include <stdlib.h>
#include "generic.h"
#include "helper.h"
#include "typesgeneric.h"

// splicer begin C_definitions
// splicer end C_definitions

// void SavePointer2(void * addr +intent(in)+value, int type +implied(type(addr))+intent(in)+value, size_t size +implied(size(addr))+intent(in)+value)
void GEN_save_pointer2(void * addr, int type, size_t size)
{
    // splicer begin function.save_pointer2
    type = convert_type(type);
    SavePointer2(addr, type, size);
    return;
    // splicer end function.save_pointer2
}

// start release allocated memory
// Release library allocated memory.
void GEN_SHROUD_memory_destructor(GEN_SHROUD_capsule_data *cap)
{
    cap->addr = NULL;
    cap->idtor = 0;  // avoid deleting again
}
// end release allocated memory
