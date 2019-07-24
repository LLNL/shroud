// wrapInterface.c
// This is generated code, do not edit
// #######################################################################
// Copyright (c) 2017-2019, Lawrence Livermore National Security, LLC.
//
// Produced at the Lawrence Livermore National Laboratory
//
// LLNL-CODE-738041.
//
// All rights reserved.
//
// This file is part of Shroud.
//
// For details about use and distribution, please read LICENSE.
//
// #######################################################################
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
