// wrapdefault_library.cpp
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
#include "wrapdefault_library.h"
#include <stdlib.h>
#include "global_header.hpp"
#include "typesdefault_library.h"


extern "C" {


void DEF_one_two_function1()
{
    one::two::function1();
    return;
}

// Release library allocated memory.
void DEF_SHROUD_memory_destructor(DEF_SHROUD_capsule_data *cap)
{
    cap->addr = NULL;
    cap->idtor = 0;  // avoid deleting again
}

}  // extern "C"
