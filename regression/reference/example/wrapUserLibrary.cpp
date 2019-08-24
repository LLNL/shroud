// wrapUserLibrary.cpp
// This is generated code, do not edit
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
#include "ExClass1.hpp"
#include "ExClass2.hpp"
#include "typesUserLibrary.h"

// splicer begin CXX_definitions
//   CXX_definitions
// splicer end CXX_definitions

extern "C" {

// splicer begin C_definitions
// splicer end C_definitions

// Release library allocated memory.
void AA_SHROUD_memory_destructor(AA_SHROUD_capsule_data *cap)
{
    void *ptr = cap->addr;
    switch (cap->idtor) {
    case 0:   // --none--
    {
        // Nothing to delete
        break;
    }
    case 1:   // example::nested::ExClass1
    {
        example::nested::ExClass1 *cxx_ptr = 
            reinterpret_cast<example::nested::ExClass1 *>(ptr);
        delete cxx_ptr;
        break;
    }
    case 2:   // example::nested::ExClass2
    {
        example::nested::ExClass2 *cxx_ptr = 
            reinterpret_cast<example::nested::ExClass2 *>(ptr);
        delete cxx_ptr;
        break;
    }
    default:
    {
        // Unexpected case in destructor
        break;
    }
    }
    cap->addr = NULL;
    cap->idtor = 0;  // avoid deleting again
}

}  // extern "C"
