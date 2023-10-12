// utilUserLibrary.cpp
// This file is generated by Shroud nowrite-version. Do not edit.
// Copyright (c) 2017-2023, Lawrence Livermore National Security, LLC and
// other Shroud Project Developers.
// See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//

// typemap
#include "ExClass1.hpp"
#include "ExClass2.hpp"
// shroud
#include "typesUserLibrary.h"
#include <cstddef>
#include <cstring>


#ifdef __cplusplus
extern "C" {
#endif

// helper copy_string
// Copy the char* or std::string in context into c_var.
// Called by Fortran to deal with allocatable character.
void AA_ShroudCopyStringAndFree(AA_SHROUD_array *data, char *c_var,
    size_t c_var_len) {
    const char *cxx_var = data->addr.ccharp;
    size_t n = c_var_len;
    if (data->elem_len < n) n = data->elem_len;
    std::strncpy(c_var, cxx_var, n);
}


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
    cap->addr = nullptr;
    cap->idtor = 0;  // avoid deleting again
}

#ifdef __cplusplus
}
#endif
