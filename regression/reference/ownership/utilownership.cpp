// utilownership.cpp
// This file is generated by Shroud nowrite-version. Do not edit.
// Copyright (c) 2017-2024, Lawrence Livermore National Security, LLC and
// other Shroud Project Developers.
// See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//

// typemap
#include "ownership.hpp"
// shroud
#include "typesownership.h"
#include <cstddef>
#include <cstring>


#ifdef __cplusplus
extern "C" {
#endif

// helper copy_array
// Copy std::vector into array c_var(c_var_size).
// Then release std::vector.
// Called from Fortran.
void OWN_ShroudCopyArray(OWN_SHROUD_array *data, void *c_var, 
    size_t c_var_size)
{
    const void *cxx_var = data->base_addr;
    int n = c_var_size < data->size ? c_var_size : data->size;
    n *= data->elem_len;
    std::memcpy(c_var, cxx_var, n);
}

// Release library allocated memory.
void OWN_SHROUD_memory_destructor(OWN_SHROUD_capsule_data *cap)
{
    void *ptr = cap->addr;
    switch (cap->idtor) {
    case 0:   // --none--
    {
        // Nothing to delete
        break;
    }
    case 1:   // Class1
    {
        Class1 *cxx_ptr = reinterpret_cast<Class1 *>(ptr);
        delete cxx_ptr;
        break;
    }
    case 2:   // int
    {
        int *cxx_ptr = reinterpret_cast<int *>(ptr);
        free(cxx_ptr);
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
