// utilns.cpp
// This file is generated by Shroud nowrite-version. Do not edit.
// Copyright (c) 2017-2024, Lawrence Livermore National Security, LLC and
// other Shroud Project Developers.
// See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//

// shroud
#include "typesns.h"
#include <cstddef>
#include <cstring>


#ifdef __cplusplus
extern "C" {
#endif

// helper copy_string
// Copy the char* or std::string in context into c_var.
// Called by Fortran to deal with allocatable character.
void NS_ShroudCopyString(NS_SHROUD_array *data, char *c_var,
    size_t c_var_len) {
    const void *cxx_var = data->base_addr;
    size_t n = c_var_len;
    if (data->elem_len < n) n = data->elem_len;
    std::memcpy(c_var, cxx_var, n);
}


// Release library allocated memory.
void NS_SHROUD_memory_destructor(NS_SHROUD_capsule_data *cap)
{
    cap->addr = nullptr;
    cap->idtor = 0;  // avoid deleting again
}

#ifdef __cplusplus
}
#endif
