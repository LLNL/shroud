// wrapmemdoc.cpp
// This is generated code, do not edit
// Copyright (c) 2017-2019, Lawrence Livermore National Security, LLC and
// other Shroud Project Developers.
// See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//
#include "wrapmemdoc.h"
#include <cstddef>
#include <stdlib.h>
#include <string>
#include "typesmemdoc.h"

// splicer begin CXX_definitions
// splicer end CXX_definitions

extern "C" {


// helper function
// start helper copy_string
// Copy the char* or std::string in context into c_var.
// Called by Fortran to deal with allocatable character.
void STR_ShroudCopyStringAndFree(STR_SHROUD_array *data, char *c_var, size_t c_var_len) {
    const char *cxx_var = data->addr.ccharp;
    size_t n = c_var_len;
    if (data->len < n) n = data->len;
    strncpy(c_var, cxx_var, n);
    STR_SHROUD_memory_destructor(&data->cxx); // delete data->cxx.addr
}
// end helper copy_string

// splicer begin C_definitions
// splicer end C_definitions

// start STR_get_const_string_ptr_alloc
const char * STR_get_const_string_ptr_alloc()
{
// splicer begin function.get_const_string_ptr_alloc
    const std::string * SHCXX_rv = getConstStringPtrAlloc();
    const char * SHC_rv = SHCXX_rv->c_str();
    return SHC_rv;
// splicer end function.get_const_string_ptr_alloc
}
// end STR_get_const_string_ptr_alloc

// start STR_get_const_string_ptr_alloc_bufferify
void STR_get_const_string_ptr_alloc_bufferify(STR_SHROUD_array *DSHF_rv)
{
// splicer begin function.get_const_string_ptr_alloc_bufferify
    const std::string * SHCXX_rv = getConstStringPtrAlloc();
    DSHF_rv->cxx.addr = static_cast<void *>(const_cast<std::string *>
        (SHCXX_rv));
    DSHF_rv->cxx.idtor = 0;
    if (SHCXX_rv->empty()) {
        DSHF_rv->addr.ccharp = NULL;
        DSHF_rv->len = 0;
    } else {
        DSHF_rv->addr.ccharp = SHCXX_rv->data();
        DSHF_rv->len = SHCXX_rv->size();
    }
    DSHF_rv->size = 1;
    return;
// splicer end function.get_const_string_ptr_alloc_bufferify
}
// end STR_get_const_string_ptr_alloc_bufferify

// start release allocated memory
// Release library allocated memory.
void STR_SHROUD_memory_destructor(STR_SHROUD_capsule_data *cap)
{
    cap->addr = NULL;
    cap->idtor = 0;  // avoid deleting again
}
// end release allocated memory

}  // extern "C"
