// wrapns.cpp
// This is generated code, do not edit
// Copyright (c) 2017-2019, Lawrence Livermore National Security, LLC and
// other Shroud Project Developers.
// See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//
#include "wrapns.h"
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <string>
#include "namespace.hpp"
#include "typesns.h"

// splicer begin CXX_definitions
// splicer end CXX_definitions

extern "C" {


// helper function
// Save str metadata into array to allow Fortran to access values.
static void ShroudStrToArray(NS_SHROUD_array *array, const std::string * src, int idtor)
{
    array->cxx.addr = static_cast<void *>(const_cast<std::string *>(src));
    array->cxx.idtor = idtor;
    if (src->empty()) {
        array->addr.ccharp = NULL;
        array->len = 0;
    } else {
        array->addr.ccharp = src->data();
        array->len = src->size();
    }
    array->size = 1;
}

// helper function
// Copy the char* or std::string in context into c_var.
// Called by Fortran to deal with allocatable character.
void NS_ShroudCopyStringAndFree(NS_SHROUD_array *data, char *c_var, size_t c_var_len) {
    const char *cxx_var = data->addr.ccharp;
    size_t n = c_var_len;
    if (data->len < n) n = data->len;
    std::strncpy(c_var, cxx_var, n);
    NS_SHROUD_memory_destructor(&data->cxx); // delete data->cxx.addr
}

// splicer begin C_definitions
// splicer end C_definitions

// const std::string & LastFunctionCalled() +deref(allocatable)
const char * NS_last_function_called()
{
// splicer begin function.last_function_called
    const std::string & SHCXX_rv = LastFunctionCalled();
    const char * SHC_rv = SHCXX_rv.c_str();
    return SHC_rv;
// splicer end function.last_function_called
}

// void LastFunctionCalled(const std::string & SHF_rv +context(DSHF_rv)+deref(allocatable)+intent(out))
void NS_last_function_called_bufferify(NS_SHROUD_array *DSHF_rv)
{
// splicer begin function.last_function_called_bufferify
    const std::string & SHCXX_rv = LastFunctionCalled();
    ShroudStrToArray(DSHF_rv, &SHCXX_rv, 0);
    return;
// splicer end function.last_function_called_bufferify
}

// void One()
void NS_one()
{
// splicer begin function.one
    One();
    return;
// splicer end function.one
}

// Release library allocated memory.
void NS_SHROUD_memory_destructor(NS_SHROUD_capsule_data *cap)
{
    cap->addr = NULL;
    cap->idtor = 0;  // avoid deleting again
}

}  // extern "C"
