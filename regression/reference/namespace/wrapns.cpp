// wrapns.cpp
// This file is generated by Shroud nowrite-version. Do not edit.
// Copyright (c) 2017-2021, Lawrence Livermore National Security, LLC and
// other Shroud Project Developers.
// See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//
#include "wrapns.h"

// cxx_header
#include "namespace.hpp"
// typemap
#include <string>
// shroud
#include "typesns.h"
#include <cstdlib>
#include <cstddef>
#include <cstring>

// splicer begin CXX_definitions
// splicer end CXX_definitions

extern "C" {


// helper ShroudStrToArray
// Save str metadata into array to allow Fortran to access values.
// CHARACTER(len=elem_size) src
static void ShroudStrToArray(NS_SHROUD_array *array, const std::string * src, int idtor)
{
    array->cxx.addr = const_cast<std::string *>(src);
    array->cxx.idtor = idtor;
    if (src->empty()) {
        array->addr.ccharp = NULL;
        array->elem_len = 0;
    } else {
        array->addr.ccharp = src->data();
        array->elem_len = src->length();
    }
    array->size = 1;
    array->rank = 0;  // scalar
}
// splicer begin C_definitions
// splicer end C_definitions

// ----------------------------------------
// Function:  const std::string & LastFunctionCalled +deref(allocatable)
// Requested: c_string_&_result
// Match:     c_string_result
const char * NS_last_function_called(void)
{
    // splicer begin function.last_function_called
    const std::string & SHCXX_rv = LastFunctionCalled();
    const char * SHC_rv = SHCXX_rv.c_str();
    return SHC_rv;
    // splicer end function.last_function_called
}

// ----------------------------------------
// Function:  void LastFunctionCalled
// Requested: c_void_scalar_result_buf
// Match:     c_default
// ----------------------------------------
// Argument:  const std::string & SHF_rv +context(DSHF_rv)+deref(allocatable)+intent(out)
// Requested: c_string_&_result_buf_allocatable
// Match:     c_string_result_buf_allocatable
void NS_last_function_called_bufferify(NS_SHROUD_array *DSHF_rv)
{
    // splicer begin function.last_function_called_bufferify
    const std::string & SHCXX_rv = LastFunctionCalled();
    ShroudStrToArray(DSHF_rv, &SHCXX_rv, 0);
    // splicer end function.last_function_called_bufferify
}

// ----------------------------------------
// Function:  void One
// Requested: c
// Match:     c_default
void NS_one(void)
{
    // splicer begin function.one
    One();
    // splicer end function.one
}

// Release library allocated memory.
void NS_SHROUD_memory_destructor(NS_SHROUD_capsule_data *cap)
{
    cap->addr = nullptr;
    cap->idtor = 0;  // avoid deleting again
}

}  // extern "C"
