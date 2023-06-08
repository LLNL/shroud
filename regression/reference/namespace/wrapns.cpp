// wrapns.cpp
// This file is generated by Shroud nowrite-version. Do not edit.
// Copyright (c) 2017-2023, Lawrence Livermore National Security, LLC and
// other Shroud Project Developers.
// See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//

// cxx_header
#include "namespace.hpp"
// typemap
#include <string>
// shroud
#include <cstddef>
#include <cstring>
#include "wrapns.h"

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
// Function:  const std::string & LastFunctionCalled
// Attrs:     +deref(allocatable)+intent(function)
// Requested: c_function_string_&_allocatable
// Match:     c_function_string_&
const char * NS_LastFunctionCalled(void)
{
    // splicer begin function.LastFunctionCalled
    const std::string & SHCXX_rv = LastFunctionCalled();
    const char * SHC_rv = SHCXX_rv.c_str();
    return SHC_rv;
    // splicer end function.LastFunctionCalled
}

// ----------------------------------------
// Function:  const std::string & LastFunctionCalled
// Attrs:     +api(cdesc)+deref(allocatable)+intent(function)
// Exact:     c_function_string_&_cdesc_allocatable
void NS_LastFunctionCalled_bufferify(NS_SHROUD_array *SHT_rv_cdesc)
{
    // splicer begin function.LastFunctionCalled_bufferify
    const std::string & SHCXX_rv = LastFunctionCalled();
    ShroudStrToArray(SHT_rv_cdesc, &SHCXX_rv, 0);
    // splicer end function.LastFunctionCalled_bufferify
}

// ----------------------------------------
// Function:  void One
// Attrs:     +intent(subroutine)
// Exact:     c_subroutine
void NS_One(void)
{
    // splicer begin function.One
    One();
    // splicer end function.One
}

}  // extern "C"
