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


// helper string_to_cdesc
// Save std::string metadata into array to allow Fortran to access values.
// CHARACTER(len=elem_size) src
static void ShroudStringToCdesc(NS_SHROUD_array *cdesc,
    const std::string * src)
{
    if (src->empty()) {
        cdesc->addr.ccharp = NULL;
        cdesc->elem_len = 0;
    } else {
        cdesc->addr.ccharp = src->data();
        cdesc->elem_len = src->length();
    }
    cdesc->size = 1;
    cdesc->rank = 0;  // scalar
}
// splicer begin C_definitions
// splicer end C_definitions

// ----------------------------------------
// Function:  const std::string & LastFunctionCalled
// Attrs:     +deref(allocatable)+intent(function)
// Statement: f_function_string_&_allocatable
const char * NS_LastFunctionCalled(void)
{
    // splicer begin function.LastFunctionCalled
    const std::string & SHCXX_rv = LastFunctionCalled();
    const char * SHC_rv = SHCXX_rv.c_str();
    return SHC_rv;
    // splicer end function.LastFunctionCalled
}

// Generated by arg_to_buffer
// ----------------------------------------
// Function:  const std::string & LastFunctionCalled
// Attrs:     +api(cdesc)+deref(allocatable)+intent(function)
// Statement: f_function_string_&_cdesc_allocatable
void NS_LastFunctionCalled_bufferify(NS_SHROUD_array *SHT_rv_cdesc,
    NS_SHROUD_capsule_data *SHT_rv_capsule)
{
    // splicer begin function.LastFunctionCalled_bufferify
    const std::string & SHCXX_rv = LastFunctionCalled();
    ShroudStringToCdesc(SHT_rv_cdesc, &SHCXX_rv);
    SHT_rv_capsule->addr  = const_cast<std::string *>(&SHCXX_rv);
    SHT_rv_capsule->idtor = 0;
    // splicer end function.LastFunctionCalled_bufferify
}

// ----------------------------------------
// Function:  void One
// Attrs:     +intent(subroutine)
// Statement: f_subroutine
void NS_One(void)
{
    // splicer begin function.One
    One();
    // splicer end function.One
}

}  // extern "C"
