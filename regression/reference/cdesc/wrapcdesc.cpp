// wrapcdesc.cpp
// This file is generated by Shroud nowrite-version. Do not edit.
// Copyright (c) 2017-2021, Lawrence Livermore National Security, LLC and
// other Shroud Project Developers.
// See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//
#include "wrapcdesc.h"

// cxx_header
#include "cdesc.hpp"
// typemap
#include <string>

// splicer begin CXX_definitions
// splicer end CXX_definitions

extern "C" {


// helper ShroudLenTrim
// Returns the length of character string src with length nsrc,
// ignoring any trailing blanks.
static int ShroudLenTrim(const char *src, int nsrc) {
    int i;

    for (i = nsrc - 1; i >= 0; i--) {
        if (src[i] != ' ') {
            break;
        }
    }

    return i + 1;
}

// splicer begin C_definitions
// splicer end C_definitions

// ----------------------------------------
// Function:  void Rank2In
// Attrs:     +intent(subroutine)
// Exact:     c_subroutine
// ----------------------------------------
// Argument:  int * arg +cdesc+intent(in)+rank(2)
// Attrs:     +intent(in)
// Exact:     c_in_native_*_cdesc
void CDE_rank2_in(CDE_SHROUD_array *SHT_arg_cdesc)
{
    // splicer begin function.rank2_in
    int * arg = static_cast<int *>
        (const_cast<void *>(SHT_arg_cdesc->addr.base));
    Rank2In(arg);
    // splicer end function.rank2_in
}

/**
 * Create several Fortran generic functions which call a single
 * C wrapper that checkes the type of the Fortran argument
 * and calls the correct templated function.
 * Adding the string argument forces a bufferified function
 * to be create.
 * XXX The non-bufferified version should not be created since
 * users will not manually create a context struct.
 */
// ----------------------------------------
// Function:  void GetScalar1
// Attrs:     +intent(subroutine)
// Exact:     c_subroutine
// ----------------------------------------
// Argument:  std::string & name +intent(in)
// Attrs:     +intent(in)
// Exact:     c_in_string_&
// ----------------------------------------
// Argument:  void * value +cdesc+intent(out)+rank(0)+value
// Attrs:     +intent(out)
// Exact:     c_out_void_*_cdesc
void CDE_get_scalar1(char * name, CDE_SHROUD_array *SHT_value_cdesc)
{
    // splicer begin function.get_scalar1
    switch(SHT_value_cdesc->type) {
    case SH_TYPE_INT: {
      *static_cast<int *>(const_cast<void *>(SHT_value_cdesc->addr.base)) = getData<int>();
      break;
    }
    case SH_TYPE_LONG: {
      *static_cast<long *>(const_cast<void *>(SHT_value_cdesc->addr.base)) = getData<long>();
      break;
    }
    case SH_TYPE_FLOAT: {
      *static_cast<float *>(const_cast<void *>(SHT_value_cdesc->addr.base)) = getData<float>();
      break;
    }
    case SH_TYPE_DOUBLE: {
      *static_cast<double *>(const_cast<void *>(SHT_value_cdesc->addr.base)) = getData<double>();
      break;
    }
    // default:
    }
    // splicer end function.get_scalar1
}

/**
 * Create several Fortran generic functions which call a single
 * C wrapper that checkes the type of the Fortran argument
 * and calls the correct templated function.
 * Adding the string argument forces a bufferified function
 * to be create.
 * XXX The non-bufferified version should not be created since
 * users will not manually create a context struct.
 */
// ----------------------------------------
// Function:  void GetScalar1
// Attrs:     +intent(subroutine)
// Exact:     c_subroutine
// ----------------------------------------
// Argument:  std::string & name +intent(in)
// Attrs:     +api(buf)+intent(in)
// Exact:     c_in_string_&_buf
// ----------------------------------------
// Argument:  int * value +cdesc+intent(out)+rank(0)
// Attrs:     +intent(out)
// Exact:     c_out_native_*_cdesc
void CDE_get_scalar1_0_bufferify(char *name, int SHT_name_len,
    CDE_SHROUD_array *SHT_value_cdesc)
{
    // splicer begin function.get_scalar1_0_bufferify
    switch(SHT_value_cdesc->type) {
    case SH_TYPE_INT: {
      *static_cast<int *>(const_cast<void *>(SHT_value_cdesc->addr.base)) = getData<int>();
      break;
    }
    case SH_TYPE_LONG: {
      *static_cast<long *>(const_cast<void *>(SHT_value_cdesc->addr.base)) = getData<long>();
      break;
    }
    case SH_TYPE_FLOAT: {
      *static_cast<float *>(const_cast<void *>(SHT_value_cdesc->addr.base)) = getData<float>();
      break;
    }
    case SH_TYPE_DOUBLE: {
      *static_cast<double *>(const_cast<void *>(SHT_value_cdesc->addr.base)) = getData<double>();
      break;
    }
    // default:
    }
    // splicer end function.get_scalar1_0_bufferify
}

/**
 * Create several Fortran generic functions which call a single
 * C wrapper that checkes the type of the Fortran argument
 * and calls the correct templated function.
 * Adding the string argument forces a bufferified function
 * to be create.
 * XXX The non-bufferified version should not be created since
 * users will not manually create a context struct.
 */
// ----------------------------------------
// Function:  void GetScalar1
// Attrs:     +intent(subroutine)
// Exact:     c_subroutine
// ----------------------------------------
// Argument:  std::string & name +intent(in)
// Attrs:     +api(buf)+intent(in)
// Exact:     c_in_string_&_buf
// ----------------------------------------
// Argument:  double * value +cdesc+intent(out)+rank(0)
// Attrs:     +intent(out)
// Exact:     c_out_native_*_cdesc
void CDE_get_scalar1_1_bufferify(char *name, int SHT_name_len,
    CDE_SHROUD_array *SHT_value_cdesc)
{
    // splicer begin function.get_scalar1_1_bufferify
    switch(SHT_value_cdesc->type) {
    case SH_TYPE_INT: {
      *static_cast<int *>(const_cast<void *>(SHT_value_cdesc->addr.base)) = getData<int>();
      break;
    }
    case SH_TYPE_LONG: {
      *static_cast<long *>(const_cast<void *>(SHT_value_cdesc->addr.base)) = getData<long>();
      break;
    }
    case SH_TYPE_FLOAT: {
      *static_cast<float *>(const_cast<void *>(SHT_value_cdesc->addr.base)) = getData<float>();
      break;
    }
    case SH_TYPE_DOUBLE: {
      *static_cast<double *>(const_cast<void *>(SHT_value_cdesc->addr.base)) = getData<double>();
      break;
    }
    // default:
    }
    // splicer end function.get_scalar1_1_bufferify
}

/**
 * Wrapper for function which is templated on the return value.
 */
// ----------------------------------------
// Function:  int getData
// Attrs:     +intent(function)
// Requested: c_function_native_scalar
// Match:     c_function
int CDE_get_data_int(void)
{
    // splicer begin function.get_data_int
    int SHC_rv = getData<int>();
    return SHC_rv;
    // splicer end function.get_data_int
}

/**
 * Wrapper for function which is templated on the return value.
 */
// ----------------------------------------
// Function:  double getData
// Attrs:     +intent(function)
// Requested: c_function_native_scalar
// Match:     c_function
double CDE_get_data_double(void)
{
    // splicer begin function.get_data_double
    double SHC_rv = getData<double>();
    return SHC_rv;
    // splicer end function.get_data_double
}

}  // extern "C"
