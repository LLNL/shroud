// wrapcdesc.cpp
// This file is generated by Shroud nowrite-version. Do not edit.
// Copyright (c) 2017-2023, Lawrence Livermore National Security, LLC and
// other Shroud Project Developers.
// See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//

// cxx_header
#include "cdesc.hpp"
// typemap
#include <string>
// shroud
#include "wrapcdesc.h"

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
// Exact:     c_in_native_*
void CDE_Rank2In(int * arg)
{
    // splicer begin function.Rank2In
    Rank2In(arg);
    // splicer end function.Rank2In
}

// ----------------------------------------
// Function:  void Rank2In
// Attrs:     +intent(subroutine)
// Exact:     c_subroutine
// ----------------------------------------
// Argument:  int * arg +cdesc+intent(in)+rank(2)
// Attrs:     +api(cdesc)+intent(in)
// Exact:     c_in_native_*_cdesc
void CDE_Rank2In_bufferify(CDE_SHROUD_array *SHT_arg_cdesc)
{
    // splicer begin function.Rank2In_bufferify
    int * arg = static_cast<int *>
        (const_cast<void *>(SHT_arg_cdesc->addr.base));
    Rank2In(arg);
    // splicer end function.Rank2In_bufferify
}

/**
 * Create several Fortran generic functions which call a single
 * C wrapper that checks the type of the Fortran argument
 * and calls the correct templated function.
 * Adding the string argument forces a bufferified function
 * to be create.
 * Argument value is intent(in). The pointer does not change, only
 * the pointee.
 * XXX - The function is virtual in the sense that GetScalar1 should
 * not need to exist but there is no way, yet, to avoid wrapping the
 * non-bufferify function.
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
// Argument:  void * value +cdesc+intent(in)+rank(0)+value
// Attrs:     +intent(in)
// Exact:     c_in_void_*
void CDE_GetScalar1(char * name, void * value)
{
    // splicer begin function.GetScalar1
    std::string SHCXX_name(name);
    GetScalar1(SHCXX_name, value);
    // splicer end function.GetScalar1
}

/**
 * Create several Fortran generic functions which call a single
 * C wrapper that checks the type of the Fortran argument
 * and calls the correct templated function.
 * Adding the string argument forces a bufferified function
 * to be create.
 * Argument value is intent(in). The pointer does not change, only
 * the pointee.
 * XXX - The function is virtual in the sense that GetScalar1 should
 * not need to exist but there is no way, yet, to avoid wrapping the
 * non-bufferify function.
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
// Attrs:     +api(cdesc)+intent(out)
// Exact:     c_out_native_*_cdesc
void CDE_GetScalar1_0_bufferify(char *name, int SHT_name_len,
    CDE_SHROUD_array *SHT_value_cdesc)
{
    // splicer begin function.GetScalar1_0_bufferify
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
    // splicer end function.GetScalar1_0_bufferify
}

/**
 * Create several Fortran generic functions which call a single
 * C wrapper that checks the type of the Fortran argument
 * and calls the correct templated function.
 * Adding the string argument forces a bufferified function
 * to be create.
 * Argument value is intent(in). The pointer does not change, only
 * the pointee.
 * XXX - The function is virtual in the sense that GetScalar1 should
 * not need to exist but there is no way, yet, to avoid wrapping the
 * non-bufferify function.
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
// Attrs:     +api(cdesc)+intent(out)
// Exact:     c_out_native_*_cdesc
void CDE_GetScalar1_1_bufferify(char *name, int SHT_name_len,
    CDE_SHROUD_array *SHT_value_cdesc)
{
    // splicer begin function.GetScalar1_1_bufferify
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
    // splicer end function.GetScalar1_1_bufferify
}

/**
 * Wrapper for function which is templated on the return value.
 */
// ----------------------------------------
// Function:  int getData
// Attrs:     +intent(function)
// Exact:     c_function_native_scalar
int CDE_getData_int(void)
{
    // splicer begin function.getData_int
    int SHC_rv = getData<int>();
    return SHC_rv;
    // splicer end function.getData_int
}

/**
 * Wrapper for function which is templated on the return value.
 */
// ----------------------------------------
// Function:  double getData
// Attrs:     +intent(function)
// Exact:     c_function_native_scalar
double CDE_getData_double(void)
{
    // splicer begin function.getData_double
    double SHC_rv = getData<double>();
    return SHC_rv;
    // splicer end function.getData_double
}

}  // extern "C"
