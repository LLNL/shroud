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


// helper char_len_trim
// Returns the length of character string src with length nsrc,
// ignoring any trailing blanks.
static int ShroudCharLenTrim(const char *src, int nsrc) {
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
// Statement: c_subroutine
// ----------------------------------------
// Argument:  int * arg +api(cdesc)+intent(in)+rank(2)
// Statement: c_in_native_*_cdesc
void CDE_Rank2In(CDE_SHROUD_array *SHT_arg_cdesc)
{
    // splicer begin function.Rank2In
    int * arg = static_cast<int *>
        (const_cast<void *>(SHT_arg_cdesc->base_addr));
    Rank2In(arg);
    // splicer end function.Rank2In
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
// Statement: c_subroutine
// ----------------------------------------
// Argument:  std::string & name +intent(in)
// Statement: c_in_string_&
// ----------------------------------------
// Argument:  void * value +api(cdesc)+intent(in)+rank(0)+value
// Statement: c_in_void_*_cdesc
void CDE_GetScalar1(char * name, CDE_SHROUD_array *SHT_value_cdesc)
{
    // splicer begin function.GetScalar1
    std::string SHCXX_name(name);
    void * value = static_cast<void *>
        (const_cast<void *>(SHT_value_cdesc->base_addr));
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
// Generated by fortran_generic
// ----------------------------------------
// Function:  void GetScalar1
// Statement: f_subroutine
// ----------------------------------------
// Argument:  std::string & name +intent(in)
// Statement: f_in_string_&_buf
// ----------------------------------------
// Argument:  int * value +api(cdesc)+intent(out)+rank(0)
// Statement: f_out_native_*_cdesc
void CDE_GetScalar1_0_bufferify(char *name, int SHT_name_len,
    CDE_SHROUD_array *SHT_value_cdesc)
{
    // splicer begin function.GetScalar1_0_bufferify
    switch(SHT_value_cdesc->type) {
    case SH_TYPE_INT: {
      *static_cast<int *>(SHT_value_cdesc->base_addr) = getData<int>();
      break;
    }
    case SH_TYPE_LONG: {
      *static_cast<long *>(SHT_value_cdesc->base_addr) = getData<long>();
      break;
    }
    case SH_TYPE_FLOAT: {
      *static_cast<float *>(SHT_value_cdesc->base_addr) = getData<float>();
      break;
    }
    case SH_TYPE_DOUBLE: {
      *static_cast<double *>(SHT_value_cdesc->base_addr) = getData<double>();
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
// Generated by fortran_generic
// ----------------------------------------
// Function:  void GetScalar1
// Statement: f_subroutine
// ----------------------------------------
// Argument:  std::string & name +intent(in)
// Statement: f_in_string_&_buf
// ----------------------------------------
// Argument:  double * value +api(cdesc)+intent(out)+rank(0)
// Statement: f_out_native_*_cdesc
void CDE_GetScalar1_1_bufferify(char *name, int SHT_name_len,
    CDE_SHROUD_array *SHT_value_cdesc)
{
    // splicer begin function.GetScalar1_1_bufferify
    switch(SHT_value_cdesc->type) {
    case SH_TYPE_INT: {
      *static_cast<int *>(SHT_value_cdesc->base_addr) = getData<int>();
      break;
    }
    case SH_TYPE_LONG: {
      *static_cast<long *>(SHT_value_cdesc->base_addr) = getData<long>();
      break;
    }
    case SH_TYPE_FLOAT: {
      *static_cast<float *>(SHT_value_cdesc->base_addr) = getData<float>();
      break;
    }
    case SH_TYPE_DOUBLE: {
      *static_cast<double *>(SHT_value_cdesc->base_addr) = getData<double>();
      break;
    }
    // default:
    }
    // splicer end function.GetScalar1_1_bufferify
}

/**
 * Wrapper for function which is templated on the return value.
 */
// Generated by cxx_template
// ----------------------------------------
// Function:  int getData
// Statement: c_function_native_scalar
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
// Generated by cxx_template
// ----------------------------------------
// Function:  double getData
// Statement: c_function_native_scalar
double CDE_getData_double(void)
{
    // splicer begin function.getData_double
    double SHC_rv = getData<double>();
    return SHC_rv;
    // splicer end function.getData_double
}

/**
 * Call a C++ function which is templated on the return value.
 * Create a Fortran function with the result passed in as an
 * argument.  Change the function call clause to directly call the
 * wrapped templated function.  fstatements is required instead of
 * splicer in order to get {stype} expanded.
 */
// Generated by fortran_generic
// ----------------------------------------
// Function:  void GetScalar2
// Statement: f_subroutine
// ----------------------------------------
// Argument:  std::string & name +intent(in)
// Statement: f_in_string_&_buf
// ----------------------------------------
// Argument:  int * value +intent(out)
// Statement: f_out_native_*
void CDE_GetScalar2_0_bufferify(char *name, int SHT_name_len,
    int * value)
{
    // splicer begin function.GetScalar2_0_bufferify
    // This function does not need to exist.
    // splicer end function.GetScalar2_0_bufferify
}

/**
 * Call a C++ function which is templated on the return value.
 * Create a Fortran function with the result passed in as an
 * argument.  Change the function call clause to directly call the
 * wrapped templated function.  fstatements is required instead of
 * splicer in order to get {stype} expanded.
 */
// Generated by fortran_generic
// ----------------------------------------
// Function:  void GetScalar2
// Statement: f_subroutine
// ----------------------------------------
// Argument:  std::string & name +intent(in)
// Statement: f_in_string_&_buf
// ----------------------------------------
// Argument:  double * value +intent(out)
// Statement: f_out_native_*
void CDE_GetScalar2_1_bufferify(char *name, int SHT_name_len,
    double * value)
{
    // splicer begin function.GetScalar2_1_bufferify
    // This function does not need to exist.
    // splicer end function.GetScalar2_1_bufferify
}

}  // extern "C"
