// wrapvectors.cpp
// This file is generated by Shroud nowrite-version. Do not edit.
// Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and
// other Shroud Project Developers.
// See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//

// cxx_header
#include "vectors.hpp"
// typemap
#include <vector>
// shroud
#include "wrapvectors.h"

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
// Function:  int vector_sum
// Attrs:     +intent(function)
// Requested: c_function_native_scalar
// Match:     c_function
// ----------------------------------------
// Argument:  const std::vector<int> & arg +rank(1)
// Attrs:     +api(buf)+intent(in)
// Exact:     c_in_vector_&_buf_targ_native_scalar
// start VEC_vector_sum_bufferify
int VEC_vector_sum_bufferify(int *arg, size_t SHT_arg_size)
{
    // splicer begin function.vector_sum_bufferify
    const std::vector<int> SHCXX_arg(arg, arg + SHT_arg_size);
    int SHC_rv = vector_sum(SHCXX_arg);
    return SHC_rv;
    // splicer end function.vector_sum_bufferify
}
// end VEC_vector_sum_bufferify

/**
 * \brief Copy vector into Fortran input array
 *
 */
// ----------------------------------------
// Function:  void vector_iota_out
// Attrs:     +intent(subroutine)
// Exact:     c_subroutine
// ----------------------------------------
// Argument:  std::vector<int> & arg +intent(out)+rank(1)
// Attrs:     +api(cdesc)+intent(out)
// Requested: c_out_vector_&_cdesc_targ_native_scalar
// Match:     c_out_vector_cdesc_targ_native_scalar
// start VEC_vector_iota_out_bufferify
void VEC_vector_iota_out_bufferify(VEC_SHROUD_array *SHT_arg_cdesc)
{
    // splicer begin function.vector_iota_out_bufferify
    std::vector<int> *SHCXX_arg = new std::vector<int>;
    vector_iota_out(*SHCXX_arg);
    SHT_arg_cdesc->cxx.addr  = SHCXX_arg;
    SHT_arg_cdesc->cxx.idtor = 1;
    SHT_arg_cdesc->addr.base = SHCXX_arg->empty() ? nullptr : &SHCXX_arg->front();
    SHT_arg_cdesc->type = SH_TYPE_INT;
    SHT_arg_cdesc->elem_len = sizeof(int);
    SHT_arg_cdesc->size = SHCXX_arg->size();
    SHT_arg_cdesc->rank = 1;
    SHT_arg_cdesc->shape[0] = SHT_arg_cdesc->size;
    // splicer end function.vector_iota_out_bufferify
}
// end VEC_vector_iota_out_bufferify

/**
 * \brief Copy vector into Fortran input array
 *
 * Convert subroutine in to a function and
 * return the number of items copied into argument
 * by setting fstatements for both C and Fortran.
 */
// ----------------------------------------
// Function:  void vector_iota_out_with_num
// Attrs:     +intent(subroutine)
// Exact:     c_subroutine
// ----------------------------------------
// Argument:  std::vector<int> & arg +intent(out)+rank(1)
// Attrs:     +api(cdesc)+intent(out)
// Requested: c_out_vector_&_cdesc_targ_native_scalar
// Match:     c_out_vector_cdesc_targ_native_scalar
// start VEC_vector_iota_out_with_num_bufferify
long VEC_vector_iota_out_with_num_bufferify(
    VEC_SHROUD_array *SHT_arg_cdesc)
{
    // splicer begin function.vector_iota_out_with_num_bufferify
    std::vector<int> *SHCXX_arg = new std::vector<int>;
    vector_iota_out_with_num(*SHCXX_arg);
    SHT_arg_cdesc->cxx.addr  = SHCXX_arg;
    SHT_arg_cdesc->cxx.idtor = 1;
    SHT_arg_cdesc->addr.base = SHCXX_arg->empty() ? nullptr : &SHCXX_arg->front();
    SHT_arg_cdesc->type = SH_TYPE_INT;
    SHT_arg_cdesc->elem_len = sizeof(int);
    SHT_arg_cdesc->size = SHCXX_arg->size();
    SHT_arg_cdesc->rank = 1;
    SHT_arg_cdesc->shape[0] = SHT_arg_cdesc->size;
    return SHT_arg_cdesc->size;
    // splicer end function.vector_iota_out_with_num_bufferify
}
// end VEC_vector_iota_out_with_num_bufferify

/**
 * \brief Copy vector into Fortran input array
 *
 * Convert subroutine in to a function.
 * Return the number of items copied into argument
 * by setting fstatements for the Fortran wrapper only.
 */
// ----------------------------------------
// Function:  void vector_iota_out_with_num2
// Attrs:     +intent(subroutine)
// Exact:     c_subroutine
// ----------------------------------------
// Argument:  std::vector<int> & arg +intent(out)+rank(1)
// Attrs:     +api(cdesc)+intent(out)
// Requested: c_out_vector_&_cdesc_targ_native_scalar
// Match:     c_out_vector_cdesc_targ_native_scalar
// start VEC_vector_iota_out_with_num2_bufferify
void VEC_vector_iota_out_with_num2_bufferify(
    VEC_SHROUD_array *SHT_arg_cdesc)
{
    // splicer begin function.vector_iota_out_with_num2_bufferify
    std::vector<int> *SHCXX_arg = new std::vector<int>;
    vector_iota_out_with_num2(*SHCXX_arg);
    SHT_arg_cdesc->cxx.addr  = SHCXX_arg;
    SHT_arg_cdesc->cxx.idtor = 1;
    SHT_arg_cdesc->addr.base = SHCXX_arg->empty() ? nullptr : &SHCXX_arg->front();
    SHT_arg_cdesc->type = SH_TYPE_INT;
    SHT_arg_cdesc->elem_len = sizeof(int);
    SHT_arg_cdesc->size = SHCXX_arg->size();
    SHT_arg_cdesc->rank = 1;
    SHT_arg_cdesc->shape[0] = SHT_arg_cdesc->size;
    // splicer end function.vector_iota_out_with_num2_bufferify
}
// end VEC_vector_iota_out_with_num2_bufferify

/**
 * \brief Copy vector into Fortran allocatable array
 *
 */
// ----------------------------------------
// Function:  void vector_iota_out_alloc
// Attrs:     +intent(subroutine)
// Exact:     c_subroutine
// ----------------------------------------
// Argument:  std::vector<int> & arg +deref(allocatable)+intent(out)+rank(1)
// Attrs:     +api(cdesc)+deref(allocatable)+intent(out)
// Requested: c_out_vector_&_cdesc_allocatable_targ_native_scalar
// Match:     c_out_vector_cdesc_targ_native_scalar
// start VEC_vector_iota_out_alloc_bufferify
void VEC_vector_iota_out_alloc_bufferify(
    VEC_SHROUD_array *SHT_arg_cdesc)
{
    // splicer begin function.vector_iota_out_alloc_bufferify
    std::vector<int> *SHCXX_arg = new std::vector<int>;
    vector_iota_out_alloc(*SHCXX_arg);
    SHT_arg_cdesc->cxx.addr  = SHCXX_arg;
    SHT_arg_cdesc->cxx.idtor = 1;
    SHT_arg_cdesc->addr.base = SHCXX_arg->empty() ? nullptr : &SHCXX_arg->front();
    SHT_arg_cdesc->type = SH_TYPE_INT;
    SHT_arg_cdesc->elem_len = sizeof(int);
    SHT_arg_cdesc->size = SHCXX_arg->size();
    SHT_arg_cdesc->rank = 1;
    SHT_arg_cdesc->shape[0] = SHT_arg_cdesc->size;
    // splicer end function.vector_iota_out_alloc_bufferify
}
// end VEC_vector_iota_out_alloc_bufferify

/**
 * \brief Copy vector into Fortran allocatable array
 *
 */
// ----------------------------------------
// Function:  void vector_iota_inout_alloc
// Attrs:     +intent(subroutine)
// Exact:     c_subroutine
// ----------------------------------------
// Argument:  std::vector<int> & arg +deref(allocatable)+intent(inout)+rank(1)
// Attrs:     +api(cdesc)+deref(allocatable)+intent(inout)
// Requested: c_inout_vector_&_cdesc_allocatable_targ_native_scalar
// Match:     c_inout_vector_cdesc_targ_native_scalar
// start VEC_vector_iota_inout_alloc_bufferify
void VEC_vector_iota_inout_alloc_bufferify(int *arg,
    size_t SHT_arg_size, VEC_SHROUD_array *SHT_arg_cdesc)
{
    // splicer begin function.vector_iota_inout_alloc_bufferify
    std::vector<int> *SHCXX_arg = new std::vector<int>(
        arg, arg + SHT_arg_size);
    vector_iota_inout_alloc(*SHCXX_arg);
    SHT_arg_cdesc->cxx.addr  = SHCXX_arg;
    SHT_arg_cdesc->cxx.idtor = 1;
    SHT_arg_cdesc->addr.base = SHCXX_arg->empty() ? nullptr : &SHCXX_arg->front();
    SHT_arg_cdesc->type = SH_TYPE_INT;
    SHT_arg_cdesc->elem_len = sizeof(int);
    SHT_arg_cdesc->size = SHCXX_arg->size();
    SHT_arg_cdesc->rank = 1;
    SHT_arg_cdesc->shape[0] = SHT_arg_cdesc->size;
    // splicer end function.vector_iota_inout_alloc_bufferify
}
// end VEC_vector_iota_inout_alloc_bufferify

// ----------------------------------------
// Function:  void vector_increment
// Attrs:     +intent(subroutine)
// Exact:     c_subroutine
// ----------------------------------------
// Argument:  std::vector<int> & arg +rank(1)
// Attrs:     +api(cdesc)+intent(inout)
// Requested: c_inout_vector_&_cdesc_targ_native_scalar
// Match:     c_inout_vector_cdesc_targ_native_scalar
void VEC_vector_increment_bufferify(int *arg, size_t SHT_arg_size,
    VEC_SHROUD_array *SHT_arg_cdesc)
{
    // splicer begin function.vector_increment_bufferify
    std::vector<int> *SHCXX_arg = new std::vector<int>(
        arg, arg + SHT_arg_size);
    vector_increment(*SHCXX_arg);
    SHT_arg_cdesc->cxx.addr  = SHCXX_arg;
    SHT_arg_cdesc->cxx.idtor = 1;
    SHT_arg_cdesc->addr.base = SHCXX_arg->empty() ? nullptr : &SHCXX_arg->front();
    SHT_arg_cdesc->type = SH_TYPE_INT;
    SHT_arg_cdesc->elem_len = sizeof(int);
    SHT_arg_cdesc->size = SHCXX_arg->size();
    SHT_arg_cdesc->rank = 1;
    SHT_arg_cdesc->shape[0] = SHT_arg_cdesc->size;
    // splicer end function.vector_increment_bufferify
}

/**
 * \brief Copy vector into Fortran input array
 *
 */
// ----------------------------------------
// Function:  void vector_iota_out_d
// Attrs:     +intent(subroutine)
// Exact:     c_subroutine
// ----------------------------------------
// Argument:  std::vector<double> & arg +intent(out)+rank(1)
// Attrs:     +api(cdesc)+intent(out)
// Requested: c_out_vector_&_cdesc_targ_native_scalar
// Match:     c_out_vector_cdesc_targ_native_scalar
void VEC_vector_iota_out_d_bufferify(VEC_SHROUD_array *SHT_arg_cdesc)
{
    // splicer begin function.vector_iota_out_d_bufferify
    std::vector<double> *SHCXX_arg = new std::vector<double>;
    vector_iota_out_d(*SHCXX_arg);
    SHT_arg_cdesc->cxx.addr  = SHCXX_arg;
    SHT_arg_cdesc->cxx.idtor = 2;
    SHT_arg_cdesc->addr.base = SHCXX_arg->empty() ? nullptr : &SHCXX_arg->front();
    SHT_arg_cdesc->type = SH_TYPE_DOUBLE;
    SHT_arg_cdesc->elem_len = sizeof(double);
    SHT_arg_cdesc->size = SHCXX_arg->size();
    SHT_arg_cdesc->rank = 1;
    SHT_arg_cdesc->shape[0] = SHT_arg_cdesc->size;
    // splicer end function.vector_iota_out_d_bufferify
}

/**
 * \brief Fortran 2-d array to vector<const double *>
 *
 */
// ----------------------------------------
// Function:  int vector_of_pointers
// Attrs:     +intent(function)
// Requested: c_function_native_scalar
// Match:     c_function
// ----------------------------------------
// Argument:  std::vector<const double * > & arg1 +intent(in)+rank(1)
// Attrs:     +api(buf)+intent(in)
// Exact:     c_in_vector_&_buf_targ_native_*
// ----------------------------------------
// Argument:  int num +value
// Attrs:     +intent(in)
// Requested: c_in_native_scalar
// Match:     c_default
int VEC_vector_of_pointers_bufferify(double *arg1, size_t SHT_arg1_len,
    size_t SHT_arg1_size, int num)
{
    // splicer begin function.vector_of_pointers_bufferify
    std::vector<const double * > SHCXX_arg1;
    for (size_t i=0; i < SHT_arg1_size; ++i) {
        SHCXX_arg1.push_back(arg1 + (SHT_arg1_len*i));
    }
    int SHC_rv = vector_of_pointers(SHCXX_arg1, num);
    return SHC_rv;
    // splicer end function.vector_of_pointers_bufferify
}

/**
 * \brief count number of underscore in vector of strings
 *
 */
// ----------------------------------------
// Function:  int vector_string_count
// Attrs:     +intent(function)
// Requested: c_function_native_scalar
// Match:     c_function
// ----------------------------------------
// Argument:  const std::vector<std::string> & arg +rank(1)
// Attrs:     +api(buf)+intent(in)
// Exact:     c_in_vector_&_buf_targ_string_scalar
int VEC_vector_string_count_bufferify(const char *arg,
    size_t SHT_arg_size, int SHT_arg_len)
{
    // splicer begin function.vector_string_count_bufferify
    std::vector<std::string> SHCXX_arg;
    {
        const char * SHC_arg_s = arg;
        std::vector<std::string>::size_type
            SHC_arg_i = 0,
            SHC_arg_n = SHT_arg_size;
        for(; SHC_arg_i < SHC_arg_n; SHC_arg_i++) {
            SHCXX_arg.push_back(std::string(SHC_arg_s,
                ShroudLenTrim(SHC_arg_s, SHT_arg_len)));
            SHC_arg_s += SHT_arg_len;
        }
    }
    int SHC_rv = vector_string_count(SHCXX_arg);
    return SHC_rv;
    // splicer end function.vector_string_count_bufferify
}

/**
 * Implement iota function.
 * Return a vector as an ALLOCATABLE array.
 * Copy results into the new array.
 */
// ----------------------------------------
// Function:  std::vector<int> ReturnVectorAlloc +rank(1)
// Attrs:     +api(cdesc)+deref(allocatable)+intent(function)
// Requested: c_function_vector_scalar_cdesc_allocatable_targ_native_scalar
// Match:     c_function_vector_scalar_cdesc_targ_native_scalar
// ----------------------------------------
// Argument:  int n +value
// Attrs:     +intent(in)
// Requested: c_in_native_scalar
// Match:     c_default
void VEC_return_vector_alloc_bufferify(int n,
    VEC_SHROUD_array *SHT_rv_cdesc)
{
    // splicer begin function.return_vector_alloc_bufferify
    std::vector<int> *SHC_rv = new std::vector<int>;
    *SHC_rv = ReturnVectorAlloc(n);
    SHT_rv_cdesc->cxx.addr  = SHC_rv;
    SHT_rv_cdesc->cxx.idtor = 1;
    SHT_rv_cdesc->addr.base = SHC_rv->empty() ? nullptr : &SHC_rv->front();
    SHT_rv_cdesc->type = SH_TYPE_OTHER;
    SHT_rv_cdesc->elem_len = sizeof(int);
    SHT_rv_cdesc->size = SHC_rv->size();
    SHT_rv_cdesc->rank = 1;
    SHT_rv_cdesc->shape[0] = SHT_rv_cdesc->size;
    // splicer end function.return_vector_alloc_bufferify
}

// ----------------------------------------
// Function:  int returnDim2
// Attrs:     +intent(function)
// Requested: c_function_native_scalar
// Match:     c_function
// ----------------------------------------
// Argument:  int * arg +intent(in)+rank(2)
// Attrs:     +intent(in)
// Requested: c_in_native_*
// Match:     c_default
// ----------------------------------------
// Argument:  int len +implied(size(arg,2))+value
// Attrs:     +intent(in)
// Requested: c_in_native_scalar
// Match:     c_default
int VEC_return_dim2(int * arg, int len)
{
    // splicer begin function.return_dim2
    int SHC_rv = returnDim2(arg, len);
    return SHC_rv;
    // splicer end function.return_dim2
}

}  // extern "C"
