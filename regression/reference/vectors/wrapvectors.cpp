// wrapvectors.cpp
// This file is generated by Shroud nowrite-version. Do not edit.
// Copyright (c) 2017-2023, Lawrence Livermore National Security, LLC and
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
#include <cstring>
#include "wrapvectors.h"

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
// Function:  int vector_sum
// Statement: c_function_native_scalar
// ----------------------------------------
// Argument:  const std::vector<int> & arg +rank(1)
// Statement: c_in_vector_&_buf_targ_native_scalar
// start VEC_vector_sum
int VEC_vector_sum(int *arg, size_t SHT_arg_size)
{
    // splicer begin function.vector_sum
    const std::vector<int> SHCXX_arg(arg, arg + SHT_arg_size);
    int SHC_rv = vector_sum(SHCXX_arg);
    return SHC_rv;
    // splicer end function.vector_sum
}
// end VEC_vector_sum

/**
 * \brief Copy vector into Fortran input array
 *
 */
// ----------------------------------------
// Function:  void vector_iota_out
// Statement: c_subroutine
// ----------------------------------------
// Argument:  std::vector<int> & arg +intent(out)+rank(1)
// Statement: c_out_vector_&_buf_copy_targ_native_scalar
// start VEC_vector_iota_out
void VEC_vector_iota_out(int *arg, size_t *SHT_arg_size)
{
    // splicer begin function.vector_iota_out
    std::vector<int> SHCXX_arg;
    vector_iota_out(SHCXX_arg);
    size_t SHC_arg_size = *SHT_arg_size < SHCXX_arg.size() ?
        *SHT_arg_size : SHCXX_arg.size();
    std::memcpy(arg, SHCXX_arg.data(),
        SHC_arg_size*sizeof(SHCXX_arg[0]));
    *SHT_arg_size = SHC_arg_size;
    // splicer end function.vector_iota_out
}
// end VEC_vector_iota_out

/**
 * \brief Copy vector into Fortran input array
 *
 */
// ----------------------------------------
// Function:  void vector_iota_out
// Statement: f_subroutine
// ----------------------------------------
// Argument:  std::vector<int> & arg +intent(out)+rank(1)
// Statement: f_out_vector_&_cdesc_targ_native_scalar
// start VEC_vector_iota_out_bufferify
void VEC_vector_iota_out_bufferify(VEC_SHROUD_array *SHT_arg_cdesc)
{
    // splicer begin function.vector_iota_out_bufferify
    std::vector<int> *SHCXX_arg = new std::vector<int>;
    vector_iota_out(*SHCXX_arg);
    SHT_arg_cdesc->base_addr = SHCXX_arg->empty() ? nullptr : &SHCXX_arg->front();
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
// Statement: c_subroutine
// ----------------------------------------
// Argument:  std::vector<int> & arg +intent(out)+rank(1)
// Statement: c_out_vector_&_buf_copy_targ_native_scalar
// start VEC_vector_iota_out_with_num
long VEC_vector_iota_out_with_num(int *arg, size_t *SHT_arg_size)
{
    // splicer begin function.vector_iota_out_with_num
    std::vector<int> SHCXX_arg;
    vector_iota_out_with_num(SHCXX_arg);
    size_t SHC_arg_size = *SHT_arg_size < SHCXX_arg.size() ?
        *SHT_arg_size : SHCXX_arg.size();
    std::memcpy(arg, SHCXX_arg.data(),
        SHC_arg_size*sizeof(SHCXX_arg[0]));
    *SHT_arg_size = SHC_arg_size;
    return SHC_arg_size;
    // splicer end function.vector_iota_out_with_num
}
// end VEC_vector_iota_out_with_num

/**
 * \brief Copy vector into Fortran input array
 *
 * Convert subroutine in to a function and
 * return the number of items copied into argument
 * by setting fstatements for both C and Fortran.
 */
// ----------------------------------------
// Function:  void vector_iota_out_with_num
// Statement: f_subroutine
// ----------------------------------------
// Argument:  std::vector<int> & arg +intent(out)+rank(1)
// Statement: f_out_vector_&_cdesc_targ_native_scalar
// start VEC_vector_iota_out_with_num_bufferify
long VEC_vector_iota_out_with_num_bufferify(
    VEC_SHROUD_array *SHT_arg_cdesc)
{
    // splicer begin function.vector_iota_out_with_num_bufferify
    std::vector<int> *SHCXX_arg = new std::vector<int>;
    vector_iota_out_with_num(*SHCXX_arg);
    SHT_arg_cdesc->base_addr = SHCXX_arg->empty() ? nullptr : &SHCXX_arg->front();
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
// Statement: c_subroutine
// ----------------------------------------
// Argument:  std::vector<int> & arg +intent(out)+rank(1)
// Statement: c_out_vector_&_buf_copy_targ_native_scalar
// start VEC_vector_iota_out_with_num2
void VEC_vector_iota_out_with_num2(int *arg, size_t *SHT_arg_size)
{
    // splicer begin function.vector_iota_out_with_num2
    std::vector<int> SHCXX_arg;
    vector_iota_out_with_num2(SHCXX_arg);
    size_t SHC_arg_size = *SHT_arg_size < SHCXX_arg.size() ?
        *SHT_arg_size : SHCXX_arg.size();
    std::memcpy(arg, SHCXX_arg.data(),
        SHC_arg_size*sizeof(SHCXX_arg[0]));
    *SHT_arg_size = SHC_arg_size;
    // splicer end function.vector_iota_out_with_num2
}
// end VEC_vector_iota_out_with_num2

/**
 * \brief Copy vector into Fortran input array
 *
 * Convert subroutine in to a function.
 * Return the number of items copied into argument
 * by setting fstatements for the Fortran wrapper only.
 */
// ----------------------------------------
// Function:  void vector_iota_out_with_num2
// Statement: f_subroutine
// ----------------------------------------
// Argument:  std::vector<int> & arg +intent(out)+rank(1)
// Statement: f_out_vector_&_cdesc_targ_native_scalar
// start VEC_vector_iota_out_with_num2_bufferify
void VEC_vector_iota_out_with_num2_bufferify(
    VEC_SHROUD_array *SHT_arg_cdesc)
{
    // splicer begin function.vector_iota_out_with_num2_bufferify
    std::vector<int> *SHCXX_arg = new std::vector<int>;
    vector_iota_out_with_num2(*SHCXX_arg);
    SHT_arg_cdesc->base_addr = SHCXX_arg->empty() ? nullptr : &SHCXX_arg->front();
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
// Statement: c_subroutine
// ----------------------------------------
// Argument:  std::vector<int> & arg +deref(allocatable)+intent(out)+rank(1)
// Statement: c_out_vector_&_buf_copy_targ_native_scalar
// start VEC_vector_iota_out_alloc
void VEC_vector_iota_out_alloc(int *arg, size_t *SHT_arg_size)
{
    // splicer begin function.vector_iota_out_alloc
    std::vector<int> SHCXX_arg;
    vector_iota_out_alloc(SHCXX_arg);
    size_t SHC_arg_size = *SHT_arg_size < SHCXX_arg.size() ?
        *SHT_arg_size : SHCXX_arg.size();
    std::memcpy(arg, SHCXX_arg.data(),
        SHC_arg_size*sizeof(SHCXX_arg[0]));
    *SHT_arg_size = SHC_arg_size;
    // splicer end function.vector_iota_out_alloc
}
// end VEC_vector_iota_out_alloc

/**
 * \brief Copy vector into Fortran allocatable array
 *
 */
// ----------------------------------------
// Function:  void vector_iota_out_alloc
// Statement: f_subroutine
// ----------------------------------------
// Argument:  std::vector<int> & arg +deref(allocatable)+intent(out)+rank(1)
// Statement: f_out_vector_&_cdesc_allocatable_targ_native_scalar
// start VEC_vector_iota_out_alloc_bufferify
void VEC_vector_iota_out_alloc_bufferify(
    VEC_SHROUD_array *SHT_arg_cdesc)
{
    // splicer begin function.vector_iota_out_alloc_bufferify
    std::vector<int> *SHCXX_arg = new std::vector<int>;
    vector_iota_out_alloc(*SHCXX_arg);
    SHT_arg_cdesc->base_addr = SHCXX_arg->empty() ? nullptr : &SHCXX_arg->front();
    SHT_arg_cdesc->type = SH_TYPE_INT;
    SHT_arg_cdesc->elem_len = sizeof(int);
    SHT_arg_cdesc->size = SHCXX_arg->size();
    SHT_arg_cdesc->rank = 1;
    SHT_arg_cdesc->shape[0] = SHT_arg_cdesc->size;
    // splicer end function.vector_iota_out_alloc_bufferify
}
// end VEC_vector_iota_out_alloc_bufferify

#if 0
! Not Implemented
/**
 * \brief Copy vector into Fortran allocatable array
 *
 */
// ----------------------------------------
// Function:  void vector_iota_inout_alloc
// Statement: c_subroutine
// ----------------------------------------
// Argument:  std::vector<int> & arg +deref(allocatable)+intent(inout)+rank(1)
// Statement: c_inout_vector_&_buf_copy_targ_native_scalar
// start VEC_vector_iota_inout_alloc
void VEC_vector_iota_inout_alloc(int *arg, size_t *SHT_arg_size)
{
    // splicer begin function.vector_iota_inout_alloc
    std::vector<int> SHCXX_arg(arg, arg + *SHT_arg_size);
    vector_iota_inout_alloc(SHCXX_arg);
    *SHT_arg_size = SHCXX_arg->size()
    // splicer end function.vector_iota_inout_alloc
}
// end VEC_vector_iota_inout_alloc
#endif

/**
 * \brief Copy vector into Fortran allocatable array
 *
 */
// ----------------------------------------
// Function:  void vector_iota_inout_alloc
// Statement: f_subroutine
// ----------------------------------------
// Argument:  std::vector<int> & arg +deref(allocatable)+intent(inout)+rank(1)
// Statement: f_inout_vector_&_cdesc_allocatable_targ_native_scalar
// start VEC_vector_iota_inout_alloc_bufferify
void VEC_vector_iota_inout_alloc_bufferify(int *arg,
    size_t SHT_arg_size, VEC_SHROUD_array *SHT_arg_cdesc)
{
    // splicer begin function.vector_iota_inout_alloc_bufferify
    std::vector<int> *SHCXX_arg = new std::vector<int>(
        arg, arg + SHT_arg_size);
    vector_iota_inout_alloc(*SHCXX_arg);
    SHT_arg_cdesc->base_addr = SHCXX_arg->empty() ? nullptr : &SHCXX_arg->front();
    SHT_arg_cdesc->type = SH_TYPE_INT;
    SHT_arg_cdesc->elem_len = sizeof(int);
    SHT_arg_cdesc->size = SHCXX_arg->size();
    SHT_arg_cdesc->rank = 1;
    SHT_arg_cdesc->shape[0] = SHT_arg_cdesc->size;
    // splicer end function.vector_iota_inout_alloc_bufferify
}
// end VEC_vector_iota_inout_alloc_bufferify

#if 0
! Not Implemented
// ----------------------------------------
// Function:  void vector_increment
// Statement: c_subroutine
// ----------------------------------------
// Argument:  std::vector<int> & arg +rank(1)
// Statement: c_inout_vector_&_buf_copy_targ_native_scalar
void VEC_vector_increment(int *arg, size_t *SHT_arg_size)
{
    // splicer begin function.vector_increment
    std::vector<int> SHCXX_arg(arg, arg + *SHT_arg_size);
    vector_increment(SHCXX_arg);
    *SHT_arg_size = SHCXX_arg->size()
    // splicer end function.vector_increment
}
#endif

// ----------------------------------------
// Function:  void vector_increment
// Statement: f_subroutine
// ----------------------------------------
// Argument:  std::vector<int> & arg +rank(1)
// Statement: f_inout_vector_&_cdesc_targ_native_scalar
void VEC_vector_increment_bufferify(int *arg, size_t SHT_arg_size,
    VEC_SHROUD_array *SHT_arg_cdesc)
{
    // splicer begin function.vector_increment_bufferify
    std::vector<int> *SHCXX_arg = new std::vector<int>(
        arg, arg + SHT_arg_size);
    vector_increment(*SHCXX_arg);
    SHT_arg_cdesc->base_addr = SHCXX_arg->empty() ? nullptr : &SHCXX_arg->front();
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
// Statement: c_subroutine
// ----------------------------------------
// Argument:  std::vector<double> & arg +intent(out)+rank(1)
// Statement: c_out_vector_&_buf_copy_targ_native_scalar
void VEC_vector_iota_out_d(double *arg, size_t *SHT_arg_size)
{
    // splicer begin function.vector_iota_out_d
    std::vector<double> SHCXX_arg;
    vector_iota_out_d(SHCXX_arg);
    size_t SHC_arg_size = *SHT_arg_size < SHCXX_arg.size() ?
        *SHT_arg_size : SHCXX_arg.size();
    std::memcpy(arg, SHCXX_arg.data(),
        SHC_arg_size*sizeof(SHCXX_arg[0]));
    *SHT_arg_size = SHC_arg_size;
    // splicer end function.vector_iota_out_d
}

/**
 * \brief Copy vector into Fortran input array
 *
 */
// ----------------------------------------
// Function:  void vector_iota_out_d
// Statement: f_subroutine
// ----------------------------------------
// Argument:  std::vector<double> & arg +intent(out)+rank(1)
// Statement: f_out_vector_&_cdesc_targ_native_scalar
void VEC_vector_iota_out_d_bufferify(VEC_SHROUD_array *SHT_arg_cdesc)
{
    // splicer begin function.vector_iota_out_d_bufferify
    std::vector<double> *SHCXX_arg = new std::vector<double>;
    vector_iota_out_d(*SHCXX_arg);
    SHT_arg_cdesc->base_addr = SHCXX_arg->empty() ? nullptr : &SHCXX_arg->front();
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
// Statement: c_function_native_scalar
// ----------------------------------------
// Argument:  std::vector<const double * > & arg1 +intent(in)+rank(1)
// Statement: c_in_vector_&_buf_targ_native_*
// ----------------------------------------
// Argument:  int num +value
// Statement: c_in_native_scalar
int VEC_vector_of_pointers(double *arg1, size_t SHT_arg1_len,
    size_t SHT_arg1_size, int num)
{
    // splicer begin function.vector_of_pointers
    std::vector<const double * > SHCXX_arg1;
    for (size_t i=0; i < SHT_arg1_size; ++i) {
        SHCXX_arg1.push_back(arg1 + (SHT_arg1_len*i));
    }
    int SHC_rv = vector_of_pointers(SHCXX_arg1, num);
    return SHC_rv;
    // splicer end function.vector_of_pointers
}

/**
 * \brief count number of underscore in vector of strings
 *
 */
// ----------------------------------------
// Function:  int vector_string_count
// Statement: c_function_native_scalar
// ----------------------------------------
// Argument:  const std::vector<std::string> & arg +rank(1)
// Statement: c_in_vector_&_buf_targ_string_scalar
int VEC_vector_string_count(const char *arg, size_t SHT_arg_size,
    int SHT_arg_len)
{
    // splicer begin function.vector_string_count
    std::vector<std::string> SHCXX_arg;
    {
        const char * SHC_arg_s = arg;
        std::vector<std::string>::size_type
            SHC_arg_i = 0,
            SHC_arg_n = SHT_arg_size;
        for(; SHC_arg_i < SHC_arg_n; SHC_arg_i++) {
            SHCXX_arg.push_back(std::string(SHC_arg_s,
                ShroudCharLenTrim(SHC_arg_s, SHT_arg_len)));
            SHC_arg_s += SHT_arg_len;
        }
    }
    int SHC_rv = vector_string_count(SHCXX_arg);
    return SHC_rv;
    // splicer end function.vector_string_count
}

#if 0
! Not Implemented
/**
 * \brief Fill in arg with some animal names
 *
 * The C++ function returns void. But the C and Fortran wrappers return
 * an int with the number of items added to arg.
 */
// ----------------------------------------
// Function:  void vector_string_fill
// Statement: c_subroutine
// ----------------------------------------
// Argument:  std::vector<std::string> & arg +intent(out)+rank(1)
// Statement: c_out_vector_&_buf_copy_targ_string_scalar
void VEC_vector_string_fill(char * arg)
{
    // splicer begin function.vector_string_fill
    vector_string_fill(*arg);
    // splicer end function.vector_string_fill
}
#endif

/**
 * \brief Fill in arg with some animal names
 *
 * The C++ function returns void. But the C and Fortran wrappers return
 * an int with the number of items added to arg.
 */
// ----------------------------------------
// Function:  void vector_string_fill
// Statement: f_subroutine
// ----------------------------------------
// Argument:  std::vector<std::string> & arg +intent(out)+rank(1)
// Statement: f_out_vector_&_cdesc_targ_string_scalar
void VEC_vector_string_fill_bufferify(VEC_SHROUD_array *SHT_arg_cdesc)
{
    // splicer begin function.vector_string_fill_bufferify
    std::vector<std::string> arg;
    vector_string_fill(arg);
    VEC_ShroudVectorStringOut(SHT_arg_cdesc, arg);
    // splicer end function.vector_string_fill_bufferify
}

#if 0
! Not Implemented
// ----------------------------------------
// Function:  void vector_string_fill_allocatable
// Statement: c_subroutine
// ----------------------------------------
// Argument:  std::vector<std::string> & arg +deref(allocatable)+intent(out)+rank(1)
// Statement: c_out_vector_&_buf_copy_targ_string_scalar
void VEC_vector_string_fill_allocatable(char * arg)
{
    // splicer begin function.vector_string_fill_allocatable
    vector_string_fill_allocatable(*arg);
    // splicer end function.vector_string_fill_allocatable
}
#endif

// ----------------------------------------
// Function:  void vector_string_fill_allocatable
// Statement: f_subroutine
// ----------------------------------------
// Argument:  std::vector<std::string> & arg +deref(allocatable)+intent(out)+rank(1)
// Statement: f_out_vector_&_cdesc_allocatable_targ_string_scalar
void VEC_vector_string_fill_allocatable_bufferify(
    VEC_SHROUD_array *SHT_arg_cdesc,
    VEC_SHROUD_capsule_data *SHT_arg_capsule)
{
    // splicer begin function.vector_string_fill_allocatable_bufferify
    std::vector<std::string> *SHCXX_arg = new std::vector<std::string>;
    vector_string_fill_allocatable(*SHCXX_arg);
    if (0 > 0) {
        SHT_arg_cdesc->elem_len = 0;
    } else {
        SHT_arg_cdesc->elem_len = VEC_ShroudVectorStringOutSize(*SHCXX_arg);
    }
    SHT_arg_cdesc->size      = SHCXX_arg->size();
    SHT_arg_capsule->addr  = SHCXX_arg;
    SHT_arg_capsule->idtor = 0;
    // splicer end function.vector_string_fill_allocatable_bufferify
}

#if 0
! Not Implemented
// ----------------------------------------
// Function:  void vector_string_fill_allocatable_len
// Statement: c_subroutine
// ----------------------------------------
// Argument:  std::vector<std::string> & arg +deref(allocatable)+intent(out)+len(20)+rank(1)
// Statement: c_out_vector_&_buf_copy_targ_string_scalar
void VEC_vector_string_fill_allocatable_len(char * arg)
{
    // splicer begin function.vector_string_fill_allocatable_len
    vector_string_fill_allocatable_len(*arg);
    // splicer end function.vector_string_fill_allocatable_len
}
#endif

// ----------------------------------------
// Function:  void vector_string_fill_allocatable_len
// Statement: f_subroutine
// ----------------------------------------
// Argument:  std::vector<std::string> & arg +deref(allocatable)+intent(out)+len(20)+rank(1)
// Statement: f_out_vector_&_cdesc_allocatable_targ_string_scalar
void VEC_vector_string_fill_allocatable_len_bufferify(
    VEC_SHROUD_array *SHT_arg_cdesc,
    VEC_SHROUD_capsule_data *SHT_arg_capsule)
{
    // splicer begin function.vector_string_fill_allocatable_len_bufferify
    std::vector<std::string> *SHCXX_arg = new std::vector<std::string>;
    vector_string_fill_allocatable_len(*SHCXX_arg);
    if (20 > 0) {
        SHT_arg_cdesc->elem_len = 20;
    } else {
        SHT_arg_cdesc->elem_len = VEC_ShroudVectorStringOutSize(*SHCXX_arg);
    }
    SHT_arg_cdesc->size      = SHCXX_arg->size();
    SHT_arg_capsule->addr  = SHCXX_arg;
    SHT_arg_capsule->idtor = 0;
    // splicer end function.vector_string_fill_allocatable_len_bufferify
}

#if 0
! Not Implemented
/**
 * Implement iota function.
 * Return a vector as an ALLOCATABLE array.
 * Copy results into the new array.
 */
// ----------------------------------------
// Function:  std::vector<int> ReturnVectorAlloc +rank(1)
// Statement: c_function_vector_scalar_targ_native_scalar
// ----------------------------------------
// Argument:  int n +value
// Statement: c_in_native_scalar
int VEC_ReturnVectorAlloc(int n)
{
    // splicer begin function.ReturnVectorAlloc
    int SHC_rv = ReturnVectorAlloc(n);
    return SHC_rv;
    // splicer end function.ReturnVectorAlloc
}
#endif

/**
 * Implement iota function.
 * Return a vector as an ALLOCATABLE array.
 * Copy results into the new array.
 */
// ----------------------------------------
// Function:  std::vector<int> ReturnVectorAlloc +rank(1)
// Statement: f_function_vector_scalar_cdesc_allocatable_targ_native_scalar
// ----------------------------------------
// Argument:  int n +value
// Statement: f_in_native_scalar
void VEC_ReturnVectorAlloc_bufferify(int n,
    VEC_SHROUD_array *SHT_rv_cdesc)
{
    // splicer begin function.ReturnVectorAlloc_bufferify
    std::vector<int> *SHC_rv = new std::vector<int>;
    *SHC_rv = ReturnVectorAlloc(n);
    SHT_rv_cdesc->base_addr = SHC_rv->empty() ? nullptr : &SHC_rv->front();
    SHT_rv_cdesc->type = SH_TYPE_OTHER;
    SHT_rv_cdesc->elem_len = sizeof(int);
    SHT_rv_cdesc->size = SHC_rv->size();
    SHT_rv_cdesc->rank = 1;
    SHT_rv_cdesc->shape[0] = SHT_rv_cdesc->size;
    // splicer end function.ReturnVectorAlloc_bufferify
}

// ----------------------------------------
// Function:  int returnDim2
// Statement: c_function_native_scalar
// ----------------------------------------
// Argument:  int * arg +intent(in)+rank(2)
// Statement: c_in_native_*
// ----------------------------------------
// Argument:  int len +implied(size(arg,2))+value
// Statement: c_in_native_scalar
int VEC_returnDim2(int * arg, int len)
{
    // splicer begin function.returnDim2
    int SHC_rv = returnDim2(arg, len);
    return SHC_rv;
    // splicer end function.returnDim2
}

}  // extern "C"
