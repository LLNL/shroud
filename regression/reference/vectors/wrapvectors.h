// wrapvectors.h
// This file is generated by Shroud nowrite-version. Do not edit.
// Copyright (c) 2017-2024, Lawrence Livermore National Security, LLC and
// other Shroud Project Developers.
// See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//
/**
 * \file wrapvectors.h
 * \brief Shroud generated wrapper for vectors library
 */
// For C users and C++ implementation

#ifndef WRAPVECTORS_H
#define WRAPVECTORS_H

// shroud
#include "typesvectors.h"

// splicer begin CXX_declarations
// splicer end CXX_declarations

#ifdef __cplusplus
extern "C" {
#endif

// splicer begin C_declarations
// splicer end C_declarations

int VEC_vector_sum(int *arg, size_t SHT_arg_size);

void VEC_vector_iota_out(int *arg, size_t *SHT_arg_size);

void VEC_vector_iota_out_bufferify(VEC_SHROUD_array *SHT_arg_cdesc);

long VEC_vector_iota_out_with_num(int *arg, size_t *SHT_arg_size);

long VEC_vector_iota_out_with_num_bufferify(
    VEC_SHROUD_array *SHT_arg_cdesc);

void VEC_vector_iota_out_with_num2(int *arg, size_t *SHT_arg_size);

void VEC_vector_iota_out_with_num2_bufferify(
    VEC_SHROUD_array *SHT_arg_cdesc);

void VEC_vector_iota_out_alloc(int **arg, size_t *SHT_arg_size);

void VEC_vector_iota_out_alloc_bufferify(
    VEC_SHROUD_array *SHT_arg_cdesc);

void VEC_vector_iota_inout_alloc(int **arg, size_t *SHT_arg_size);

void VEC_vector_iota_inout_alloc_bufferify(int *arg,
    size_t SHT_arg_size, VEC_SHROUD_array *SHT_arg_cdesc);

void VEC_vector_increment_bufferify(int *arg, size_t SHT_arg_size,
    VEC_SHROUD_array *SHT_arg_cdesc);

void VEC_vector_iota_out_d(double *arg, size_t *SHT_arg_size);

void VEC_vector_iota_out_d_bufferify(VEC_SHROUD_array *SHT_arg_cdesc);

int VEC_vector_of_pointers(double *arg1, size_t SHT_arg1_len,
    size_t SHT_arg1_size, int num);

int VEC_vector_string_count(const char *arg, size_t SHT_arg_size,
    int SHT_arg_len);

void VEC_vector_string_fill_bufferify(VEC_SHROUD_array *SHT_arg_cdesc);

void VEC_vector_string_fill_allocatable_bufferify(
    VEC_SHROUD_array *SHT_arg_cdesc,
    VEC_SHROUD_capsule_data *SHT_arg_capsule);

void VEC_vector_string_fill_allocatable_len_bufferify(
    VEC_SHROUD_array *SHT_arg_cdesc,
    VEC_SHROUD_capsule_data *SHT_arg_capsule);

int * VEC_ReturnVectorAlloc(int n, size_t *SHT_rv_size);

void VEC_ReturnVectorAlloc_bufferify(int n,
    VEC_SHROUD_array *SHT_rv_cdesc);

int VEC_returnDim2(int * arg, int len);

#ifdef __cplusplus
}
#endif

#endif  // WRAPVECTORS_H
