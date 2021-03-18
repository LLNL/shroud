// wrapgeneric.h
// This file is generated by Shroud nowrite-version. Do not edit.
// Copyright (c) 2017-2021, Lawrence Livermore National Security, LLC and
// other Shroud Project Developers.
// See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//
/**
 * \file wrapgeneric.h
 * \brief Shroud generated wrapper for generic library
 */
// For C users and C implementation

#ifndef WRAPGENERIC_H
#define WRAPGENERIC_H

// typemap
#include <stddef.h>
// shroud
#include "ISO_Fortran_binding.h"
#include "typesgeneric.h"

// splicer begin C_declarations
// splicer end C_declarations

int GEN_sum_values_CFI(CFI_cdesc_t *values, int nvalues);

void GEN_assign_values_scalar(const int * from, int nfrom, int * to,
    int nto);

void GEN_assign_values_broadcast(const int * from, int nfrom, int * to,
    int nto);

void GEN_assign_values_copy(const int * from, int nfrom, int * to,
    int nto);

#if 1
void GEN_save_pointer_float1d(float * addr, int type, size_t size);
#endif

#if 1
void GEN_save_pointer_float2d(float * addr, int type, size_t size);
#endif

void GEN_save_pointer2(void * addr, int type, size_t size);

void GEN_save_pointer2_float1d(float * addr, int type, size_t size);

void GEN_save_pointer2_float2d(float * addr, int type, size_t size);

#if 0
void GEN_get_pointer_as_pointer_float1d_bufferify(
    GEN_SHROUD_array *addr_temp0, int * type, size_t * size);
#endif

#if 0
void GEN_get_pointer_as_pointer_float2d_bufferify(
    GEN_SHROUD_array *addr_temp0, int * type, size_t * size);
#endif

#endif  // WRAPGENERIC_H
