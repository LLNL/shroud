// wrapstruct.h
// This file is generated by Shroud nowrite-version. Do not edit.
// Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and
// other Shroud Project Developers.
// See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//
/**
 * \file wrapstruct.h
 * \brief Shroud generated wrapper for struct library
 */
// For C users and C implementation

#ifndef WRAPSTRUCT_H
#define WRAPSTRUCT_H

// typemap
#include "struct.h"
// shroud
#include "typesstruct.h"

// splicer begin C_declarations
// splicer end C_declarations

int STR_pass_struct2_bufferify(const Cstruct1 * s1, char *outbuf,
    int SHT_outbuf_len);

Cstruct1 * STR_return_struct_ptr2_bufferify(int i, double d,
    char *outbuf, int SHT_outbuf_len);

STR_Cstruct_as_class * STR_create__cstruct_as_class(
    STR_Cstruct_as_class * SHC_rv);

STR_Cstruct_as_class * STR_create__cstruct_as_class_args(int x, int y,
    STR_Cstruct_as_class * SHC_rv);

int STR_cstruct_as_class_sum(STR_Cstruct_as_class * point);

STR_Cstruct_as_subclass * STR_create__cstruct_as_subclass_args(int x,
    int y, int z, STR_Cstruct_as_subclass * SHC_rv);

const double * STR_cstruct_ptr_get_const_dvalue_bufferify(
    Cstruct_ptr * SH_this);

void STR_cstruct_ptr_set_const_dvalue(Cstruct_ptr * SH_this,
    const double * val);

void STR_cstruct_list_get_ivalue_bufferify(Cstruct_list * SH_this,
    STR_SHROUD_array *SHT_rv_cdesc);

void STR_cstruct_list_set_ivalue(Cstruct_list * SH_this, int * val);

void STR_cstruct_list_get_dvalue_bufferify(Cstruct_list * SH_this,
    STR_SHROUD_array *SHT_rv_cdesc);

void STR_cstruct_list_set_dvalue(Cstruct_list * SH_this, double * val);

#endif  // WRAPSTRUCT_H
