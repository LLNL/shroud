// wrapstruct.h
// This file is generated by Shroud nowrite-version. Do not edit.
// Copyright (c) 2017-2023, Lawrence Livermore National Security, LLC and
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

int STR_passStruct2_bufferify(const Cstruct1 * s1, char *outbuf,
    int SHT_outbuf_len);

void STR_returnStructByValue(int i, double d, Cstruct1 *SHC_rv);

Cstruct1 * STR_returnStructPtr1(int i, double d);

void STR_returnStructPtr1_bufferify(int i, double d,
    STR_SHROUD_array *SHT_rv_cdesc);

Cstruct1 * STR_returnStructPtr2(int i, double d, char * outbuf);

void STR_returnStructPtr2_bufferify(int i, double d, char *outbuf,
    int SHT_outbuf_len, STR_SHROUD_array *SHT_rv_cdesc);

Cstruct1 * STR_returnStructPtrArray(void);

void STR_returnStructPtrArray_bufferify(STR_SHROUD_array *SHT_rv_cdesc);

Cstruct_list * STR_get_global_struct_list(void);

void STR_get_global_struct_list_bufferify(
    STR_SHROUD_array *SHT_rv_cdesc);

STR_Cstruct_as_class * STR_Create_Cstruct_as_class(
    STR_Cstruct_as_class * SHC_rv);

STR_Cstruct_as_class * STR_Create_Cstruct_as_class_args(int x, int y,
    STR_Cstruct_as_class * SHC_rv);

int STR_Cstruct_as_class_sum(STR_Cstruct_as_class * point);

STR_Cstruct_as_subclass * STR_Create_Cstruct_as_subclass_args(int x,
    int y, int z, STR_Cstruct_as_subclass * SHC_rv);

const double * STR_Cstruct_ptr_get_const_dvalue(Cstruct_ptr * SH_this);

void STR_Cstruct_ptr_set_const_dvalue(Cstruct_ptr * SH_this,
    const double * val);

void STR_Cstruct_list_get_ivalue(Cstruct_list * SH_this,
    STR_SHROUD_array *SHT_rv_cdesc);

void STR_Cstruct_list_set_ivalue(Cstruct_list * SH_this, int * val);

void STR_Cstruct_list_get_dvalue(Cstruct_list * SH_this,
    STR_SHROUD_array *SHT_rv_cdesc);

void STR_Cstruct_list_set_dvalue(Cstruct_list * SH_this, double * val);

#endif  // WRAPSTRUCT_H
