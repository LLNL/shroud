// wrapstruct.h
// This file is generated by Shroud nowrite-version. Do not edit.
// Copyright (c) 2017-2021, Lawrence Livermore National Security, LLC and
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

void STR_create__cstruct_as_class(STR_Cstruct_as_class * SHC_rv);

void STR_create__cstruct_as_class_args(STR_Cstruct_as_class * SHC_rv,
    int x, int y);

int STR_cstruct_as_class_sum(STR_Cstruct_as_class * point);

void STR_create__cstruct_as_subclass_args(
    STR_Cstruct_as_subclass * SHC_rv, int x, int y, int z);

#endif  // WRAPSTRUCT_H
