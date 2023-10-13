// wrapexample_nested_ExClass2.h
// This file is generated by Shroud nowrite-version. Do not edit.
// Copyright (c) 2017-2023, Lawrence Livermore National Security, LLC and
// other Shroud Project Developers.
// See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//
/**
 * \file wrapexample_nested_ExClass2.h
 * \brief Shroud generated wrapper for ExClass2 class
 */
// For C users and C++ implementation

#ifndef WRAPEXAMPLE_NESTED_EXCLASS2_H
#define WRAPEXAMPLE_NESTED_EXCLASS2_H

// typemap
#include "wrapUserLibrary.h"
#ifdef __cplusplus
#include "sidre/SidreWrapperHelpers.hpp"
#else
#include "sidre/SidreTypes.h"
#endif
// shroud
#include "typesUserLibrary.h"

// splicer begin namespace.example::nested.class.ExClass2.CXX_declarations
// splicer end namespace.example::nested.class.ExClass2.CXX_declarations

#ifdef __cplusplus
extern "C" {
#endif

// splicer begin namespace.example::nested.class.ExClass2.C_declarations
// splicer end namespace.example::nested.class.ExClass2.C_declarations

AA_example_nested_ExClass2 * AA_example_nested_ExClass2_ctor(
    const char * name, AA_example_nested_ExClass2 * SHC_rv);

AA_example_nested_ExClass2 * AA_example_nested_ExClass2_ctor_bufferify(
    char *name, int SHT_name_len, AA_example_nested_ExClass2 * SHC_rv);

void AA_example_nested_ExClass2_dtor(AA_example_nested_ExClass2 * self);

const char * AA_example_nested_ExClass2_getName(
    const AA_example_nested_ExClass2 * self);

void AA_example_nested_ExClass2_getName_bufferify(
    const AA_example_nested_ExClass2 * self, char *SHC_rv,
    int SHT_rv_len);

const char * AA_example_nested_ExClass2_getName2(
    AA_example_nested_ExClass2 * self);

void AA_example_nested_ExClass2_getName2_bufferify(
    AA_example_nested_ExClass2 * self, AA_SHROUD_array *SHT_rv_cdesc,
    AA_SHROUD_capsule_data *SHT_rv_capsule);

char * AA_example_nested_ExClass2_getName3(
    const AA_example_nested_ExClass2 * self);

void AA_example_nested_ExClass2_getName3_bufferify(
    const AA_example_nested_ExClass2 * self,
    AA_SHROUD_array *SHT_rv_cdesc,
    AA_SHROUD_capsule_data *SHT_rv_capsule);

char * AA_example_nested_ExClass2_getName4(
    AA_example_nested_ExClass2 * self);

void AA_example_nested_ExClass2_getName4_bufferify(
    AA_example_nested_ExClass2 * self, AA_SHROUD_array *SHT_rv_cdesc,
    AA_SHROUD_capsule_data *SHT_rv_capsule);

int AA_example_nested_ExClass2_GetNameLength(
    const AA_example_nested_ExClass2 * self);

AA_example_nested_ExClass1 * AA_example_nested_ExClass2_get_class1(
    AA_example_nested_ExClass2 * self, AA_example_nested_ExClass1 * in,
    AA_example_nested_ExClass1 * SHC_rv);

void AA_example_nested_ExClass2_declare_0(
    AA_example_nested_ExClass2 * self, AA_TypeID type);

void AA_example_nested_ExClass2_declare_1(
    AA_example_nested_ExClass2 * self, AA_TypeID type,
    SIDRE_SidreLength len);

void AA_example_nested_ExClass2_destroyall(
    AA_example_nested_ExClass2 * self);

AA_TypeID AA_example_nested_ExClass2_getTypeID(
    const AA_example_nested_ExClass2 * self);

void AA_example_nested_ExClass2_setValue_int(
    AA_example_nested_ExClass2 * self, int value);

void AA_example_nested_ExClass2_setValue_long(
    AA_example_nested_ExClass2 * self, long value);

void AA_example_nested_ExClass2_setValue_float(
    AA_example_nested_ExClass2 * self, float value);

void AA_example_nested_ExClass2_setValue_double(
    AA_example_nested_ExClass2 * self, double value);

int AA_example_nested_ExClass2_getValue_int(
    AA_example_nested_ExClass2 * self);

double AA_example_nested_ExClass2_getValue_double(
    AA_example_nested_ExClass2 * self);

#ifdef __cplusplus
}
#endif

#endif  // WRAPEXAMPLE_NESTED_EXCLASS2_H
