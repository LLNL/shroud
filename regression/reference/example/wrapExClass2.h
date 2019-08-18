// wrapExClass2.h
// This is generated code, do not edit
// Copyright (c) 2017-2019, Lawrence Livermore National Security, LLC.
//
// Produced at the Lawrence Livermore National Laboratory
//
// LLNL-CODE-738041.
//
// All rights reserved.
//
// This file is part of Shroud.
//
// For details about use and distribution, please read LICENSE.
//
// #######################################################################
/**
 * \file wrapExClass2.h
 * \brief Shroud generated wrapper for ExClass2 class
 */
// For C users and C++ implementation

#ifndef WRAPEXCLASS2_H
#define WRAPEXCLASS2_H

#include "sidre/SidreTypes.h"
#include "typesUserLibrary.h"

// splicer begin class.ExClass2.CXX_declarations
// splicer end class.ExClass2.CXX_declarations

#ifdef __cplusplus
extern "C" {
#endif

// splicer begin class.ExClass2.C_declarations
// splicer end class.ExClass2.C_declarations

AA_exclass2 * AA_example_nested_ExClass2_ctor(const char * name,
    AA_exclass2 * SHC_rv);

AA_exclass2 * AA_example_nested_ExClass2_ctor_bufferify(
    const char * name, int trim_name, AA_exclass2 * SHC_rv);

void AA_example_nested_ExClass2_dtor(AA_exclass2 * self);

const char * AA_example_nested_ExClass2_get_name(
    const AA_exclass2 * self);

void AA_example_nested_ExClass2_get_name_bufferify(
    const AA_exclass2 * self, char * SHF_rv, int NSHF_rv);

const char * AA_example_nested_ExClass2_get_name2(AA_exclass2 * self);

void AA_example_nested_ExClass2_get_name2_bufferify(AA_exclass2 * self,
    AA_SHROUD_array *DSHF_rv);

char * AA_example_nested_ExClass2_get_name3(const AA_exclass2 * self);

void AA_example_nested_ExClass2_get_name3_bufferify(
    const AA_exclass2 * self, AA_SHROUD_array *DSHF_rv);

char * AA_example_nested_ExClass2_get_name4(AA_exclass2 * self);

void AA_example_nested_ExClass2_get_name4_bufferify(AA_exclass2 * self,
    AA_SHROUD_array *DSHF_rv);

int AA_example_nested_ExClass2_get_name_length(
    const AA_exclass2 * self);

AA_exclass1 * AA_example_nested_ExClass2_get_class1(AA_exclass2 * self,
    const AA_exclass1 * in, AA_exclass1 * SHC_rv);

void AA_example_nested_ExClass2_declare_0(AA_exclass2 * self, int type);

void AA_example_nested_ExClass2_declare_1(AA_exclass2 * self, int type,
    SIDRE_SidreLength len);

void AA_example_nested_ExClass2_destroyall(AA_exclass2 * self);

int AA_example_nested_ExClass2_get_type_id(const AA_exclass2 * self);

void AA_example_nested_ExClass2_set_value_int(AA_exclass2 * self,
    int value);

void AA_example_nested_ExClass2_set_value_long(AA_exclass2 * self,
    long value);

void AA_example_nested_ExClass2_set_value_float(AA_exclass2 * self,
    float value);

void AA_example_nested_ExClass2_set_value_double(AA_exclass2 * self,
    double value);

int AA_example_nested_ExClass2_get_value_int(AA_exclass2 * self);

double AA_example_nested_ExClass2_get_value_double(AA_exclass2 * self);

#ifdef __cplusplus
}
#endif

#endif  // WRAPEXCLASS2_H
