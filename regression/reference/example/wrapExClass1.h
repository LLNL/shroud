// wrapExClass1.h
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
 * \file wrapExClass1.h
 * \brief Shroud generated wrapper for ExClass1 class
 */
// For C users and C++ implementation

#ifndef WRAPEXCLASS1_H
#define WRAPEXCLASS1_H

#include "typesUserLibrary.h"

// splicer begin class.ExClass1.CXX_declarations
// splicer end class.ExClass1.CXX_declarations

#ifdef __cplusplus
extern "C" {
#endif

// splicer begin class.ExClass1.C_declarations
// splicer end class.ExClass1.C_declarations

AA_exclass1 * AA_example_nested_exclass1_ctor_0(AA_exclass1 * SHC_rv);

AA_exclass1 * AA_example_nested_exclass1_ctor_1(const char * name,
    AA_exclass1 * SHC_rv);

AA_exclass1 * AA_example_nested_exclass1_ctor_1_bufferify(
    const char * name, int Lname, AA_exclass1 * SHC_rv);

void AA_example_nested_exclass1_dtor(AA_exclass1 * self);

int AA_example_nested_exclass1_increment_count(AA_exclass1 * self,
    int incr);

const char * AA_example_nested_exclass1_get_name_error_pattern(
    const AA_exclass1 * self);

void AA_example_nested_exclass1_get_name_error_pattern_bufferify(
    const AA_exclass1 * self, char * SHF_rv, int NSHF_rv);

int AA_example_nested_exclass1_get_name_length(
    const AA_exclass1 * self);

const char * AA_example_nested_exclass1_get_name_error_check(
    const AA_exclass1 * self);

void AA_example_nested_exclass1_get_name_error_check_bufferify(
    const AA_exclass1 * self, AA_SHROUD_array *DSHF_rv);

const char * AA_example_nested_exclass1_get_name_arg(
    const AA_exclass1 * self);

void AA_example_nested_exclass1_get_name_arg_bufferify(
    const AA_exclass1 * self, char * name, int Nname);

void * AA_example_nested_exclass1_get_root(AA_exclass1 * self);

int AA_example_nested_exclass1_get_value_from_int(AA_exclass1 * self,
    int value);

long AA_example_nested_exclass1_get_value_1(AA_exclass1 * self,
    long value);

void * AA_example_nested_exclass1_get_addr(AA_exclass1 * self);

bool AA_example_nested_exclass1_has_addr(AA_exclass1 * self, bool in);

void AA_example_nested_exclass1_splicer_special(AA_exclass1 * self);

#ifdef __cplusplus
}
#endif

#endif  // WRAPEXCLASS1_H
