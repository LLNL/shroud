// top.h
// This file is generated by Shroud nowrite-version. Do not edit.
// Copyright (c) 2017-2023, Lawrence Livermore National Security, LLC and
// other Shroud Project Developers.
// See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//
/**
 * \file top.h
 * \brief Shroud generated wrapper for testnames library
 */
// For C users and C++ implementation

#ifndef TOP_H
#define TOP_H

// file_code
#ifdef __cplusplus
#include "cxxfoo.hpp"
#else
#include "cfoo.h"
#endif
// shroud
#include "typestestnames.hh"

// splicer begin CXX_declarations
// splicer end CXX_declarations

#ifdef __cplusplus
extern "C" {
#endif

//  Color
enum TES_Color {
    TES_RED,
    TES_BLUE,
    TES_WHITE
};

// splicer begin C_declarations
// splicer end C_declarations

void TES_getName(char * name);

void TES_getName_bufferify(char *name, int SHT_name_len);

void YYY_TES_function1(void);

void c_name_special(void);

void YYY_TES_function3a_0(int i);

void YYY_TES_function3a_1(long i);

int YYY_TES_function4(const char * rv);

int YYY_TES_function4_bufferify(char *rv, int SHT_rv_len);

void YYY_TES_fiveplus(void);

void TES_TestMultilineSplicer(char * name, int * value);

void TES_TestMultilineSplicer_bufferify(char *name, int SHT_name_len,
    int * value);

void c_name_instantiation1(int arg1, long arg2);

void TES_FunctionTU_instantiation2(float arg1, double arg2);

int TES_UseImplWorker_instantiation3(void);

int TES_Cstruct_as_class_sum(TES_Cstruct_as_class * point);

void TES_external_funcs(const char * rdbase, const char * pkg,
    const char * name, void ( * alloc)(double * arr, int * err),
    void ( * afree)(double * arr), void ( * assoc)(double * arr,
    int * err));

void TES_bindtest(int arg1);

#ifdef __cplusplus
}
#endif

#endif  // TOP_H
