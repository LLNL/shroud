// wrapTutorial.h
// This is generated code, do not edit
// Copyright (c) 2017-2019, Lawrence Livermore National Security, LLC and
// other Shroud Project Developers.
// See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//
/**
 * \file wrapTutorial.h
 * \brief Shroud generated wrapper for Tutorial library
 */
// For C users and C++ implementation

#ifndef WRAPTUTORIAL_H
#define WRAPTUTORIAL_H

#include "typesTutorial.h"

// splicer begin CXX_declarations
// splicer end CXX_declarations

#ifdef __cplusplus
extern "C" {
#endif

//  tutorial::Color
enum TUT_tutorial_Color {
    TUT_tutorial_Color_RED,
    TUT_tutorial_Color_BLUE,
    TUT_tutorial_Color_WHITE
};

// splicer begin C_declarations
// splicer end C_declarations

void TUT_no_return_no_arguments();

double TUT_pass_by_value(double arg1, int arg2);

void TUT_concatenate_strings_bufferify(const char * arg1, int Larg1,
    const char * arg2, int Larg2, TUT_SHROUD_array *DSHF_rv);

double TUT_use_default_arguments();

double TUT_use_default_arguments_arg1(double arg1);

double TUT_use_default_arguments_arg1_arg2(double arg1, bool arg2);

void TUT_overloaded_function_from_name(const char * name);

void TUT_overloaded_function_from_name_bufferify(const char * name,
    int Lname);

void TUT_overloaded_function_from_index(int indx);

void TUT_template_argument_int(int arg);

void TUT_template_argument_double(double arg);

int TUT_template_return_int();

double TUT_template_return_double();

void TUT_fortran_generic(double arg);

void TUT_fortran_generic_overloaded_0();

void TUT_fortran_generic_overloaded_1(const char * name, double arg2);

void TUT_fortran_generic_overloaded_1_bufferify(const char * name,
    int Lname, double arg2);

int TUT_use_default_overload_num(int num);

int TUT_use_default_overload_num_offset(int num, int offset);

int TUT_use_default_overload_num_offset_stride(int num, int offset,
    int stride);

int TUT_use_default_overload_3(double type, int num);

int TUT_use_default_overload_4(double type, int num, int offset);

int TUT_use_default_overload_5(double type, int num, int offset,
    int stride);

int TUT_typefunc(int arg);

int TUT_enumfunc(int arg);

int TUT_colorfunc(int arg);

void TUT_get_min_max(int * min, int * max);

int TUT_direction_func(int arg);

void TUT_pass_class_by_value(TUT_class1 arg);

int TUT_useclass(const TUT_class1 * arg);

TUT_class1 * TUT_getclass2(TUT_class1 * SHC_rv);

TUT_class1 * TUT_getclass3(TUT_class1 * SHC_rv);

TUT_class1 * TUT_get_class_copy(int flag, TUT_class1 * SHC_rv);

int TUT_callback1(int in, int ( * incr)(int));

void TUT_set_global_flag(int arg);

int TUT_get_global_flag();

const char * TUT_last_function_called();

void TUT_last_function_called_bufferify(char * SHF_rv, int NSHF_rv);

#ifdef __cplusplus
}
#endif

#endif  // WRAPTUTORIAL_H
