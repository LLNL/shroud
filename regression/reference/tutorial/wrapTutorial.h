// wrapTutorial.h
// This is generated code, do not edit
// #######################################################################
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
 * \file wrapTutorial.h
 * \brief Shroud generated wrapper for Tutorial library
 */
// For C users and C++ implementation

#ifndef WRAPTUTORIAL_H
#define WRAPTUTORIAL_H

#include <stddef.h>
#include "typesTutorial.h"

// splicer begin CXX_declarations
// splicer end CXX_declarations

#ifdef __cplusplus
extern "C" {
#endif

//  Color
enum TUT_Color {
    RED,
    BLUE,
    WHITE
};


struct s_TUT_struct1 {
    int ifield;
    double dfield;
};
typedef struct s_TUT_struct1 TUT_struct1;

// splicer begin C_declarations
// splicer end C_declarations

void TUT_function1();

double TUT_function2(double arg1, int arg2);

void TUT_sum(size_t len, int * values, int * result);

long long TUT_type_long_long(long long arg1);

bool TUT_function3(bool arg);

void TUT_function3b(const bool arg1, bool * arg2, bool * arg3);

void TUT_function4a_bufferify(const char * arg1, int Larg1,
    const char * arg2, int Larg2, char * SHF_rv, int NSHF_rv);

const char * TUT_function4b(const char * arg1, const char * arg2);

void TUT_function4b_bufferify(const char * arg1, int Larg1,
    const char * arg2, int Larg2, char * output, int Noutput);

void TUT_function4c_bufferify(const char * arg1, int Larg1,
    const char * arg2, int Larg2, TUT_SHROUD_array *DSHF_rv);

const char * TUT_function4d();

void TUT_function4d_bufferify(TUT_SHROUD_array *DSHF_rv);

double TUT_function5();

double TUT_function5_arg1(double arg1);

double TUT_function5_arg1_arg2(double arg1, bool arg2);

void TUT_function6_from_name(const char * name);

void TUT_function6_from_name_bufferify(const char * name, int Lname);

void TUT_function6_from_index(int indx);

void TUT_function7_int(int arg);

void TUT_function7_double(double arg);

int TUT_function8_int();

double TUT_function8_double();

void TUT_function9(double arg);

void TUT_function10_0();

void TUT_function10_1(const char * name, double arg2);

void TUT_function10_1_bufferify(const char * name, int Lname,
    double arg2);

int TUT_overload1_num(int num);

int TUT_overload1_num_offset(int num, int offset);

int TUT_overload1_num_offset_stride(int num, int offset, int stride);

int TUT_overload1_3(double type, int num);

int TUT_overload1_4(double type, int num, int offset);

int TUT_overload1_5(double type, int num, int offset, int stride);

int TUT_typefunc(int arg);

int TUT_enumfunc(int arg);

int TUT_colorfunc(int arg);

void TUT_get_min_max(int * min, int * max);

int TUT_direction_func(int arg);

int TUT_useclass(const TUT_class1 * arg);

TUT_class1 * TUT_getclass2(TUT_class1 * SHC_rv);

TUT_class1 * TUT_getclass3(TUT_class1 * SHC_rv);

TUT_class1 * TUT_get_class_copy(int flag, TUT_class1 * SHC_rv);

int TUT_callback1(int in, int ( * incr)(int));

TUT_struct1 TUT_return_struct(int i, double d);

TUT_struct1 * TUT_return_struct_ptr(int i, double d);

double TUT_accept_struct_in(TUT_struct1 arg);

double TUT_accept_struct_in_ptr(TUT_struct1 * arg);

void TUT_accept_struct_out_ptr(TUT_struct1 * arg, int i, double d);

void TUT_accept_struct_in_out_ptr(TUT_struct1 * arg);

const char * TUT_last_function_called();

void TUT_last_function_called_bufferify(char * SHF_rv, int NSHF_rv);

#ifdef __cplusplus
}
#endif

#endif  // WRAPTUTORIAL_H
