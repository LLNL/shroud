// wrapstruct.h
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
 * \file wrapstruct.h
 * \brief Shroud generated wrapper for struct library
 */
// For C users and C++ implementation

#ifndef WRAPSTRUCT_H
#define WRAPSTRUCT_H

#include "typesstruct.h"

// splicer begin CXX_declarations
// splicer end CXX_declarations

#ifdef __cplusplus
extern "C" {
#endif


struct s_STR_cstruct1 {
    int ifield;
    double dfield;
};
typedef struct s_STR_cstruct1 STR_cstruct1;

// splicer begin C_declarations
// splicer end C_declarations

int STR_pass_struct_by_value(STR_cstruct1 arg);

int STR_pass_struct1(STR_cstruct1 * arg);

int STR_pass_struct2(STR_cstruct1 * s1, char * outbuf);

int STR_pass_struct2_bufferify(STR_cstruct1 * s1, char * outbuf,
    int Noutbuf);

int STR_accept_struct_in_ptr(STR_cstruct1 * arg);

void STR_accept_struct_out_ptr(STR_cstruct1 * arg, int i, double d);

void STR_accept_struct_in_out_ptr(STR_cstruct1 * arg);

STR_cstruct1 STR_return_struct(int i, double d);

STR_cstruct1 * STR_return_struct_ptr1(int i, double d);

STR_cstruct1 * STR_return_struct_ptr2(int i, double d, char * outbuf);

STR_cstruct1 * STR_return_struct_ptr2_bufferify(int i, double d,
    char * outbuf, int Noutbuf);

#ifdef __cplusplus
}
#endif

#endif  // WRAPSTRUCT_H
