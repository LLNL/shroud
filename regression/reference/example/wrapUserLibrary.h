// wrapUserLibrary.h
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
 * \file wrapUserLibrary.h
 * \brief Shroud generated wrapper for UserLibrary library
 */
// For C users and C++ implementation

#ifndef WRAPUSERLIBRARY_H
#define WRAPUSERLIBRARY_H

#include <stddef.h>
#ifdef USE_MPI
#include "mpi.h"
#endif
#include "sidre/wrapGroup.h"
#include "typesUserLibrary.h"

// splicer begin CXX_declarations
// splicer end CXX_declarations

#ifdef __cplusplus
extern "C" {
#endif

// splicer begin C_declarations
// splicer end C_declarations

void AA_local_function1();

bool AA_is_name_valid(const char * name);

bool AA_is_name_valid_bufferify(const char * name, int Lname);

bool AA_is_initialized();

void AA_test_names(const char * name);

void AA_test_names_bufferify(const char * name, int Lname);

void AA_test_names_flag(const char * name, int flag);

void AA_test_names_flag_bufferify(const char * name, int Lname,
    int flag);

void AA_testoptional_0();

void AA_testoptional_1(int i);

void AA_testoptional_2(int i, long j);

size_t AA_test_size_t();

#ifdef HAVE_MPI
void AA_testmpi_mpi(MPI_Fint comm);
#endif

#ifndef HAVE_MPI
void AA_testmpi_serial();
#endif

void AA_testgroup1(SIDRE_group * grp);

void AA_testgroup2(const SIDRE_group * grp);

void AA_func_ptr1(void ( * get)());

void AA_func_ptr2(double * ( * get)());

void AA_func_ptr3(double ( * get)(int i, int));

void AA_func_ptr4(double ( * get)(double, int));

void AA_func_ptr5(void ( * get)(int verylongname1, int verylongname2,
    int verylongname3, int verylongname4, int verylongname5,
    int verylongname6, int verylongname7, int verylongname8,
    int verylongname9, int verylongname10));

void AA_verylongfunctionname1(int * verylongname1, int * verylongname2,
    int * verylongname3, int * verylongname4, int * verylongname5,
    int * verylongname6, int * verylongname7, int * verylongname8,
    int * verylongname9, int * verylongname10);

int AA_verylongfunctionname2(int verylongname1, int verylongname2,
    int verylongname3, int verylongname4, int verylongname5,
    int verylongname6, int verylongname7, int verylongname8,
    int verylongname9, int verylongname10);

void AA_cos_doubles(double * in, double * out, int sizein);

#ifdef __cplusplus
}
#endif

#endif  // WRAPUSERLIBRARY_H
