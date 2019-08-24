// wrapUserLibrary_example_nested.h
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
 * \file wrapUserLibrary_example_nested.h
 * \brief Shroud generated wrapper for nested namespace
 */
// For C users and C++ implementation

#ifndef WRAPUSERLIBRARY_EXAMPLE_NESTED_H
#define WRAPUSERLIBRARY_EXAMPLE_NESTED_H

#include <stddef.h>
#ifdef USE_MPI
#include "mpi.h"
#endif
#include "sidre/wrapGroup.h"
#include "typesUserLibrary.h"

// splicer begin namespace.example::nested.CXX_declarations
// splicer end namespace.example::nested.CXX_declarations

#ifdef __cplusplus
extern "C" {
#endif

// splicer begin namespace.example::nested.C_declarations
// splicer end namespace.example::nested.C_declarations

void AA_example_nested_local_function1();

bool AA_example_nested_is_name_valid(const char * name);

bool AA_example_nested_is_name_valid_bufferify(const char * name,
    int Lname);

bool AA_example_nested_is_initialized();

void AA_example_nested_test_names(const char * name);

void AA_example_nested_test_names_bufferify(const char * name,
    int Lname);

void AA_example_nested_test_names_flag(const char * name, int flag);

void AA_example_nested_test_names_flag_bufferify(const char * name,
    int Lname, int flag);

void AA_example_nested_testoptional_0();

void AA_example_nested_testoptional_1(int i);

void AA_example_nested_testoptional_2(int i, long j);

size_t AA_example_nested_test_size_t();

#ifdef HAVE_MPI
void AA_example_nested_testmpi_mpi(MPI_Fint comm);
#endif

#ifndef HAVE_MPI
void AA_example_nested_testmpi_serial();
#endif

void AA_example_nested_testgroup1(SIDRE_group * grp);

void AA_example_nested_testgroup2(const SIDRE_group * grp);

void AA_example_nested_func_ptr1(void ( * get)());

void AA_example_nested_func_ptr2(double * ( * get)());

void AA_example_nested_func_ptr3(double ( * get)(int i, int));

void AA_example_nested_func_ptr4(double ( * get)(double, int));

void AA_example_nested_func_ptr5(void ( * get)(int verylongname1,
    int verylongname2, int verylongname3, int verylongname4,
    int verylongname5, int verylongname6, int verylongname7,
    int verylongname8, int verylongname9, int verylongname10));

void AA_example_nested_verylongfunctionname1(int * verylongname1,
    int * verylongname2, int * verylongname3, int * verylongname4,
    int * verylongname5, int * verylongname6, int * verylongname7,
    int * verylongname8, int * verylongname9, int * verylongname10);

int AA_example_nested_verylongfunctionname2(int verylongname1,
    int verylongname2, int verylongname3, int verylongname4,
    int verylongname5, int verylongname6, int verylongname7,
    int verylongname8, int verylongname9, int verylongname10);

void AA_example_nested_cos_doubles(double * in, double * out,
    int sizein);

#ifdef __cplusplus
}
#endif

#endif  // WRAPUSERLIBRARY_EXAMPLE_NESTED_H
