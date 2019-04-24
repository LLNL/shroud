// wrapClibrary.h
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
 * \file wrapClibrary.h
 * \brief Shroud generated wrapper for Clibrary library
 */
// For C users and C implementation

#ifndef WRAPCLIBRARY_H
#define WRAPCLIBRARY_H

#include "clibrary.h"
#include "typesClibrary.h"

// splicer begin C_declarations
// splicer end C_declarations

void CLI_function4a_bufferify(const char * arg1, int Larg1,
    const char * arg2, int Larg2, char * SHF_rv, int NSHF_rv);

void CLI_bind_c2_bufferify(const char * name, int Lname);

int CLI_pass_struct2_bufferify(Cstruct1 * s1, const char * name,
    int Lname);

Cstruct1 * CLI_return_struct_ptr2_bufferify(int ifield,
    const char * name, int Lname);

#endif  // WRAPCLIBRARY_H
