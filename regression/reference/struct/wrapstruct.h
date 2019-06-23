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
// For C users and C implementation

#ifndef WRAPSTRUCT_H
#define WRAPSTRUCT_H

#include "struct.h"
#include "typesstruct.h"

// splicer begin C_declarations
// splicer end C_declarations

int STR_pass_struct2_bufferify(Cstruct1 * s1, char * outbuf,
    int Noutbuf);

Cstruct1 * STR_return_struct_ptr2_bufferify(int i, double d,
    char * outbuf, int Noutbuf);

#endif  // WRAPSTRUCT_H
