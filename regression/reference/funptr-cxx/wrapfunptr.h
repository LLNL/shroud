// wrapfunptr.h
// This file is generated by Shroud nowrite-version. Do not edit.
// Copyright (c) 2017-2024, Lawrence Livermore National Security, LLC and
// other Shroud Project Developers.
// See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//
/**
 * \file wrapfunptr.h
 * \brief Shroud generated wrapper for funptr library
 */
// For C users and C++ implementation

#ifndef WRAPFUNPTR_H
#define WRAPFUNPTR_H

#include "wrapfunptr.h"
#include "typesfunptr.h"

// splicer begin CXX_declarations
// splicer end CXX_declarations

#ifdef __cplusplus
extern "C" {
#endif

// typedef incrtype
// splicer begin typedef.incrtype
typedef void ( * FUN_incrtype)(int i);
// splicer end typedef.incrtype

// splicer begin C_declarations
// splicer end C_declarations

void FUN_callback1(void ( * incr)(void));

void FUN_callback1_wrap(void ( * incr)(void));

void FUN_callback1_external(void ( * incr)(void));

void FUN_callback1_funptr(void ( * incr)(void));

void FUN_callback2(const char * name, int ival, FUN_incrtype incr);

void FUN_callback2_external(const char * name, int ival,
    FUN_incrtype incr);

#ifdef __cplusplus
}
#endif

#endif  // WRAPFUNPTR_H
