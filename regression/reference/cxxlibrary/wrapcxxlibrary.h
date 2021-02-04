// wrapcxxlibrary.h
// This file is generated by Shroud nowrite-version. Do not edit.
// Copyright (c) 2017-2021, Lawrence Livermore National Security, LLC and
// other Shroud Project Developers.
// See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//
/**
 * \file wrapcxxlibrary.h
 * \brief Shroud generated wrapper for cxxlibrary library
 */
// For C users and C++ implementation

#ifndef WRAPCXXLIBRARY_H
#define WRAPCXXLIBRARY_H

// typemap
#ifndef __cplusplus
#include <stdbool.h>
#endif
// shroud
#include "typescxxlibrary.h"

// splicer begin CXX_declarations
// splicer end CXX_declarations

#ifdef __cplusplus
extern "C" {
#endif

// splicer begin C_declarations
// splicer end C_declarations

bool CXX_default_ptr_is_null_0(void);

bool CXX_default_ptr_is_null_1(double * data);

void CXX_default_args_in_out_0(int in1, int * out1, int * out2);

void CXX_default_args_in_out_1(int in1, int * out1, int * out2,
    bool flag);

#ifdef __cplusplus
}
#endif

#endif  // WRAPCXXLIBRARY_H
