// wrapstd_vector_int.h
// This is generated code, do not edit
// Copyright (c) 2017-2020, Lawrence Livermore National Security, LLC and
// other Shroud Project Developers.
// See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//
/**
 * \file wrapstd_vector_int.h
 * \brief Shroud generated wrapper for vector class
 */
// For C users and C++ implementation

#ifndef WRAPSTD_VECTOR_INT_H
#define WRAPSTD_VECTOR_INT_H

#include "typestemplates.h"
#ifdef __cplusplus
#include <cstddef>
#else
#include <stddef.h>
#endif

// splicer begin namespace.std.class.vector.CXX_declarations
// splicer end namespace.std.class.vector.CXX_declarations

#ifdef __cplusplus
extern "C" {
#endif

// splicer begin namespace.std.class.vector.C_declarations
// splicer end namespace.std.class.vector.C_declarations

TEM_vector_int * TEM_vector_int_ctor(TEM_vector_int * SHC_rv);

void TEM_vector_int_dtor(TEM_vector_int * self);

void TEM_vector_int_push_back(TEM_vector_int * self, const int * value);

int * TEM_vector_int_at(TEM_vector_int * self, size_t n);

#ifdef __cplusplus
}
#endif

#endif  // WRAPSTD_VECTOR_INT_H
