// wrapvector_int.h
// This is generated code, do not edit
/**
 * \file wrapvector_int.h
 * \brief Shroud generated wrapper for vector class
 */
// For C users and C++ implementation

#ifndef WRAPVECTOR_INT_H
#define WRAPVECTOR_INT_H

#include "typestemplates.h"
#ifdef __cplusplus
#include <cstddef>
#include <vector>
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

#endif  // WRAPVECTOR_INT_H
