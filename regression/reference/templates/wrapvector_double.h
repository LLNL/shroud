// wrapvector_double.h
// This is generated code, do not edit
/**
 * \file wrapvector_double.h
 * \brief Shroud generated wrapper for vector class
 */
// For C users and C++ implementation

#ifndef WRAPVECTOR_DOUBLE_H
#define WRAPVECTOR_DOUBLE_H

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

TEM_vector_double * TEM_vector_double_ctor(TEM_vector_double * SHC_rv);

void TEM_vector_double_dtor(TEM_vector_double * self);

void TEM_vector_double_push_back(TEM_vector_double * self,
    const double * value);

double * TEM_vector_double_at(TEM_vector_double * self, size_t n);

#ifdef __cplusplus
}
#endif

#endif  // WRAPVECTOR_DOUBLE_H
