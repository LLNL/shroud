// wrapvector_double.h
// This is generated code, do not edit
/**
 * \file wrapvector_double.h
 * \brief Shroud generated wrapper for vector class
 */
// For C users and C++ implementation

#ifndef WRAPVECTOR_DOUBLE_H
#define WRAPVECTOR_DOUBLE_H

#include <stddef.h>
#include "typestemplates.h"

// splicer begin class.vector.CXX_declarations
// splicer end class.vector.CXX_declarations

#ifdef __cplusplus
extern "C" {
#endif

// splicer begin class.vector.C_declarations
// splicer end class.vector.C_declarations

TEM_vector_double TEM_vector_double_ctor();

void TEM_vector_double_dtor(TEM_vector_double * self);

void TEM_vector_double_push_back(TEM_vector_double * self,
    const double * value);

double * TEM_vector_double_at(TEM_vector_double * self, size_t n);

#ifdef __cplusplus
}
#endif

#endif  // WRAPVECTOR_DOUBLE_H
