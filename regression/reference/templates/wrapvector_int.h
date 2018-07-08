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

// splicer begin class.vector.CXX_declarations
// splicer end class.vector.CXX_declarations

#ifdef __cplusplus
extern "C" {
#endif

// splicer begin class.vector.C_declarations
// splicer end class.vector.C_declarations

TEM_vector TEM_vector_int_ctor();

void TEM_vector_int_dtor(TEM_vector_int * self);

void TEM_vector_int_push_back_XXXX(TEM_vector_int * self,
    const int value);

#ifdef __cplusplus
}
#endif

#endif  // WRAPVECTOR_INT_H
