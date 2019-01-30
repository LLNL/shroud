// wrapvectors.h
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
 * \file wrapvectors.h
 * \brief Shroud generated wrapper for vectors library
 */
// For C users and C++ implementation

#ifndef WRAPVECTORS_H
#define WRAPVECTORS_H

#include "typesvectors.h"

// splicer begin CXX_declarations
// splicer end CXX_declarations

#ifdef __cplusplus
extern "C" {
#endif

// splicer begin C_declarations
// splicer end C_declarations

int VEC_vector_sum_bufferify(const int * arg, long Sarg);

void VEC_vector_iota_out_bufferify(VEC_SHROUD_array *Darg);

void VEC_vector_iota_out_alloc_bufferify(VEC_SHROUD_array *Darg);

void VEC_vector_iota_inout_alloc_bufferify(int * arg, long Sarg,
    VEC_SHROUD_array *Darg);

void VEC_vector_increment_bufferify(int * arg, long Sarg,
    VEC_SHROUD_array *Darg);

int VEC_vector_string_count_bufferify(const char * arg, long Sarg,
    int Narg);

#ifdef __cplusplus
}
#endif

#endif  // WRAPVECTORS_H
