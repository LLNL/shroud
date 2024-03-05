// wrapcdesc.h
// This file is generated by Shroud nowrite-version. Do not edit.
// Copyright (c) 2017-2024, Lawrence Livermore National Security, LLC and
// other Shroud Project Developers.
// See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//
/**
 * \file wrapcdesc.h
 * \brief Shroud generated wrapper for cdesc library
 */
// For C users and C++ implementation

#ifndef WRAPCDESC_H
#define WRAPCDESC_H

// shroud
#include "typescdesc.h"

// splicer begin CXX_declarations
// splicer end CXX_declarations

#ifdef __cplusplus
extern "C" {
#endif

// splicer begin C_declarations
// splicer end C_declarations

void CDE_Rank2In(int * arg);

void CDE_Rank2In_bufferify(CDE_SHROUD_array *SHT_arg_cdesc);

void CDE_GetScalar1(char * name, void * value);

void CDE_GetScalar1_0_bufferify(char *name, int SHT_name_len,
    CDE_SHROUD_array *SHT_value_cdesc);

void CDE_GetScalar1_1_bufferify(char *name, int SHT_name_len,
    CDE_SHROUD_array *SHT_value_cdesc);

int CDE_getData_int(void);

double CDE_getData_double(void);

void CDE_GetScalar2_0_bufferify(char *name, int SHT_name_len,
    int * value);

void CDE_GetScalar2_1_bufferify(char *name, int SHT_name_len,
    double * value);

#ifdef __cplusplus
}
#endif

#endif  // WRAPCDESC_H
