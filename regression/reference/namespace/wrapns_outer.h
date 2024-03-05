// wrapns_outer.h
// This file is generated by Shroud nowrite-version. Do not edit.
// Copyright (c) 2017-2024, Lawrence Livermore National Security, LLC and
// other Shroud Project Developers.
// See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//
/**
 * \file wrapns_outer.h
 * \brief Shroud generated wrapper for outer namespace
 */
// For C users and C++ implementation

#ifndef WRAPNS_OUTER_H
#define WRAPNS_OUTER_H

// shroud
#include "typesns.h"

// splicer begin namespace.outer.CXX_declarations
// splicer end namespace.outer.CXX_declarations

#ifdef __cplusplus
extern "C" {
#endif

#ifdef __cplusplus
}
#endif
#ifdef __cplusplus
using NS_cstruct1 = outer::Cstruct1;
#else  // __cplusplus

typedef struct s_NS_cstruct1 NS_cstruct1;
struct s_NS_cstruct1 {
    int ifield;
    double dfield;
};
#endif  // __cplusplus

#ifdef __cplusplus
extern "C" {
#endif

// splicer begin namespace.outer.C_declarations
// splicer end namespace.outer.C_declarations

void NS_outer_One(void);

#ifdef __cplusplus
}
#endif

#endif  // WRAPNS_OUTER_H
