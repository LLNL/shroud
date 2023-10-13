// wrapns.h
// This file is generated by Shroud nowrite-version. Do not edit.
// Copyright (c) 2017-2023, Lawrence Livermore National Security, LLC and
// other Shroud Project Developers.
// See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//
/**
 * \file wrapns.h
 * \brief Shroud generated wrapper for ns library
 */
// For C users and C++ implementation

#ifndef WRAPNS_H
#define WRAPNS_H

// shroud
#include "typesns.h"

// splicer begin CXX_declarations
// splicer end CXX_declarations

#ifdef __cplusplus
extern "C" {
#endif

// splicer begin C_declarations
// splicer end C_declarations

const char * NS_LastFunctionCalled(void);

void NS_LastFunctionCalled_bufferify(NS_SHROUD_array *SHT_rv_cdesc,
    NS_SHROUD_capsule_data *SHT_rv_capsule);

void NS_One(void);

#ifdef __cplusplus
}
#endif

#endif  // WRAPNS_H
