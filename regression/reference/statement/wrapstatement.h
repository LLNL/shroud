// wrapstatement.h
// This file is generated by Shroud nowrite-version. Do not edit.
// Copyright (c) 2017-2021, Lawrence Livermore National Security, LLC and
// other Shroud Project Developers.
// See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//
/**
 * \file wrapstatement.h
 * \brief Shroud generated wrapper for statement library
 */
// For C users and C++ implementation

#ifndef WRAPSTATEMENT_H
#define WRAPSTATEMENT_H

// shroud
#include "typesstatement.h"

// splicer begin CXX_declarations
// splicer end CXX_declarations

#ifdef __cplusplus
extern "C" {
#endif

// splicer begin C_declarations
// splicer end C_declarations

int STMT_get_name_length(void);

const char * STMT_get_name_error_pattern(void);

void STMT_get_name_error_pattern_bufferify(char * SHF_rv, int NSHF_rv);

#ifdef __cplusplus
}
#endif

#endif  // WRAPSTATEMENT_H
