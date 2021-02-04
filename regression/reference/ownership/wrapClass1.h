// wrapClass1.h
// This file is generated by Shroud nowrite-version. Do not edit.
// Copyright (c) 2017-2021, Lawrence Livermore National Security, LLC and
// other Shroud Project Developers.
// See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//
/**
 * \file wrapClass1.h
 * \brief Shroud generated wrapper for Class1 class
 */
// For C users and C++ implementation

#ifndef WRAPCLASS1_H
#define WRAPCLASS1_H

// shroud
#include "typesownership.h"

// splicer begin class.Class1.CXX_declarations
// splicer end class.Class1.CXX_declarations

#ifdef __cplusplus
extern "C" {
#endif

// splicer begin class.Class1.C_declarations
// splicer end class.Class1.C_declarations

void OWN_Class1_dtor(OWN_Class1 * self);

int OWN_Class1_get_flag(OWN_Class1 * self);

#ifdef __cplusplus
}
#endif

#endif  // WRAPCLASS1_H
