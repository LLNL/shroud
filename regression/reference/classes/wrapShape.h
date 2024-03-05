// wrapShape.h
// This file is generated by Shroud nowrite-version. Do not edit.
// Copyright (c) 2017-2024, Lawrence Livermore National Security, LLC and
// other Shroud Project Developers.
// See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//
/**
 * \file wrapShape.h
 * \brief Shroud generated wrapper for Shape class
 */
// For C users and C++ implementation

#ifndef WRAPSHAPE_H
#define WRAPSHAPE_H

// shroud
#include "typesclasses.h"

// splicer begin class.Shape.CXX_declarations
// splicer end class.Shape.CXX_declarations

#ifdef __cplusplus
extern "C" {
#endif

// splicer begin class.Shape.C_declarations
// splicer end class.Shape.C_declarations

CLA_Shape * CLA_Shape_ctor(CLA_Shape * SHC_rv);

int CLA_Shape_get_ivar(const CLA_Shape * self);

#ifdef __cplusplus
}
#endif

#endif  // WRAPSHAPE_H
