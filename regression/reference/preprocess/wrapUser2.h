// wrapUser2.h
// This file is generated by Shroud nowrite-version. Do not edit.
// Copyright (c) 2017-2023, Lawrence Livermore National Security, LLC and
// other Shroud Project Developers.
// See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//
/**
 * \file wrapUser2.h
 * \brief Shroud generated wrapper for User2 class
 */
// For C users and C++ implementation

#ifndef WRAPUSER2_H
#define WRAPUSER2_H
#ifdef USE_USER2

// shroud
#include "typespreprocess.h"

// splicer begin class.User2.CXX_declarations
// splicer end class.User2.CXX_declarations

#ifdef __cplusplus
extern "C" {
#endif

// splicer begin class.User2.C_declarations
// splicer end class.User2.C_declarations

#ifdef USE_CLASS3_A
void PRE_User2_exfunc_0(PRE_User2 * self);
#endif

#ifndef USE_CLASS3_A
void PRE_User2_exfunc_1(PRE_User2 * self, int flag);
#endif

#ifdef __cplusplus
}
#endif
#endif  // ifdef USE_USER2

#endif  // WRAPUSER2_H
