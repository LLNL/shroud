// wrapExClass3.h
// This is generated code, do not edit
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
 * \file wrapExClass3.h
 * \brief Shroud generated wrapper for ExClass3 class
 */
// For C users and C++ implementation

#ifndef WRAPEXCLASS3_H
#define WRAPEXCLASS3_H
#ifdef USE_CLASS3

#include "typesUserLibrary.h"

// splicer begin class.ExClass3.CXX_declarations
// splicer end class.ExClass3.CXX_declarations

#ifdef __cplusplus
extern "C" {
#endif

// splicer begin class.ExClass3.C_declarations
// splicer end class.ExClass3.C_declarations

#ifdef USE_CLASS3_A
void AA_exclass3_exfunc_0(AA_exclass3 * self);
#endif

#ifndef USE_CLASS3_A
void AA_exclass3_exfunc_1(AA_exclass3 * self, int flag);
#endif

#ifdef __cplusplus
}
#endif
#endif  // ifdef USE_CLASS3

#endif  // WRAPEXCLASS3_H
