// wrapClass2.h
// This is generated code, do not edit
// #######################################################################
// Copyright (c) 2018-2019, Lawrence Livermore National Security, LLC.
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
 * \file wrapClass2.h
 * \brief Shroud generated wrapper for Class2 class
 */
// For C users and C++ implementation

#ifndef WRAPCLASS2_H
#define WRAPCLASS2_H

#include "typesforward.h"

// splicer begin class.Class2.CXX_declarations
// splicer end class.Class2.CXX_declarations

#ifdef __cplusplus
extern "C" {
#endif

// splicer begin class.Class2.C_declarations
// splicer end class.Class2.C_declarations

FOR_Class2 * FOR_Class2_ctor(FOR_Class2 * SHC_rv);

void FOR_Class2_dtor(FOR_Class2 * self);

void FOR_Class2_func1(FOR_Class2 * self, TUT_class1 * arg);

void FOR_Class2_accept_class3(FOR_Class2 * self, FOR_Class3 * arg);

#ifdef __cplusplus
}
#endif

#endif  // WRAPCLASS2_H
