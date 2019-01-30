// wrapSingleton.h
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
 * \file wrapSingleton.h
 * \brief Shroud generated wrapper for Singleton class
 */
// For C users and C++ implementation

#ifndef WRAPSINGLETON_H
#define WRAPSINGLETON_H

#include "typesTutorial.h"

// splicer begin class.Singleton.CXX_declarations
// splicer end class.Singleton.CXX_declarations

#ifdef __cplusplus
extern "C" {
#endif

// splicer begin class.Singleton.C_declarations
// splicer end class.Singleton.C_declarations

TUT_singleton * TUT_singleton_get_reference(TUT_singleton * SHC_rv);

#ifdef __cplusplus
}
#endif

#endif  // WRAPSINGLETON_H
