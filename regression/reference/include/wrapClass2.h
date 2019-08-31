// wrapClass2.h
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
 * \file wrapClass2.h
 * \brief Shroud generated wrapper for Class2 class
 */
// For C users and C++ implementation

#ifndef WRAPCLASS2_H
#define WRAPCLASS2_H

#include "mpi.h"
#include "typeslibrary.h"


#ifdef __cplusplus
extern "C" {
#endif


void LIB_Class2_method1(LIB_Class2 * self, MPI_Fint comm);

void LIB_Class2_method2(LIB_Class2 * self, LIB_three_Class1 * c2);

#ifdef __cplusplus
}
#endif

#endif  // WRAPCLASS2_H
