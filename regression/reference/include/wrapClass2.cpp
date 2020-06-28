// wrapClass2.cpp
// This file is generated by Shroud 0.12.0. Do not edit.
// Copyright (c) 2017-2020, Lawrence Livermore National Security, LLC and
// other Shroud Project Developers.
// See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//
#include "wrapClass2.h"
#include "class_header.hpp"
#include "global_header.hpp"


extern "C" {


// ----------------------------------------
// Function:  void method1
// Requested: c
// Match:     c_default
// ----------------------------------------
// Argument:  MPI_Comm comm +intent(in)+value
// Requested: c_unknown_scalar_in
// Match:     c_default
void LIB_Class2_method1(LIB_Class2 * self, MPI_Fint comm)
{
    Class2 *SH_this = static_cast<Class2 *>(self->addr);
    MPI_Comm SHCXX_comm = MPI_Comm_f2c(comm);
    SH_this->method1(SHCXX_comm);
}

// ----------------------------------------
// Function:  void method2
// Requested: c
// Match:     c_default
// ----------------------------------------
// Argument:  three::Class1 * c2 +intent(inout)
// Requested: c_shadow_*_inout
// Match:     c_shadow_inout
void LIB_Class2_method2(LIB_Class2 * self, LIB_three_Class1 * c2)
{
    Class2 *SH_this = static_cast<Class2 *>(self->addr);
    three::Class1 * SHCXX_c2 = static_cast<three::Class1 *>(c2->addr);
    SH_this->method2(SHCXX_c2);
}

}  // extern "C"
