// wrapstructAsClass_double.cpp
// This file is generated by Shroud nowrite-version. Do not edit.
// Copyright (c) 2017-2023, Lawrence Livermore National Security, LLC and
// other Shroud Project Developers.
// See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//

// cxx_header
#include "templates.hpp"
// shroud
#include "wrapstructAsClass_double.h"

// splicer begin class.structAsClass.CXX_definitions
// splicer end class.structAsClass.CXX_definitions

extern "C" {

// splicer begin class.structAsClass.C_definitions
// splicer end class.structAsClass.C_definitions

// ----------------------------------------
// Function:  structAsClass
// Statement: c_ctor_shadow_scalar_capptr
TEM_structAsClass_double * TEM_structAsClass_double_ctor(
    TEM_structAsClass_double * SHC_rv)
{
    // splicer begin class.structAsClass.method.ctor
    structAsClass<double> *SHCXX_rv = new structAsClass<double>();
    SHC_rv->addr = static_cast<void *>(SHCXX_rv);
    SHC_rv->idtor = 4;
    return SHC_rv;
    // splicer end class.structAsClass.method.ctor
}

// ----------------------------------------
// Function:  void set_npts
// Statement: c_subroutine
// ----------------------------------------
// Argument:  int n
// Statement: c_in_native_scalar
void TEM_structAsClass_double_set_npts(TEM_structAsClass_double * self,
    int n)
{
    structAsClass<double> *SH_this =
        static_cast<structAsClass<double> *>(self->addr);
    // splicer begin class.structAsClass.method.set_npts
    SH_this->set_npts(n);
    // splicer end class.structAsClass.method.set_npts
}

// ----------------------------------------
// Function:  int get_npts
// Statement: c_function_native_scalar
int TEM_structAsClass_double_get_npts(TEM_structAsClass_double * self)
{
    structAsClass<double> *SH_this =
        static_cast<structAsClass<double> *>(self->addr);
    // splicer begin class.structAsClass.method.get_npts
    int SHC_rv = SH_this->get_npts();
    return SHC_rv;
    // splicer end class.structAsClass.method.get_npts
}

// Generated by cxx_template
// ----------------------------------------
// Function:  void set_value
// Statement: c_subroutine
// ----------------------------------------
// Argument:  double v
// Statement: c_in_native_scalar
void TEM_structAsClass_double_set_value(TEM_structAsClass_double * self,
    double v)
{
    structAsClass<double> *SH_this =
        static_cast<structAsClass<double> *>(self->addr);
    // splicer begin class.structAsClass.method.set_value
    SH_this->set_value(v);
    // splicer end class.structAsClass.method.set_value
}

// Generated by cxx_template
// ----------------------------------------
// Function:  double get_value
// Statement: c_function_native_scalar
double TEM_structAsClass_double_get_value(
    TEM_structAsClass_double * self)
{
    structAsClass<double> *SH_this =
        static_cast<structAsClass<double> *>(self->addr);
    // splicer begin class.structAsClass.method.get_value
    double SHC_rv = SH_this->get_value();
    return SHC_rv;
    // splicer end class.structAsClass.method.get_value
}

}  // extern "C"
