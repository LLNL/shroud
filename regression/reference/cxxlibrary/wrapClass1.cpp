// wrapClass1.cpp
// This file is generated by Shroud nowrite-version. Do not edit.
// Copyright (c) 2017-2023, Lawrence Livermore National Security, LLC and
// other Shroud Project Developers.
// See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//

// cxx_header
#include "cxxlibrary.hpp"
// shroud
#include "wrapClass1.h"

// splicer begin class.Class1.CXX_definitions
// splicer end class.Class1.CXX_definitions

extern "C" {

// splicer begin class.Class1.C_definitions
// splicer end class.Class1.C_definitions

// ----------------------------------------
// Function:  Class1
// Statement: c_ctor_shadow_scalar_capptr
CXX_Class1 * CXX_Class1_ctor(CXX_Class1 * SHC_rv)
{
    // splicer begin class.Class1.method.ctor
    Class1 *SHCXX_rv = new Class1();
    SHC_rv->addr = static_cast<void *>(SHCXX_rv);
    SHC_rv->idtor = 1;
    return SHC_rv;
    // splicer end class.Class1.method.ctor
}

/**
 * \brief Test fortran_generic with default arguments.
 *
 */
// Generated by has_default_arg
// ----------------------------------------
// Function:  int check_length
// Statement: c_function_native_scalar
int CXX_Class1_check_length_0(CXX_Class1 * self)
{
    Class1 *SH_this = static_cast<Class1 *>(self->addr);
    // splicer begin class.Class1.method.check_length_0
    int SHC_rv = SH_this->check_length();
    return SHC_rv;
    // splicer end class.Class1.method.check_length_0
}

/**
 * \brief Test fortran_generic with default arguments.
 *
 */
// ----------------------------------------
// Function:  int check_length
// Statement: c_function_native_scalar
// ----------------------------------------
// Argument:  int length=1
// Statement: c_in_native_scalar
int CXX_Class1_check_length_1(CXX_Class1 * self, int length)
{
    Class1 *SH_this = static_cast<Class1 *>(self->addr);
    // splicer begin class.Class1.method.check_length_1
    int SHC_rv = SH_this->check_length(length);
    return SHC_rv;
    // splicer end class.Class1.method.check_length_1
}

// Generated by has_default_arg
// ----------------------------------------
// Function:  Class1 * declare
// Statement: c_function_shadow_*_this
// ----------------------------------------
// Argument:  int flag
// Statement: c_in_native_scalar
void CXX_Class1_declare_0(CXX_Class1 * self, int flag)
{
    Class1 *SH_this = static_cast<Class1 *>(self->addr);
    // splicer begin class.Class1.method.declare_0
    SH_this->declare(flag);
    // splicer end class.Class1.method.declare_0
}

// ----------------------------------------
// Function:  Class1 * declare
// Statement: c_function_shadow_*_this
// ----------------------------------------
// Argument:  int flag
// Statement: c_in_native_scalar
// ----------------------------------------
// Argument:  int length=1
// Statement: c_in_native_scalar
void CXX_Class1_declare_1(CXX_Class1 * self, int flag, int length)
{
    Class1 *SH_this = static_cast<Class1 *>(self->addr);
    // splicer begin class.Class1.method.declare_1
    SH_this->declare(flag, length);
    // splicer end class.Class1.method.declare_1
}

// Generated by getter/setter
// ----------------------------------------
// Function:  int get_length +intent(getter)
// Statement: f_getter_native_scalar
int CXX_Class1_get_length(CXX_Class1 * self)
{
    Class1 *SH_this = static_cast<Class1 *>(self->addr);
    // splicer begin class.Class1.method.get_length
    // skip call c_getter
    return SH_this->m_length;
    // splicer end class.Class1.method.get_length
}

}  // extern "C"
