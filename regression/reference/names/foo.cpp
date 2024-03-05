// foo.cpp
// This file is generated by Shroud nowrite-version. Do not edit.
// Copyright (c) 2017-2024, Lawrence Livermore National Security, LLC and
// other Shroud Project Developers.
// See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//

// shroud
#include "foo.h"

// splicer begin namespace.ns0.class.Names.CXX_definitions
// CXX_definitions for ns0 class Names
// allow YAML multiline strings
// splicer end namespace.ns0.class.Names.CXX_definitions

extern "C" {

// splicer begin namespace.ns0.class.Names.C_definitions
// splicer end namespace.ns0.class.Names.C_definitions

// ----------------------------------------
// Function:  Names +name(defaultctor)
// Statement: c_ctor_shadow_scalar_capptr
TES_ns0_Names * XXX_TES_ns0_Names_defaultctor(TES_ns0_Names * SHC_rv)
{
    // splicer begin namespace.ns0.class.Names.method.defaultctor
    ns0::Names *ARG_rv = new ns0::Names();
    SHC_rv->addr = static_cast<void *>(ARG_rv);
    SHC_rv->idtor = 1;
    return SHC_rv;
    // splicer end namespace.ns0.class.Names.method.defaultctor
}

// ----------------------------------------
// Function:  void method1
// Statement: c_subroutine
void XXX_TES_ns0_Names_method1(TES_ns0_Names * self)
{
    ns0::Names *SH_this = static_cast<ns0::Names *>(self->addr);
    // splicer begin namespace.ns0.class.Names.method.method1
    SH_this->method1();
    // splicer end namespace.ns0.class.Names.method.method1
}

// ----------------------------------------
// Function:  void method2
// Statement: c_subroutine
void XXX_TES_ns0_Names_method2(TES_ns0_Names * self2)
{
    ns0::Names *SH_this2 = static_cast<ns0::Names *>(self2->addr);
    // splicer begin namespace.ns0.class.Names.method.method2
    SH_this->method2();
    // splicer end namespace.ns0.class.Names.method.method2
}

}  // extern "C"
