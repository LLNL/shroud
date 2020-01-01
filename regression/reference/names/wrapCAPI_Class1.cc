// wrapCAPI_Class1.cc
// This is generated code, do not edit
// Copyright (c) 2017-2020, Lawrence Livermore National Security, LLC and
// other Shroud Project Developers.
// See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//
#include "wrapCAPI_Class1.hh"

// splicer begin namespace.CAPI.class.Class1.CXX_definitions
// splicer end namespace.CAPI.class.Class1.CXX_definitions

extern "C" {

// splicer begin namespace.CAPI.class.Class1.C_definitions
// splicer end namespace.CAPI.class.Class1.C_definitions

// void Member1()
void TES_capi_class1_member1(TES_capi_class1 * self)
{
    CAPI::Class1 *SH_this = static_cast<CAPI::Class1 *>(self->addr);
    // splicer begin namespace.CAPI.class.Class1.method.member1
    SH_this->Member1();
    return;
    // splicer end namespace.CAPI.class.Class1.method.member1
}

}  // extern "C"
