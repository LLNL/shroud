// wrapCAPI_Class1.cc
// This file is generated by Shroud nowrite-version. Do not edit.
// Copyright (c) 2017-2024, Lawrence Livermore National Security, LLC and
// other Shroud Project Developers.
// See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//

// shroud
#include "wrapCAPI_Class1.hh"

// splicer begin namespace.CAPI.class.Class1.CXX_definitions
// splicer end namespace.CAPI.class.Class1.CXX_definitions

extern "C" {

// splicer begin namespace.CAPI.class.Class1.C_definitions
// splicer end namespace.CAPI.class.Class1.C_definitions

// ----------------------------------------
// Function:  void Member1
// Statement: c_subroutine
void TES_capi_class1_member1(TES_capi_class1 * self)
{
    CAPI::Class1 *SH_this = static_cast<CAPI::Class1 *>(self->addr);
    // splicer begin namespace.CAPI.class.Class1.method.Member1
    SH_this->Member1();
    // splicer end namespace.CAPI.class.Class1.method.Member1
}

}  // extern "C"
