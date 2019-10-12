// wrapClass1.cc
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
#include "wrapClass1.hh"

// splicer begin namespace.CAPI.class.Class1.CXX_definitions
// splicer end namespace.CAPI.class.Class1.CXX_definitions

extern "C" {

// splicer begin namespace.CAPI.class.Class1.C_definitions
// splicer end namespace.CAPI.class.Class1.C_definitions

// void Member1()
void TES_capi_class1_member1(TES_capi_class1 * self)
{
// splicer begin namespace.CAPI.class.Class1.method.member1
    CAPI::Class1 *SH_this = static_cast<CAPI::Class1 *>(self->addr);
    SH_this->Member1();
    return;
// splicer end namespace.CAPI.class.Class1.method.member1
}

}  // extern "C"
