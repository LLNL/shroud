// foo.cpp
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
#include "foo.h"

// splicer begin class.Names.CXX_definitions
// splicer end class.Names.CXX_definitions

extern "C" {

// splicer begin class.Names.C_definitions
// splicer end class.Names.C_definitions

// void method1()
void XXX_TES_names_method1(TES_names * self)
{
// splicer begin class.Names.method.method1
    ns0::Names *SH_this = static_cast<ns0::Names *>(self->addr);
    SH_this->method1();
    return;
// splicer end class.Names.method.method1
}

// void method2()
void XXX_TES_names_method2(TES_names * self2)
{
// splicer begin class.Names.method.method2
    ns0::Names *SH_this2 = static_cast<ns0::Names *>(self2->addr);
    SH_this->method2();
    return;
// splicer end class.Names.method.method2
}

}  // extern "C"
