// wrapClass2.cpp
// This is generated code, do not edit
// #######################################################################
// Copyright (c) 2018-2019, Lawrence Livermore National Security, LLC.
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
#include "wrapClass2.h"
#include <stdlib.h>
#include "tutorial.hpp"

// splicer begin class.Class2.CXX_definitions
// splicer end class.Class2.CXX_definitions

extern "C" {

// splicer begin class.Class2.C_definitions
// splicer end class.Class2.C_definitions

// Class2()
FOR_Class2 * FOR_Class2_ctor(FOR_Class2 * SHC_rv)
{
// splicer begin class.Class2.method.ctor
    tutorial::Class2 *SHCXX_rv = new tutorial::Class2();
    SHC_rv->addr = static_cast<void *>(SHCXX_rv);
    SHC_rv->idtor = 0;
    return SHC_rv;
// splicer end class.Class2.method.ctor
}

// ~Class2()
void FOR_Class2_dtor(FOR_Class2 * self)
{
// splicer begin class.Class2.method.dtor
    tutorial::Class2 *SH_this =
        static_cast<tutorial::Class2 *>(self->addr);
    delete SH_this;
    self->addr = NULL;
    return;
// splicer end class.Class2.method.dtor
}

// void func1(Class1 * arg +intent(in))
void FOR_Class2_func1(FOR_Class2 * self, TUT_class1 * arg)
{
// splicer begin class.Class2.method.func1
    tutorial::Class2 *SH_this =
        static_cast<tutorial::Class2 *>(self->addr);
    tutorial::Class1 * SHCXX_arg =
        static_cast<tutorial::Class1 *>(arg->addr);
    SH_this->func1(SHCXX_arg);
    return;
// splicer end class.Class2.method.func1
}

// void acceptClass3(Class3 * arg +intent(in))
void FOR_Class2_accept_class3(FOR_Class2 * self, FOR_Class3 * arg)
{
// splicer begin class.Class2.method.accept_class3
    tutorial::Class2 *SH_this =
        static_cast<tutorial::Class2 *>(self->addr);
    tutorial::Class3 * SHCXX_arg =
        static_cast<tutorial::Class3 *>(arg->addr);
    SH_this->acceptClass3(SHCXX_arg);
    return;
// splicer end class.Class2.method.accept_class3
}

}  // extern "C"
