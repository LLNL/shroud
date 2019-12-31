// wrapUser1.cpp
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
#include "wrapUser1.h"
#include "preprocess.hpp"

// splicer begin class.User1.CXX_definitions
// splicer end class.User1.CXX_definitions

extern "C" {

// splicer begin class.User1.C_definitions
// splicer end class.User1.C_definitions

// void method1()
void PRE_User1_method1(PRE_User1 * self)
{
    User1 *SH_this = static_cast<User1 *>(self->addr);
    // splicer begin class.User1.method.method1
    SH_this->method1();
    return;
    // splicer end class.User1.method.method1
}

// void method2()
#if defined(USE_TWO)
void PRE_User1_method2(PRE_User1 * self)
{
    User1 *SH_this = static_cast<User1 *>(self->addr);
    // splicer begin class.User1.method.method2
    SH_this->method2();
    return;
    // splicer end class.User1.method.method2
}
#endif  // if defined(USE_TWO)

// void method3def()
#if defined(USE_THREE)
void PRE_User1_method3def_0(PRE_User1 * self)
{
    User1 *SH_this = static_cast<User1 *>(self->addr);
    // splicer begin class.User1.method.method3def_0
    SH_this->method3def();
    return;
    // splicer end class.User1.method.method3def_0
}
#endif  // if defined(USE_THREE)

// void method3def(int i=0 +intent(in)+value)
#if defined(USE_THREE)
void PRE_User1_method3def_1(PRE_User1 * self, int i)
{
    User1 *SH_this = static_cast<User1 *>(self->addr);
    // splicer begin class.User1.method.method3def_1
    SH_this->method3def(i);
    return;
    // splicer end class.User1.method.method3def_1
}
#endif  // if defined(USE_THREE)

}  // extern "C"
