// wrapUser2.cpp
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
#ifdef USE_USER2
#include "wrapUser2.h"
#include "User2.hpp"

// splicer begin class.User2.CXX_definitions
// splicer end class.User2.CXX_definitions

extern "C" {

// splicer begin class.User2.C_definitions
// splicer end class.User2.C_definitions

// void exfunc()
#ifdef USE_CLASS3_A
void PRE_user2_exfunc_0(PRE_user2 * self)
{
// splicer begin class.User2.method.exfunc_0
    User2 *SH_this = static_cast<User2 *>(self->addr);
    SH_this->exfunc();
    return;
// splicer end class.User2.method.exfunc_0
}
#endif  // ifdef USE_CLASS3_A

// void exfunc(int flag +intent(in)+value)
#ifndef USE_CLASS3_A
void PRE_user2_exfunc_1(PRE_user2 * self, int flag)
{
// splicer begin class.User2.method.exfunc_1
    User2 *SH_this = static_cast<User2 *>(self->addr);
    SH_this->exfunc(flag);
    return;
// splicer end class.User2.method.exfunc_1
}
#endif  // ifndef USE_CLASS3_A

}  // extern "C"
#endif  // ifdef USE_USER2
