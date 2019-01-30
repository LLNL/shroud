// wrapExClass3.cpp
// This is generated code, do not edit
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
#ifdef USE_CLASS3
#include "wrapExClass3.h"

// splicer begin class.ExClass3.CXX_definitions
// splicer end class.ExClass3.CXX_definitions

extern "C" {

// splicer begin class.ExClass3.C_definitions
// splicer end class.ExClass3.C_definitions

// void exfunc()
#ifdef USE_CLASS3_A
void AA_exclass3_exfunc_0(AA_exclass3 * self)
{
// splicer begin class.ExClass3.method.exfunc_0
    example::nested::ExClass3 *SH_this =
        static_cast<example::nested::ExClass3 *>(self->addr);
    SH_this->exfunc();
    return;
// splicer end class.ExClass3.method.exfunc_0
}
#endif  // ifdef USE_CLASS3_A

// void exfunc(int flag +intent(in)+value)
#ifndef USE_CLASS3_A
void AA_exclass3_exfunc_1(AA_exclass3 * self, int flag)
{
// splicer begin class.ExClass3.method.exfunc_1
    example::nested::ExClass3 *SH_this =
        static_cast<example::nested::ExClass3 *>(self->addr);
    SH_this->exfunc(flag);
    return;
// splicer end class.ExClass3.method.exfunc_1
}
#endif  // ifndef USE_CLASS3_A

}  // extern "C"
#endif  // ifdef USE_CLASS3
