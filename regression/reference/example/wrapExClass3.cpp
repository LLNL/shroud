// wrapExClass3.cpp
// This is generated code, do not edit
// Copyright (c) 2017-2018, Lawrence Livermore National Security, LLC.
// Produced at the Lawrence Livermore National Laboratory
//
// LLNL-CODE-738041.
// All rights reserved.
//
// This file is part of Shroud.  For details, see
// https://github.com/LLNL/shroud. Please also read shroud/LICENSE.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
// * Redistributions of source code must retain the above copyright
//   notice, this list of conditions and the disclaimer below.
//
// * Redistributions in binary form must reproduce the above copyright
//   notice, this list of conditions and the disclaimer (as noted below)
//   in the documentation and/or other materials provided with the
//   distribution.
//
// * Neither the name of the LLNS/LLNL nor the names of its contributors
//   may be used to endorse or promote products derived from this
//   software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
// A PARTICULAR PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL LAWRENCE
// LIVERMORE NATIONAL SECURITY, LLC, THE U.S. DEPARTMENT OF ENERGY OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
// LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
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
