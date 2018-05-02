// wrapClass2.cpp
// This is generated code, do not edit
// #######################################################################
// Copyright (c) 2018, Lawrence Livermore National Security, LLC.
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
#include "wrapClass2.h"
#include <stdlib.h>
#include "tutorial.hpp"

// splicer begin class.Class2.CXX_definitions
// splicer end class.Class2.CXX_definitions

extern "C" {

// splicer begin class.Class2.C_definitions
// splicer end class.Class2.C_definitions

// Class2()
// function_index=0
FOR_class2 * FOR_class2_ctor()
{
// splicer begin class.Class2.method.ctor
    tutorial::Class2 *SHCXX_rv = new tutorial::Class2();
    FOR_class2 *SHC_rv = (FOR_class2 *) malloc(sizeof(FOR_class2));
    SHC_rv->addr = static_cast<void *>(SHCXX_rv);
    SHC_rv->idtor = 0;
    return SHC_rv;
// splicer end class.Class2.method.ctor
}

// ~Class2()
// function_index=1
void FOR_class2_dtor(FOR_class2 * self)
{
// splicer begin class.Class2.method.dtor
    tutorial::Class2 *SH_this = static_cast<tutorial::
        Class2 *>(self->addr);
    delete SH_this;
    self->addr = NULL;
    self->idtor = 0;
    return;
// splicer end class.Class2.method.dtor
}

// void func1(Class1 * arg +intent(in)+value)
// function_index=2
void FOR_class2_func1(FOR_class2 * self, TUT_class1 * arg)
{
// splicer begin class.Class2.method.func1
    tutorial::Class2 *SH_this = static_cast<tutorial::
        Class2 *>(self->addr);
    tutorial::Class1 * SHCXX_arg = 
        static_cast<tutorial::Class1 *>(arg->addr);
    SH_this->func1(SHCXX_arg);
    return;
// splicer end class.Class2.method.func1
}

}  // extern "C"
