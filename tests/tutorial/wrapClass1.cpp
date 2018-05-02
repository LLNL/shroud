// wrapClass1.cpp
// This is generated code, do not edit
// #######################################################################
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
#include "wrapClass1.h"
#include <stdlib.h>
#include "tutorial.hpp"

// splicer begin class.Class1.CXX_definitions
// splicer end class.Class1.CXX_definitions

extern "C" {

// splicer begin class.Class1.C_definitions
// splicer end class.Class1.C_definitions

// Class1() +name(new)
// function_index=0
TUT_class1 * TUT_class1_new_default()
{
// splicer begin class.Class1.method.new_default
    tutorial::Class1 *SHCXX_rv = new tutorial::Class1();
    TUT_class1 *SHC_rv = (TUT_class1 *) malloc(sizeof(TUT_class1));
    SHC_rv->addr = static_cast<void *>(SHCXX_rv);
    SHC_rv->idtor = 0;
    return SHC_rv;
// splicer end class.Class1.method.new_default
}

// Class1(int flag +intent(in)+value) +name(new)
// function_index=1
TUT_class1 * TUT_class1_new_flag(int flag)
{
// splicer begin class.Class1.method.new_flag
    tutorial::Class1 *SHCXX_rv = new tutorial::Class1(flag);
    TUT_class1 *SHC_rv = (TUT_class1 *) malloc(sizeof(TUT_class1));
    SHC_rv->addr = static_cast<void *>(SHCXX_rv);
    SHC_rv->idtor = 0;
    return SHC_rv;
// splicer end class.Class1.method.new_flag
}

// ~Class1() +name(delete)
// function_index=2
void TUT_class1_delete(TUT_class1 * self)
{
// splicer begin class.Class1.method.delete
    tutorial::Class1 *SH_this = static_cast<tutorial::
        Class1 *>(self->addr);
    delete SH_this;
    self->addr = NULL;
    self->idtor = 0;
    return;
// splicer end class.Class1.method.delete
}

// int Method1()
// function_index=3
/**
 * \brief returns the value of flag member
 *
 */
int TUT_class1_method1(TUT_class1 * self)
{
// splicer begin class.Class1.method.method1
    tutorial::Class1 *SH_this = static_cast<tutorial::
        Class1 *>(self->addr);
    int SHC_rv = SH_this->Method1();
    return SHC_rv;
// splicer end class.Class1.method.method1
}

// bool equivalent(const Class1 & obj2 +intent(in)+value) const
// function_index=4
/**
 * \brief Pass in reference to instance
 *
 */
bool TUT_class1_equivalent(const TUT_class1 * self,
    const TUT_class1 * obj2)
{
// splicer begin class.Class1.method.equivalent
    const tutorial::Class1 *SH_this = static_cast<const tutorial::
        Class1 *>(self->addr);
    const tutorial::Class1 * SHCXX_obj2 = static_cast<const tutorial::
        Class1 *>(obj2->addr);
    bool SHC_rv = SH_this->equivalent(*SHCXX_obj2);
    return SHC_rv;
// splicer end class.Class1.method.equivalent
}

// Class1 * returnThis()
// function_index=5
/**
 * \brief Return pointer to 'this' to allow chaining calls
 *
 */
void TUT_class1_return_this(TUT_class1 * self)
{
// splicer begin class.Class1.method.return_this
    tutorial::Class1 *SH_this = static_cast<tutorial::
        Class1 *>(self->addr);
    SH_this->returnThis();
    return;
// splicer end class.Class1.method.return_this
}

// DIRECTION directionFunc(DIRECTION arg +intent(in)+value)
// function_index=6
int TUT_class1_direction_func(TUT_class1 * self, int arg)
{
// splicer begin class.Class1.method.direction_func
    tutorial::Class1 *SH_this = static_cast<tutorial::
        Class1 *>(self->addr);
    tutorial::Class1::DIRECTION SHCXX_arg = static_cast<tutorial::
        Class1::DIRECTION>(arg);
    tutorial::Class1::DIRECTION SHCXX_rv = SH_this->directionFunc(
        SHCXX_arg);
    int SHC_rv = static_cast<int>(SHCXX_rv);
    return SHC_rv;
// splicer end class.Class1.method.direction_func
}

// int getM_flag()
// function_index=7
int TUT_class1_get_m_flag(TUT_class1 * self)
{
// splicer begin class.Class1.method.get_m_flag
    tutorial::Class1 *SH_this = static_cast<tutorial::
        Class1 *>(self->addr);
    return SH_this->m_flag;
// splicer end class.Class1.method.get_m_flag
}

// int getTest()
// function_index=8
int TUT_class1_get_test(TUT_class1 * self)
{
// splicer begin class.Class1.method.get_test
    tutorial::Class1 *SH_this = static_cast<tutorial::
        Class1 *>(self->addr);
    return SH_this->m_test;
// splicer end class.Class1.method.get_test
}

// void setTest(int val +intent(in)+value)
// function_index=9
void TUT_class1_set_test(TUT_class1 * self, int val)
{
// splicer begin class.Class1.method.set_test
    tutorial::Class1 *SH_this = static_cast<tutorial::
        Class1 *>(self->addr);
    SH_this->m_test = val;
    return;
// splicer end class.Class1.method.set_test
}

}  // extern "C"
