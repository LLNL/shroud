// wrapExClass1.cpp
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
#include "wrapExClass1.h"
#include <cstring>
#include <string>
#include "ExClass1.hpp"

// splicer begin class.ExClass1.CXX_definitions
// splicer end class.ExClass1.CXX_definitions

extern "C" {


// Copy s into a, blank fill to la characters
// Truncate if a is too short.
static void ShroudStrCopy(char *a, int la, const char *s)
{
   int ls,nm;
   ls = std::strlen(s);
   nm = ls < la ? ls : la;
   std::memcpy(a,s,nm);
   if(la > nm) std::memset(a+nm,' ',la-nm);
}
// splicer begin class.ExClass1.C_definitions
// splicer end class.ExClass1.C_definitions

// ExClass1()
// function_index=0
AA_exclass1 AA_exclass1_ctor_0()
{
// splicer begin class.ExClass1.method.ctor_0
    example::nested::ExClass1 *SHCXX_rv = new example::nested::
        ExClass1();
    AA_exclass1 SHC_rv = { static_cast<void *>(SHCXX_rv), 0 };
    return SHC_rv;
// splicer end class.ExClass1.method.ctor_0
}

// ExClass1(const string * name +intent(in))
// function_index=1
/**
 * \brief constructor
 *
 * longer description
 * usually multiple lines
 *
 * \return return new instance
 */
AA_exclass1 AA_exclass1_ctor_1(const char * name)
{
// splicer begin class.ExClass1.method.ctor_1
    const std::string SH_name(name);
    example::nested::ExClass1 *SHCXX_rv = new example::nested::
        ExClass1(&SH_name);
    AA_exclass1 SHC_rv = { static_cast<void *>(SHCXX_rv), 0 };
    return SHC_rv;
// splicer end class.ExClass1.method.ctor_1
}

// ExClass1(const string * name +intent(in)+len_trim(Lname))
// function_index=14
/**
 * \brief constructor
 *
 * longer description
 * usually multiple lines
 *
 * \return return new instance
 */
AA_exclass1 AA_exclass1_ctor_1_bufferify(const char * name, int Lname)
{
// splicer begin class.ExClass1.method.ctor_1_bufferify
    const std::string SH_name(name, Lname);
    example::nested::ExClass1 *SHCXX_rv = new example::nested::
        ExClass1(&SH_name);
    AA_exclass1 SHC_rv = { static_cast<void *>(SHCXX_rv), 0 };
    return SHC_rv;
// splicer end class.ExClass1.method.ctor_1_bufferify
}

// ~ExClass1()
// function_index=2
/**
 * \brief destructor
 *
 * longer description joined with previous line
 */
void AA_exclass1_dtor(AA_exclass1 * self)
{
// splicer begin class.ExClass1.method.dtor
    example::nested::ExClass1 *SH_this = static_cast<example::nested::
        ExClass1 *>(self->addr);
    delete SH_this;
    return;
// splicer end class.ExClass1.method.dtor
}

// int incrementCount(int incr +intent(in)+value)
// function_index=3
int AA_exclass1_increment_count(AA_exclass1 * self, int incr)
{
// splicer begin class.ExClass1.method.increment_count
    example::nested::ExClass1 *SH_this = static_cast<example::nested::
        ExClass1 *>(self->addr);
    int SHC_rv = SH_this->incrementCount(incr);
    return SHC_rv;
// splicer end class.ExClass1.method.increment_count
}

// const string & getNameErrorPattern() const +len(aa_exclass1_get_name_length({F_this}%{F_derived_member}))
// function_index=4
const char * AA_exclass1_get_name_error_pattern(
    const AA_exclass1 * self)
{
// splicer begin class.ExClass1.method.get_name_error_pattern
    const example::nested::ExClass1 *SH_this = 
        static_cast<const example::nested::ExClass1 *>(self->addr);
    const std::string & SHCXX_rv = SH_this->getNameErrorPattern();
    // C_error_pattern
    if (! isNameValid(SHCXX_rv)) {
        return NULL;
    }

    const char * SHC_rv = SHCXX_rv.c_str();
    return SHC_rv;
// splicer end class.ExClass1.method.get_name_error_pattern
}

// void getNameErrorPattern(string & SHF_rv +intent(out)+len(NSHF_rv)) const +len(aa_exclass1_get_name_length({F_this}%{F_derived_member}))
// function_index=15
void AA_exclass1_get_name_error_pattern_bufferify(
    const AA_exclass1 * self, char * SHF_rv, int NSHF_rv)
{
// splicer begin class.ExClass1.method.get_name_error_pattern_bufferify
    const example::nested::ExClass1 *SH_this = 
        static_cast<const example::nested::ExClass1 *>(self->addr);
    const std::string & SHCXX_rv = SH_this->getNameErrorPattern();
    if (SHCXX_rv.empty()) {
        std::memset(SHF_rv, ' ', NSHF_rv);
    } else {
        ShroudStrCopy(SHF_rv, NSHF_rv, SHCXX_rv.c_str());
    }
    return;
// splicer end class.ExClass1.method.get_name_error_pattern_bufferify
}

// int GetNameLength() const
// function_index=5
/**
 * \brief helper function for Fortran to get length of name.
 *
 */
int AA_exclass1_get_name_length(const AA_exclass1 * self)
{
// splicer begin class.ExClass1.method.get_name_length
    const example::nested::ExClass1 *SH_this = 
        static_cast<const example::nested::ExClass1 *>(self->addr);
    return SH_this->getName().length();

// splicer end class.ExClass1.method.get_name_length
}

// const string & getNameErrorCheck() const
// function_index=6
const char * AA_exclass1_get_name_error_check(const AA_exclass1 * self)
{
// splicer begin class.ExClass1.method.get_name_error_check
    const example::nested::ExClass1 *SH_this = 
        static_cast<const example::nested::ExClass1 *>(self->addr);
    const std::string & SHCXX_rv = SH_this->getNameErrorCheck();
    const char * SHC_rv = SHCXX_rv.c_str();
    return SHC_rv;
// splicer end class.ExClass1.method.get_name_error_check
}

// void getNameErrorCheck(string & SHF_rv +intent(out)+len(NSHF_rv)) const
// function_index=16
void AA_exclass1_get_name_error_check_bufferify(
    const AA_exclass1 * self, char * SHF_rv, int NSHF_rv)
{
// splicer begin class.ExClass1.method.get_name_error_check_bufferify
    const example::nested::ExClass1 *SH_this = 
        static_cast<const example::nested::ExClass1 *>(self->addr);
    const std::string & SHCXX_rv = SH_this->getNameErrorCheck();
    if (SHCXX_rv.empty()) {
        std::memset(SHF_rv, ' ', NSHF_rv);
    } else {
        ShroudStrCopy(SHF_rv, NSHF_rv, SHCXX_rv.c_str());
    }
    return;
// splicer end class.ExClass1.method.get_name_error_check_bufferify
}

// const string & getNameArg() const
// function_index=7
const char * AA_exclass1_get_name_arg(const AA_exclass1 * self)
{
// splicer begin class.ExClass1.method.get_name_arg
    const example::nested::ExClass1 *SH_this = 
        static_cast<const example::nested::ExClass1 *>(self->addr);
    const std::string & SHCXX_rv = SH_this->getNameArg();
    const char * SHC_rv = SHCXX_rv.c_str();
    return SHC_rv;
// splicer end class.ExClass1.method.get_name_arg
}

// void getNameArg(string & name +intent(out)+len(Nname)) const
// function_index=17
void AA_exclass1_get_name_arg_bufferify(const AA_exclass1 * self,
    char * name, int Nname)
{
// splicer begin class.ExClass1.method.get_name_arg_bufferify
    const example::nested::ExClass1 *SH_this = 
        static_cast<const example::nested::ExClass1 *>(self->addr);
    const std::string & SHCXX_rv = SH_this->getNameArg();
    if (SHCXX_rv.empty()) {
        std::memset(name, ' ', Nname);
    } else {
        ShroudStrCopy(name, Nname, SHCXX_rv.c_str());
    }
    return;
// splicer end class.ExClass1.method.get_name_arg_bufferify
}

// void * getRoot()
// function_index=8
void * AA_exclass1_get_root(AA_exclass1 * self)
{
// splicer begin class.ExClass1.method.get_root
    example::nested::ExClass1 *SH_this = static_cast<example::nested::
        ExClass1 *>(self->addr);
    void * SHC_rv = SH_this->getRoot();
    return SHC_rv;
// splicer end class.ExClass1.method.get_root
}

// int getValue(int value +intent(in)+value)
// function_index=9
int AA_exclass1_get_value_from_int(AA_exclass1 * self, int value)
{
// splicer begin class.ExClass1.method.get_value_from_int
    example::nested::ExClass1 *SH_this = static_cast<example::nested::
        ExClass1 *>(self->addr);
    int SHC_rv = SH_this->getValue(value);
    return SHC_rv;
// splicer end class.ExClass1.method.get_value_from_int
}

// long getValue(long value +intent(in)+value)
// function_index=10
long AA_exclass1_get_value_1(AA_exclass1 * self, long value)
{
// splicer begin class.ExClass1.method.get_value_1
    example::nested::ExClass1 *SH_this = static_cast<example::nested::
        ExClass1 *>(self->addr);
    long SHC_rv = SH_this->getValue(value);
    return SHC_rv;
// splicer end class.ExClass1.method.get_value_1
}

// void * getAddr()
// function_index=11
void * AA_exclass1_get_addr(AA_exclass1 * self)
{
// splicer begin class.ExClass1.method.get_addr
    example::nested::ExClass1 *SH_this = static_cast<example::nested::
        ExClass1 *>(self->addr);
    void * SHC_rv = SH_this->getAddr();
    return SHC_rv;
// splicer end class.ExClass1.method.get_addr
}

// bool hasAddr(bool in +intent(in)+value)
// function_index=12
bool AA_exclass1_has_addr(AA_exclass1 * self, bool in)
{
// splicer begin class.ExClass1.method.has_addr
    example::nested::ExClass1 *SH_this = static_cast<example::nested::
        ExClass1 *>(self->addr);
    bool SHC_rv = SH_this->hasAddr(in);
    return SHC_rv;
// splicer end class.ExClass1.method.has_addr
}

// void SplicerSpecial()
// function_index=13
void AA_exclass1_splicer_special(AA_exclass1 * self)
{
// splicer begin class.ExClass1.method.splicer_special
//   splicer for SplicerSpecial
// splicer end class.ExClass1.method.splicer_special
}

}  // extern "C"
