// wrapExClass2.cpp
// This is generated code, do not edit
// Copyright (c) 2017, Lawrence Livermore National Security, LLC.
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
#include "wrapExClass2.h"
#include <cstring>
#include <string>
#include "ExClass2.hpp"
#include "sidre/SidreWrapperHelpers.hpp"

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

namespace example {
namespace nested {

// splicer begin class.ExClass2.CXX_definitions
// splicer end class.ExClass2.CXX_definitions

extern "C" {

// splicer begin class.ExClass2.C_definitions
// splicer end class.ExClass2.C_definitions

// ExClass2(const string * name +intent(in)+len_trim(trim_name))
// function_index=19
/**
 * \brief constructor
 *
 */
AA_exclass2 * AA_exclass2_ctor(const char * name)
{
// splicer begin class.ExClass2.method.ctor
    const std::string SH_name(name);
    ExClass2 * SHT_rv = new ExClass2(&SH_name);
    return static_cast<AA_exclass2 *>(static_cast<void *>(SHT_rv));
// splicer end class.ExClass2.method.ctor
}

// ExClass2(const string * name +intent(in)+len_trim(trim_name))
// function_index=39
/**
 * \brief constructor
 *
 */
AA_exclass2 * AA_exclass2_ctor_bufferify(const char * name,
    int trim_name)
{
// splicer begin class.ExClass2.method.ctor_bufferify
    const std::string SH_name(name, trim_name);
    ExClass2 * SHT_rv = new ExClass2(&SH_name);
    return static_cast<AA_exclass2 *>(static_cast<void *>(SHT_rv));
// splicer end class.ExClass2.method.ctor_bufferify
}

// ~ExClass2()
// function_index=20
/**
 * \brief destructor
 *
 */
void AA_exclass2_dtor(AA_exclass2 * self)
{
// splicer begin class.ExClass2.method.dtor
    ExClass2 *SH_this = static_cast<ExClass2 *>(static_cast<void *>(
        self));
    delete SH_this;
    return;
// splicer end class.ExClass2.method.dtor
}

// const string & getName +len(aa_exclass2_get_name_length({F_this}%{F_derived_member}))() const
// function_index=21
const char * AA_exclass2_get_name(const AA_exclass2 * self)
{
// splicer begin class.ExClass2.method.get_name
    const ExClass2 *SH_this = static_cast<const ExClass2 *>(
        static_cast<const void *>(self));
    const std::string & SHT_rv = SH_this->getName();
    const char * XSHT_rv = SHT_rv.c_str();
    return XSHT_rv;
// splicer end class.ExClass2.method.get_name
}

// void getName +len(aa_exclass2_get_name_length({F_this}%{F_derived_member}))(string & SHF_rv +intent(out)+len(NSHF_rv)) const
// function_index=40
void AA_exclass2_get_name_bufferify(const AA_exclass2 * self,
    char * SHF_rv, int NSHF_rv)
{
// splicer begin class.ExClass2.method.get_name_bufferify
    const ExClass2 *SH_this = static_cast<const ExClass2 *>(
        static_cast<const void *>(self));
    const std::string & SHT_rv = SH_this->getName();
    if (SHT_rv.empty()) {
        std::memset(SHF_rv, ' ', NSHF_rv);
    } else {
        ShroudStrCopy(SHF_rv, NSHF_rv, SHT_rv.c_str());
    }
    return;
// splicer end class.ExClass2.method.get_name_bufferify
}

// const string & getName2()
// function_index=22
const char * AA_exclass2_get_name2(AA_exclass2 * self)
{
// splicer begin class.ExClass2.method.get_name2
    ExClass2 *SH_this = static_cast<ExClass2 *>(static_cast<void *>(
        self));
    const std::string & SHT_rv = SH_this->getName2();
    const char * XSHT_rv = SHT_rv.c_str();
    return XSHT_rv;
// splicer end class.ExClass2.method.get_name2
}

// void getName2(string & SHF_rv +intent(out)+len(NSHF_rv))
// function_index=41
void AA_exclass2_get_name2_bufferify(AA_exclass2 * self, char * SHF_rv,
    int NSHF_rv)
{
// splicer begin class.ExClass2.method.get_name2_bufferify
    ExClass2 *SH_this = static_cast<ExClass2 *>(static_cast<void *>(
        self));
    const std::string & SHT_rv = SH_this->getName2();
    if (SHT_rv.empty()) {
        std::memset(SHF_rv, ' ', NSHF_rv);
    } else {
        ShroudStrCopy(SHF_rv, NSHF_rv, SHT_rv.c_str());
    }
    return;
// splicer end class.ExClass2.method.get_name2_bufferify
}

// string & getName3() const
// function_index=23
char * AA_exclass2_get_name3(const AA_exclass2 * self)
{
// splicer begin class.ExClass2.method.get_name3
    const ExClass2 *SH_this = static_cast<const ExClass2 *>(
        static_cast<const void *>(self));
    std::string & SHT_rv = SH_this->getName3();
    char * XSHT_rv = SHT_rv.c_str();
    return XSHT_rv;
// splicer end class.ExClass2.method.get_name3
}

// void getName3(string & SHF_rv +intent(out)+len(NSHF_rv)) const
// function_index=42
void AA_exclass2_get_name3_bufferify(const AA_exclass2 * self,
    char * SHF_rv, int NSHF_rv)
{
// splicer begin class.ExClass2.method.get_name3_bufferify
    const ExClass2 *SH_this = static_cast<const ExClass2 *>(
        static_cast<const void *>(self));
    std::string & SHT_rv = SH_this->getName3();
    if (SHT_rv.empty()) {
        std::memset(SHF_rv, ' ', NSHF_rv);
    } else {
        ShroudStrCopy(SHF_rv, NSHF_rv, SHT_rv.c_str());
    }
    return;
// splicer end class.ExClass2.method.get_name3_bufferify
}

// string & getName4()
// function_index=24
char * AA_exclass2_get_name4(AA_exclass2 * self)
{
// splicer begin class.ExClass2.method.get_name4
    ExClass2 *SH_this = static_cast<ExClass2 *>(static_cast<void *>(
        self));
    std::string & SHT_rv = SH_this->getName4();
    char * XSHT_rv = SHT_rv.c_str();
    return XSHT_rv;
// splicer end class.ExClass2.method.get_name4
}

// void getName4(string & SHF_rv +intent(out)+len(NSHF_rv))
// function_index=43
void AA_exclass2_get_name4_bufferify(AA_exclass2 * self, char * SHF_rv,
    int NSHF_rv)
{
// splicer begin class.ExClass2.method.get_name4_bufferify
    ExClass2 *SH_this = static_cast<ExClass2 *>(static_cast<void *>(
        self));
    std::string & SHT_rv = SH_this->getName4();
    if (SHT_rv.empty()) {
        std::memset(SHF_rv, ' ', NSHF_rv);
    } else {
        ShroudStrCopy(SHF_rv, NSHF_rv, SHT_rv.c_str());
    }
    return;
// splicer end class.ExClass2.method.get_name4_bufferify
}

// int GetNameLength() const
// function_index=25
/**
 * \brief helper function for Fortran
 *
 */
int AA_exclass2_get_name_length(const AA_exclass2 * self)
{
// splicer begin class.ExClass2.method.get_name_length
    const ExClass2 *SH_this = static_cast<const ExClass2 *>(
        static_cast<const void *>(self));
    return SH_this->getName().length();

// splicer end class.ExClass2.method.get_name_length
}

// ExClass1 * get_class1(const ExClass1 * in +intent(in)+value)
// function_index=26
AA_exclass1 * AA_exclass2_get_class1(AA_exclass2 * self,
    const AA_exclass1 * in)
{
// splicer begin class.ExClass2.method.get_class1
    ExClass2 *SH_this = static_cast<ExClass2 *>(static_cast<void *>(
        self));
    const ExClass1 *SH_in = static_cast<const ExClass1 *>(
        static_cast<const void *>(in));
    ExClass1 * SHT_rv = SH_this->get_class1(SH_in);
    AA_exclass1 * XSHT_rv = static_cast<AA_exclass1 *>(
        static_cast<void *>(SHT_rv));
    return XSHT_rv;
// splicer end class.ExClass2.method.get_class1
}

// void * declare(TypeID type +intent(in)+value)
// function_index=32
void AA_exclass2_declare_0(AA_exclass2 * self, int type)
{
// splicer begin class.ExClass2.method.declare_0
    ExClass2 *SH_this = static_cast<ExClass2 *>(static_cast<void *>(
        self));
    SH_this->declare(getTypeID(type));
    return;
// splicer end class.ExClass2.method.declare_0
}

// void * declare(TypeID type +intent(in)+value, SidreLength len=1 +intent(in)+value)
// function_index=27
void AA_exclass2_declare_1(AA_exclass2 * self, int type,
    SIDRE_SidreLength len)
{
// splicer begin class.ExClass2.method.declare_1
    ExClass2 *SH_this = static_cast<ExClass2 *>(static_cast<void *>(
        self));
    SH_this->declare(getTypeID(type), len);
    return;
// splicer end class.ExClass2.method.declare_1
}

// void destroyall()
// function_index=28
void AA_exclass2_destroyall(AA_exclass2 * self)
{
// splicer begin class.ExClass2.method.destroyall
    ExClass2 *SH_this = static_cast<ExClass2 *>(static_cast<void *>(
        self));
    SH_this->destroyall();
    return;
// splicer end class.ExClass2.method.destroyall
}

// TypeID getTypeID() const
// function_index=29
int AA_exclass2_get_type_id(const AA_exclass2 * self)
{
// splicer begin class.ExClass2.method.get_type_id
    const ExClass2 *SH_this = static_cast<const ExClass2 *>(
        static_cast<const void *>(self));
    TypeID SHT_rv = SH_this->getTypeID();
    int XSHT_rv = static_cast<int>(SHT_rv);
    return XSHT_rv;
// splicer end class.ExClass2.method.get_type_id
}

// void setValue(int value +intent(in)+value)
// function_index=33
void AA_exclass2_set_value_int(AA_exclass2 * self, int value)
{
// splicer begin class.ExClass2.method.set_value_int
    ExClass2 *SH_this = static_cast<ExClass2 *>(static_cast<void *>(
        self));
    SH_this->setValue<int>(value);
    return;
// splicer end class.ExClass2.method.set_value_int
}

// void setValue(long value +intent(in)+value)
// function_index=34
void AA_exclass2_set_value_long(AA_exclass2 * self, long value)
{
// splicer begin class.ExClass2.method.set_value_long
    ExClass2 *SH_this = static_cast<ExClass2 *>(static_cast<void *>(
        self));
    SH_this->setValue<long>(value);
    return;
// splicer end class.ExClass2.method.set_value_long
}

// void setValue(float value +intent(in)+value)
// function_index=35
void AA_exclass2_set_value_float(AA_exclass2 * self, float value)
{
// splicer begin class.ExClass2.method.set_value_float
    ExClass2 *SH_this = static_cast<ExClass2 *>(static_cast<void *>(
        self));
    SH_this->setValue<float>(value);
    return;
// splicer end class.ExClass2.method.set_value_float
}

// void setValue(double value +intent(in)+value)
// function_index=36
void AA_exclass2_set_value_double(AA_exclass2 * self, double value)
{
// splicer begin class.ExClass2.method.set_value_double
    ExClass2 *SH_this = static_cast<ExClass2 *>(static_cast<void *>(
        self));
    SH_this->setValue<double>(value);
    return;
// splicer end class.ExClass2.method.set_value_double
}

// int getValue()
// function_index=37
int AA_exclass2_get_value_int(AA_exclass2 * self)
{
// splicer begin class.ExClass2.method.get_value_int
    ExClass2 *SH_this = static_cast<ExClass2 *>(static_cast<void *>(
        self));
    int SHT_rv = SH_this->getValue<int>();
    return SHT_rv;
// splicer end class.ExClass2.method.get_value_int
}

// double getValue()
// function_index=38
double AA_exclass2_get_value_double(AA_exclass2 * self)
{
// splicer begin class.ExClass2.method.get_value_double
    ExClass2 *SH_this = static_cast<ExClass2 *>(static_cast<void *>(
        self));
    double SHT_rv = SH_this->getValue<double>();
    return SHT_rv;
// splicer end class.ExClass2.method.get_value_double
}

}  // extern "C"

}  // namespace nested
}  // namespace example
