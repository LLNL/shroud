// wrapTutorial.cpp
// This is generated code, do not edit
// #######################################################################
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
#include "wrapTutorial.h"
#include <cstring>
#include <string>
#include "tutorial.hpp"

// Returns the length of character string a with length ls,
// ignoring any trailing blanks.
int ShroudLenTrim(const char *s, int ls) {
    int i;

    for (i = ls - 1; i >= 0; i--) {
        if (s[i] != ' ') {
            break;
        }
    }

    return i + 1;
}


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

namespace tutorial {

// splicer begin CXX_definitions
// splicer end CXX_definitions

extern "C" {

// splicer begin C_definitions
// splicer end C_definitions

// void Function1()
// function_index=3
void TUT_function1()
{
// splicer begin function.function1
    Function1();
    return;
// splicer end function.function1
}

// double Function2(double arg1 +intent(in)+value, int arg2 +intent(in)+value)
// function_index=4
double TUT_function2(double arg1, int arg2)
{
// splicer begin function.function2
    double SHT_rv = Function2(arg1, arg2);
    return SHT_rv;
// splicer end function.function2
}

// void Sum(int len +intent(in)+value, int * values +dimension(len)+intent(in), int * result +intent(out))
// function_index=5
void TUT_sum(int len, int * values, int * result)
{
// splicer begin function.sum
    Sum(len, values, result);
    return;
// splicer end function.sum
}

// long long TypeLongLong(long long arg1 +intent(in)+value)
// function_index=6
long long TUT_type_long_long(long long arg1)
{
// splicer begin function.type_long_long
    long long SHT_rv = TypeLongLong(arg1);
    return SHT_rv;
// splicer end function.type_long_long
}

// bool Function3(bool arg +intent(in)+value)
// function_index=7
bool TUT_function3(bool arg)
{
// splicer begin function.function3
    bool SHT_rv = Function3(arg);
    return SHT_rv;
// splicer end function.function3
}

// void Function3b(const bool arg1 +intent(in)+value, bool * arg2 +intent(out), bool * arg3 +intent(inout))
// function_index=8
void TUT_function3b(const bool arg1, bool * arg2, bool * arg3)
{
// splicer begin function.function3b
    Function3b(arg1, arg2, arg3);
    return;
// splicer end function.function3b
}

// void Function4a +len(30)(const std::string & arg1 +intent(in)+len_trim(Larg1), const std::string & arg2 +intent(in)+len_trim(Larg2), std::string * SHF_rv +intent(out)+len(NSHF_rv))
// function_index=41
void TUT_function4a_bufferify(const char * arg1, int Larg1, 
    const char * arg2, int Larg2, char * SHF_rv, int NSHF_rv)
{
// splicer begin function.function4a_bufferify
    const std::string SH_arg1(arg1, Larg1);
    const std::string SH_arg2(arg2, Larg2);
    const std::string SHT_rv = Function4a(SH_arg1, SH_arg2);
    if (SHT_rv.empty()) {
        std::memset(SHF_rv, ' ', NSHF_rv);
    } else {
        ShroudStrCopy(SHF_rv, NSHF_rv, SHT_rv.c_str());
    }
    return;
// splicer end function.function4a_bufferify
}

// const std::string & Function4b(const std::string & arg1 +intent(in), const std::string & arg2 +intent(in))
// function_index=10
const char * TUT_function4b(const char * arg1, const char * arg2)
{
// splicer begin function.function4b
    const std::string SH_arg1(arg1);
    const std::string SH_arg2(arg2);
    const std::string & SHT_rv = Function4b(SH_arg1, SH_arg2);
    const char * XSHT_rv = SHT_rv.c_str();
    return XSHT_rv;
// splicer end function.function4b
}

// void Function4b(const std::string & arg1 +intent(in)+len_trim(Larg1), const std::string & arg2 +intent(in)+len_trim(Larg2), std::string & output +intent(out)+len(Noutput))
// function_index=42
void TUT_function4b_bufferify(const char * arg1, int Larg1, 
    const char * arg2, int Larg2, char * output, int Noutput)
{
// splicer begin function.function4b_bufferify
    const std::string SH_arg1(arg1, Larg1);
    const std::string SH_arg2(arg2, Larg2);
    const std::string & SHT_rv = Function4b(SH_arg1, SH_arg2);
    if (SHT_rv.empty()) {
        std::memset(output, ' ', Noutput);
    } else {
        ShroudStrCopy(output, Noutput, SHT_rv.c_str());
    }
    return;
// splicer end function.function4b_bufferify
}

// double Function5()
// function_index=31
double TUT_function5()
{
// splicer begin function.function5
    double SHT_rv = Function5();
    return SHT_rv;
// splicer end function.function5
}

// double Function5(double arg1=3.1415 +intent(in)+value)
// function_index=32
double TUT_function5_arg1(double arg1)
{
// splicer begin function.function5_arg1
    double SHT_rv = Function5(arg1);
    return SHT_rv;
// splicer end function.function5_arg1
}

// double Function5(double arg1=3.1415 +intent(in)+value, bool arg2=true +intent(in)+value)
// function_index=11
double TUT_function5_arg1_arg2(double arg1, bool arg2)
{
// splicer begin function.function5_arg1_arg2
    double SHT_rv = Function5(arg1, arg2);
    return SHT_rv;
// splicer end function.function5_arg1_arg2
}

// void Function6(const std::string & name +intent(in))
// function_index=12
void TUT_function6_from_name(const char * name)
{
// splicer begin function.function6_from_name
    const std::string SH_name(name);
    Function6(SH_name);
    return;
// splicer end function.function6_from_name
}

// void Function6(const std::string & name +intent(in)+len_trim(Lname))
// function_index=44
void TUT_function6_from_name_bufferify(const char * name, int Lname)
{
// splicer begin function.function6_from_name_bufferify
    const std::string SH_name(name, Lname);
    Function6(SH_name);
    return;
// splicer end function.function6_from_name_bufferify
}

// void Function6(int indx +intent(in)+value)
// function_index=13
void TUT_function6_from_index(int indx)
{
// splicer begin function.function6_from_index
    Function6(indx);
    return;
// splicer end function.function6_from_index
}

// void Function7(int arg +intent(in)+value)
// function_index=33
void TUT_function7_int(int arg)
{
// splicer begin function.function7_int
    Function7<int>(arg);
    return;
// splicer end function.function7_int
}

// void Function7(double arg +intent(in)+value)
// function_index=34
void TUT_function7_double(double arg)
{
// splicer begin function.function7_double
    Function7<double>(arg);
    return;
// splicer end function.function7_double
}

// int Function8()
// function_index=35
int TUT_function8_int()
{
// splicer begin function.function8_int
    int SHT_rv = Function8<int>();
    return SHT_rv;
// splicer end function.function8_int
}

// double Function8()
// function_index=36
double TUT_function8_double()
{
// splicer begin function.function8_double
    double SHT_rv = Function8<double>();
    return SHT_rv;
// splicer end function.function8_double
}

// void Function9(double arg +intent(in)+value)
// function_index=16
void TUT_function9(double arg)
{
// splicer begin function.function9
    Function9(arg);
    return;
// splicer end function.function9
}

// void Function10()
// function_index=17
void TUT_function10_0()
{
// splicer begin function.function10_0
    Function10();
    return;
// splicer end function.function10_0
}

// void Function10(const std::string & name +intent(in), double arg2 +intent(in)+value)
// function_index=18
void TUT_function10_1(const char * name, double arg2)
{
// splicer begin function.function10_1
    const std::string SH_name(name);
    Function10(SH_name, arg2);
    return;
// splicer end function.function10_1
}

// void Function10(const std::string & name +intent(in)+len_trim(Lname), double arg2 +intent(in)+value)
// function_index=45
void TUT_function10_1_bufferify(const char * name, int Lname, 
    double arg2)
{
// splicer begin function.function10_1_bufferify
    const std::string SH_name(name, Lname);
    Function10(SH_name, arg2);
    return;
// splicer end function.function10_1_bufferify
}

// int overload1(int num +intent(in)+value)
// function_index=37
int TUT_overload1_num(int num)
{
// splicer begin function.overload1_num
    int SHT_rv = overload1(num);
    return SHT_rv;
// splicer end function.overload1_num
}

// int overload1(int num +intent(in)+value, int offset=0 +intent(in)+value)
// function_index=38
int TUT_overload1_num_offset(int num, int offset)
{
// splicer begin function.overload1_num_offset
    int SHT_rv = overload1(num, offset);
    return SHT_rv;
// splicer end function.overload1_num_offset
}

// int overload1(int num +intent(in)+value, int offset=0 +intent(in)+value, int stride=1 +intent(in)+value)
// function_index=19
int TUT_overload1_num_offset_stride(int num, int offset, int stride)
{
// splicer begin function.overload1_num_offset_stride
    int SHT_rv = overload1(num, offset, stride);
    return SHT_rv;
// splicer end function.overload1_num_offset_stride
}

// int overload1(double type +intent(in)+value, int num +intent(in)+value)
// function_index=39
int TUT_overload1_3(double type, int num)
{
// splicer begin function.overload1_3
    int SHT_rv = overload1(type, num);
    return SHT_rv;
// splicer end function.overload1_3
}

// int overload1(double type +intent(in)+value, int num +intent(in)+value, int offset=0 +intent(in)+value)
// function_index=40
int TUT_overload1_4(double type, int num, int offset)
{
// splicer begin function.overload1_4
    int SHT_rv = overload1(type, num, offset);
    return SHT_rv;
// splicer end function.overload1_4
}

// int overload1(double type +intent(in)+value, int num +intent(in)+value, int offset=0 +intent(in)+value, int stride=1 +intent(in)+value)
// function_index=20
int TUT_overload1_5(double type, int num, int offset, int stride)
{
// splicer begin function.overload1_5
    int SHT_rv = overload1(type, num, offset, stride);
    return SHT_rv;
// splicer end function.overload1_5
}

// TypeID typefunc(TypeID arg +intent(in)+value)
// function_index=21
int TUT_typefunc(int arg)
{
// splicer begin function.typefunc
    TypeID SHT_rv = typefunc(arg);
    return SHT_rv;
// splicer end function.typefunc
}

// EnumTypeID enumfunc(EnumTypeID arg +intent(in)+value)
// function_index=22
int TUT_enumfunc(int arg)
{
// splicer begin function.enumfunc
    EnumTypeID SHT_rv = enumfunc(static_cast<EnumTypeID>(arg));
    int XSHT_rv = static_cast<int>(SHT_rv);
    return XSHT_rv;
// splicer end function.enumfunc
}

// void useclass(const Class1 * arg1 +intent(in)+value)
// function_index=23
void TUT_useclass(const TUT_class1 * arg1)
{
// splicer begin function.useclass
    const Class1 *SH_arg1 = static_cast<const Class1 *>(static_cast<const void *>(arg1));
    useclass(SH_arg1);
    return;
// splicer end function.useclass
}

// int vector_sum(const std::vector<int> & arg +dimension(:)+intent(in)+size(Sarg))
// function_index=46
int TUT_vector_sum_bufferify(const int * arg, long Sarg)
{
// splicer begin function.vector_sum_bufferify
    const std::vector<int> SH_arg(arg, arg + Sarg);
    int SHT_rv = vector_sum(SH_arg);
    return SHT_rv;
// splicer end function.vector_sum_bufferify
}

// void vector_iota(std::vector<int> & arg +dimension(:)+intent(out)+size(Sarg))
// function_index=47
void TUT_vector_iota_bufferify(int * arg, long Sarg)
{
// splicer begin function.vector_iota_bufferify
    std::vector<int> SH_arg(Sarg);
    vector_iota(SH_arg);
    {
        std::vector<int>::size_type
            SHT_i = 0,
            SHT_n = Sarg;
        SHT_n = std::min(SH_arg.size(), SHT_n);
        for(; SHT_i < SHT_n; SHT_i++) {
            arg[SHT_i] = SH_arg[SHT_i];
        }
    }
    return;
// splicer end function.vector_iota_bufferify
}

// void vector_increment(std::vector<int> & arg +dimension(:)+intent(inout)+size(Sarg))
// function_index=48
void TUT_vector_increment_bufferify(int * arg, long Sarg)
{
// splicer begin function.vector_increment_bufferify
    std::vector<int> SH_arg(arg, arg + Sarg);
    vector_increment(SH_arg);
    {
        std::vector<int>::size_type
            SHT_i = 0,
            SHT_n = Sarg;
        SHT_n = std::min(SH_arg.size(), SHT_n);
        for(; SHT_i < SHT_n; SHT_i++) {
              arg[SHT_i] = SH_arg[SHT_i];
        }
    }
    return;
// splicer end function.vector_increment_bufferify
}

// int vector_string_count(const std::vector<std::string> & arg +dimension(:)+intent(in)+len(Narg)+size(Sarg))
// function_index=49
/**
 * \brief count number of underscore in vector of strings
 *
 */
int TUT_vector_string_count_bufferify(const char * arg, long Sarg, 
    int Narg)
{
// splicer begin function.vector_string_count_bufferify
    std::vector<std::string> SH_arg;
    {
          const char * BBB = arg;
          std::vector<std::string>::size_type
            SHT_i = 0,
            SHT_n = Sarg;
        for(; SHT_i < SHT_n; SHT_i++) {
            SH_arg.push_back(std::string(BBB,ShroudLenTrim(BBB, Narg)));
            BBB += Narg;
        }
    }
    int SHT_rv = vector_string_count(SH_arg);
    return SHT_rv;
// splicer end function.vector_string_count_bufferify
}

// void vector_string_fill(std::vector<std::string> & arg +dimension(:)+intent(out)+len(Narg)+size(Sarg))
// function_index=50
/**
 * \brief Fill in arg with some animal names
 *
 * The C++ function returns void. But the C and Fortran wrappers return
 * an int with the number of items added to arg.
 */
int TUT_vector_string_fill_bufferify(char * arg, long Sarg, int Narg)
{
// splicer begin function.vector_string_fill_bufferify
    std::vector<std::string> SH_arg;
    vector_string_fill(SH_arg);
    {
        char * BBB = arg;
        std::vector<std::string>::size_type
            SHT_i = 0,
            SHT_n = Sarg;
        SHT_n = std::min(SH_arg.size(),SHT_n);
        for(; SHT_i < SHT_n; SHT_i++) {
            ShroudStrCopy(BBB, Narg, SH_arg[SHT_i].c_str());
            BBB += Narg;
        }
    }
    return SH_arg.size();
// splicer end function.vector_string_fill_bufferify
}

// void vector_string_append(std::vector<std::string> & arg +dimension(:)+intent(inout)+len(Narg)+size(Sarg))
// function_index=51
/**
 * \brief append '-like' to names.
 *
 */
void TUT_vector_string_append_bufferify(char * arg, long Sarg, int Narg)
{
// splicer begin function.vector_string_append_bufferify
    std::vector<std::string> SH_arg;
    {
        char * BBB = arg;
        std::vector<std::string>::size_type
            SHT_i = 0,
            SHT_n = Sarg;
        for(; SHT_i < SHT_n; SHT_i++) {
            SH_arg.push_back(std::string(BBB,ShroudLenTrim(BBB, Narg)));
            BBB += Narg;
        }
    }
    vector_string_append(SH_arg);
    {
        char * BBB = arg;
        std::vector<std::string>::size_type
            SHT_i = 0,
            SHT_n = Sarg;
        SHT_n = std::min(SH_arg.size(),SHT_n);
        for(; SHT_i < SHT_n; SHT_i++) {
            ShroudStrCopy(BBB, Narg, SH_arg[SHT_i].c_str());
            BBB += Narg;
        }
    }
    return;
// splicer end function.vector_string_append_bufferify
}

// const std::string & LastFunctionCalled() +pure
// function_index=30
const char * TUT_last_function_called()
{
// splicer begin function.last_function_called
    const std::string & SHT_rv = LastFunctionCalled();
    const char * XSHT_rv = SHT_rv.c_str();
    return XSHT_rv;
// splicer end function.last_function_called
}

// void LastFunctionCalled(std::string & SHF_rv +intent(out)+len(NSHF_rv)) +pure
// function_index=52
void TUT_last_function_called_bufferify(char * SHF_rv, int NSHF_rv)
{
// splicer begin function.last_function_called_bufferify
    const std::string & SHT_rv = LastFunctionCalled();
    if (SHT_rv.empty()) {
        std::memset(SHF_rv, ' ', NSHF_rv);
    } else {
        ShroudStrCopy(SHF_rv, NSHF_rv, SHT_rv.c_str());
    }
    return;
// splicer end function.last_function_called_bufferify
}

}  // extern "C"

}  // namespace tutorial
