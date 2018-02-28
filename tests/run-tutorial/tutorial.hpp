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
//
// tutorial.hpp - wrapped routines
//

#ifndef TUTORIAL_HPP
#define TUTORIAL_HPP

#include <string>
#include <vector>

namespace tutorial
{

enum EnumTypeID {
    ENUM0,
    ENUM1,
    ENUM2
};

typedef int TypeID;

void Function1();

double Function2(double arg1, int arg2);

bool Function3(bool arg);
void Function3b(const bool arg1, bool *arg2, bool *arg3);

const std::string  Function4a(const std::string& arg1, const std::string& arg2);
const std::string& Function4b(const std::string& arg1, const std::string& arg2);

const std::string  Function4a_alloc(const std::string& arg1, const std::string& arg2);
const std::string& Function4b_alloc(const std::string& arg1, const std::string& arg2);

double Function5(double arg1 = 3.1415, bool arg2 = true);

void Function6(const std::string& name);
void Function6(int indx);

// specialize for int and double in tutorial.cpp
template<typename ArgType>
void Function7(ArgType arg);

// specialize for int and double in tutorial.cpp
template<typename RetType>
RetType Function8();

void Function9(double arg);

void Function10();
void Function10(const std::string &name, double arg2);

void Sum(size_t len, int * values, int *result);

long long TypeLongLong(long long arg1);

int overload1(int num, int offset = 0, int stride = 1);
int overload1(double type, int num, int offset = 0, int stride = 1);

TypeID typefunc(TypeID arg);

EnumTypeID enumfunc(EnumTypeID arg);

void getMinMax(int &min, int &max);

const std::string& LastFunctionCalled();

class Class1
{
public:
    int m_flag;
    Class1()         : m_flag(0)    {};
    Class1(int flag) : m_flag(flag) {};
    int Method1();
    bool equivalent(Class1 const &obj2) const;
};

int useclass(const Class1 *arg);
void getclass(const Class1 **arg);
const Class1 * getclass2();
Class1 * getclass3();

int vector_sum(const std::vector<int> &arg);
void vector_iota(std::vector<int> &arg);
void vector_increment(std::vector<int> &arg);

int vector_string_count(const std::vector< std::string > &arg);
void vector_string_fill(std::vector< std::string > &arg);
void vector_string_append(std::vector< std::string > &arg);

int callback1(int in, int (*incr)(int));

} /* end namespace tutorial */

#endif // TUTORIAL_HPP
