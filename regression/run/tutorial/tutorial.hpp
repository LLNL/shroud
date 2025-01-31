// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
// other Shroud Project Developers.
// See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//
// tutorial.hpp - wrapped routines
//

#ifndef TUTORIAL_HPP
#define TUTORIAL_HPP

#include <string>

extern int global_flag;

namespace tutorial
{

enum Color {
    RED,
    BLUE,
    WHITE,
};

typedef int TypeID;

extern int tutorial_flag;

void NoReturnNoArguments();

double PassByValue(double arg1, int arg2);

const std::string  ConcatenateStrings(const std::string& arg1, const std::string& arg2);

// start UseDefaultArguments
double UseDefaultArguments(double arg1 = 3.1415, bool arg2 = true);
// end UseDefaultArguments

void OverloadedFunction(const std::string& name);
void OverloadedFunction(int indx);

// specialize for int and double in tutorial.cpp
template<typename ArgType>
void TemplateArgument(ArgType arg);

// specialize for int and double in tutorial.cpp
template<typename RetType>
RetType TemplateReturn();

void FortranGeneric(double arg);

void FortranGenericOverloaded();
void FortranGenericOverloaded(const std::string &name, double arg2);

int UseDefaultOverload(int num, int offset = 0, int stride = 1);
int UseDefaultOverload(double type, int num, int offset = 0, int stride = 1);

TypeID typefunc(TypeID arg);

Color colorfunc(Color arg);

void getMinMax(int &min, int &max);

int callback1(int in, int (*incr)(int));

void set_global_flag(int arg);
int get_global_flag();
const std::string& LastFunctionCalled();

} /* end namespace tutorial */


#endif // TUTORIAL_HPP
