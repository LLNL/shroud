// Copyright (c) 2017-2020, Lawrence Livermore National Security, LLC and
// other Shroud Project Developers.
// See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//
// tutorial.cpp - wrapped routines
//

#include "tutorial.hpp"

namespace tutorial
{

int tutorial_flag;

static std::string last_function_called;

// These variables exist to avoid warning errors
static std::string global_str;
static int global_int;
static double global_double;


void NoReturnNoArguments()
{
    last_function_called = "NoReturnNoArguments";
    return;
}

double PassByValue(double arg1, int arg2)
{
    last_function_called = "PassByValue";
    return arg1 + arg2;
}

const std::string ConcatenateStrings(const std::string& arg1, const std::string& arg2)
{
    last_function_called = "Function4c";
    return arg1 + arg2;
}

double UseDefaultArguments(double arg1, bool arg2)
{
    last_function_called = "UseDefautArguments";
    if (arg2) {
	return arg1 + 10.0;
    } else {
	return arg1;
    }
}

void OverloadedFunction(const std::string& name)
{
    last_function_called = "OverloadedFunction(string)";
    global_str = name;
    return;
}
void OverloadedFunction(int indx)
{
    last_function_called = "OverloadedFunction(int)";
    global_int = indx;
    return;
}

template<>
void TemplateArgument<int>(int arg)
{
    last_function_called = "TemplateArgument<int>";
    global_int = arg;
}

template<>
void TemplateArgument<double>(double arg)
{
    last_function_called = "TemplateArgument<double>";
    global_double = arg;
}

template<>
int TemplateReturn<int>()
{
    last_function_called = "TemplateReturn<int>";
    return global_int;
}

template<>
double TemplateReturn<double>()
{
    last_function_called = "TemplateReturn<double>";
    return global_double;
}

void FortranGeneric(double arg)
{
    last_function_called = "FortranGeneric";
    global_double = arg;
    return;
}

void FortranGenericOverloaded()
{
    last_function_called = "FortranGenericOverloaded_0";
}

void FortranGenericOverloaded(const std::string &name, double arg2)
{
    last_function_called = "FortranGenericOverloaded_1";
    global_str = name;
    global_double = arg2;
}

// default values and overloaded
// int UseDefaultOverload(int num, int offset = 0, int stride = 1);
int UseDefaultOverload(int num, int offset, int stride)
{
    last_function_called = "UseDefaultOverload_0";
    return num + offset * stride;
    
}

// default values and overloaded
// int UseDefaultOverload(double type, int num, int offset = 0, int stride = 1);
int UseDefaultOverload(double type, int num, int offset, int stride)
{
    last_function_called = "UseDefaultOverload_1";
    global_double = type;
    return num + offset * stride;
}

TypeID typefunc(TypeID arg)
{
    last_function_called = "typefunc";
    return static_cast<int>(arg);
}

EnumTypeID enumfunc(EnumTypeID arg)
{
    last_function_called = "enumfunc";
    switch (arg) {
    default:
	return ENUM2;
    }
}

Color colorfunc(Color arg)
{
    last_function_called = "colorfunc";
    return RED;
}

// start getMinMaxa
void getMinMax(int &min, int &max)
{
  min = -1;
  max = 100;
}
// end getMinMaxa

//----------------------------------------------------------------------

// start callback1
int callback1(int in, int (*incr)(int))
{
  return incr(in);
}
// end callback1

//----------------------------------------------------------------------

const std::string& LastFunctionCalled()
{
    return last_function_called;
}

} /* end namespace tutorial */
