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
//
// tutorial.hpp - wrapped routines
//

#include "tutorial.hpp"

int global_flag;

namespace tutorial
{

int tutorial_flag;

static std::string last_function_called;

// These variables exist to avoid warning errors
static std::string global_str;
static int global_int;
static double global_double;
static const Class1 *global_class1;
static struct1 global_struct1;


void Function1()
{
    last_function_called = "Function1";
    return;
}

double Function2(double arg1, int arg2)
{
    last_function_called = "Function2";
    return arg1 + arg2;
}

bool Function3(bool arg)
{
    last_function_called = "Function3";
    return ! arg;
}

void Function3b(const bool arg1, bool *arg2, bool *arg3)
{
    last_function_called = "Function3b";
    *arg2 = ! arg1;
    *arg3 = ! *arg3;
    return;
}

const std::string Function4a(const std::string& arg1, const std::string& arg2)
{
    last_function_called = "Function4a";
    return arg1 + arg2;
}

const std::string& Function4b(const std::string& arg1, const std::string& arg2)
{
    last_function_called = "Function4b";
    global_str = arg1 + arg2;
    return global_str;
}

const std::string Function4c(const std::string& arg1, const std::string& arg2)
{
    last_function_called = "Function4c";
    return arg1 + arg2;
}

const std::string * Function4d()
{
    const std::string * rv = new std::string("Function4d");
    return rv;
}

double Function5(double arg1, bool arg2)
{
    last_function_called = "Function5";
    if (arg2) {
	return arg1 + 10.0;
    } else {
	return arg1;
    }
}

void Function6(const std::string& name)
{
    last_function_called = "Function6(string)";
    global_str = name;
    return;
}
void Function6(int indx)
{
    last_function_called = "Function6(int)";
    global_int = indx;
    return;
}

template<>
void Function7<int>(int arg)
{
    last_function_called = "Function7<int>";
    global_int = arg;
}

template<>
void Function7<double>(double arg)
{
    last_function_called = "Function7<double>";
    global_double = arg;
}

template<>
int Function8<int>()
{
    last_function_called = "Function8<int>";
    return global_int;
}

template<>
double Function8<double>()
{
    last_function_called = "Function8<double>";
    return global_double;
}

void Function9(double arg)
{
    last_function_called = "Function9";
    global_double = arg;
    return;
}

void Function10()
{
    last_function_called = "Function10_0";
}

void Function10(const std::string &name, double arg2)
{
    last_function_called = "Function10_1";
    global_str = name;
    global_double = arg2;
}

void Sum(size_t len, int *values, int *result)
{
    last_function_called = "Sum";

    int sum = 0;
    for (size_t i=0; i < len; i++) {
	sum += values[i];
    }
    *result = sum;
    return;
}

long long TypeLongLong(long long arg1)
{
  return arg1 + 2;
}


// default values and overloaded
// int overload1(int num, int offset = 0, int stride = 1);
int overload1(int num, int offset, int stride)
{
    last_function_called = "overload1_0";
    return num + offset * stride;
    
}

// default values and overloaded
// int overload1(double type, int num, int offset = 0, int stride = 1);
int overload1(double type, int num, int offset, int stride)
{
    last_function_called = "overload1_1";
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

void getMinMax(int &min, int &max)
{
  min = -1;
  max = 100;
}

int useclass(const Class1 *arg)
{
    last_function_called = "useclass";
    global_class1 = arg;
    return arg->m_flag;
}

void getclass(const Class1 **arg)
{
    last_function_called = "getclass";
    *arg = global_class1;
}

const Class1 * getclass2()
{
    last_function_called = "getclass2";
    return global_class1;
}

Class1 * getclass3()
{
    last_function_called = "getclass3";
    return const_cast<Class1 *>(global_class1);
}

/* Return class instance by value */
Class1 getClassCopy(int flag)
{
    Class1 node(flag);
    last_function_called = "getClassCopy";
    return node;
}

//----------------------------------------------------------------------
// class methods

int Class1::Method1()
{
    last_function_called = "Class1::Method1";
    return m_flag;
}

bool Class1::equivalent(Class1 const &obj2) const
{
    last_function_called = "Class1::equivalent";
    return m_flag == obj2.m_flag;
}

Class1 * Class1::returnThis()
{
    last_function_called = "Class1::returnThis";
    return this;
}

Class1 *Class1::returnThisBuffer(std::string & name, bool flag)
{
    global_str = name;
    last_function_called = "Class1::getThisBuffer";
    return this;
}

Class1 * Class1::getclass3() const
{
    last_function_called = "Class1::getclass3";
    return const_cast<Class1 *>(global_class1);
}

Class1::DIRECTION Class1::directionFunc(Class1::DIRECTION arg)
{
    last_function_called = "Class1::directionFunc";
    return Class1::LEFT;
}

// This method is not in the class but uses the class enum
Class1::DIRECTION directionFunc(Class1::DIRECTION arg)
{
    last_function_called = "directionFunc";
    return Class1::RIGHT;
}

//----------------------------------------------------------------------

int callback1(int in, int (*incr)(int))
{
  return incr(in);
}

//----------------------------------------------------------------------

struct1 returnStruct(int i, double d)
{
  struct1 s = {i, d};
  return s;
}

struct1 *returnStructPtr(int i, double d)
{
  global_struct1.ifield = i;
  global_struct1.dfield = d;
  return &global_struct1;
}

struct1 *returnStructPtrNew(int i, double d)
{
  struct1 *s = new struct1;
  s->ifield = i;
  s->dfield = d;
  return s;
}

void freeStruct(struct1 *arg1)
{
  delete arg1;
}

// return sum of fields as a check
double acceptStructIn(struct1 arg)
{
  return arg.ifield + arg.dfield;
}

// return sum of fields as a check
double acceptStructInPtr(struct1 *arg)
{
  return arg->ifield + arg->dfield;
}

void acceptStructOutPtr(struct1 *arg, int i, double d)
{
  arg->ifield = i;
  arg->dfield = d;
  return;
}

void acceptStructInOutPtr(struct1 *arg)
{
  arg->ifield += 1;
  arg->dfield += 1.0;
  return;
}

//----------------------------------------------------------------------

const std::string& LastFunctionCalled()
{
    return last_function_called;
}

} /* end namespace tutorial */
