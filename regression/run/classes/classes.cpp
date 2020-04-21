// Copyright (c) 2017-2020, Lawrence Livermore National Security, LLC and
// other Shroud Project Developers.
// See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//
// classes.hpp - wrapped routines
//

#include "classes.hpp"

int global_flag;

namespace classes
{

static std::string last_function_called;
static std::string class1_name("Class1");
static std::string class2_name("Class2");

static const Class1 *global_class1;
  
// Save arg flag value in global flag.
// Used by test drive to make sure arg was passed correctly.
void passClassByValue(Class1 arg)
{
    last_function_called = "passClassByValue";
    global_flag = arg.m_test;
    return;
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

const Class1 &getConstClassReference()
{
    last_function_called = "getConstClassReference";
    return *global_class1;
}

Class1 &getClassReference()
{
    last_function_called = "getClassReference";
    return *const_cast<Class1 *>(global_class1);
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
  //    global_str = name;
    last_function_called = "Class1::getThisBuffer";
    return this;
}

Class1 * Class1::getclass3() const
{
    last_function_called = "Class1::getclass3";
    return const_cast<Class1 *>(global_class1);
}

// The wrappers for these two functions will require the copy_string helper.
// This is a global function since it is implemented in C but must be
// called from Fortran.
// This test is to ensure there is only one copy of the function generated.
const std::string& Class1:: getName()
{
    return class1_name;
}
const std::string& Class2:: getName()
{
    return class1_name;
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

void set_global_flag(int arg)
{
  global_flag = arg;
}

int get_global_flag()
{
  return global_flag;
}

const std::string& LastFunctionCalled()
{
    return last_function_called;
}

} /* end namespace classes */
