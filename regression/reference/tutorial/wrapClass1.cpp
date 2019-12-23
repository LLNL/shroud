// wrapClass1.cpp
// This is generated code, do not edit
// Copyright (c) 2017-2019, Lawrence Livermore National Security, LLC and
// other Shroud Project Developers.
// See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//
#include "wrapClass1.h"
#include <string>
#include "tutorial.hpp"

// splicer begin class.Class1.CXX_definitions
// splicer end class.Class1.CXX_definitions

extern "C" {

// splicer begin class.Class1.C_definitions
// splicer end class.Class1.C_definitions

// Class1() +name(new)
// start TUT_Class1_new_default
TUT_Class1 * TUT_Class1_new_default(TUT_Class1 * SHC_rv)
{
// splicer begin class.Class1.method.new_default
    tutorial::Class1 *SHCXX_rv = new tutorial::Class1();
    SHC_rv->addr = static_cast<void *>(SHCXX_rv);
    SHC_rv->idtor = 0;
    return SHC_rv;
// splicer end class.Class1.method.new_default
}
// end TUT_Class1_new_default

// Class1(int flag +intent(in)+value) +name(new)
// start TUT_Class1_new_flag
TUT_Class1 * TUT_Class1_new_flag(int flag, TUT_Class1 * SHC_rv)
{
// splicer begin class.Class1.method.new_flag
    tutorial::Class1 *SHCXX_rv = new tutorial::Class1(flag);
    SHC_rv->addr = static_cast<void *>(SHCXX_rv);
    SHC_rv->idtor = 0;
    return SHC_rv;
// splicer end class.Class1.method.new_flag
}
// end TUT_Class1_new_flag

// ~Class1() +name(delete)
// start TUT_Class1_delete
void TUT_Class1_delete(TUT_Class1 * self)
{
// splicer begin class.Class1.method.delete
    tutorial::Class1 *SH_this =
        static_cast<tutorial::Class1 *>(self->addr);
    delete SH_this;
    self->addr = NULL;
    return;
// splicer end class.Class1.method.delete
}
// end TUT_Class1_delete

// int Method1()
/**
 * \brief returns the value of flag member
 *
 */
// start TUT_Class1_method1
int TUT_Class1_method1(TUT_Class1 * self)
{
// splicer begin class.Class1.method.method1
    tutorial::Class1 *SH_this =
        static_cast<tutorial::Class1 *>(self->addr);
    int SHC_rv = SH_this->Method1();
    return SHC_rv;
// splicer end class.Class1.method.method1
}
// end TUT_Class1_method1

// bool equivalent(const Class1 & obj2 +intent(in)) const
/**
 * \brief Pass in reference to instance
 *
 */
// start TUT_Class1_equivalent
bool TUT_Class1_equivalent(const TUT_Class1 * self, TUT_Class1 * obj2)
{
// splicer begin class.Class1.method.equivalent
    const tutorial::Class1 *SH_this =
        static_cast<const tutorial::Class1 *>(self->addr);
    const tutorial::Class1 * SHCXX_obj2 =
        static_cast<const tutorial::Class1 *>(obj2->addr);
    bool SHC_rv = SH_this->equivalent(*SHCXX_obj2);
    return SHC_rv;
// splicer end class.Class1.method.equivalent
}
// end TUT_Class1_equivalent

// Class1 * returnThis()
/**
 * \brief Return pointer to 'this' to allow chaining calls
 *
 */
// start TUT_Class1_return_this
void TUT_Class1_return_this(TUT_Class1 * self)
{
// splicer begin class.Class1.method.return_this
    tutorial::Class1 *SH_this =
        static_cast<tutorial::Class1 *>(self->addr);
    SH_this->returnThis();
    return;
// splicer end class.Class1.method.return_this
}
// end TUT_Class1_return_this

// Class1 * returnThisBuffer(std::string & name +intent(in), bool flag +intent(in)+value)
/**
 * \brief Return pointer to 'this' to allow chaining calls
 *
 */
// start TUT_Class1_return_this_buffer
TUT_Class1 * TUT_Class1_return_this_buffer(TUT_Class1 * self,
    char * name, bool flag, TUT_Class1 * SHC_rv)
{
// splicer begin class.Class1.method.return_this_buffer
    tutorial::Class1 *SH_this =
        static_cast<tutorial::Class1 *>(self->addr);
    std::string SHCXX_name(name);
    tutorial::Class1 * SHCXX_rv = SH_this->returnThisBuffer(SHCXX_name,
        flag);
    SHC_rv->addr = static_cast<void *>(SHCXX_rv);
    SHC_rv->idtor = 0;
    return SHC_rv;
// splicer end class.Class1.method.return_this_buffer
}
// end TUT_Class1_return_this_buffer

// Class1 * returnThisBuffer(std::string & name +intent(in)+len_trim(Lname), bool flag +intent(in)+value)
/**
 * \brief Return pointer to 'this' to allow chaining calls
 *
 */
// start TUT_Class1_return_this_buffer_bufferify
TUT_Class1 * TUT_Class1_return_this_buffer_bufferify(TUT_Class1 * self,
    char * name, int Lname, bool flag, TUT_Class1 * SHC_rv)
{
// splicer begin class.Class1.method.return_this_buffer_bufferify
    tutorial::Class1 *SH_this =
        static_cast<tutorial::Class1 *>(self->addr);
    std::string SHCXX_name(name, Lname);
    tutorial::Class1 * SHCXX_rv = SH_this->returnThisBuffer(SHCXX_name,
        flag);
    SHC_rv->addr = static_cast<void *>(SHCXX_rv);
    SHC_rv->idtor = 0;
    return SHC_rv;
// splicer end class.Class1.method.return_this_buffer_bufferify
}
// end TUT_Class1_return_this_buffer_bufferify

// Class1 * getclass3() const
/**
 * \brief Test const method
 *
 */
// start TUT_Class1_getclass3
TUT_Class1 * TUT_Class1_getclass3(const TUT_Class1 * self,
    TUT_Class1 * SHC_rv)
{
// splicer begin class.Class1.method.getclass3
    const tutorial::Class1 *SH_this =
        static_cast<const tutorial::Class1 *>(self->addr);
    tutorial::Class1 * SHCXX_rv = SH_this->getclass3();
    SHC_rv->addr = static_cast<void *>(SHCXX_rv);
    SHC_rv->idtor = 0;
    return SHC_rv;
// splicer end class.Class1.method.getclass3
}
// end TUT_Class1_getclass3

// DIRECTION directionFunc(DIRECTION arg +intent(in)+value)
// start TUT_Class1_direction_func
int TUT_Class1_direction_func(TUT_Class1 * self, int arg)
{
// splicer begin class.Class1.method.direction_func
    tutorial::Class1 *SH_this =
        static_cast<tutorial::Class1 *>(self->addr);
    tutorial::Class1::DIRECTION SHCXX_arg =
        static_cast<tutorial::Class1::DIRECTION>(arg);
    tutorial::Class1::DIRECTION SHCXX_rv = SH_this->directionFunc(
        SHCXX_arg);
    int SHC_rv = static_cast<int>(SHCXX_rv);
    return SHC_rv;
// splicer end class.Class1.method.direction_func
}
// end TUT_Class1_direction_func

// int getM_flag()
// start TUT_Class1_get_m_flag
int TUT_Class1_get_m_flag(TUT_Class1 * self)
{
// splicer begin class.Class1.method.get_m_flag
    tutorial::Class1 *SH_this =
        static_cast<tutorial::Class1 *>(self->addr);
    return SH_this->m_flag;
// splicer end class.Class1.method.get_m_flag
}
// end TUT_Class1_get_m_flag

// int getTest()
// start TUT_Class1_get_test
int TUT_Class1_get_test(TUT_Class1 * self)
{
// splicer begin class.Class1.method.get_test
    tutorial::Class1 *SH_this =
        static_cast<tutorial::Class1 *>(self->addr);
    return SH_this->m_test;
// splicer end class.Class1.method.get_test
}
// end TUT_Class1_get_test

// void setTest(int val +intent(in)+value)
// start TUT_Class1_set_test
void TUT_Class1_set_test(TUT_Class1 * self, int val)
{
// splicer begin class.Class1.method.set_test
    tutorial::Class1 *SH_this =
        static_cast<tutorial::Class1 *>(self->addr);
    SH_this->m_test = val;
    return;
// splicer end class.Class1.method.set_test
}
// end TUT_Class1_set_test

}  // extern "C"
