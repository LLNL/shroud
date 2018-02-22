// wrapUserLibrary.cpp
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
#include "wrapUserLibrary.h"
#include <string>
#include "sidre/Group.hpp"

namespace example {
namespace nested {

// splicer begin CXX_definitions
// splicer end CXX_definitions

extern "C" {

// splicer begin C_definitions
// splicer end C_definitions

// void local_function1()
// function_index=49
void AA_local_function1()
{
// splicer begin function.local_function1
    local_function1();
    return;
// splicer end function.local_function1
}

// bool isNameValid(const std::string & name +intent(in))
// function_index=50
bool AA_is_name_valid(const char * name)
{
// splicer begin function.is_name_valid
    return name != NULL;
// splicer end function.is_name_valid
}

// bool isNameValid(const std::string & name +intent(in)+len_trim(Lname))
// function_index=70
bool AA_is_name_valid_bufferify(const char * name, int Lname)
{
// splicer begin function.is_name_valid_bufferify
    return name != NULL;
// splicer end function.is_name_valid_bufferify
}

// bool isInitialized()
// function_index=51
bool AA_is_initialized()
{
// splicer begin function.is_initialized
    bool SHC_rv = isInitialized();
    return SHC_rv;
// splicer end function.is_initialized
}

// void checkBool(bool arg1 +intent(in)+value, bool * arg2 +intent(out), bool * arg3 +intent(inout))
// function_index=52
void AA_check_bool(bool arg1, bool * arg2, bool * arg3)
{
// splicer begin function.check_bool
    checkBool(arg1, arg2, arg3);
    return;
// splicer end function.check_bool
}

// void test_names(const std::string & name +intent(in))
// function_index=53
void AA_test_names(const char * name)
{
// splicer begin function.test_names
    const std::string SH_name(name);
    test_names(SH_name);
    return;
// splicer end function.test_names
}

// void test_names(const std::string & name +intent(in)+len_trim(Lname))
// function_index=71
void AA_test_names_bufferify(const char * name, int Lname)
{
// splicer begin function.test_names_bufferify
    const std::string SH_name(name, Lname);
    test_names(SH_name);
    return;
// splicer end function.test_names_bufferify
}

// void test_names(const std::string & name +intent(in), int flag +intent(in)+value)
// function_index=54
void AA_test_names_flag(const char * name, int flag)
{
// splicer begin function.test_names_flag
    const std::string SH_name(name);
    test_names(SH_name, flag);
    return;
// splicer end function.test_names_flag
}

// void test_names(const std::string & name +intent(in)+len_trim(Lname), int flag +intent(in)+value)
// function_index=72
void AA_test_names_flag_bufferify(const char * name, int Lname,
    int flag)
{
// splicer begin function.test_names_flag_bufferify
    const std::string SH_name(name, Lname);
    test_names(SH_name, flag);
    return;
// splicer end function.test_names_flag_bufferify
}

// void testoptional()
// function_index=68
void AA_testoptional_0()
{
// splicer begin function.testoptional_0
    testoptional();
    return;
// splicer end function.testoptional_0
}

// void testoptional(int i=1 +intent(in)+value)
// function_index=69
void AA_testoptional_1(int i)
{
// splicer begin function.testoptional_1
    testoptional(i);
    return;
// splicer end function.testoptional_1
}

// void testoptional(int i=1 +intent(in)+value, long j=2 +intent(in)+value)
// function_index=55
void AA_testoptional_2(int i, long j)
{
// splicer begin function.testoptional_2
    testoptional(i, j);
    return;
// splicer end function.testoptional_2
}

// size_t test_size_t()
// function_index=56
size_t AA_test_size_t()
{
// splicer begin function.test_size_t
    size_t SHC_rv = test_size_t();
    return SHC_rv;
// splicer end function.test_size_t
}

// void testmpi(MPI_Comm comm +intent(in)+value)
// function_index=57
#ifdef HAVE_MPI
void AA_testmpi(MPI_Fint comm)
{
// splicer begin function.testmpi
    MPI_Comm SHCXX_comm = MPI_Comm_f2c(comm);
    testmpi(SHCXX_comm);
    return;
// splicer end function.testmpi
}
#endif  // ifdef HAVE_MPI

// void testgroup1(DataGroup * grp +intent(in)+value)
// function_index=58
void AA_testgroup1(SIDRE_group * grp)
{
// splicer begin function.testgroup1
    axom::sidre::Group * SHCXX_grp = static_cast<axom::sidre::Group *>(
        static_cast<void *>(grp));
    testgroup1(SHCXX_grp);
    return;
// splicer end function.testgroup1
}

// void testgroup2(const DataGroup * grp +intent(in)+value)
// function_index=59
void AA_testgroup2(const SIDRE_group * grp)
{
// splicer begin function.testgroup2
    const axom::sidre::Group * SHCXX_grp = 
        static_cast<const axom::sidre::Group *>(
        static_cast<const void *>(grp));
    testgroup2(SHCXX_grp);
    return;
// splicer end function.testgroup2
}

// void FuncPtr1(void ( * get) +intent(in)+value())
// function_index=60
/**
 * \brief subroutine
 *
 */
void AA_func_ptr1(void ( * get)())
{
// splicer begin function.func_ptr1
    FuncPtr1(get);
    return;
// splicer end function.func_ptr1
}

// void FuncPtr2(double * ( * get) +intent(in)())
// function_index=61
/**
 * \brief return a pointer
 *
 */
void AA_func_ptr2(double * ( * get)())
{
// splicer begin function.func_ptr2
    FuncPtr2(get);
    return;
// splicer end function.func_ptr2
}

// void FuncPtr3(double ( * get) +intent(in)+value(int i +value, int +value))
// function_index=62
/**
 * \brief abstract argument
 *
 */
void AA_func_ptr3(double ( * get)(int i, int))
{
// splicer begin function.func_ptr3
    FuncPtr3(get);
    return;
// splicer end function.func_ptr3
}

// void FuncPtr4(double ( * get) +intent(in)+value(double +value, int +value))
// function_index=63
/**
 * \brief abstract argument
 *
 */
void AA_func_ptr4(double ( * get)(double, int))
{
// splicer begin function.func_ptr4
    FuncPtr4(get);
    return;
// splicer end function.func_ptr4
}

// void FuncPtr5(void ( * get) +intent(in)+value(int verylongname1 +value, int verylongname2 +value, int verylongname3 +value, int verylongname4 +value, int verylongname5 +value, int verylongname6 +value, int verylongname7 +value, int verylongname8 +value, int verylongname9 +value, int verylongname10 +value))
// function_index=64
void AA_func_ptr5(void ( * get)(int verylongname1, int verylongname2,
    int verylongname3, int verylongname4, int verylongname5,
    int verylongname6, int verylongname7, int verylongname8,
    int verylongname9, int verylongname10))
{
// splicer begin function.func_ptr5
    FuncPtr5(get);
    return;
// splicer end function.func_ptr5
}

// void verylongfunctionname1(int * verylongname1 +intent(inout), int * verylongname2 +intent(inout), int * verylongname3 +intent(inout), int * verylongname4 +intent(inout), int * verylongname5 +intent(inout), int * verylongname6 +intent(inout), int * verylongname7 +intent(inout), int * verylongname8 +intent(inout), int * verylongname9 +intent(inout), int * verylongname10 +intent(inout))
// function_index=65
void AA_verylongfunctionname1(int * verylongname1, int * verylongname2,
    int * verylongname3, int * verylongname4, int * verylongname5,
    int * verylongname6, int * verylongname7, int * verylongname8,
    int * verylongname9, int * verylongname10)
{
// splicer begin function.verylongfunctionname1
    verylongfunctionname1(verylongname1, verylongname2, verylongname3,
        verylongname4, verylongname5, verylongname6, verylongname7,
        verylongname8, verylongname9, verylongname10);
    return;
// splicer end function.verylongfunctionname1
}

// int verylongfunctionname2(int verylongname1 +intent(in)+value, int verylongname2 +intent(in)+value, int verylongname3 +intent(in)+value, int verylongname4 +intent(in)+value, int verylongname5 +intent(in)+value, int verylongname6 +intent(in)+value, int verylongname7 +intent(in)+value, int verylongname8 +intent(in)+value, int verylongname9 +intent(in)+value, int verylongname10 +intent(in)+value)
// function_index=66
int AA_verylongfunctionname2(int verylongname1, int verylongname2,
    int verylongname3, int verylongname4, int verylongname5,
    int verylongname6, int verylongname7, int verylongname8,
    int verylongname9, int verylongname10)
{
// splicer begin function.verylongfunctionname2
    int SHC_rv = verylongfunctionname2(verylongname1, verylongname2,
        verylongname3, verylongname4, verylongname5, verylongname6,
        verylongname7, verylongname8, verylongname9, verylongname10);
    return SHC_rv;
// splicer end function.verylongfunctionname2
}

// void cos_doubles(double * in +dimension(:,:)+intent(in), double * out +allocatable(mold=in)+dimension(:,:)+intent(out), int sizein +implied(size(in))+intent(in)+value)
// function_index=67
/**
 * \brief Test multidimensional arrays with allocatable
 *
 */
void AA_cos_doubles(double * in, double * out, int sizein)
{
// splicer begin function.cos_doubles
    cos_doubles(in, out, sizein);
    return;
// splicer end function.cos_doubles
}

}  // extern "C"

}  // namespace nested
}  // namespace example
