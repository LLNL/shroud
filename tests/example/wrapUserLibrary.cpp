// wrapUserLibrary.cpp
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
#include "wrapUserLibrary.h"
#include <stdlib.h>
#include <string>
#include "ExClass1.hpp"
#include "ExClass2.hpp"
#include "sidre/Group.hpp"

// splicer begin CXX_definitions
// splicer end CXX_definitions

extern "C" {

// splicer begin C_definitions
// splicer end C_definitions

// void local_function1()
// function_index=50
void AA_local_function1()
{
// splicer begin function.local_function1
    example::nested::local_function1();
    return;
// splicer end function.local_function1
}

// bool isNameValid(const std::string & name +intent(in))
// function_index=51
bool AA_is_name_valid(const char * name)
{
// splicer begin function.is_name_valid
    return name != NULL;
// splicer end function.is_name_valid
}

// bool isNameValid(const std::string & name +intent(in)+len_trim(Lname))
// function_index=72
bool AA_is_name_valid_bufferify(const char * name, int Lname)
{
// splicer begin function.is_name_valid_bufferify
    return name != NULL;
// splicer end function.is_name_valid_bufferify
}

// bool isInitialized()
// function_index=52
bool AA_is_initialized()
{
// splicer begin function.is_initialized
    bool SHC_rv = example::nested::isInitialized();
    return SHC_rv;
// splicer end function.is_initialized
}

// void checkBool(bool arg1 +intent(in)+value, bool * arg2 +intent(out), bool * arg3 +intent(inout))
// function_index=53
void AA_check_bool(bool arg1, bool * arg2, bool * arg3)
{
// splicer begin function.check_bool
    example::nested::checkBool(arg1, arg2, arg3);
    return;
// splicer end function.check_bool
}

// void test_names(const std::string & name +intent(in))
// function_index=54
void AA_test_names(const char * name)
{
// splicer begin function.test_names
    const std::string SH_name(name);
    example::nested::test_names(SH_name);
    return;
// splicer end function.test_names
}

// void test_names(const std::string & name +intent(in)+len_trim(Lname))
// function_index=73
void AA_test_names_bufferify(const char * name, int Lname)
{
// splicer begin function.test_names_bufferify
    const std::string SH_name(name, Lname);
    example::nested::test_names(SH_name);
    return;
// splicer end function.test_names_bufferify
}

// void test_names(const std::string & name +intent(in), int flag +intent(in)+value)
// function_index=55
void AA_test_names_flag(const char * name, int flag)
{
// splicer begin function.test_names_flag
    const std::string SH_name(name);
    example::nested::test_names(SH_name, flag);
    return;
// splicer end function.test_names_flag
}

// void test_names(const std::string & name +intent(in)+len_trim(Lname), int flag +intent(in)+value)
// function_index=74
void AA_test_names_flag_bufferify(const char * name, int Lname,
    int flag)
{
// splicer begin function.test_names_flag_bufferify
    const std::string SH_name(name, Lname);
    example::nested::test_names(SH_name, flag);
    return;
// splicer end function.test_names_flag_bufferify
}

// void testoptional()
// function_index=70
void AA_testoptional_0()
{
// splicer begin function.testoptional_0
    example::nested::testoptional();
    return;
// splicer end function.testoptional_0
}

// void testoptional(int i=1 +intent(in)+value)
// function_index=71
void AA_testoptional_1(int i)
{
// splicer begin function.testoptional_1
    example::nested::testoptional(i);
    return;
// splicer end function.testoptional_1
}

// void testoptional(int i=1 +intent(in)+value, long j=2 +intent(in)+value)
// function_index=56
void AA_testoptional_2(int i, long j)
{
// splicer begin function.testoptional_2
    example::nested::testoptional(i, j);
    return;
// splicer end function.testoptional_2
}

// size_t test_size_t()
// function_index=57
size_t AA_test_size_t()
{
// splicer begin function.test_size_t
    size_t SHC_rv = example::nested::test_size_t();
    return SHC_rv;
// splicer end function.test_size_t
}

// void testmpi(MPI_Comm comm +intent(in)+value)
// function_index=58
#ifdef HAVE_MPI
void AA_testmpi_mpi(MPI_Fint comm)
{
// splicer begin function.testmpi_mpi
    MPI_Comm SHCXX_comm = MPI_Comm_f2c(comm);
    example::nested::testmpi(SHCXX_comm);
    return;
// splicer end function.testmpi_mpi
}
#endif  // ifdef HAVE_MPI

// void testmpi()
// function_index=59
#ifndef HAVE_MPI
void AA_testmpi_serial()
{
// splicer begin function.testmpi_serial
    example::nested::testmpi();
    return;
// splicer end function.testmpi_serial
}
#endif  // ifndef HAVE_MPI

// void testgroup1(axom::sidre::Group * grp +intent(in)+value)
// function_index=60
void AA_testgroup1(SIDRE_group * grp)
{
// splicer begin function.testgroup1
    axom::sidre::Group * SHCXX_grp = 
        static_cast<axom::sidre::Group *>(grp->addr);
    example::nested::testgroup1(SHCXX_grp);
    return;
// splicer end function.testgroup1
}

// void testgroup2(const axom::sidre::Group * grp +intent(in)+value)
// function_index=61
void AA_testgroup2(const SIDRE_group * grp)
{
// splicer begin function.testgroup2
    const axom::sidre::Group * SHCXX_grp = 
        static_cast<const axom::sidre::Group *>(grp->addr);
    example::nested::testgroup2(SHCXX_grp);
    return;
// splicer end function.testgroup2
}

// void FuncPtr1(void ( * get)() +intent(in)+value)
// function_index=62
/**
 * \brief subroutine
 *
 */
void AA_func_ptr1(void ( * get)())
{
// splicer begin function.func_ptr1
    example::nested::FuncPtr1(get);
    return;
// splicer end function.func_ptr1
}

// void FuncPtr2(double * ( * get)() +intent(in))
// function_index=63
/**
 * \brief return a pointer
 *
 */
void AA_func_ptr2(double * ( * get)())
{
// splicer begin function.func_ptr2
    example::nested::FuncPtr2(get);
    return;
// splicer end function.func_ptr2
}

// void FuncPtr3(double ( * get)(int i +value, int +value) +intent(in)+value)
// function_index=64
/**
 * \brief abstract argument
 *
 */
void AA_func_ptr3(double ( * get)(int i, int))
{
// splicer begin function.func_ptr3
    example::nested::FuncPtr3(get);
    return;
// splicer end function.func_ptr3
}

// void FuncPtr4(double ( * get)(double +value, int +value) +intent(in)+value)
// function_index=65
/**
 * \brief abstract argument
 *
 */
void AA_func_ptr4(double ( * get)(double, int))
{
// splicer begin function.func_ptr4
    example::nested::FuncPtr4(get);
    return;
// splicer end function.func_ptr4
}

// void FuncPtr5(void ( * get)(int verylongname1 +value, int verylongname2 +value, int verylongname3 +value, int verylongname4 +value, int verylongname5 +value, int verylongname6 +value, int verylongname7 +value, int verylongname8 +value, int verylongname9 +value, int verylongname10 +value) +intent(in)+value)
// function_index=66
void AA_func_ptr5(void ( * get)(int verylongname1, int verylongname2,
    int verylongname3, int verylongname4, int verylongname5,
    int verylongname6, int verylongname7, int verylongname8,
    int verylongname9, int verylongname10))
{
// splicer begin function.func_ptr5
    example::nested::FuncPtr5(get);
    return;
// splicer end function.func_ptr5
}

// void verylongfunctionname1(int * verylongname1 +intent(inout), int * verylongname2 +intent(inout), int * verylongname3 +intent(inout), int * verylongname4 +intent(inout), int * verylongname5 +intent(inout), int * verylongname6 +intent(inout), int * verylongname7 +intent(inout), int * verylongname8 +intent(inout), int * verylongname9 +intent(inout), int * verylongname10 +intent(inout))
// function_index=67
void AA_verylongfunctionname1(int * verylongname1, int * verylongname2,
    int * verylongname3, int * verylongname4, int * verylongname5,
    int * verylongname6, int * verylongname7, int * verylongname8,
    int * verylongname9, int * verylongname10)
{
// splicer begin function.verylongfunctionname1
    example::nested::verylongfunctionname1(verylongname1, verylongname2,
        verylongname3, verylongname4, verylongname5, verylongname6,
        verylongname7, verylongname8, verylongname9, verylongname10);
    return;
// splicer end function.verylongfunctionname1
}

// int verylongfunctionname2(int verylongname1 +intent(in)+value, int verylongname2 +intent(in)+value, int verylongname3 +intent(in)+value, int verylongname4 +intent(in)+value, int verylongname5 +intent(in)+value, int verylongname6 +intent(in)+value, int verylongname7 +intent(in)+value, int verylongname8 +intent(in)+value, int verylongname9 +intent(in)+value, int verylongname10 +intent(in)+value)
// function_index=68
int AA_verylongfunctionname2(int verylongname1, int verylongname2,
    int verylongname3, int verylongname4, int verylongname5,
    int verylongname6, int verylongname7, int verylongname8,
    int verylongname9, int verylongname10)
{
// splicer begin function.verylongfunctionname2
    int SHC_rv = example::nested::verylongfunctionname2(verylongname1,
        verylongname2, verylongname3, verylongname4, verylongname5,
        verylongname6, verylongname7, verylongname8, verylongname9,
        verylongname10);
    return SHC_rv;
// splicer end function.verylongfunctionname2
}

// void cos_doubles(double * in +dimension(:,:)+intent(in), double * out +allocatable(mold=in)+dimension(:,:)+intent(out), int sizein +implied(size(in))+intent(in)+value)
// function_index=69
/**
 * \brief Test multidimensional arrays with allocatable
 *
 */
void AA_cos_doubles(double * in, double * out, int sizein)
{
// splicer begin function.cos_doubles
    example::nested::cos_doubles(in, out, sizein);
    return;
// splicer end function.cos_doubles
}

// Release C++ allocated memory if refcount reaches 0.
void AA_SHROUD_array_destructor_function
    (SHROUD_capsule_data *cap, bool gc)
{
    void *ptr = cap->addr;
    switch (cap->idtor) {
    case 0:
    {
        // Nothing to delete
        break;
    }
    case 1:
    {
        example::nested::ExClass1 *cxx_ptr = 
            reinterpret_cast<example::nested::ExClass1 *>(ptr);
        delete cxx_ptr;
        break;
    }
    case 2:
    {
        example::nested::ExClass2 *cxx_ptr = 
            reinterpret_cast<example::nested::ExClass2 *>(ptr);
        delete cxx_ptr;
        break;
    }
    default:
    {
        // Unexpected case in destructor
        break;
    }
    }
    if (gc) {
        free(cap);
    } else {
        cap->addr = NULL;
        cap->idtor = 0;  // avoid deleting again
    }
}

}  // extern "C"
