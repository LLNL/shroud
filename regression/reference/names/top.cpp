// top.cpp
// This is generated code, do not edit
// #######################################################################
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
#include "top.h"
#include <cstdlib>
#include <string>
#include "typestestnames.hh"

// splicer begin CXX_definitions
// Add some text from splicer
// And another line
// splicer end CXX_definitions

extern "C" {

// splicer begin C_definitions
// splicer end C_definitions

// void function1()
void YYY_TES_function1()
{
    // splicer begin function.function1
    function1();
    return;
    // splicer end function.function1
}

// void function2()
void c_name_special()
{
    // splicer begin function.function2
    function2();
    return;
    // splicer end function.function2
}

// void function3a(int i +intent(in)+value)
void YYY_TES_function3a_0(int i)
{
    // splicer begin function.function3a_0
    function3a(i);
    return;
    // splicer end function.function3a_0
}

// void function3a(long i +intent(in)+value)
void YYY_TES_function3a_1(long i)
{
    // splicer begin function.function3a_1
    function3a(i);
    return;
    // splicer end function.function3a_1
}

// int function4(const std::string & rv +intent(in))
int YYY_TES_function4(const char * rv)
{
    // splicer begin function.function4
    const std::string ARG_rv(rv);
    int SHC_rv = function4(ARG_rv);
    return SHC_rv;
    // splicer end function.function4
}

// int function4(const std::string & rv +intent(in)+len_trim(Lrv))
int YYY_TES_function4_bufferify(const char * rv, int Lrv)
{
    // splicer begin function.function4_bufferify
    const std::string ARG_rv(rv, Lrv);
    int SHC_rv = function4(ARG_rv);
    return SHC_rv;
    // splicer end function.function4_bufferify
}

// void function5() +name(fiveplus)
void YYY_TES_fiveplus()
{
    // splicer begin function.fiveplus
    fiveplus();
    return;
    // splicer end function.fiveplus
}

// void FunctionTU(int arg1 +intent(in)+value, long arg2 +intent(in)+value)
/**
 * \brief Function template with two template parameters.
 *
 */
void c_name_instantiation1(int arg1, long arg2)
{
    // splicer begin function.function_tu_0
    FunctionTU<int, long>(arg1, arg2);
    return;
    // splicer end function.function_tu_0
}

// void FunctionTU(float arg1 +intent(in)+value, double arg2 +intent(in)+value)
/**
 * \brief Function template with two template parameters.
 *
 */
void TES_function_tu_instantiation2(float arg1, double arg2)
{
    // splicer begin function.function_tu_instantiation2
    FunctionTU<float, double>(arg1, arg2);
    return;
    // splicer end function.function_tu_instantiation2
}

// int UseImplWorker()
/**
 * \brief Function which uses a templated T in the implemetation.
 *
 */
int TES_use_impl_worker_instantiation3()
{
    // splicer begin function.use_impl_worker_instantiation3
    int SHC_rv = UseImplWorker<internal::ImplWorker1>();
    return SHC_rv;
    // splicer end function.use_impl_worker_instantiation3
}

// Release library allocated memory.
void TES_SHROUD_memory_destructor(TES_SHROUD_capsule_data *cap)
{
    cap->addr = NULL;
    cap->idtor = 0;  // avoid deleting again
}

}  // extern "C"
