// wrapTutorial.cpp
// This is generated code, do not edit
// #######################################################################
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
#include "wrapTutorial.h"
#include <cstddef>
#include <cstring>
#include <stdlib.h>
#include <string>
#include "tutorial.hpp"
#include "typesTutorial.h"

// splicer begin CXX_definitions
// splicer end CXX_definitions

extern "C" {


typedef union {
  tutorial::struct1 cxx;
  TUT_struct1 c;
} SH_union_0_t;


// helper function
// Copy src into dest, blank fill to ndest characters
// Truncate if dest is too short.
// dest will not be NULL terminated.
static void ShroudStrCopy(char *dest, int ndest, const char *src, int nsrc)
{
   int nm = nsrc < ndest ? nsrc : ndest;
   std::memcpy(dest,src,nm);
   if(ndest > nm) std::memset(dest+nm,' ',ndest-nm);
}

// helper function
// Copy the char* or std::string in context into c_var.
// Called by Fortran to deal with allocatable character.
void TUT_ShroudCopyStringAndFree(TUT_SHROUD_array *data, char *c_var, size_t c_var_len) {
    const char *cxx_var = data->addr.ccharp;
    size_t n = c_var_len;
    if (data->len < n) n = data->len;
    strncpy(c_var, cxx_var, n);
    TUT_SHROUD_memory_destructor(&data->cxx); // delete data->cxx.addr
}

// splicer begin C_definitions
// splicer end C_definitions

// void Function1()
void TUT_function1()
{
// splicer begin function.function1
    tutorial::Function1();
    return;
// splicer end function.function1
}

// double Function2(double arg1 +intent(in)+value, int arg2 +intent(in)+value)
double TUT_function2(double arg1, int arg2)
{
// splicer begin function.function2
    double SHC_rv = tutorial::Function2(arg1, arg2);
    return SHC_rv;
// splicer end function.function2
}

// void Sum(size_t len +implied(size(values))+intent(in)+value, int * values +dimension(:)+intent(in), int * result +intent(out))
void TUT_sum(size_t len, int * values, int * result)
{
// splicer begin function.sum
    tutorial::Sum(len, values, result);
    return;
// splicer end function.sum
}

// long long TypeLongLong(long long arg1 +intent(in)+value)
long long TUT_type_long_long(long long arg1)
{
// splicer begin function.type_long_long
    long long SHC_rv = tutorial::TypeLongLong(arg1);
    return SHC_rv;
// splicer end function.type_long_long
}

// bool Function3(bool arg +intent(in)+value)
bool TUT_function3(bool arg)
{
// splicer begin function.function3
    bool SHC_rv = tutorial::Function3(arg);
    return SHC_rv;
// splicer end function.function3
}

// void Function3b(const bool arg1 +intent(in)+value, bool * arg2 +intent(out), bool * arg3 +intent(inout))
void TUT_function3b(const bool arg1, bool * arg2, bool * arg3)
{
// splicer begin function.function3b
    tutorial::Function3b(arg1, arg2, arg3);
    return;
// splicer end function.function3b
}

// void Function4a(const std::string & arg1 +intent(in)+len_trim(Larg1), const std::string & arg2 +intent(in)+len_trim(Larg2), std::string * SHF_rv +intent(out)+len(NSHF_rv)) +len(30)
/**
 * Since +len(30) is provided, the result of the function
 * will be copied directly into memory provided by Fortran.
 * The function will not be ALLOCATABLE.
 */
void TUT_function4a_bufferify(const char * arg1, int Larg1,
    const char * arg2, int Larg2, char * SHF_rv, int NSHF_rv)
{
// splicer begin function.function4a_bufferify
    const std::string SH_arg1(arg1, Larg1);
    const std::string SH_arg2(arg2, Larg2);
    const std::string SHCXX_rv = tutorial::Function4a(SH_arg1, SH_arg2);
    if (SHCXX_rv.empty()) {
        std::memset(SHF_rv, ' ', NSHF_rv);
    } else {
        ShroudStrCopy(SHF_rv, NSHF_rv, SHCXX_rv.data(),
            SHCXX_rv.size());
    }
    return;
// splicer end function.function4a_bufferify
}

// const std::string & Function4b(const std::string & arg1 +intent(in), const std::string & arg2 +intent(in)) +deref(result_as_arg)
const char * TUT_function4b(const char * arg1, const char * arg2)
{
// splicer begin function.function4b
    const std::string SH_arg1(arg1);
    const std::string SH_arg2(arg2);
    const std::string & SHCXX_rv = tutorial::Function4b(SH_arg1,
        SH_arg2);
    const char * SHC_rv = SHCXX_rv.c_str();
    return SHC_rv;
// splicer end function.function4b
}

// void Function4b(const std::string & arg1 +intent(in)+len_trim(Larg1), const std::string & arg2 +intent(in)+len_trim(Larg2), std::string & output +intent(out)+len(Noutput))
void TUT_function4b_bufferify(const char * arg1, int Larg1,
    const char * arg2, int Larg2, char * output, int Noutput)
{
// splicer begin function.function4b_bufferify
    const std::string SH_arg1(arg1, Larg1);
    const std::string SH_arg2(arg2, Larg2);
    const std::string & SHCXX_rv = tutorial::Function4b(SH_arg1,
        SH_arg2);
    if (SHCXX_rv.empty()) {
        std::memset(output, ' ', Noutput);
    } else {
        ShroudStrCopy(output, Noutput, SHCXX_rv.data(),
            SHCXX_rv.size());
    }
    return;
// splicer end function.function4b_bufferify
}

// const std::string & Function4c(const std::string & arg1 +intent(in), const std::string & arg2 +intent(in)) +deref(allocatable)
/**
 * Note that since a reference is returned, no intermediate string
 * is allocated.  It is assumed +owner(library).
 */
const char * TUT_function4c(const char * arg1, const char * arg2)
{
// splicer begin function.function4c
    const std::string SH_arg1(arg1);
    const std::string SH_arg2(arg2);
    const std::string & SHCXX_rv = tutorial::Function4c(SH_arg1,
        SH_arg2);
    const char * SHC_rv = SHCXX_rv.c_str();
    return SHC_rv;
// splicer end function.function4c
}

// void Function4c(const std::string & arg1 +intent(in)+len_trim(Larg1), const std::string & arg2 +intent(in)+len_trim(Larg2), const stringout * SHF_rv +context(DSHF_rv)+deref(allocatable)+intent(out))
/**
 * Note that since a reference is returned, no intermediate string
 * is allocated.  It is assumed +owner(library).
 */
void TUT_function4c_bufferify(const char * arg1, int Larg1,
    const char * arg2, int Larg2, TUT_SHROUD_array *DSHF_rv)
{
// splicer begin function.function4c_bufferify
    const std::string SH_arg1(arg1, Larg1);
    const std::string SH_arg2(arg2, Larg2);
    const std::string & SHCXX_rv = tutorial::Function4c(SH_arg1,
        SH_arg2);
    DSHF_rv->cxx.addr = static_cast<void *>(const_cast<std::string *>
        (&SHCXX_rv));
    DSHF_rv->cxx.idtor = 0;
    if (SHCXX_rv.empty()) {
        DSHF_rv->addr.ccharp = NULL;
        DSHF_rv->len = 0;
    } else {
        DSHF_rv->addr.ccharp = SHCXX_rv.data();
        DSHF_rv->len = SHCXX_rv.size();
    }
    DSHF_rv->size = 1;
    return;
// splicer end function.function4c_bufferify
}

// const std::string * Function4d() +deref(allocatable)+owner(caller)
/**
 * A string is allocated by the library is must be deleted
 * by the caller.
 */
const char * TUT_function4d()
{
// splicer begin function.function4d
    const std::string * SHCXX_rv = tutorial::Function4d();
    const char * SHC_rv = SHCXX_rv->c_str();
    return SHC_rv;
// splicer end function.function4d
}

// void Function4d(const stringout * SHF_rv +context(DSHF_rv)+deref(allocatable)+intent(out)+owner(caller))
/**
 * A string is allocated by the library is must be deleted
 * by the caller.
 */
void TUT_function4d_bufferify(TUT_SHROUD_array *DSHF_rv)
{
// splicer begin function.function4d_bufferify
    const std::string * SHCXX_rv = tutorial::Function4d();
    DSHF_rv->cxx.addr = static_cast<void *>(const_cast<std::string *>
        (SHCXX_rv));
    DSHF_rv->cxx.idtor = 2;
    if (SHCXX_rv->empty()) {
        DSHF_rv->addr.ccharp = NULL;
        DSHF_rv->len = 0;
    } else {
        DSHF_rv->addr.ccharp = SHCXX_rv->data();
        DSHF_rv->len = SHCXX_rv->size();
    }
    DSHF_rv->size = 1;
    return;
// splicer end function.function4d_bufferify
}

// double Function5()
double TUT_function5()
{
// splicer begin function.function5
    double SHC_rv = tutorial::Function5();
    return SHC_rv;
// splicer end function.function5
}

// double Function5(double arg1=3.1415 +intent(in)+value)
double TUT_function5_arg1(double arg1)
{
// splicer begin function.function5_arg1
    double SHC_rv = tutorial::Function5(arg1);
    return SHC_rv;
// splicer end function.function5_arg1
}

// double Function5(double arg1=3.1415 +intent(in)+value, bool arg2=true +intent(in)+value)
double TUT_function5_arg1_arg2(double arg1, bool arg2)
{
// splicer begin function.function5_arg1_arg2
    double SHC_rv = tutorial::Function5(arg1, arg2);
    return SHC_rv;
// splicer end function.function5_arg1_arg2
}

// void Function6(const std::string & name +intent(in))
void TUT_function6_from_name(const char * name)
{
// splicer begin function.function6_from_name
    const std::string SH_name(name);
    tutorial::Function6(SH_name);
    return;
// splicer end function.function6_from_name
}

// void Function6(const std::string & name +intent(in)+len_trim(Lname))
void TUT_function6_from_name_bufferify(const char * name, int Lname)
{
// splicer begin function.function6_from_name_bufferify
    const std::string SH_name(name, Lname);
    tutorial::Function6(SH_name);
    return;
// splicer end function.function6_from_name_bufferify
}

// void Function6(int indx +intent(in)+value)
void TUT_function6_from_index(int indx)
{
// splicer begin function.function6_from_index
    tutorial::Function6(indx);
    return;
// splicer end function.function6_from_index
}

// void Function7(int arg +intent(in)+value)
void TUT_function7_int(int arg)
{
// splicer begin function.function7_int
    tutorial::Function7<int>(arg);
    return;
// splicer end function.function7_int
}

// void Function7(double arg +intent(in)+value)
void TUT_function7_double(double arg)
{
// splicer begin function.function7_double
    tutorial::Function7<double>(arg);
    return;
// splicer end function.function7_double
}

// int Function8()
int TUT_function8_int()
{
// splicer begin function.function8_int
    int SHC_rv = tutorial::Function8<int>();
    return SHC_rv;
// splicer end function.function8_int
}

// double Function8()
double TUT_function8_double()
{
// splicer begin function.function8_double
    double SHC_rv = tutorial::Function8<double>();
    return SHC_rv;
// splicer end function.function8_double
}

// void Function9(double arg +intent(in)+value)
void TUT_function9(double arg)
{
// splicer begin function.function9
    tutorial::Function9(arg);
    return;
// splicer end function.function9
}

// void Function10()
void TUT_function10_0()
{
// splicer begin function.function10_0
    tutorial::Function10();
    return;
// splicer end function.function10_0
}

// void Function10(const std::string & name +intent(in), double arg2 +intent(in)+value)
void TUT_function10_1(const char * name, double arg2)
{
// splicer begin function.function10_1
    const std::string SH_name(name);
    tutorial::Function10(SH_name, arg2);
    return;
// splicer end function.function10_1
}

// void Function10(const std::string & name +intent(in)+len_trim(Lname), double arg2 +intent(in)+value)
void TUT_function10_1_bufferify(const char * name, int Lname,
    double arg2)
{
// splicer begin function.function10_1_bufferify
    const std::string SH_name(name, Lname);
    tutorial::Function10(SH_name, arg2);
    return;
// splicer end function.function10_1_bufferify
}

// int overload1(int num +intent(in)+value)
int TUT_overload1_num(int num)
{
// splicer begin function.overload1_num
    int SHC_rv = tutorial::overload1(num);
    return SHC_rv;
// splicer end function.overload1_num
}

// int overload1(int num +intent(in)+value, int offset=0 +intent(in)+value)
int TUT_overload1_num_offset(int num, int offset)
{
// splicer begin function.overload1_num_offset
    int SHC_rv = tutorial::overload1(num, offset);
    return SHC_rv;
// splicer end function.overload1_num_offset
}

// int overload1(int num +intent(in)+value, int offset=0 +intent(in)+value, int stride=1 +intent(in)+value)
int TUT_overload1_num_offset_stride(int num, int offset, int stride)
{
// splicer begin function.overload1_num_offset_stride
    int SHC_rv = tutorial::overload1(num, offset, stride);
    return SHC_rv;
// splicer end function.overload1_num_offset_stride
}

// int overload1(double type +intent(in)+value, int num +intent(in)+value)
int TUT_overload1_3(double type, int num)
{
// splicer begin function.overload1_3
    int SHC_rv = tutorial::overload1(type, num);
    return SHC_rv;
// splicer end function.overload1_3
}

// int overload1(double type +intent(in)+value, int num +intent(in)+value, int offset=0 +intent(in)+value)
int TUT_overload1_4(double type, int num, int offset)
{
// splicer begin function.overload1_4
    int SHC_rv = tutorial::overload1(type, num, offset);
    return SHC_rv;
// splicer end function.overload1_4
}

// int overload1(double type +intent(in)+value, int num +intent(in)+value, int offset=0 +intent(in)+value, int stride=1 +intent(in)+value)
int TUT_overload1_5(double type, int num, int offset, int stride)
{
// splicer begin function.overload1_5
    int SHC_rv = tutorial::overload1(type, num, offset, stride);
    return SHC_rv;
// splicer end function.overload1_5
}

// TypeID typefunc(TypeID arg +intent(in)+value)
int TUT_typefunc(int arg)
{
// splicer begin function.typefunc
    tutorial::TypeID SHC_rv = tutorial::typefunc(arg);
    return SHC_rv;
// splicer end function.typefunc
}

// EnumTypeID enumfunc(EnumTypeID arg +intent(in)+value)
int TUT_enumfunc(int arg)
{
// splicer begin function.enumfunc
    tutorial::EnumTypeID SHCXX_arg =
        static_cast<tutorial::EnumTypeID>(arg);
    tutorial::EnumTypeID SHCXX_rv = tutorial::enumfunc(SHCXX_arg);
    int SHC_rv = static_cast<int>(SHCXX_rv);
    return SHC_rv;
// splicer end function.enumfunc
}

// Color colorfunc(Color arg +intent(in)+value)
int TUT_colorfunc(int arg)
{
// splicer begin function.colorfunc
    tutorial::Color SHCXX_arg = static_cast<tutorial::Color>(arg);
    tutorial::Color SHCXX_rv = tutorial::colorfunc(SHCXX_arg);
    int SHC_rv = static_cast<int>(SHCXX_rv);
    return SHC_rv;
// splicer end function.colorfunc
}

// void getMinMax(int & min +intent(out), int & max +intent(out))
/**
 * \brief Pass in reference to scalar
 *
 */
void TUT_get_min_max(int * min, int * max)
{
// splicer begin function.get_min_max
    tutorial::getMinMax(*min, *max);
    return;
// splicer end function.get_min_max
}

// Class1::DIRECTION directionFunc(Class1::DIRECTION arg +intent(in)+value)
int TUT_direction_func(int arg)
{
// splicer begin function.direction_func
    tutorial::Class1::DIRECTION SHCXX_arg =
        static_cast<tutorial::Class1::DIRECTION>(arg);
    tutorial::Class1::DIRECTION SHCXX_rv = tutorial::directionFunc(
        SHCXX_arg);
    int SHC_rv = static_cast<int>(SHCXX_rv);
    return SHC_rv;
// splicer end function.direction_func
}

// int useclass(const Class1 * arg1 +intent(in))
int TUT_useclass(const TUT_class1 * arg1)
{
// splicer begin function.useclass
    const tutorial::Class1 * SHCXX_arg1 =
        static_cast<const tutorial::Class1 *>(arg1->addr);
    int SHC_rv = tutorial::useclass(SHCXX_arg1);
    return SHC_rv;
// splicer end function.useclass
}

// const Class1 * getclass2()
TUT_class1 TUT_getclass2()
{
// splicer begin function.getclass2
    const tutorial::Class1 * SHCXX_rv = tutorial::getclass2();
    TUT_class1 SHC_rv;
    SHC_rv.addr = static_cast<void *>(const_cast<tutorial::Class1 *>
        (SHCXX_rv));
    SHC_rv.idtor = 0;
    return SHC_rv;
// splicer end function.getclass2
}

// Class1 * getclass3()
TUT_class1 TUT_getclass3()
{
// splicer begin function.getclass3
    tutorial::Class1 * SHCXX_rv = tutorial::getclass3();
    TUT_class1 SHC_rv;
    SHC_rv.addr = static_cast<void *>(SHCXX_rv);
    SHC_rv.idtor = 0;
    return SHC_rv;
// splicer end function.getclass3
}

// Class1 getClassCopy(int flag +intent(in)+value)
/**
 * \brief Return Class1 instance by value, uses copy constructor
 *
 */
TUT_class1 TUT_get_class_copy(int flag)
{
// splicer begin function.get_class_copy
    tutorial::Class1 * SHCXX_rv = new tutorial::Class1;
    *SHCXX_rv = tutorial::getClassCopy(flag);
    TUT_class1 SHC_rv;
    SHC_rv.addr = static_cast<void *>(SHCXX_rv);
    SHC_rv.idtor = 1;
    return SHC_rv;
// splicer end function.get_class_copy
}

// int callback1(int in +intent(in)+value, int ( * incr)(int +value) +intent(in)+value)
int TUT_callback1(int in, int ( * incr)(int))
{
// splicer begin function.callback1
    int SHC_rv = tutorial::callback1(in, incr);
    return SHC_rv;
// splicer end function.callback1
}

// struct1 returnStruct(int i +intent(in)+value, double d +intent(in)+value)
TUT_struct1 TUT_return_struct(int i, double d)
{
// splicer begin function.return_struct
    SH_union_0_t SHC_rv = {tutorial::returnStruct(i, d)};
    return SHC_rv.c;
// splicer end function.return_struct
}

// struct1 * returnStructPtr(int i +intent(in)+value, double d +intent(in)+value)
TUT_struct1 * TUT_return_struct_ptr(int i, double d)
{
// splicer begin function.return_struct_ptr
    tutorial::struct1 * SHCXX_rv = tutorial::returnStructPtr(i, d);
    TUT_struct1 * SHC_rv = static_cast<TUT_struct1 *>(
        static_cast<void *>(SHCXX_rv));
    return SHC_rv;
// splicer end function.return_struct_ptr
}

// double acceptStructIn(struct1 arg +intent(in)+value)
double TUT_accept_struct_in(TUT_struct1 arg)
{
// splicer begin function.accept_struct_in
    tutorial::struct1 * SHCXX_arg = static_cast<tutorial::struct1 *>(
        static_cast<void *>(&arg));
    double SHC_rv = tutorial::acceptStructIn(*SHCXX_arg);
    return SHC_rv;
// splicer end function.accept_struct_in
}

// double acceptStructInPtr(struct1 * arg +intent(in))
double TUT_accept_struct_in_ptr(TUT_struct1 * arg)
{
// splicer begin function.accept_struct_in_ptr
    tutorial::struct1 * SHCXX_arg = static_cast<tutorial::struct1 *>(
        static_cast<void *>(arg));
    double SHC_rv = tutorial::acceptStructInPtr(SHCXX_arg);
    return SHC_rv;
// splicer end function.accept_struct_in_ptr
}

// void acceptStructOutPtr(struct1 * arg +intent(out), int i +intent(in)+value, double d +intent(in)+value)
void TUT_accept_struct_out_ptr(TUT_struct1 * arg, int i, double d)
{
// splicer begin function.accept_struct_out_ptr
    tutorial::struct1 * SHCXX_arg = static_cast<tutorial::struct1 *>(
        static_cast<void *>(arg));
    tutorial::acceptStructOutPtr(SHCXX_arg, i, d);
    return;
// splicer end function.accept_struct_out_ptr
}

// void acceptStructInOutPtr(struct1 * arg +intent(inout))
void TUT_accept_struct_in_out_ptr(TUT_struct1 * arg)
{
// splicer begin function.accept_struct_in_out_ptr
    tutorial::struct1 * SHCXX_arg = static_cast<tutorial::struct1 *>(
        static_cast<void *>(arg));
    tutorial::acceptStructInOutPtr(SHCXX_arg);
    return;
// splicer end function.accept_struct_in_out_ptr
}

// const std::string & LastFunctionCalled() +deref(result_as_arg)+len(30)
const char * TUT_last_function_called()
{
// splicer begin function.last_function_called
    const std::string & SHCXX_rv = tutorial::LastFunctionCalled();
    const char * SHC_rv = SHCXX_rv.c_str();
    return SHC_rv;
// splicer end function.last_function_called
}

// void LastFunctionCalled(std::string & SHF_rv +intent(out)+len(NSHF_rv)) +len(30)
void TUT_last_function_called_bufferify(char * SHF_rv, int NSHF_rv)
{
// splicer begin function.last_function_called_bufferify
    const std::string & SHCXX_rv = tutorial::LastFunctionCalled();
    if (SHCXX_rv.empty()) {
        std::memset(SHF_rv, ' ', NSHF_rv);
    } else {
        ShroudStrCopy(SHF_rv, NSHF_rv, SHCXX_rv.data(),
            SHCXX_rv.size());
    }
    return;
// splicer end function.last_function_called_bufferify
}

// Release C++ allocated memory.
void TUT_SHROUD_memory_destructor(TUT_SHROUD_capsule_data *cap)
{
    void *ptr = cap->addr;
    switch (cap->idtor) {
    case 0:   // --none--
    {
        // Nothing to delete
        break;
    }
    case 1:   // tutorial::Class1
    {
        tutorial::Class1 *cxx_ptr = 
            reinterpret_cast<tutorial::Class1 *>(ptr);
        delete cxx_ptr;
        break;
    }
    case 2:   // std::string
    {
        std::string *cxx_ptr = reinterpret_cast<std::string *>(ptr);
        delete cxx_ptr;
        break;
    }
    default:
    {
        // Unexpected case in destructor
        break;
    }
    }
    cap->addr = NULL;
    cap->idtor = 0;  // avoid deleting again
}

}  // extern "C"
