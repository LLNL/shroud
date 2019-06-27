// wrapstruct.cpp
// This is generated code, do not edit
// Copyright (c) 2017-2019, Lawrence Livermore National Security, LLC and
// other Shroud Project Developers.
// See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//
#include "wrapstruct.h"
#include <cstring>
#include <stdlib.h>
#include "struct.h"
#include "typesstruct.h"

// splicer begin CXX_definitions
// splicer end CXX_definitions

extern "C" {


typedef union {
  Cstruct1 cxx;
  STR_cstruct1 c;
} SH_union_0_t;


// helper function
// blank fill dest starting at trailing NULL.
static void ShroudStrBlankFill(char *dest, int ndest)
{
   int nm = std::strlen(dest);
   if(ndest > nm) std::memset(dest+nm,' ',ndest-nm);
}
// splicer begin C_definitions
// splicer end C_definitions

// int passStructByValue(Cstruct1 arg +intent(in)+value)
int STR_pass_struct_by_value(STR_cstruct1 arg)
{
// splicer begin function.pass_struct_by_value
    Cstruct1 * SHCXX_arg = static_cast<Cstruct1 *>(static_cast<void *>(
        &arg));
    int SHC_rv = passStructByValue(*SHCXX_arg);
    return SHC_rv;
// splicer end function.pass_struct_by_value
}

// int passStruct1(Cstruct1 * arg +intent(in))
int STR_pass_struct1(STR_cstruct1 * arg)
{
// splicer begin function.pass_struct1
    Cstruct1 * SHCXX_arg = static_cast<Cstruct1 *>(static_cast<void *>(
        arg));
    int SHC_rv = passStruct1(SHCXX_arg);
    return SHC_rv;
// splicer end function.pass_struct1
}

// int passStruct2(Cstruct1 * s1 +intent(in), char * outbuf +charlen(LENOUTBUF)+intent(out))
/**
 * Pass name argument which will build a bufferify function.
 */
int STR_pass_struct2(STR_cstruct1 * s1, char * outbuf)
{
// splicer begin function.pass_struct2
    Cstruct1 * SHCXX_s1 = static_cast<Cstruct1 *>(static_cast<void *>(
        s1));
    int SHC_rv = passStruct2(SHCXX_s1, outbuf);
    return SHC_rv;
// splicer end function.pass_struct2
}

// int passStruct2(Cstruct1 * s1 +intent(in), char * outbuf +charlen(LENOUTBUF)+intent(out)+len(Noutbuf))
/**
 * Pass name argument which will build a bufferify function.
 */
int STR_pass_struct2_bufferify(STR_cstruct1 * s1, char * outbuf,
    int Noutbuf)
{
// splicer begin function.pass_struct2_bufferify
    Cstruct1 * SHCXX_s1 = static_cast<Cstruct1 *>(static_cast<void *>(
        s1));
    int SHC_rv = passStruct2(SHCXX_s1, outbuf);
    ShroudStrBlankFill(outbuf, Noutbuf);
    return SHC_rv;
// splicer end function.pass_struct2_bufferify
}

// int acceptStructInPtr(Cstruct1 * arg +intent(in))
int STR_accept_struct_in_ptr(STR_cstruct1 * arg)
{
// splicer begin function.accept_struct_in_ptr
    Cstruct1 * SHCXX_arg = static_cast<Cstruct1 *>(static_cast<void *>(
        arg));
    int SHC_rv = acceptStructInPtr(SHCXX_arg);
    return SHC_rv;
// splicer end function.accept_struct_in_ptr
}

// void acceptStructOutPtr(Cstruct1 * arg +intent(out), int i +intent(in)+value, double d +intent(in)+value)
/**
 * Pass name argument which will build a bufferify function.
 */
void STR_accept_struct_out_ptr(STR_cstruct1 * arg, int i, double d)
{
// splicer begin function.accept_struct_out_ptr
    Cstruct1 * SHCXX_arg = static_cast<Cstruct1 *>(static_cast<void *>(
        arg));
    acceptStructOutPtr(SHCXX_arg, i, d);
    return;
// splicer end function.accept_struct_out_ptr
}

// void acceptStructInOutPtr(Cstruct1 * arg +intent(inout))
void STR_accept_struct_in_out_ptr(STR_cstruct1 * arg)
{
// splicer begin function.accept_struct_in_out_ptr
    Cstruct1 * SHCXX_arg = static_cast<Cstruct1 *>(static_cast<void *>(
        arg));
    acceptStructInOutPtr(SHCXX_arg);
    return;
// splicer end function.accept_struct_in_out_ptr
}

// Cstruct1 returnStructByValue(int i +intent(in)+value, double d +intent(in)+value)
STR_cstruct1 STR_return_struct_by_value(int i, double d)
{
// splicer begin function.return_struct_by_value
    SH_union_0_t SHC_rv = {returnStructByValue(i, d)};
    return SHC_rv.c;
// splicer end function.return_struct_by_value
}

// Cstruct1 * returnStructPtr1(int i +intent(in)+value, double d +intent(in)+value)
/**
 * \brief Return a pointer to a struct
 *
 * Does not generate a bufferify C wrapper.
 */
STR_cstruct1 * STR_return_struct_ptr1(int i, double d)
{
// splicer begin function.return_struct_ptr1
    Cstruct1 * SHCXX_rv = returnStructPtr1(i, d);
    STR_cstruct1 * SHC_rv = static_cast<STR_cstruct1 *>(
        static_cast<void *>(SHCXX_rv));
    return SHC_rv;
// splicer end function.return_struct_ptr1
}

// Cstruct1 * returnStructPtr2(int i +intent(in)+value, double d +intent(in)+value, char * outbuf +intent(out))
/**
 * \brief Return a pointer to a struct
 *
 * Generates a bufferify C wrapper function.
 */
STR_cstruct1 * STR_return_struct_ptr2(int i, double d, char * outbuf)
{
// splicer begin function.return_struct_ptr2
    Cstruct1 * SHCXX_rv = returnStructPtr2(i, d, outbuf);
    STR_cstruct1 * SHC_rv = static_cast<STR_cstruct1 *>(
        static_cast<void *>(SHCXX_rv));
    return SHC_rv;
// splicer end function.return_struct_ptr2
}

// Cstruct1 * returnStructPtr2(int i +intent(in)+value, double d +intent(in)+value, char * outbuf +intent(out)+len(Noutbuf))
/**
 * \brief Return a pointer to a struct
 *
 * Generates a bufferify C wrapper function.
 */
STR_cstruct1 * STR_return_struct_ptr2_bufferify(int i, double d,
    char * outbuf, int Noutbuf)
{
// splicer begin function.return_struct_ptr2_bufferify
    Cstruct1 * SHCXX_rv = returnStructPtr2(i, d, outbuf);
    ShroudStrBlankFill(outbuf, Noutbuf);
    STR_cstruct1 * SHC_rv = static_cast<STR_cstruct1 *>(
        static_cast<void *>(SHCXX_rv));
    return SHC_rv;
// splicer end function.return_struct_ptr2_bufferify
}

// Release C++ allocated memory.
void STR_SHROUD_memory_destructor(STR_SHROUD_capsule_data *cap)
{
    cap->addr = NULL;
    cap->idtor = 0;  // avoid deleting again
}

}  // extern "C"
