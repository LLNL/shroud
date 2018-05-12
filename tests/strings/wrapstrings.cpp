// wrapstrings.cpp
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
#include "wrapstrings.h"
#include <cstring>
#include <stdlib.h>
#include <string>
#include "strings.hpp"

// splicer begin CXX_definitions
// splicer end CXX_definitions

extern "C" {


// Copy s into a, blank fill to la characters
// Truncate if a is too short.
static void ShroudStrCopy(char *a, int la, const char *s)
{
   int ls,nm;
   ls = std::strlen(s);
   nm = ls < la ? ls : la;
   std::memcpy(a,s,nm);
   if(la > nm) std::memset(a+nm,' ',la-nm);
}

// Copy the std::string in context into c_var.
// Called by Fortran to deal with allocatable character.
void STR_ShroudStringCopyAndFree(STR_SHROUD_vector_context *data, char *c_var, long c_var_len) {
    std::string * cxxstr = static_cast<std::string *>(data->cxx);

    strncpy(c_var, cxxstr->data(), cxxstr->size());
    // free the string?
}

// splicer begin C_definitions
// splicer end C_definitions

// void passChar(char_scalar status +intent(in)+value)
// function_index=0
/**
 * \brief pass a single char argument as a scalar.
 *
 */
void STR_pass_char(char status)
{
// splicer begin function.pass_char
    passChar(status);
    return;
// splicer end function.pass_char
}

// char_scalar returnChar()
// function_index=1
/**
 * \brief return a char argument (non-pointer)
 *
 */
char STR_return_char()
{
// splicer begin function.return_char
    char SHC_rv = returnChar();
    return SHC_rv;
// splicer end function.return_char
}

// void returnChar(char_scalar * SHF_rv +intent(out)+len(NSHF_rv))
// function_index=29
/**
 * \brief return a char argument (non-pointer)
 *
 */
void STR_return_char_bufferify(char * SHF_rv, int NSHF_rv)
{
// splicer begin function.return_char_bufferify
    char SHC_rv = returnChar();
    std::memset(SHF_rv, ' ', NSHF_rv);
    SHF_rv[0] = SHC_rv;
    return;
// splicer end function.return_char_bufferify
}

// void passCharPtr(char * dest +intent(out), const char * src +intent(in))
// function_index=2
/**
 * \brief strcpy like behavior
 *
 * dest is marked intent(OUT) to override the intent(INOUT) default
 * This avoid a copy-in on dest.
 */
void STR_pass_char_ptr(char * dest, const char * src)
{
// splicer begin function.pass_char_ptr
    passCharPtr(dest, src);
    return;
// splicer end function.pass_char_ptr
}

// void passCharPtr(char * dest +intent(out)+len(Ndest), const char * src +intent(in)+len_trim(Lsrc))
// function_index=30
/**
 * \brief strcpy like behavior
 *
 * dest is marked intent(OUT) to override the intent(INOUT) default
 * This avoid a copy-in on dest.
 */
void STR_pass_char_ptr_bufferify(char * dest, int Ndest,
    const char * src, int Lsrc)
{
// splicer begin function.pass_char_ptr_bufferify
    char * SH_dest = (char *) malloc(Ndest + 1);
    char * SH_src = (char *) malloc(Lsrc + 1);
    std::memcpy(SH_src, src, Lsrc);
    SH_src[Lsrc] = '\0';
    passCharPtr(SH_dest, SH_src);
    ShroudStrCopy(dest, Ndest, SH_dest);
    free(SH_dest);
    free(SH_src);
    return;
// splicer end function.pass_char_ptr_bufferify
}

// void passCharPtrInOut(char * s +intent(inout))
// function_index=3
/**
 * \brief toupper
 *
 * Change a string in-place.
 * For Python, return a new string since strings are immutable.
 */
void STR_pass_char_ptr_in_out(char * s)
{
// splicer begin function.pass_char_ptr_in_out
    passCharPtrInOut(s);
    return;
// splicer end function.pass_char_ptr_in_out
}

// void passCharPtrInOut(char * s +intent(inout)+len(Ns)+len_trim(Ls))
// function_index=31
/**
 * \brief toupper
 *
 * Change a string in-place.
 * For Python, return a new string since strings are immutable.
 */
void STR_pass_char_ptr_in_out_bufferify(char * s, int Ls, int Ns)
{
// splicer begin function.pass_char_ptr_in_out_bufferify
    char * SH_s = (char *) malloc(Ns + 1);
    std::memcpy(SH_s, s, Ls);
    SH_s[Ls] = '\0';
    passCharPtrInOut(SH_s);
    ShroudStrCopy(s, Ns, SH_s);
    free(SH_s);
    return;
// splicer end function.pass_char_ptr_in_out_bufferify
}

// const char * getCharPtr1() +pure
// function_index=4
/**
 * \brief return a 'const char *' as character(*)
 *
 */
const char * STR_get_char_ptr1()
{
// splicer begin function.get_char_ptr1
    const char * SHC_rv = getCharPtr1();
    return SHC_rv;
// splicer end function.get_char_ptr1
}

// void getCharPtr1(char * SHF_rv +intent(out)+len(NSHF_rv)+pure) +pure
// function_index=32
/**
 * \brief return a 'const char *' as character(*)
 *
 */
void STR_get_char_ptr1_bufferify(char * SHF_rv, int NSHF_rv)
{
// splicer begin function.get_char_ptr1_bufferify
    const char * SHC_rv = getCharPtr1();
    if (SHC_rv == NULL) {
        std::memset(SHF_rv, ' ', NSHF_rv);
    } else {
        ShroudStrCopy(SHF_rv, NSHF_rv, SHC_rv);
    }
    return;
// splicer end function.get_char_ptr1_bufferify
}

// const char * getCharPtr2() +len(30)
// function_index=5
/**
 * \brief return 'const char *' with fixed size (len=30)
 *
 */
const char * STR_get_char_ptr2()
{
// splicer begin function.get_char_ptr2
    const char * SHC_rv = getCharPtr2();
    return SHC_rv;
// splicer end function.get_char_ptr2
}

// void getCharPtr2(char * SHF_rv +intent(out)+len(NSHF_rv)) +len(30)
// function_index=33
/**
 * \brief return 'const char *' with fixed size (len=30)
 *
 */
void STR_get_char_ptr2_bufferify(char * SHF_rv, int NSHF_rv)
{
// splicer begin function.get_char_ptr2_bufferify
    const char * SHC_rv = getCharPtr2();
    if (SHC_rv == NULL) {
        std::memset(SHF_rv, ' ', NSHF_rv);
    } else {
        ShroudStrCopy(SHF_rv, NSHF_rv, SHC_rv);
    }
    return;
// splicer end function.get_char_ptr2_bufferify
}

// const char * getCharPtr3()
// function_index=6
/**
 * \brief return a 'const char *' as argument
 *
 */
const char * STR_get_char_ptr3()
{
// splicer begin function.get_char_ptr3
    const char * SHC_rv = getCharPtr3();
    return SHC_rv;
// splicer end function.get_char_ptr3
}

// void getCharPtr3(char * output +intent(out)+len(Noutput))
// function_index=34
/**
 * \brief return a 'const char *' as argument
 *
 */
void STR_get_char_ptr3_bufferify(char * output, int Noutput)
{
// splicer begin function.get_char_ptr3_bufferify
    const char * SHC_rv = getCharPtr3();
    if (SHC_rv == NULL) {
        std::memset(output, ' ', Noutput);
    } else {
        ShroudStrCopy(output, Noutput, SHC_rv);
    }
    return;
// splicer end function.get_char_ptr3_bufferify
}

// void getConstStringLen(string * SHF_rv +intent(out)+len(NSHF_rv)) +len(30)
// function_index=36
/**
 * \brief return a 'const string' as argument
 *
 */
void STR_get_const_string_len_bufferify(char * SHF_rv, int NSHF_rv)
{
// splicer begin function.get_const_string_len_bufferify
    const std::string SHCXX_rv = getConstStringLen();
    if (SHCXX_rv.empty()) {
        std::memset(SHF_rv, ' ', NSHF_rv);
    } else {
        ShroudStrCopy(SHF_rv, NSHF_rv, SHCXX_rv.c_str());
    }
    return;
// splicer end function.get_const_string_len_bufferify
}

// void getConstStringAsArg(string * output +intent(out)+len(Noutput))
// function_index=37
/**
 * \brief return a 'const string' as argument
 *
 */
void STR_get_const_string_as_arg_bufferify(char * output, int Noutput)
{
// splicer begin function.get_const_string_as_arg_bufferify
    const std::string SHCXX_rv = getConstStringAsArg();
    if (SHCXX_rv.empty()) {
        std::memset(output, ' ', Noutput);
    } else {
        ShroudStrCopy(output, Noutput, SHCXX_rv.c_str());
    }
    return;
// splicer end function.get_const_string_as_arg_bufferify
}

// void getConstStringAlloc(const stringout * * SHF_rv +context(DSHF_rv)+intent(out)) +allocatable
// function_index=39
void STR_get_const_string_alloc_bufferify(
    STR_SHROUD_vector_context *DSHF_rv)
{
// splicer begin function.get_const_string_alloc_bufferify
    std::string * SHCXX_rv = new std::string;
    *SHCXX_rv = getConstStringAlloc();
    DSHF_rv->cxx = static_cast<void *>(const_cast<std::string *>
        (SHCXX_rv));
    DSHF_rv->len = SHCXX_rv->size();
    return;
// splicer end function.get_const_string_alloc_bufferify
}

// const string & getConstStringRefPure() +pure
// function_index=10
/**
 * \brief return a 'const string&' as character(*)
 *
 */
const char * STR_get_const_string_ref_pure()
{
// splicer begin function.get_const_string_ref_pure
    const std::string & SHCXX_rv = getConstStringRefPure();
    const char * SHC_rv = SHCXX_rv.c_str();
    return SHC_rv;
// splicer end function.get_const_string_ref_pure
}

// void getConstStringRefPure(string & SHF_rv +intent(out)+len(NSHF_rv)+pure) +pure
// function_index=40
/**
 * \brief return a 'const string&' as character(*)
 *
 */
void STR_get_const_string_ref_pure_bufferify(char * SHF_rv, int NSHF_rv)
{
// splicer begin function.get_const_string_ref_pure_bufferify
    const std::string & SHCXX_rv = getConstStringRefPure();
    if (SHCXX_rv.empty()) {
        std::memset(SHF_rv, ' ', NSHF_rv);
    } else {
        ShroudStrCopy(SHF_rv, NSHF_rv, SHCXX_rv.c_str());
    }
    return;
// splicer end function.get_const_string_ref_pure_bufferify
}

// const string & getConstStringRefLen() +len(30)
// function_index=11
/**
 * \brief return 'const string&' with fixed size (len=30)
 *
 */
const char * STR_get_const_string_ref_len()
{
// splicer begin function.get_const_string_ref_len
    const std::string & SHCXX_rv = getConstStringRefLen();
    // C_error_pattern
    if (SHCXX_rv.empty()) {
        return NULL;
    }

    const char * SHC_rv = SHCXX_rv.c_str();
    return SHC_rv;
// splicer end function.get_const_string_ref_len
}

// void getConstStringRefLen(string & SHF_rv +intent(out)+len(NSHF_rv)) +len(30)
// function_index=41
/**
 * \brief return 'const string&' with fixed size (len=30)
 *
 */
void STR_get_const_string_ref_len_bufferify(char * SHF_rv, int NSHF_rv)
{
// splicer begin function.get_const_string_ref_len_bufferify
    const std::string & SHCXX_rv = getConstStringRefLen();
    if (SHCXX_rv.empty()) {
        std::memset(SHF_rv, ' ', NSHF_rv);
    } else {
        ShroudStrCopy(SHF_rv, NSHF_rv, SHCXX_rv.c_str());
    }
    return;
// splicer end function.get_const_string_ref_len_bufferify
}

// const string & getConstStringRefAsArg()
// function_index=12
/**
 * \brief return a 'const string&' as argument
 *
 */
const char * STR_get_const_string_ref_as_arg()
{
// splicer begin function.get_const_string_ref_as_arg
    const std::string & SHCXX_rv = getConstStringRefAsArg();
    // C_error_pattern
    if (SHCXX_rv.empty()) {
        return NULL;
    }

    const char * SHC_rv = SHCXX_rv.c_str();
    return SHC_rv;
// splicer end function.get_const_string_ref_as_arg
}

// void getConstStringRefAsArg(string & output +intent(out)+len(Noutput))
// function_index=42
/**
 * \brief return a 'const string&' as argument
 *
 */
void STR_get_const_string_ref_as_arg_bufferify(char * output,
    int Noutput)
{
// splicer begin function.get_const_string_ref_as_arg_bufferify
    const std::string & SHCXX_rv = getConstStringRefAsArg();
    if (SHCXX_rv.empty()) {
        std::memset(output, ' ', Noutput);
    } else {
        ShroudStrCopy(output, Noutput, SHCXX_rv.c_str());
    }
    return;
// splicer end function.get_const_string_ref_as_arg_bufferify
}

// const string & getConstStringRefLenEmpty() +len(30)
// function_index=13
/**
 * \brief Test returning empty string reference
 *
 */
const char * STR_get_const_string_ref_len_empty()
{
// splicer begin function.get_const_string_ref_len_empty
    const std::string & SHCXX_rv = getConstStringRefLenEmpty();
    // C_error_pattern
    if (SHCXX_rv.empty()) {
        return NULL;
    }

    const char * SHC_rv = SHCXX_rv.c_str();
    return SHC_rv;
// splicer end function.get_const_string_ref_len_empty
}

// void getConstStringRefLenEmpty(string & SHF_rv +intent(out)+len(NSHF_rv)) +len(30)
// function_index=44
/**
 * \brief Test returning empty string reference
 *
 */
void STR_get_const_string_ref_len_empty_bufferify(char * SHF_rv,
    int NSHF_rv)
{
// splicer begin function.get_const_string_ref_len_empty_bufferify
    const std::string & SHCXX_rv = getConstStringRefLenEmpty();
    if (SHCXX_rv.empty()) {
        std::memset(SHF_rv, ' ', NSHF_rv);
    } else {
        ShroudStrCopy(SHF_rv, NSHF_rv, SHCXX_rv.c_str());
    }
    return;
// splicer end function.get_const_string_ref_len_empty_bufferify
}

// const std::string & getConstStringRefAlloc() +allocatable
// function_index=14
const char * STR_get_const_string_ref_alloc()
{
// splicer begin function.get_const_string_ref_alloc
    const std::string & SHCXX_rv = getConstStringRefAlloc();
    const char * SHC_rv = SHCXX_rv.c_str();
    return SHC_rv;
// splicer end function.get_const_string_ref_alloc
}

// void getConstStringRefAlloc(const stringout * * SHF_rv +context(DSHF_rv)+intent(out)) +allocatable
// function_index=45
void STR_get_const_string_ref_alloc_bufferify(
    STR_SHROUD_vector_context *DSHF_rv)
{
// splicer begin function.get_const_string_ref_alloc_bufferify
    const std::string & SHCXX_rv = getConstStringRefAlloc();
    DSHF_rv->cxx = static_cast<void *>(const_cast<std::string *>
        (&SHCXX_rv));
    DSHF_rv->len = SHCXX_rv.size();
    return;
// splicer end function.get_const_string_ref_alloc_bufferify
}

// const string * getConstStringPtrLen() +len(30)
// function_index=15
/**
 * \brief return a 'const string *' as character(*)
 *
 */
const char * STR_get_const_string_ptr_len()
{
// splicer begin function.get_const_string_ptr_len
    const std::string * SHCXX_rv = getConstStringPtrLen();
    const char * SHC_rv = SHCXX_rv->c_str();
    return SHC_rv;
// splicer end function.get_const_string_ptr_len
}

// void getConstStringPtrLen(string * SHF_rv +intent(out)+len(NSHF_rv)) +len(30)
// function_index=46
/**
 * \brief return a 'const string *' as character(*)
 *
 */
void STR_get_const_string_ptr_len_bufferify(char * SHF_rv, int NSHF_rv)
{
// splicer begin function.get_const_string_ptr_len_bufferify
    const std::string * SHCXX_rv = getConstStringPtrLen();
    if (SHCXX_rv->empty()) {
        std::memset(SHF_rv, ' ', NSHF_rv);
    } else {
        ShroudStrCopy(SHF_rv, NSHF_rv, SHCXX_rv->c_str());
    }
    {
        // C_finalize
        delete SHCXX_rv;
    }
    return;
// splicer end function.get_const_string_ptr_len_bufferify
}

// const std::string * getConstStringPtrAlloc() +allocatable
// function_index=16
const char * STR_get_const_string_ptr_alloc()
{
// splicer begin function.get_const_string_ptr_alloc
    const std::string * SHCXX_rv = getConstStringPtrAlloc();
    const char * SHC_rv = SHCXX_rv->c_str();
    return SHC_rv;
// splicer end function.get_const_string_ptr_alloc
}

// void getConstStringPtrAlloc(const stringout * * SHF_rv +context(DSHF_rv)+intent(out)) +allocatable
// function_index=47
void STR_get_const_string_ptr_alloc_bufferify(
    STR_SHROUD_vector_context *DSHF_rv)
{
// splicer begin function.get_const_string_ptr_alloc_bufferify
    const std::string * SHCXX_rv = getConstStringPtrAlloc();
    DSHF_rv->cxx = static_cast<void *>(const_cast<std::string *>
        (SHCXX_rv));
    DSHF_rv->len = SHCXX_rv->size();
    return;
// splicer end function.get_const_string_ptr_alloc_bufferify
}

// const std::string * getConstStringPtrOwnsAlloc() +allocatable
// function_index=17
const char * STR_get_const_string_ptr_owns_alloc()
{
// splicer begin function.get_const_string_ptr_owns_alloc
    const std::string * SHCXX_rv = getConstStringPtrOwnsAlloc();
    const char * SHC_rv = SHCXX_rv->c_str();
    return SHC_rv;
// splicer end function.get_const_string_ptr_owns_alloc
}

// void getConstStringPtrOwnsAlloc(const stringout * * SHF_rv +context(DSHF_rv)+intent(out)) +allocatable
// function_index=48
void STR_get_const_string_ptr_owns_alloc_bufferify(
    STR_SHROUD_vector_context *DSHF_rv)
{
// splicer begin function.get_const_string_ptr_owns_alloc_bufferify
    const std::string * SHCXX_rv = getConstStringPtrOwnsAlloc();
    DSHF_rv->cxx = static_cast<void *>(const_cast<std::string *>
        (SHCXX_rv));
    DSHF_rv->len = SHCXX_rv->size();
    return;
// splicer end function.get_const_string_ptr_owns_alloc_bufferify
}

// void acceptStringConstReference(const std::string & arg1 +intent(in))
// function_index=18
/**
 * \brief Accept a const string reference
 *
 * Save contents of arg1.
 * arg1 is assumed to be intent(IN) since it is const
 * Will copy in.
 */
void STR_accept_string_const_reference(const char * arg1)
{
// splicer begin function.accept_string_const_reference
    const std::string SH_arg1(arg1);
    acceptStringConstReference(SH_arg1);
    return;
// splicer end function.accept_string_const_reference
}

// void acceptStringConstReference(const std::string & arg1 +intent(in)+len_trim(Larg1))
// function_index=49
/**
 * \brief Accept a const string reference
 *
 * Save contents of arg1.
 * arg1 is assumed to be intent(IN) since it is const
 * Will copy in.
 */
void STR_accept_string_const_reference_bufferify(const char * arg1,
    int Larg1)
{
// splicer begin function.accept_string_const_reference_bufferify
    const std::string SH_arg1(arg1, Larg1);
    acceptStringConstReference(SH_arg1);
    return;
// splicer end function.accept_string_const_reference_bufferify
}

// void acceptStringReferenceOut(std::string & arg1 +intent(out))
// function_index=19
/**
 * \brief Accept a string reference
 *
 * Set out to a constant string.
 * arg1 is intent(OUT)
 * Must copy out.
 */
void STR_accept_string_reference_out(char * arg1)
{
// splicer begin function.accept_string_reference_out
    std::string SH_arg1;
    acceptStringReferenceOut(SH_arg1);
    strcpy(arg1, SH_arg1.c_str());
    return;
// splicer end function.accept_string_reference_out
}

// void acceptStringReferenceOut(std::string & arg1 +intent(out)+len(Narg1))
// function_index=50
/**
 * \brief Accept a string reference
 *
 * Set out to a constant string.
 * arg1 is intent(OUT)
 * Must copy out.
 */
void STR_accept_string_reference_out_bufferify(char * arg1, int Narg1)
{
// splicer begin function.accept_string_reference_out_bufferify
    std::string SH_arg1;
    acceptStringReferenceOut(SH_arg1);
    ShroudStrCopy(arg1, Narg1, SH_arg1.c_str());
    return;
// splicer end function.accept_string_reference_out_bufferify
}

// void acceptStringReference(std::string & arg1 +intent(inout))
// function_index=20
/**
 * \brief Accept a string reference
 *
 * Append "dog" to the end of arg1.
 * arg1 is assumed to be intent(INOUT)
 * Must copy in and copy out.
 */
void STR_accept_string_reference(char * arg1)
{
// splicer begin function.accept_string_reference
    std::string SH_arg1(arg1);
    acceptStringReference(SH_arg1);
    strcpy(arg1, SH_arg1.c_str());
    return;
// splicer end function.accept_string_reference
}

// void acceptStringReference(std::string & arg1 +intent(inout)+len(Narg1)+len_trim(Larg1))
// function_index=51
/**
 * \brief Accept a string reference
 *
 * Append "dog" to the end of arg1.
 * arg1 is assumed to be intent(INOUT)
 * Must copy in and copy out.
 */
void STR_accept_string_reference_bufferify(char * arg1, int Larg1,
    int Narg1)
{
// splicer begin function.accept_string_reference_bufferify
    std::string SH_arg1(arg1, Larg1);
    acceptStringReference(SH_arg1);
    ShroudStrCopy(arg1, Narg1, SH_arg1.c_str());
    return;
// splicer end function.accept_string_reference_bufferify
}

// void acceptStringPointer(std::string * arg1 +intent(inout))
// function_index=21
/**
 * \brief Accept a string pointer
 *
 */
void STR_accept_string_pointer(char * arg1)
{
// splicer begin function.accept_string_pointer
    std::string SH_arg1(arg1);
    acceptStringPointer(&SH_arg1);
    strcpy(arg1, SH_arg1.c_str());
    return;
// splicer end function.accept_string_pointer
}

// void acceptStringPointer(std::string * arg1 +intent(inout)+len(Narg1)+len_trim(Larg1))
// function_index=52
/**
 * \brief Accept a string pointer
 *
 */
void STR_accept_string_pointer_bufferify(char * arg1, int Larg1,
    int Narg1)
{
// splicer begin function.accept_string_pointer_bufferify
    std::string SH_arg1(arg1, Larg1);
    acceptStringPointer(&SH_arg1);
    ShroudStrCopy(arg1, Narg1, SH_arg1.c_str());
    return;
// splicer end function.accept_string_pointer_bufferify
}

// void explicit1(char * name +intent(in)+len_trim(AAlen))
// function_index=24
void STR_explicit1(char * name)
{
// splicer begin function.explicit1
    explicit1(name);
    return;
// splicer end function.explicit1
}

// void explicit1(char * name +intent(in)+len_trim(AAlen))
// function_index=53
void STR_explicit1_BUFFER(char * name, int AAlen)
{
// splicer begin function.explicit1_BUFFER
    char * SH_name = (char *) malloc(AAlen + 1);
    std::memcpy(SH_name, name, AAlen);
    SH_name[AAlen] = '\0';
    explicit1(SH_name);
    free(SH_name);
    return;
// splicer end function.explicit1_BUFFER
}

// void explicit2(char * name +intent(out)+len(AAtrim))
// function_index=25
void STR_explicit2(char * name)
{
// splicer begin function.explicit2
    explicit2(name);
    return;
// splicer end function.explicit2
}

// void explicit2(char * name +intent(out)+len(AAtrim))
// function_index=54
void STR_explicit2_bufferify(char * name, int AAtrim)
{
// splicer begin function.explicit2_bufferify
    char * SH_name = (char *) malloc(AAtrim + 1);
    explicit2(SH_name);
    ShroudStrCopy(name, AAtrim, SH_name);
    free(SH_name);
    return;
// splicer end function.explicit2_bufferify
}

// void CreturnChar(char_scalar * SHF_rv +intent(out)+len(NSHF_rv))
// function_index=55
/**
 * \brief return a char argument (non-pointer), extern "C"
 *
 */
void STR_creturn_char_bufferify(char * SHF_rv, int NSHF_rv)
{
// splicer begin function.creturn_char_bufferify
    char SHC_rv = CreturnChar();
    std::memset(SHF_rv, ' ', NSHF_rv);
    SHF_rv[0] = SHC_rv;
    return;
// splicer end function.creturn_char_bufferify
}

// void CpassCharPtr(char * dest +intent(out)+len(Ndest), const char * src +intent(in)+len_trim(Lsrc))
// function_index=56
/**
 * \brief strcpy like behavior
 *
 * dest is marked intent(OUT) to override the intent(INOUT) default
 * This avoid a copy-in on dest.
 * extern "C"
 */
void STR_cpass_char_ptr_bufferify(char * dest, int Ndest,
    const char * src, int Lsrc)
{
// splicer begin function.cpass_char_ptr_bufferify
    char * SH_dest = (char *) malloc(Ndest + 1);
    char * SH_src = (char *) malloc(Lsrc + 1);
    std::memcpy(SH_src, src, Lsrc);
    SH_src[Lsrc] = '\0';
    CpassCharPtr(SH_dest, SH_src);
    ShroudStrCopy(dest, Ndest, SH_dest);
    free(SH_dest);
    free(SH_src);
    return;
// splicer end function.cpass_char_ptr_bufferify
}

}  // extern "C"
