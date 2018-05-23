// wrapstrings.h
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
/**
 * \file wrapstrings.h
 * \brief Shroud generated wrapper for strings library
 */
// For C users and C++ implementation

#ifndef WRAPSTRINGS_H
#define WRAPSTRINGS_H

#include <stddef.h>
#include "typesstrings.h"

// splicer begin CXX_declarations
// splicer end CXX_declarations

#ifdef __cplusplus
extern "C" {
#endif

// splicer begin C_declarations
// splicer end C_declarations

void STR_pass_char(char status);

char STR_return_char();

void STR_return_char_bufferify(char * SHF_rv, int NSHF_rv);

void STR_pass_char_ptr(char * dest, const char * src);

void STR_pass_char_ptr_bufferify(char * dest, int Ndest,
    const char * src, int Lsrc);

void STR_pass_char_ptr_in_out(char * s);

void STR_pass_char_ptr_in_out_bufferify(char * s, int Ls, int Ns);

const char * STR_get_char_ptr1();

void STR_get_char_ptr1_bufferify(char * SHF_rv, int NSHF_rv);

const char * STR_get_char_ptr2();

void STR_get_char_ptr2_bufferify(char * SHF_rv, int NSHF_rv);

const char * STR_get_char_ptr3();

void STR_get_char_ptr3_bufferify(char * output, int Noutput);

void STR_get_const_string_len_bufferify(char * SHF_rv, int NSHF_rv);

void STR_get_const_string_as_arg_bufferify(char * output, int Noutput);

void STR_get_const_string_alloc_bufferify(STR_SHROUD_array *DSHF_rv);

const char * STR_get_const_string_ref_pure();

void STR_get_const_string_ref_pure_bufferify(char * SHF_rv,
    int NSHF_rv);

const char * STR_get_const_string_ref_len();

void STR_get_const_string_ref_len_bufferify(char * SHF_rv, int NSHF_rv);

const char * STR_get_const_string_ref_as_arg();

void STR_get_const_string_ref_as_arg_bufferify(char * output,
    int Noutput);

const char * STR_get_const_string_ref_len_empty();

void STR_get_const_string_ref_len_empty_bufferify(char * SHF_rv,
    int NSHF_rv);

const char * STR_get_const_string_ref_alloc();

void STR_get_const_string_ref_alloc_bufferify(
    STR_SHROUD_array *DSHF_rv);

const char * STR_get_const_string_ptr_len();

void STR_get_const_string_ptr_len_bufferify(char * SHF_rv, int NSHF_rv);

const char * STR_get_const_string_ptr_alloc();

void STR_get_const_string_ptr_alloc_bufferify(
    STR_SHROUD_array *DSHF_rv);

const char * STR_get_const_string_ptr_owns_alloc();

void STR_get_const_string_ptr_owns_alloc_bufferify(
    STR_SHROUD_array *DSHF_rv);

void STR_accept_string_const_reference(const char * arg1);

void STR_accept_string_const_reference_bufferify(const char * arg1,
    int Larg1);

void STR_accept_string_reference_out(char * arg1);

void STR_accept_string_reference_out_bufferify(char * arg1, int Narg1);

void STR_accept_string_reference(char * arg1);

void STR_accept_string_reference_bufferify(char * arg1, int Larg1,
    int Narg1);

void STR_accept_string_pointer(char * arg1);

void STR_accept_string_pointer_bufferify(char * arg1, int Larg1,
    int Narg1);

void STR_explicit1(char * name);

void STR_explicit1_BUFFER(char * name, int AAlen);

void STR_explicit2(char * name);

void STR_explicit2_bufferify(char * name, int AAtrim);

void STR_creturn_char_bufferify(char * SHF_rv, int NSHF_rv);

void STR_cpass_char_ptr_bufferify(char * dest, int Ndest,
    const char * src, int Lsrc);

#ifdef __cplusplus
}
#endif

#endif  // WRAPSTRINGS_H
