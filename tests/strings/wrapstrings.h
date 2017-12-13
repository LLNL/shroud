// wrapstrings.h
// This is generated code, do not edit
// #######################################################################
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
/**
 * \file wrapstrings.h
 * \brief Shroud generated wrapper for strings library
 */
// For C users and C++ implementation

#ifndef WRAPSTRINGS_H
#define WRAPSTRINGS_H

// splicer begin CXX_declarations
// splicer end CXX_declarations

#ifdef __cplusplus
extern "C" {
#endif

// declaration of wrapped types

// splicer begin C_declarations
// splicer end C_declarations

void STR_pass_char(char status);

char STR_return_char();

void STR_return_char_bufferify(char * SHF_rv, int NSHF_rv);

void STR_pass_char_ptr(char * dest, const char * src);

void STR_pass_char_ptr_bufferify(char * dest, int Ndest, const char * src, int Lsrc);

void STR_pass_char_ptr_in_out(char * s);

void STR_pass_char_ptr_in_out_bufferify(char * s, int Ls, int Ns);

const char * STR_get_char1();

void STR_get_char1_bufferify(char * SHF_rv, int NSHF_rv);

const char * STR_get_char2();

void STR_get_char2_bufferify(char * SHF_rv, int NSHF_rv);

const char * STR_get_char3();

void STR_get_char3_bufferify(char * output, int Noutput);

const char * STR_get_string1();

void STR_get_string1_bufferify(char * SHF_rv, int NSHF_rv);

const char * STR_get_string2();

void STR_get_string2_bufferify(char * SHF_rv, int NSHF_rv);

const char * STR_get_string3();

void STR_get_string3_bufferify(char * output, int Noutput);

const char * STR_get_string2_empty();

void STR_get_string2_empty_bufferify(char * SHF_rv, int NSHF_rv);

void STR_get_string5_bufferify(char * SHF_rv, int NSHF_rv);

void STR_get_string6_bufferify(char * output, int Noutput);

void STR_accept_string_const_reference(const char * arg1);

void STR_accept_string_const_reference_bufferify(const char * arg1, int Larg1);

void STR_accept_string_reference_out(char * arg1);

void STR_accept_string_reference_out_bufferify(char * arg1, int Narg1);

void STR_accept_string_reference(char * arg1);

void STR_accept_string_reference_bufferify(char * arg1, int Larg1, int Narg1);

void STR_explicit1(char * name);

void STR_explicit1_BUFFER(char * name, int AAlen);

void STR_explicit2(char * name);

void STR_explicit2_bufferify(char * name, int AAtrim);

void STR_creturn_char_bufferify(char * SHF_rv, int NSHF_rv);

void STR_cpass_char_ptr_bufferify(char * dest, int Ndest, const char * src, int Lsrc);

#ifdef __cplusplus
}
#endif

#endif  // WRAPSTRINGS_H
