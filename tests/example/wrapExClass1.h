// wrapExClass1.h
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
/**
 * \file wrapExClass1.h
 * \brief Shroud generated wrapper for ExClass1 class
 */
// For C users and C++ implementation

#ifndef WRAPEXCLASS1_H
#define WRAPEXCLASS1_H

// splicer begin class.ExClass1.CXX_declarations
// splicer end class.ExClass1.CXX_declarations

#ifdef __cplusplus
extern "C" {
#endif

// declaration of shadow types
struct s_AA_exclass1;
typedef struct s_AA_exclass1 AA_exclass1;

// splicer begin class.ExClass1.C_declarations
// splicer end class.ExClass1.C_declarations

AA_exclass1 * AA_exclass1_ctor_0();

AA_exclass1 * AA_exclass1_ctor_1(const char * name);

AA_exclass1 * AA_exclass1_ctor_1_bufferify(const char * name,
    int Lname);

void AA_exclass1_dtor(AA_exclass1 * self);

int AA_exclass1_increment_count(AA_exclass1 * self, int incr);

const char * AA_exclass1_get_name_error_pattern(
    const AA_exclass1 * self);

void AA_exclass1_get_name_error_pattern_bufferify(
    const AA_exclass1 * self, char * SHF_rv, int NSHF_rv);

int AA_exclass1_get_name_length(const AA_exclass1 * self);

const char * AA_exclass1_get_name_error_check(const AA_exclass1 * self);

void AA_exclass1_get_name_error_check_bufferify(
    const AA_exclass1 * self, char * SHF_rv, int NSHF_rv);

const char * AA_exclass1_get_name_arg(const AA_exclass1 * self);

void AA_exclass1_get_name_arg_bufferify(const AA_exclass1 * self,
    char * name, int Nname);

void * AA_exclass1_get_root(AA_exclass1 * self);

int AA_exclass1_get_value_from_int(AA_exclass1 * self, int value);

long AA_exclass1_get_value_1(AA_exclass1 * self, long value);

void * AA_exclass1_get_addr(AA_exclass1 * self);

bool AA_exclass1_has_addr(AA_exclass1 * self, bool in);

void AA_exclass1_splicer_special(AA_exclass1 * self);

#ifdef __cplusplus
}
#endif

#endif  // WRAPEXCLASS1_H
