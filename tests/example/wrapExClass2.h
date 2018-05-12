// wrapExClass2.h
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
 * \file wrapExClass2.h
 * \brief Shroud generated wrapper for ExClass2 class
 */
// For C users and C++ implementation

#ifndef WRAPEXCLASS2_H
#define WRAPEXCLASS2_H

#include "sidre/SidreTypes.h"
#include "typesUserLibrary.h"

// splicer begin class.ExClass2.CXX_declarations
// splicer end class.ExClass2.CXX_declarations

#ifdef __cplusplus
extern "C" {
#endif

// splicer begin class.ExClass2.C_declarations
// splicer end class.ExClass2.C_declarations

AA_exclass2 AA_exclass2_ctor(const char * name);

AA_exclass2 AA_exclass2_ctor_bufferify(const char * name,
    int trim_name);

void AA_exclass2_dtor(AA_exclass2 * self);

const char * AA_exclass2_get_name(const AA_exclass2 * self);

void AA_exclass2_get_name_bufferify(const AA_exclass2 * self,
    char * SHF_rv, int NSHF_rv);

const char * AA_exclass2_get_name2(AA_exclass2 * self);

void AA_exclass2_get_name2_bufferify(AA_exclass2 * self, char * SHF_rv,
    int NSHF_rv);

char * AA_exclass2_get_name3(const AA_exclass2 * self);

void AA_exclass2_get_name3_bufferify(const AA_exclass2 * self,
    char * SHF_rv, int NSHF_rv);

char * AA_exclass2_get_name4(AA_exclass2 * self);

void AA_exclass2_get_name4_bufferify(AA_exclass2 * self, char * SHF_rv,
    int NSHF_rv);

int AA_exclass2_get_name_length(const AA_exclass2 * self);

AA_exclass1 AA_exclass2_get_class1(AA_exclass2 * self,
    const AA_exclass1 * in);

void AA_exclass2_declare_0(AA_exclass2 * self, int type);

void AA_exclass2_declare_1(AA_exclass2 * self, int type,
    SIDRE_SidreLength len);

void AA_exclass2_destroyall(AA_exclass2 * self);

int AA_exclass2_get_type_id(const AA_exclass2 * self);

void AA_exclass2_set_value_int(AA_exclass2 * self, int value);

void AA_exclass2_set_value_long(AA_exclass2 * self, long value);

void AA_exclass2_set_value_float(AA_exclass2 * self, float value);

void AA_exclass2_set_value_double(AA_exclass2 * self, double value);

int AA_exclass2_get_value_int(AA_exclass2 * self);

double AA_exclass2_get_value_double(AA_exclass2 * self);

#ifdef __cplusplus
}
#endif

#endif  // WRAPEXCLASS2_H
