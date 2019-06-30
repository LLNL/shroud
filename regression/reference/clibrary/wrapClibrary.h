// wrapClibrary.h
// This is generated code, do not edit
// Copyright (c) 2017-2019, Lawrence Livermore National Security, LLC and
// other Shroud Project Developers.
// See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//
/**
 * \file wrapClibrary.h
 * \brief Shroud generated wrapper for Clibrary library
 */
// For C users and C implementation

#ifndef WRAPCLIBRARY_H
#define WRAPCLIBRARY_H

#include "typesClibrary.h"

// splicer begin C_declarations
// splicer end C_declarations

void CLI_function4a_bufferify(const char * arg1, const char * arg2,
    char * SHF_rv, int NSHF_rv);

void CLI_return_one_name_bufferify(char * name1, int Nname1);

void CLI_return_two_names_bufferify(char * name1, int Nname1,
    char * name2, int Nname2);

void CLI_implied_text_len_bufferify(char * text, int Ntext, int ltext);

void CLI_bind_c2_bufferify(char * outbuf, int Noutbuf);

int CLI_pass_assumed_type_buf_bufferify(void * arg, char * outbuf,
    int Noutbuf);

void CLI_callback3_bufferify(const char * type, void * in,
    void ( * incr)(int *), char * outbuf, int Noutbuf);

#endif  // WRAPCLIBRARY_H
