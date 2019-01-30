// luaUserLibrarymodule.hpp
// This is generated code, do not edit
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
#ifndef LUAUSERLIBRARYMODULE_HPP
#define LUAUSERLIBRARYMODULE_HPP
#ifdef __cplusplus
extern "C" {
#endif
#include "lua.h"
// splicer begin class.ExClass1.C_declaration
// splicer end class.ExClass1.C_declaration

typedef struct {
    example::nested::ExClass1 * self;
    // splicer begin class.ExClass1.C_object
    // splicer end class.ExClass1.C_object
} l_ExClass1_Type;
// splicer begin class.ExClass2.C_declaration
// splicer end class.ExClass2.C_declaration

typedef struct {
    example::nested::ExClass2 * self;
    // splicer begin class.ExClass2.C_object
    // splicer end class.ExClass2.C_object
} l_ExClass2_Type;
// splicer begin class.ExClass3.C_declaration
// splicer end class.ExClass3.C_declaration

typedef struct {
    example::nested::ExClass3 * self;
    // splicer begin class.ExClass3.C_object
    // splicer end class.ExClass3.C_object
} l_ExClass3_Type;

int luaopen_userlibrary(lua_State *L);

#ifdef __cplusplus
}
#endif
#endif  /* LUAUSERLIBRARYMODULE_HPP */
