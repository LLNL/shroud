// luaUserLibrarymodule.hpp
// This file is generated by Shroud 0.12.0. Do not edit.
// Copyright (c) 2017-2020, Lawrence Livermore National Security, LLC and
// other Shroud Project Developers.
// See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//
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

int luaopen_userlibrary(lua_State *L);

#ifdef __cplusplus
}
#endif
#endif  /* LUAUSERLIBRARYMODULE_HPP */
