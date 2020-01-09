// luaownershipmodule.hpp
// This is generated code, do not edit
// Copyright (c) 2017-2020, Lawrence Livermore National Security, LLC and
// other Shroud Project Developers.
// See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//
#ifndef LUAOWNERSHIPMODULE_HPP
#define LUAOWNERSHIPMODULE_HPP
#ifdef __cplusplus
extern "C" {
#endif
#include "ownership.hpp"
#include "lua.h"
// splicer begin class.Class1.C_declaration
// splicer end class.Class1.C_declaration

typedef struct {
    Class1 * self;
    // splicer begin class.Class1.C_object
    // splicer end class.Class1.C_object
} l_Class1_Type;

int luaopen_ownership(lua_State *L);

#ifdef __cplusplus
}
#endif
#endif  /* LUAOWNERSHIPMODULE_HPP */
