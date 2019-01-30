// luaownershipmodule.hpp
// This is generated code, do not edit
// #######################################################################
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
