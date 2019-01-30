// luaforwardmodule.hpp
// This is generated code, do not edit
// #######################################################################
// Copyright (c) 2018-2019, Lawrence Livermore National Security, LLC.
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
#ifndef LUAFORWARDMODULE_HPP
#define LUAFORWARDMODULE_HPP
#ifdef __cplusplus
extern "C" {
#endif
#include "tutorial.hpp"
#include "lua.h"
// splicer begin class.Class3.C_declaration
// splicer end class.Class3.C_declaration

typedef struct {
    tutorial::Class3 * self;
    // splicer begin class.Class3.C_object
    // splicer end class.Class3.C_object
} l_Class3_Type;
// splicer begin class.Class2.C_declaration
// splicer end class.Class2.C_declaration

typedef struct {
    tutorial::Class2 * self;
    // splicer begin class.Class2.C_object
    // splicer end class.Class2.C_object
} l_Class2_Type;

int luaopen_forward(lua_State *L);

#ifdef __cplusplus
}
#endif
#endif  /* LUAFORWARDMODULE_HPP */
