// luaTutorialmodule.hpp
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
#ifndef LUATUTORIALMODULE_HPP
#define LUATUTORIALMODULE_HPP
#ifdef __cplusplus
extern "C" {
#endif
#include "tutorial.hpp"
#include "lua.h"
// splicer begin class.struct1.C_declaration
// splicer end class.struct1.C_declaration

typedef struct {
    tutorial::struct1 * self;
    // splicer begin class.struct1.C_object
    // splicer end class.struct1.C_object
} l_struct1_Type;
// splicer begin class.Class1.C_declaration
// splicer end class.Class1.C_declaration

typedef struct {
    tutorial::Class1 * self;
    // splicer begin class.Class1.C_object
    // splicer end class.Class1.C_object
} l_Class1_Type;

int luaopen_tutorial(lua_State *L);

#ifdef __cplusplus
}
#endif
#endif  /* LUATUTORIALMODULE_HPP */
