// luaforwardmodule.cpp
// This is generated code, do not edit
// #######################################################################
// Copyright (c) 2018, Lawrence Livermore National Security, LLC.
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
#include "tutorial.hpp"
#include "luaforwardmodule.hpp"
#ifdef __cplusplus
extern "C" {
#endif
#include "lauxlib.h"
#ifdef __cplusplus
}
#endif
// splicer begin include
// splicer end include
// splicer begin C_definition
// splicer end C_definition

// splicer begin class.Class3.additional_functions
// splicer end class.Class3.additional_functions

static const struct luaL_Reg l_Class3_Reg [] = {
    // splicer begin class.Class3.register
    // splicer end class.Class3.register
    {NULL, NULL}   /*sentinel */
};

// Class2()
static int l_class2_ctor(lua_State *L)
{
    // splicer begin class.Class2.method.ctor
    l_Class2_Type * SH_this =
        (l_Class2_Type *) lua_newuserdata(L, sizeof(*SH_this));
    SH_this->self = new tutorial::Class2();
    /* Add the metatable to the stack. */
    luaL_getmetatable(L, "Class2.metatable");
    /* Set the metatable on the userdata. */
    lua_setmetatable(L, -2);
    return 1;
    // splicer end class.Class2.method.ctor
}

// ~Class2()
static int l_class2_dtor(lua_State *L)
{
    // splicer begin class.Class2.method.__gc
    l_Class2_Type * SH_this = (l_Class2_Type *) luaL_checkudata(
        L, 1, "Class2.metatable");
    delete SH_this->self;
    SH_this->self = NULL;
    return 0;
    // splicer end class.Class2.method.__gc
}

// void func1(Class1 * arg +intent(in))
static int l_class2_func1(lua_State *L)
{
    // splicer begin class.Class2.method.func1
    tutorial::Class1 * arg = static_cast<tutorial::Class1 *>(
        (l_Class2_Type *) luaL_checkudata(
        L, 1, "Class2.metatable")->addr);
    l_Class2_Type * SH_this = (l_Class2_Type *) luaL_checkudata(
        L, 1, "Class2.metatable");
    SH_this->self->func1(arg);
    return 0;
    // splicer end class.Class2.method.func1
}

// void acceptClass3(Class3 * arg +intent(in))
static int l_class2_accept_class3(lua_State *L)
{
    // splicer begin class.Class2.method.acceptClass3
    tutorial::Class3 * arg = static_cast<tutorial::Class3 *>(
        (l_Class2_Type *) luaL_checkudata(
        L, 1, "Class2.metatable")->addr);
    l_Class2_Type * SH_this = (l_Class2_Type *) luaL_checkudata(
        L, 1, "Class2.metatable");
    SH_this->self->acceptClass3(arg);
    return 0;
    // splicer end class.Class2.method.acceptClass3
}

// splicer begin class.Class2.additional_functions
// splicer end class.Class2.additional_functions

static const struct luaL_Reg l_Class2_Reg [] = {
    {"__gc", l_class2_dtor},
    {"func1", l_class2_func1},
    {"acceptClass3", l_class2_accept_class3},
    // splicer begin class.Class2.register
    // splicer end class.Class2.register
    {NULL, NULL}   /*sentinel */
};

// splicer begin additional_functions
// splicer end additional_functions

static const struct luaL_Reg l_forward_Reg [] = {
    {"Class2", l_class2_ctor},
    // splicer begin register
    // splicer end register
    {NULL, NULL}   /*sentinel */
};

#ifdef __cplusplus
extern "C" {
#endif
int luaopen_forward(lua_State *L) {

    /* Create the metatable and put it on the stack. */
    luaL_newmetatable(L, "Class3.metatable");
    /* Duplicate the metatable on the stack (We now have 2). */
    lua_pushvalue(L, -1);
    /* Pop the first metatable off the stack and assign it to __index
     * of the second one. We set the metatable for the table to itself.
     * This is equivalent to the following in lua:
     * metatable = {}
     * metatable.__index = metatable
     */
    lua_setfield(L, -2, "__index");

    /* Set the methods to the metatable that should be accessed via object:func */
#if LUA_VERSION_NUM < 502
    luaL_register(L, NULL, l_Class3_Reg);
#else
    luaL_setfuncs(L, l_Class3_Reg, 0);
#endif


    /* Create the metatable and put it on the stack. */
    luaL_newmetatable(L, "Class2.metatable");
    /* Duplicate the metatable on the stack (We now have 2). */
    lua_pushvalue(L, -1);
    /* Pop the first metatable off the stack and assign it to __index
     * of the second one. We set the metatable for the table to itself.
     * This is equivalent to the following in lua:
     * metatable = {}
     * metatable.__index = metatable
     */
    lua_setfield(L, -2, "__index");

    /* Set the methods to the metatable that should be accessed via object:func */
#if LUA_VERSION_NUM < 502
    luaL_register(L, NULL, l_Class2_Reg);
#else
    luaL_setfuncs(L, l_Class2_Reg, 0);
#endif


#if LUA_VERSION_NUM < 502
    luaL_register(L, "forward", l_forward_Reg);
#else
    luaL_newlib(L, l_forward_Reg);
#endif
    return 1;
}
#ifdef __cplusplus
}
#endif
