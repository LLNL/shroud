// luaTutorialmodule.cpp
// This is generated code, do not edit
// #######################################################################
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
#include "tutorial.hpp"
#include "luaTutorialmodule.hpp"
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

// splicer begin class.struct1.additional_functions
// splicer end class.struct1.additional_functions

static const struct luaL_Reg l_struct1_Reg [] = {
    // splicer begin class.struct1.register
    // splicer end class.struct1.register
    {NULL, NULL}   /*sentinel */
};

// Class1() +name(new)
// Class1(int flag +intent(in)+value) +name(new)
static int l_class1_new(lua_State *L)
{
    // splicer begin class.Class1.method.new
    int SH_nresult = 0;
    int SH_nargs = lua_gettop(L);
    int SH_itype1 = lua_type(L, 1);
    switch (SH_nargs) {
    case 0:
        {
            l_Class1_Type * SH_this =
                (l_Class1_Type *) lua_newuserdata(L, sizeof(*SH_this));
            SH_this->self = new tutorial::Class1();
            /* Add the metatable to the stack. */
            luaL_getmetatable(L, "Class1.metatable");
            /* Set the metatable on the userdata. */
            lua_setmetatable(L, -2);
            SH_nresult = 1;
        }
        break;
    case 1:
        if (SH_itype1 == LUA_TNUMBER) {
            int flag = lua_tointeger(L, 1);
            l_Class1_Type * SH_this =
                (l_Class1_Type *) lua_newuserdata(L, sizeof(*SH_this));
            SH_this->self = new tutorial::Class1(flag);
            /* Add the metatable to the stack. */
            luaL_getmetatable(L, "Class1.metatable");
            /* Set the metatable on the userdata. */
            lua_setmetatable(L, -2);
            SH_nresult = 1;
        }
        else {
            luaL_error(L, "error with arguments");
        }
        break;
    default:
        luaL_error(L, "error with arguments");
        break;
    }
    return SH_nresult;
    // splicer end class.Class1.method.new
}

// ~Class1() +name(delete)
static int l_class1_delete(lua_State *L)
{
    // splicer begin class.Class1.method.__gc
    l_Class1_Type * SH_this = (l_Class1_Type *) luaL_checkudata(
        L, 1, "Class1.metatable");
    delete SH_this->self;
    SH_this->self = NULL;
    return 0;
    // splicer end class.Class1.method.__gc
}

// int Method1()
/**
 * \brief returns the value of flag member
 *
 */
static int l_class1_method1(lua_State *L)
{
    // splicer begin class.Class1.method.Method1
    l_Class1_Type * SH_this = (l_Class1_Type *) luaL_checkudata(
        L, 1, "Class1.metatable");
    int SHCXX_rv = SH_this->self->Method1();
    lua_pushinteger(L, SHCXX_rv);
    return 1;
    // splicer end class.Class1.method.Method1
}

// DIRECTION directionFunc(DIRECTION arg +intent(in)+value)
static int l_class1_direction_func(lua_State *L)
{
    // splicer begin class.Class1.method.directionFunc
    tutorial::Class1::DIRECTION arg =
        static_cast<tutorial::Class1::DIRECTION>(lua_tointeger(L, 1));
    l_Class1_Type * SH_this = (l_Class1_Type *) luaL_checkudata(
        L, 1, "Class1.metatable");
    tutorial::Class1::DIRECTION SHCXX_rv =
        SH_this->self->directionFunc(arg);
    lua_pushinteger(L, static_cast<int>(SHCXX_rv));
    return 1;
    // splicer end class.Class1.method.directionFunc
}

// splicer begin class.Class1.additional_functions
// splicer end class.Class1.additional_functions

static const struct luaL_Reg l_Class1_Reg [] = {
    {"__gc", l_class1_delete},
    {"Method1", l_class1_method1},
    {"directionFunc", l_class1_direction_func},
    // splicer begin class.Class1.register
    // splicer end class.Class1.register
    {NULL, NULL}   /*sentinel */
};

// void Function1()
static int l_function1(lua_State *)
{
    // splicer begin function.Function1
    tutorial::Function1();
    return 0;
    // splicer end function.Function1
}

// double Function2(double arg1 +intent(in)+value, int arg2 +intent(in)+value)
static int l_function2(lua_State *L)
{
    // splicer begin function.Function2
    double arg1 = lua_tonumber(L, 1);
    int arg2 = lua_tointeger(L, 2);
    double SHCXX_rv = tutorial::Function2(arg1, arg2);
    lua_pushnumber(L, SHCXX_rv);
    return 1;
    // splicer end function.Function2
}

// bool Function3(bool arg +intent(in)+value)
static int l_function3(lua_State *L)
{
    // splicer begin function.Function3
    bool arg = lua_toboolean(L, 1);
    bool SHCXX_rv = tutorial::Function3(arg);
    lua_pushboolean(L, SHCXX_rv);
    return 1;
    // splicer end function.Function3
}

// const std::string Function4a(const std::string & arg1 +intent(in), const std::string & arg2 +intent(in)) +deref(result_as_arg)+len(30)
/**
 * Since +len(30) is provided, the result of the function
 * will be copied directly into memory provided by Fortran.
 * The function will not be ALLOCATABLE.
 */
static int l_function4a(lua_State *L)
{
    // splicer begin function.Function4a
    const char * arg1 = lua_tostring(L, 1);
    const char * arg2 = lua_tostring(L, 2);
    const std::string SHCXX_rv = tutorial::Function4a(arg1, arg2);
    lua_pushstring(L, SHCXX_rv.c_str());
    return 1;
    // splicer end function.Function4a
}

// const std::string & Function4b(const std::string & arg1 +intent(in), const std::string & arg2 +intent(in)) +deref(result_as_arg)
static int l_function4b(lua_State *L)
{
    // splicer begin function.Function4b
    const char * arg1 = lua_tostring(L, 1);
    const char * arg2 = lua_tostring(L, 2);
    const std::string & SHCXX_rv = tutorial::Function4b(arg1, arg2);
    lua_pushstring(L, SHCXX_rv.c_str());
    return 1;
    // splicer end function.Function4b
}

// const std::string Function4c(const std::string & arg1 +intent(in), const std::string & arg2 +intent(in)) +deref(allocatable)
/**
 * Note that since a reference is returned, no intermediate string
 * is allocated.  It is assumed +owner(library).
 */
static int l_function4c(lua_State *L)
{
    // splicer begin function.Function4c
    const char * arg1 = lua_tostring(L, 1);
    const char * arg2 = lua_tostring(L, 2);
    const std::string SHCXX_rv = tutorial::Function4c(arg1, arg2);
    lua_pushstring(L, SHCXX_rv.c_str());
    return 1;
    // splicer end function.Function4c
}

// const std::string * Function4d() +deref(allocatable)+owner(caller)
/**
 * A string is allocated by the library is must be deleted
 * by the caller.
 */
static int l_function4d(lua_State *L)
{
    // splicer begin function.Function4d
    const std::string * SHCXX_rv = tutorial::Function4d();
    lua_pushstring(L, SHCXX_rv->c_str());
    return 1;
    // splicer end function.Function4d
}

// double Function5(double arg1=3.1415 +intent(in)+value, bool arg2=true +intent(in)+value)
static int l_function5(lua_State *L)
{
    // splicer begin function.Function5
    int SH_nresult = 0;
    int SH_nargs = lua_gettop(L);
    int SH_itype1 = lua_type(L, 1);
    int SH_itype2 = lua_type(L, 2);
    switch (SH_nargs) {
    case 0:
        {
            double SHCXX_rv = tutorial::Function5();
            lua_pushnumber(L, SHCXX_rv);
            SH_nresult = 1;
        }
        break;
    case 1:
        if (SH_itype1 == LUA_TNUMBER) {
            double arg1 = lua_tonumber(L, 1);
            double SHCXX_rv = tutorial::Function5(arg1);
            lua_pushnumber(L, SHCXX_rv);
            SH_nresult = 1;
        }
        else {
            luaL_error(L, "error with arguments");
        }
        break;
    case 2:
        if (SH_itype1 == LUA_TNUMBER &&
            SH_itype2 == LUA_TBOOLEAN) {
            double arg1 = lua_tonumber(L, 1);
            bool arg2 = lua_toboolean(L, 2);
            double SHCXX_rv = tutorial::Function5(arg1, arg2);
            lua_pushnumber(L, SHCXX_rv);
            SH_nresult = 1;
        }
        else {
            luaL_error(L, "error with arguments");
        }
        break;
    default:
        luaL_error(L, "error with arguments");
        break;
    }
    return SH_nresult;
    // splicer end function.Function5
}

// void Function6(const std::string & name +intent(in))
// void Function6(int indx +intent(in)+value)
static int l_function6(lua_State *L)
{
    // splicer begin function.Function6
    int SH_nresult = 0;
    int SH_nargs = lua_gettop(L);
    int SH_itype1 = lua_type(L, 1);
    switch (SH_nargs) {
    case 1:
        if (SH_itype1 == LUA_TSTRING) {
            const char * name = lua_tostring(L, 1);
            tutorial::Function6(name);
            SH_nresult = 0;
        }
        else if (SH_itype1 == LUA_TNUMBER) {
            int indx = lua_tointeger(L, 1);
            tutorial::Function6(indx);
            SH_nresult = 0;
        }
        else {
            luaL_error(L, "error with arguments");
        }
        break;
    default:
        luaL_error(L, "error with arguments");
        break;
    }
    return SH_nresult;
    // splicer end function.Function6
}

// void Function7(int arg +intent(in)+value)
// void Function7(double arg +intent(in)+value)
static int l_function7(lua_State *L)
{
    // splicer begin function.Function7
    int SH_nresult = 0;
    int SH_nargs = lua_gettop(L);
    int SH_itype1 = lua_type(L, 1);
    switch (SH_nargs) {
    case 1:
        if (SH_itype1 == LUA_TNUMBER) {
            int arg = lua_tointeger(L, 1);
            tutorial::Function7(arg);
            SH_nresult = 0;
        }
        else if (SH_itype1 == LUA_TNUMBER) {
            double arg = lua_tonumber(L, 1);
            tutorial::Function7(arg);
            SH_nresult = 0;
        }
        else {
            luaL_error(L, "error with arguments");
        }
        break;
    default:
        luaL_error(L, "error with arguments");
        break;
    }
    return SH_nresult;
    // splicer end function.Function7
}

// void Function9(double arg +intent(in)+value)
static int l_function9(lua_State *L)
{
    // splicer begin function.Function9
    double arg = lua_tonumber(L, 1);
    tutorial::Function9(arg);
    return 0;
    // splicer end function.Function9
}

// void Function10()
// void Function10(const std::string & name +intent(in), double arg2 +intent(in)+value)
static int l_function10(lua_State *L)
{
    // splicer begin function.Function10
    int SH_nresult = 0;
    int SH_nargs = lua_gettop(L);
    int SH_itype1 = lua_type(L, 1);
    int SH_itype2 = lua_type(L, 2);
    switch (SH_nargs) {
    case 0:
        {
            tutorial::Function10();
            SH_nresult = 0;
        }
        break;
    case 2:
        if (SH_itype1 == LUA_TSTRING &&
            SH_itype2 == LUA_TNUMBER) {
            const char * name = lua_tostring(L, 1);
            double arg2 = lua_tonumber(L, 2);
            tutorial::Function10(name, arg2);
            SH_nresult = 0;
        }
        else {
            luaL_error(L, "error with arguments");
        }
        break;
    default:
        luaL_error(L, "error with arguments");
        break;
    }
    return SH_nresult;
    // splicer end function.Function10
}

// int overload1(int num +intent(in)+value, int offset=0 +intent(in)+value, int stride=1 +intent(in)+value)
// int overload1(double type +intent(in)+value, int num +intent(in)+value, int offset=0 +intent(in)+value, int stride=1 +intent(in)+value)
static int l_overload1(lua_State *L)
{
    // splicer begin function.overload1
    int SH_nresult = 0;
    int SH_nargs = lua_gettop(L);
    int SH_itype1 = lua_type(L, 1);
    int SH_itype2 = lua_type(L, 2);
    int SH_itype3 = lua_type(L, 3);
    int SH_itype4 = lua_type(L, 4);
    switch (SH_nargs) {
    case 1:
        if (SH_itype1 == LUA_TNUMBER) {
            int num = lua_tointeger(L, 1);
            int SHCXX_rv = tutorial::overload1(num);
            lua_pushinteger(L, SHCXX_rv);
            SH_nresult = 1;
        }
        else {
            luaL_error(L, "error with arguments");
        }
        break;
    case 2:
        if (SH_itype1 == LUA_TNUMBER &&
            SH_itype2 == LUA_TNUMBER) {
            int num = lua_tointeger(L, 1);
            int offset = lua_tointeger(L, 2);
            int SHCXX_rv = tutorial::overload1(num, offset);
            lua_pushinteger(L, SHCXX_rv);
            SH_nresult = 1;
        }
        else if (SH_itype1 == LUA_TNUMBER &&
            SH_itype2 == LUA_TNUMBER) {
            double type = lua_tonumber(L, 1);
            int num = lua_tointeger(L, 2);
            int SHCXX_rv = tutorial::overload1(type, num);
            lua_pushinteger(L, SHCXX_rv);
            SH_nresult = 1;
        }
        else {
            luaL_error(L, "error with arguments");
        }
        break;
    case 3:
        if (SH_itype1 == LUA_TNUMBER &&
            SH_itype2 == LUA_TNUMBER &&
            SH_itype3 == LUA_TNUMBER) {
            int num = lua_tointeger(L, 1);
            int offset = lua_tointeger(L, 2);
            int stride = lua_tointeger(L, 3);
            int SHCXX_rv = tutorial::overload1(num, offset, stride);
            lua_pushinteger(L, SHCXX_rv);
            SH_nresult = 1;
        }
        else if (SH_itype1 == LUA_TNUMBER &&
            SH_itype2 == LUA_TNUMBER &&
            SH_itype3 == LUA_TNUMBER) {
            double type = lua_tonumber(L, 1);
            int num = lua_tointeger(L, 2);
            int offset = lua_tointeger(L, 3);
            int SHCXX_rv = tutorial::overload1(type, num, offset);
            lua_pushinteger(L, SHCXX_rv);
            SH_nresult = 1;
        }
        else {
            luaL_error(L, "error with arguments");
        }
        break;
    case 4:
        if (SH_itype1 == LUA_TNUMBER &&
            SH_itype2 == LUA_TNUMBER &&
            SH_itype3 == LUA_TNUMBER &&
            SH_itype4 == LUA_TNUMBER) {
            double type = lua_tonumber(L, 1);
            int num = lua_tointeger(L, 2);
            int offset = lua_tointeger(L, 3);
            int stride = lua_tointeger(L, 4);
            int SHCXX_rv = tutorial::overload1(type, num, offset,
                stride);
            lua_pushinteger(L, SHCXX_rv);
            SH_nresult = 1;
        }
        else {
            luaL_error(L, "error with arguments");
        }
        break;
    default:
        luaL_error(L, "error with arguments");
        break;
    }
    return SH_nresult;
    // splicer end function.overload1
}

// TypeID typefunc(TypeID arg +intent(in)+value)
static int l_typefunc(lua_State *L)
{
    // splicer begin function.typefunc
    tutorial::TypeID arg = lua_tointeger(L, 1);
    tutorial::TypeID SHCXX_rv = tutorial::typefunc(arg);
    lua_pushinteger(L, SHCXX_rv);
    return 1;
    // splicer end function.typefunc
}

// EnumTypeID enumfunc(EnumTypeID arg +intent(in)+value)
static int l_enumfunc(lua_State *L)
{
    // splicer begin function.enumfunc
    tutorial::EnumTypeID arg =
        static_cast<tutorial::EnumTypeID>(lua_tointeger(L, 1));
    tutorial::EnumTypeID SHCXX_rv = tutorial::enumfunc(arg);
    lua_pushinteger(L, static_cast<int>(SHCXX_rv));
    return 1;
    // splicer end function.enumfunc
}

// Color colorfunc(Color arg +intent(in)+value)
static int l_colorfunc(lua_State *L)
{
    // splicer begin function.colorfunc
    tutorial::Color arg =
        static_cast<tutorial::Color>(lua_tointeger(L, 1));
    tutorial::Color SHCXX_rv = tutorial::colorfunc(arg);
    lua_pushinteger(L, static_cast<int>(SHCXX_rv));
    return 1;
    // splicer end function.colorfunc
}

// Class1::DIRECTION directionFunc(Class1::DIRECTION arg +intent(in)+value)
static int l_direction_func(lua_State *L)
{
    // splicer begin function.directionFunc
    tutorial::Class1::DIRECTION arg =
        static_cast<tutorial::Class1::DIRECTION>(lua_tointeger(L, 1));
    tutorial::Class1::DIRECTION SHCXX_rv = tutorial::directionFunc(arg);
    lua_pushinteger(L, static_cast<int>(SHCXX_rv));
    return 1;
    // splicer end function.directionFunc
}

// const std::string & LastFunctionCalled() +deref(result_as_arg)+len(30)
static int l_last_function_called(lua_State *L)
{
    // splicer begin function.LastFunctionCalled
    const std::string & SHCXX_rv = tutorial::LastFunctionCalled();
    lua_pushstring(L, SHCXX_rv.c_str());
    return 1;
    // splicer end function.LastFunctionCalled
}

// splicer begin additional_functions
// splicer end additional_functions

static const struct luaL_Reg l_Tutorial_Reg [] = {
    {"Class1", l_class1_new},
    {"Function1", l_function1},
    {"Function2", l_function2},
    {"Function3", l_function3},
    {"Function4a", l_function4a},
    {"Function4b", l_function4b},
    {"Function4c", l_function4c},
    {"Function4d", l_function4d},
    {"Function5", l_function5},
    {"Function6", l_function6},
    {"Function7", l_function7},
    {"Function9", l_function9},
    {"Function10", l_function10},
    {"overload1", l_overload1},
    {"typefunc", l_typefunc},
    {"enumfunc", l_enumfunc},
    {"colorfunc", l_colorfunc},
    {"directionFunc", l_direction_func},
    {"LastFunctionCalled", l_last_function_called},
    // splicer begin register
    // splicer end register
    {NULL, NULL}   /*sentinel */
};

#ifdef __cplusplus
extern "C" {
#endif
int luaopen_tutorial(lua_State *L) {

    /* Create the metatable and put it on the stack. */
    luaL_newmetatable(L, "struct1.metatable");
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
    luaL_register(L, NULL, l_struct1_Reg);
#else
    luaL_setfuncs(L, l_struct1_Reg, 0);
#endif


    /* Create the metatable and put it on the stack. */
    luaL_newmetatable(L, "Class1.metatable");
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
    luaL_register(L, NULL, l_Class1_Reg);
#else
    luaL_setfuncs(L, l_Class1_Reg, 0);
#endif


#if LUA_VERSION_NUM < 502
    luaL_register(L, "tutorial", l_Tutorial_Reg);
#else
    luaL_newlib(L, l_Tutorial_Reg);
#endif
    return 1;
}
#ifdef __cplusplus
}
#endif
