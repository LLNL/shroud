// luaclassesmodule.cpp
// This file is generated by Shroud nowrite-version. Do not edit.
// Copyright (c) 2017-2021, Lawrence Livermore National Security, LLC and
// other Shroud Project Developers.
// See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//

// cxx_header
#include "classes.hpp"
// shroud
#include "luaclassesmodule.hpp"
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

// Class1(void)
// Class1(int flag +value)
// ----------------------------------------
// Function:  Class1
// Attrs:     +intent(result)
// Requested: lua_shadow_scalar_ctor
// Match:     lua_shadow_ctor
// ----------------------------------------
// Function:  Class1
// Attrs:     +intent(result)
// Requested: lua_shadow_scalar_ctor
// Match:     lua_shadow_ctor
// ----------------------------------------
// Argument:  int flag +value
// Attrs:     +intent(in)
// Exact:     lua_native_scalar_in
static int l_Class1_ctor(lua_State *L)
{
    // splicer begin class.Class1.method.ctor
    int SH_nresult = 0;
    int SH_nargs = lua_gettop(L);
    int SH_itype1 = lua_type(L, 1);
    switch (SH_nargs) {
    case 0:
        {
            l_Class1_Type * SH_this =
                (l_Class1_Type *) lua_newuserdata(L, sizeof(*SH_this));
            SH_this->self = new classes::Class1();
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
            SH_this->self = new classes::Class1(flag);
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
    // splicer end class.Class1.method.ctor
}

// ~Class1(void) +name(delete)
// ----------------------------------------
// Function:  ~Class1 +name(delete)
// Requested: lua_shadow_scalar_dtor
// Match:     lua_shadow_dtor
static int l_Class1_delete(lua_State *L)
{
    // splicer begin class.Class1.method.__gc
    l_Class1_Type * SH_this = (l_Class1_Type *) luaL_checkudata(
        L, 1, "Class1.metatable");
    delete SH_this->self;
    SH_this->self = NULL;
    return 0;
    // splicer end class.Class1.method.__gc
}

// int Method1(void)
// ----------------------------------------
// Function:  int Method1
// Attrs:     +intent(result)
// Exact:     lua_native_scalar_result
/**
 * \brief returns the value of flag member
 *
 */
static int l_Class1_method1(lua_State *L)
{
    // splicer begin class.Class1.method.Method1
    l_Class1_Type * SH_this = (l_Class1_Type *) luaL_checkudata(
        L, 1, "Class1.metatable");
    int SHCXX_rv = SH_this->self->Method1();
    lua_pushinteger(L, SHCXX_rv);
    return 1;
    // splicer end class.Class1.method.Method1
}

// const std::string & getName(void) +deref(allocatable)
// ----------------------------------------
// Function:  const std::string & getName +deref(allocatable)
// Attrs:     +deref(allocatable)+intent(result)
// Exact:     lua_string_&_result
/**
 * \brief test helper
 *
 */
static int l_Class1_get_name(lua_State *L)
{
    // splicer begin class.Class1.method.getName
    l_Class1_Type * SH_this = (l_Class1_Type *) luaL_checkudata(
        L, 1, "Class1.metatable");
    const std::string & SHCXX_rv = SH_this->self->getName();
    lua_pushstring(L, SHCXX_rv.c_str());
    return 1;
    // splicer end class.Class1.method.getName
}

// DIRECTION directionFunc(DIRECTION arg +value)
// ----------------------------------------
// Function:  DIRECTION directionFunc
// Attrs:     +intent(result)
// Exact:     lua_native_scalar_result
// ----------------------------------------
// Argument:  DIRECTION arg +value
// Attrs:     +intent(in)
// Exact:     lua_native_scalar_in
static int l_Class1_direction_func(lua_State *L)
{
    // splicer begin class.Class1.method.directionFunc
    classes::Class1::DIRECTION arg =
        static_cast<classes::Class1::DIRECTION>(lua_tointeger(L, 1));
    l_Class1_Type * SH_this = (l_Class1_Type *) luaL_checkudata(
        L, 1, "Class1.metatable");
    classes::Class1::DIRECTION SHCXX_rv =
        SH_this->self->directionFunc(arg);
    lua_pushinteger(L, static_cast<int>(SHCXX_rv));
    return 1;
    // splicer end class.Class1.method.directionFunc
}

// splicer begin class.Class1.additional_functions
// splicer end class.Class1.additional_functions

static const struct luaL_Reg l_Class1_Reg [] = {
    {"__gc", l_Class1_delete},
    {"Method1", l_Class1_method1},
    {"getName", l_Class1_get_name},
    {"directionFunc", l_Class1_direction_func},
    // splicer begin class.Class1.register
    // splicer end class.Class1.register
    {NULL, NULL}   /*sentinel */
};

// const std::string & getName(void) +deref(allocatable)
// ----------------------------------------
// Function:  const std::string & getName +deref(allocatable)
// Attrs:     +deref(allocatable)+intent(result)
// Exact:     lua_string_&_result
/**
 * \brief test helper
 *
 */
static int l_Class2_get_name(lua_State *L)
{
    // splicer begin class.Class2.method.getName
    l_Class2_Type * SH_this = (l_Class2_Type *) luaL_checkudata(
        L, 1, "Class2.metatable");
    const std::string & SHCXX_rv = SH_this->self->getName();
    lua_pushstring(L, SHCXX_rv.c_str());
    return 1;
    // splicer end class.Class2.method.getName
}

// splicer begin class.Class2.additional_functions
// splicer end class.Class2.additional_functions

static const struct luaL_Reg l_Class2_Reg [] = {
    {"getName", l_Class2_get_name},
    // splicer begin class.Class2.register
    // splicer end class.Class2.register
    {NULL, NULL}   /*sentinel */
};

// Shape(void)
// ----------------------------------------
// Function:  Shape
// Attrs:     +intent(result)
// Requested: lua_shadow_scalar_ctor
// Match:     lua_shadow_ctor
static int l_Shape_ctor(lua_State *L)
{
    // splicer begin class.Shape.method.ctor
    l_Shape_Type * SH_this =
        (l_Shape_Type *) lua_newuserdata(L, sizeof(*SH_this));
    SH_this->self = new classes::Shape();
    /* Add the metatable to the stack. */
    luaL_getmetatable(L, "Shape.metatable");
    /* Set the metatable on the userdata. */
    lua_setmetatable(L, -2);
    return 1;
    // splicer end class.Shape.method.ctor
}

// int get_ivar(void) const
// ----------------------------------------
// Function:  int get_ivar
// Attrs:     +intent(result)
// Exact:     lua_native_scalar_result
static int l_Shape_get_ivar(lua_State *L)
{
    // splicer begin class.Shape.method.get_ivar
    l_Shape_Type * SH_this = (l_Shape_Type *) luaL_checkudata(
        L, 1, "Shape.metatable");
    int SHCXX_rv = SH_this->self->get_ivar();
    lua_pushinteger(L, SHCXX_rv);
    return 1;
    // splicer end class.Shape.method.get_ivar
}

// splicer begin class.Shape.additional_functions
// splicer end class.Shape.additional_functions

static const struct luaL_Reg l_Shape_Reg [] = {
    {"get_ivar", l_Shape_get_ivar},
    // splicer begin class.Shape.register
    // splicer end class.Shape.register
    {NULL, NULL}   /*sentinel */
};

// Circle(void)
// ----------------------------------------
// Function:  Circle
// Attrs:     +intent(result)
// Requested: lua_shadow_scalar_ctor
// Match:     lua_shadow_ctor
static int l_Circle_ctor(lua_State *L)
{
    // splicer begin class.Circle.method.ctor
    l_Circle_Type * SH_this =
        (l_Circle_Type *) lua_newuserdata(L, sizeof(*SH_this));
    SH_this->self = new classes::Circle();
    /* Add the metatable to the stack. */
    luaL_getmetatable(L, "Circle.metatable");
    /* Set the metatable on the userdata. */
    lua_setmetatable(L, -2);
    return 1;
    // splicer end class.Circle.method.ctor
}

// splicer begin class.Circle.additional_functions
// splicer end class.Circle.additional_functions

static const struct luaL_Reg l_Circle_Reg [] = {
    // splicer begin class.Circle.register
    // splicer end class.Circle.register
    {NULL, NULL}   /*sentinel */
};

// Class1::DIRECTION directionFunc(Class1::DIRECTION arg +value)
// ----------------------------------------
// Function:  Class1::DIRECTION directionFunc
// Attrs:     +intent(result)
// Exact:     lua_native_scalar_result
// ----------------------------------------
// Argument:  Class1::DIRECTION arg +value
// Attrs:     +intent(in)
// Exact:     lua_native_scalar_in
static int l_direction_func(lua_State *L)
{
    // splicer begin function.directionFunc
    classes::Class1::DIRECTION arg =
        static_cast<classes::Class1::DIRECTION>(lua_tointeger(L, 1));
    classes::Class1::DIRECTION SHCXX_rv = classes::directionFunc(arg);
    lua_pushinteger(L, static_cast<int>(SHCXX_rv));
    return 1;
    // splicer end function.directionFunc
}

// void set_global_flag(int arg +value)
// ----------------------------------------
// Function:  void set_global_flag
// Exact:     lua_subroutine
// ----------------------------------------
// Argument:  int arg +value
// Attrs:     +intent(in)
// Exact:     lua_native_scalar_in
static int l_set_global_flag(lua_State *L)
{
    // splicer begin function.set_global_flag
    int arg = lua_tointeger(L, 1);
    classes::set_global_flag(arg);
    return 0;
    // splicer end function.set_global_flag
}

// int get_global_flag(void)
// ----------------------------------------
// Function:  int get_global_flag
// Attrs:     +intent(result)
// Exact:     lua_native_scalar_result
static int l_get_global_flag(lua_State *L)
{
    // splicer begin function.get_global_flag
    int SHCXX_rv = classes::get_global_flag();
    lua_pushinteger(L, SHCXX_rv);
    return 1;
    // splicer end function.get_global_flag
}

// const std::string & LastFunctionCalled(void) +deref(result-as-arg)+len(30)
// ----------------------------------------
// Function:  const std::string & LastFunctionCalled +deref(result-as-arg)+len(30)
// Attrs:     +deref(result-as-arg)+intent(result)
// Exact:     lua_string_&_result
static int l_last_function_called(lua_State *L)
{
    // splicer begin function.LastFunctionCalled
    const std::string & SHCXX_rv = classes::LastFunctionCalled();
    lua_pushstring(L, SHCXX_rv.c_str());
    return 1;
    // splicer end function.LastFunctionCalled
}

// splicer begin additional_functions
// splicer end additional_functions

static const struct luaL_Reg l_classes_Reg [] = {
    {"Class1", l_Class1_ctor},
    {"Shape", l_Shape_ctor},
    {"Circle", l_Circle_ctor},
    {"directionFunc", l_direction_func},
    {"set_global_flag", l_set_global_flag},
    {"get_global_flag", l_get_global_flag},
    {"LastFunctionCalled", l_last_function_called},
    // splicer begin register
    // splicer end register
    {NULL, NULL}   /*sentinel */
};

#ifdef __cplusplus
extern "C" {
#endif
int luaopen_classes(lua_State *L) {

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


    /* Create the metatable and put it on the stack. */
    luaL_newmetatable(L, "Shape.metatable");
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
    luaL_register(L, NULL, l_Shape_Reg);
#else
    luaL_setfuncs(L, l_Shape_Reg, 0);
#endif


    /* Create the metatable and put it on the stack. */
    luaL_newmetatable(L, "Circle.metatable");
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
    luaL_register(L, NULL, l_Circle_Reg);
#else
    luaL_setfuncs(L, l_Circle_Reg, 0);
#endif


#if LUA_VERSION_NUM < 502
    luaL_register(L, "classes", l_classes_Reg);
#else
    luaL_newlib(L, l_classes_Reg);
#endif
    return 1;
}
#ifdef __cplusplus
}
#endif
