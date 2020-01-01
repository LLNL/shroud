// luaTutorialmodule.cpp
// This is generated code, do not edit
// Copyright (c) 2017-2020, Lawrence Livermore National Security, LLC and
// other Shroud Project Developers.
// See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//
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

// Class1() +name(new)
// Class1(int flag +intent(in)+value) +name(new)
static int l_Class1_new(lua_State *L)
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

// int Method1()
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

// DIRECTION directionFunc(DIRECTION arg +intent(in)+value)
static int l_Class1_direction_func(lua_State *L)
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
    {"__gc", l_Class1_delete},
    {"Method1", l_Class1_method1},
    {"directionFunc", l_Class1_direction_func},
    // splicer begin class.Class1.register
    // splicer end class.Class1.register
    {NULL, NULL}   /*sentinel */
};

// void NoReturnNoArguments()
static int l_no_return_no_arguments(lua_State *)
{
    // splicer begin function.NoReturnNoArguments
    tutorial::NoReturnNoArguments();
    return 0;
    // splicer end function.NoReturnNoArguments
}

// double PassByValue(double arg1 +intent(in)+value, int arg2 +intent(in)+value)
static int l_pass_by_value(lua_State *L)
{
    // splicer begin function.PassByValue
    double arg1 = lua_tonumber(L, 1);
    int arg2 = lua_tointeger(L, 2);
    double SHCXX_rv = tutorial::PassByValue(arg1, arg2);
    lua_pushnumber(L, SHCXX_rv);
    return 1;
    // splicer end function.PassByValue
}

// const std::string ConcatenateStrings(const std::string & arg1 +intent(in), const std::string & arg2 +intent(in)) +deref(allocatable)
/**
 * Note that since a reference is returned, no intermediate string
 * is allocated.  It is assumed +owner(library).
 */
static int l_concatenate_strings(lua_State *L)
{
    // splicer begin function.ConcatenateStrings
    const char * arg1 = lua_tostring(L, 1);
    const char * arg2 = lua_tostring(L, 2);
    const std::string SHCXX_rv = tutorial::ConcatenateStrings(arg1,
        arg2);
    lua_pushstring(L, SHCXX_rv.c_str());
    return 1;
    // splicer end function.ConcatenateStrings
}

// double UseDefaultArguments(double arg1=3.1415 +intent(in)+value, bool arg2=true +intent(in)+value)
static int l_use_default_arguments(lua_State *L)
{
    // splicer begin function.UseDefaultArguments
    int SH_nresult = 0;
    int SH_nargs = lua_gettop(L);
    int SH_itype1 = lua_type(L, 1);
    int SH_itype2 = lua_type(L, 2);
    switch (SH_nargs) {
    case 0:
        {
            double SHCXX_rv = tutorial::UseDefaultArguments();
            lua_pushnumber(L, SHCXX_rv);
            SH_nresult = 1;
        }
        break;
    case 1:
        if (SH_itype1 == LUA_TNUMBER) {
            double arg1 = lua_tonumber(L, 1);
            double SHCXX_rv = tutorial::UseDefaultArguments(arg1);
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
            double SHCXX_rv = tutorial::UseDefaultArguments(arg1, arg2);
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
    // splicer end function.UseDefaultArguments
}

// void OverloadedFunction(const std::string & name +intent(in))
// void OverloadedFunction(int indx +intent(in)+value)
static int l_overloaded_function(lua_State *L)
{
    // splicer begin function.OverloadedFunction
    int SH_nresult = 0;
    int SH_nargs = lua_gettop(L);
    int SH_itype1 = lua_type(L, 1);
    switch (SH_nargs) {
    case 1:
        if (SH_itype1 == LUA_TSTRING) {
            const char * name = lua_tostring(L, 1);
            tutorial::OverloadedFunction(name);
            SH_nresult = 0;
        }
        else if (SH_itype1 == LUA_TNUMBER) {
            int indx = lua_tointeger(L, 1);
            tutorial::OverloadedFunction(indx);
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
    // splicer end function.OverloadedFunction
}

// void TemplateArgument(int arg +intent(in)+value)
// void TemplateArgument(double arg +intent(in)+value)
static int l_template_argument(lua_State *L)
{
    // splicer begin function.TemplateArgument
    int SH_nresult = 0;
    int SH_nargs = lua_gettop(L);
    int SH_itype1 = lua_type(L, 1);
    switch (SH_nargs) {
    case 1:
        if (SH_itype1 == LUA_TNUMBER) {
            int arg = lua_tointeger(L, 1);
            tutorial::TemplateArgument(arg);
            SH_nresult = 0;
        }
        else if (SH_itype1 == LUA_TNUMBER) {
            double arg = lua_tonumber(L, 1);
            tutorial::TemplateArgument(arg);
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
    // splicer end function.TemplateArgument
}

// void FortranGenericOverloaded()
// void FortranGenericOverloaded(const std::string & name +intent(in), double arg2 +intent(in)+value)
static int l_fortran_generic_overloaded(lua_State *L)
{
    // splicer begin function.FortranGenericOverloaded
    int SH_nresult = 0;
    int SH_nargs = lua_gettop(L);
    int SH_itype1 = lua_type(L, 1);
    int SH_itype2 = lua_type(L, 2);
    switch (SH_nargs) {
    case 0:
        {
            tutorial::FortranGenericOverloaded();
            SH_nresult = 0;
        }
        break;
    case 2:
        if (SH_itype1 == LUA_TSTRING &&
            SH_itype2 == LUA_TNUMBER) {
            const char * name = lua_tostring(L, 1);
            double arg2 = lua_tonumber(L, 2);
            tutorial::FortranGenericOverloaded(name, arg2);
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
    // splicer end function.FortranGenericOverloaded
}

// int UseDefaultOverload(int num +intent(in)+value, int offset=0 +intent(in)+value, int stride=1 +intent(in)+value)
// int UseDefaultOverload(double type +intent(in)+value, int num +intent(in)+value, int offset=0 +intent(in)+value, int stride=1 +intent(in)+value)
static int l_use_default_overload(lua_State *L)
{
    // splicer begin function.UseDefaultOverload
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
            int SHCXX_rv = tutorial::UseDefaultOverload(num);
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
            int SHCXX_rv = tutorial::UseDefaultOverload(num, offset);
            lua_pushinteger(L, SHCXX_rv);
            SH_nresult = 1;
        }
        else if (SH_itype1 == LUA_TNUMBER &&
            SH_itype2 == LUA_TNUMBER) {
            double type = lua_tonumber(L, 1);
            int num = lua_tointeger(L, 2);
            int SHCXX_rv = tutorial::UseDefaultOverload(type, num);
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
            int SHCXX_rv = tutorial::UseDefaultOverload(num, offset,
                stride);
            lua_pushinteger(L, SHCXX_rv);
            SH_nresult = 1;
        }
        else if (SH_itype1 == LUA_TNUMBER &&
            SH_itype2 == LUA_TNUMBER &&
            SH_itype3 == LUA_TNUMBER) {
            double type = lua_tonumber(L, 1);
            int num = lua_tointeger(L, 2);
            int offset = lua_tointeger(L, 3);
            int SHCXX_rv = tutorial::UseDefaultOverload(type, num,
                offset);
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
            int SHCXX_rv = tutorial::UseDefaultOverload(type, num,
                offset, stride);
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
    // splicer end function.UseDefaultOverload
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

// void set_global_flag(int arg +intent(in)+value)
static int l_set_global_flag(lua_State *L)
{
    // splicer begin function.set_global_flag
    int arg = lua_tointeger(L, 1);
    tutorial::set_global_flag(arg);
    return 0;
    // splicer end function.set_global_flag
}

// int get_global_flag()
static int l_get_global_flag(lua_State *L)
{
    // splicer begin function.get_global_flag
    int SHCXX_rv = tutorial::get_global_flag();
    lua_pushinteger(L, SHCXX_rv);
    return 1;
    // splicer end function.get_global_flag
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
    {"Class1", l_Class1_new},
    {"NoReturnNoArguments", l_no_return_no_arguments},
    {"PassByValue", l_pass_by_value},
    {"ConcatenateStrings", l_concatenate_strings},
    {"UseDefaultArguments", l_use_default_arguments},
    {"OverloadedFunction", l_overloaded_function},
    {"TemplateArgument", l_template_argument},
    {"FortranGenericOverloaded", l_fortran_generic_overloaded},
    {"UseDefaultOverload", l_use_default_overload},
    {"typefunc", l_typefunc},
    {"enumfunc", l_enumfunc},
    {"colorfunc", l_colorfunc},
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
int luaopen_tutorial(lua_State *L) {

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
