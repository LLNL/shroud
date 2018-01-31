// luaUserLibrarymodule.cpp
// This is generated code, do not edit
// Copyright (c) 2017, Lawrence Livermore National Security, LLC.
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
#include "luaUserLibrarymodule.hpp"
#ifdef __cplusplus
extern "C" {
#endif
#include "lauxlib.h"
#ifdef __cplusplus
}
#endif
// splicer begin include
// splicer end include

namespace example {
namespace nested {
// splicer begin C_definition
// splicer end C_definition

static int l_exclass1_ctor(lua_State *L)
{
    // splicer begin class.ExClass1.method.ctor
    int SH_nresult = 0;
    int SH_nargs = lua_gettop(L);
    int SH_itype1 = lua_type(L, 1);
    switch (SH_nargs) {
    case 0:
        {
            l_ExClass1_Type * SH_this = (l_ExClass1_Type *) lua_newuserdata(L, sizeof(*SH_this));
            SH_this->self = new ExClass1();
            /* Add the metatable to the stack. */
            luaL_getmetatable(L, "ExClass1.metatable");
            /* Set the metatable on the userdata. */
            lua_setmetatable(L, -2);
            SH_nresult = 1;
        }
        break;
    case 1:
        if (SH_itype1 == LUA_TSTRING) {
            const char * name = lua_tostring(L, 1);
            l_ExClass1_Type * SH_this = (l_ExClass1_Type *) lua_newuserdata(L, sizeof(*SH_this));
            SH_this->self = new ExClass1(name);
            /* Add the metatable to the stack. */
            luaL_getmetatable(L, "ExClass1.metatable");
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
    // splicer end class.ExClass1.method.ctor
}

static int l_exclass1_dtor(lua_State *L)
{
    // splicer begin class.ExClass1.method.__gc
    l_ExClass1_Type * SH_this = (l_ExClass1_Type *)luaL_checkudata(L, 1, "ExClass1.metatable");
    delete SH_this->self;
    SH_this->self = NULL;
    return 0;
    // splicer end class.ExClass1.method.__gc
}

static int l_exclass1_increment_count(lua_State *L)
{
    // splicer begin class.ExClass1.method.incrementCount
    int incr = lua_tointeger(L, 1);
    l_ExClass1_Type * SH_this = (l_ExClass1_Type *)luaL_checkudata(L, 1, "ExClass1.metatable");
    int rv = SH_this->self->incrementCount(incr);
    lua_pushinteger(L, rv);
    return 1;
    // splicer end class.ExClass1.method.incrementCount
}

static int l_exclass1_get_name(lua_State *L)
{
    // splicer begin class.ExClass1.method.getName
    l_ExClass1_Type * SH_this = (l_ExClass1_Type *)luaL_checkudata(L, 1, "ExClass1.metatable");
    const std::string & rv = SH_this->self->getName();
    lua_pushstring(L, rv.c_str());
    return 1;
    // splicer end class.ExClass1.method.getName
}

static int l_exclass1_get_name_length(lua_State *L)
{
    // splicer begin class.ExClass1.method.GetNameLength
    l_ExClass1_Type * SH_this = (l_ExClass1_Type *)luaL_checkudata(L, 1, "ExClass1.metatable");
    int rv = SH_this->self->GetNameLength();
    lua_pushinteger(L, rv);
    return 1;
    // splicer end class.ExClass1.method.GetNameLength
}

static int l_exclass1_get_name_error_check(lua_State *L)
{
    // splicer begin class.ExClass1.method.getNameErrorCheck
    l_ExClass1_Type * SH_this = (l_ExClass1_Type *)luaL_checkudata(L, 1, "ExClass1.metatable");
    const std::string & rv = SH_this->self->getNameErrorCheck();
    lua_pushstring(L, rv.c_str());
    return 1;
    // splicer end class.ExClass1.method.getNameErrorCheck
}

static int l_exclass1_get_name_arg(lua_State *L)
{
    // splicer begin class.ExClass1.method.getNameArg
    l_ExClass1_Type * SH_this = (l_ExClass1_Type *)luaL_checkudata(L, 1, "ExClass1.metatable");
    const std::string & rv = SH_this->self->getNameArg();
    lua_pushstring(L, rv.c_str());
    return 1;
    // splicer end class.ExClass1.method.getNameArg
}

static int l_exclass1_get_root(lua_State *L)
{
    // splicer begin class.ExClass1.method.getRoot
    l_ExClass1_Type * SH_this = (l_ExClass1_Type *)luaL_checkudata(L, 1, "ExClass1.metatable");
    ExClass2 * rv = SH_this->self->getRoot();
    PUSH;
    return 1;
    // splicer end class.ExClass1.method.getRoot
}

static int l_exclass1_get_value(lua_State *L)
{
    // splicer begin class.ExClass1.method.getValue
    int SH_nresult = 0;
    int SH_nargs = lua_gettop(L);
    int SH_itype1 = lua_type(L, 1);
    switch (SH_nargs) {
    case 1:
        if (SH_itype1 == LUA_TNUMBER) {
            int value = lua_tointeger(L, 1);
            l_ExClass1_Type * SH_this = (l_ExClass1_Type *)luaL_checkudata(L, 1, "ExClass1.metatable");
            int rv = SH_this->self->getValue(value);
            lua_pushinteger(L, rv);
            SH_nresult = 1;
        }
        else if (SH_itype1 == LUA_TNUMBER) {
            long value = lua_tointeger(L, 1);
            l_ExClass1_Type * SH_this = (l_ExClass1_Type *)luaL_checkudata(L, 1, "ExClass1.metatable");
            long rv = SH_this->self->getValue(value);
            lua_pushinteger(L, rv);
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
    // splicer end class.ExClass1.method.getValue
}

static int l_exclass1_get_addr(lua_State *L)
{
    // splicer begin class.ExClass1.method.getAddr
    l_ExClass1_Type * SH_this = (l_ExClass1_Type *)luaL_checkudata(L, 1, "ExClass1.metatable");
    void * rv = SH_this->self->getAddr();
    PUSH;
    return 1;
    // splicer end class.ExClass1.method.getAddr
}

static int l_exclass1_has_addr(lua_State *L)
{
    // splicer begin class.ExClass1.method.hasAddr
    bool in = lua_toboolean(L, 1);
    l_ExClass1_Type * SH_this = (l_ExClass1_Type *)luaL_checkudata(L, 1, "ExClass1.metatable");
    bool rv = SH_this->self->hasAddr(in);
    lua_pushboolean(L, rv);
    return 1;
    // splicer end class.ExClass1.method.hasAddr
}

static int l_exclass1_splicer_special(lua_State *L)
{
    // splicer begin class.ExClass1.method.SplicerSpecial
    l_ExClass1_Type * SH_this = (l_ExClass1_Type *)luaL_checkudata(L, 1, "ExClass1.metatable");
    SH_this->self->SplicerSpecial();
    return 0;
    // splicer end class.ExClass1.method.SplicerSpecial
}

// splicer begin class.ExClass1.additional_functions
// splicer end class.ExClass1.additional_functions

static const struct luaL_Reg l_ExClass1_Reg [] = {
    {"__gc", l_exclass1_dtor},
    {"incrementCount", l_exclass1_increment_count},
    {"getName", l_exclass1_get_name},
    {"GetNameLength", l_exclass1_get_name_length},
    {"getNameErrorCheck", l_exclass1_get_name_error_check},
    {"getNameArg", l_exclass1_get_name_arg},
    {"getRoot", l_exclass1_get_root},
    {"getValue", l_exclass1_get_value},
    {"getAddr", l_exclass1_get_addr},
    {"hasAddr", l_exclass1_has_addr},
    {"SplicerSpecial", l_exclass1_splicer_special},
    // splicer begin class.ExClass1.register
    // splicer end class.ExClass1.register
    {NULL, NULL}   /*sentinel */
};

static int l_exclass2_ctor(lua_State *L)
{
    // splicer begin class.ExClass2.method.ctor
    const char * name = lua_tostring(L, 1);
    l_ExClass2_Type * SH_this = (l_ExClass2_Type *) lua_newuserdata(L, sizeof(*SH_this));
    SH_this->self = new ExClass2(name);
    /* Add the metatable to the stack. */
    luaL_getmetatable(L, "ExClass2.metatable");
    /* Set the metatable on the userdata. */
    lua_setmetatable(L, -2);
    return 1;
    // splicer end class.ExClass2.method.ctor
}

static int l_exclass2_dtor(lua_State *L)
{
    // splicer begin class.ExClass2.method.__gc
    l_ExClass2_Type * SH_this = (l_ExClass2_Type *)luaL_checkudata(L, 1, "ExClass2.metatable");
    delete SH_this->self;
    SH_this->self = NULL;
    return 0;
    // splicer end class.ExClass2.method.__gc
}

static int l_exclass2_get_name(lua_State *L)
{
    // splicer begin class.ExClass2.method.getName
    l_ExClass2_Type * SH_this = (l_ExClass2_Type *)luaL_checkudata(L, 1, "ExClass2.metatable");
    const std::string & rv = SH_this->self->getName();
    lua_pushstring(L, rv.c_str());
    return 1;
    // splicer end class.ExClass2.method.getName
}

static int l_exclass2_get_name2(lua_State *L)
{
    // splicer begin class.ExClass2.method.getName2
    l_ExClass2_Type * SH_this = (l_ExClass2_Type *)luaL_checkudata(L, 1, "ExClass2.metatable");
    const std::string & rv = SH_this->self->getName2();
    lua_pushstring(L, rv.c_str());
    return 1;
    // splicer end class.ExClass2.method.getName2
}

static int l_exclass2_get_name3(lua_State *L)
{
    // splicer begin class.ExClass2.method.getName3
    l_ExClass2_Type * SH_this = (l_ExClass2_Type *)luaL_checkudata(L, 1, "ExClass2.metatable");
    std::string & rv = SH_this->self->getName3();
    lua_pushstring(L, rv.c_str());
    return 1;
    // splicer end class.ExClass2.method.getName3
}

static int l_exclass2_get_name4(lua_State *L)
{
    // splicer begin class.ExClass2.method.getName4
    l_ExClass2_Type * SH_this = (l_ExClass2_Type *)luaL_checkudata(L, 1, "ExClass2.metatable");
    std::string & rv = SH_this->self->getName4();
    lua_pushstring(L, rv.c_str());
    return 1;
    // splicer end class.ExClass2.method.getName4
}

static int l_exclass2_get_name_length(lua_State *L)
{
    // splicer begin class.ExClass2.method.GetNameLength
    l_ExClass2_Type * SH_this = (l_ExClass2_Type *)luaL_checkudata(L, 1, "ExClass2.metatable");
    int rv = SH_this->self->GetNameLength();
    lua_pushinteger(L, rv);
    return 1;
    // splicer end class.ExClass2.method.GetNameLength
}

static int l_exclass2_get_class1(lua_State *L)
{
    // splicer begin class.ExClass2.method.get_class1
    const ExClass1 * in = static_cast<ExClass1 *>(static_cast<void *>((l_ExClass2_Type *)luaL_checkudata(L, 1, "ExClass2.metatable")));
    l_ExClass2_Type * SH_this = (l_ExClass2_Type *)luaL_checkudata(L, 1, "ExClass2.metatable");
    ExClass1 * rv = SH_this->self->get_class1(in);
    PUSH;
    return 1;
    // splicer end class.ExClass2.method.get_class1
}

static int l_exclass2_declare(lua_State *L)
{
    // splicer begin class.ExClass2.method.declare
    int SH_nresult = 0;
    int SH_nargs = lua_gettop(L);
    int SH_itype1 = lua_type(L, 1);
    int SH_itype2 = lua_type(L, 2);
    switch (SH_nargs) {
    case 1:
        if (SH_itype1 == LUA_TNUMBER) {
            TypeID type = getTypeID(lua_tointeger(L, 1));
            l_ExClass2_Type * SH_this = (l_ExClass2_Type *)luaL_checkudata(L, 1, "ExClass2.metatable");
            SH_this->self->declare(type);
            SH_nresult = 0;
        }
        else {
            luaL_error(L, "error with arguments");
        }
        break;
    case 2:
        if (SH_itype1 == LUA_TNUMBER &&
            SH_itype2 == LUA_TNUMBER) {
            TypeID type = getTypeID(lua_tointeger(L, 1));
            SidreLength len = lua_tointeger(L, 2);
            l_ExClass2_Type * SH_this = (l_ExClass2_Type *)luaL_checkudata(L, 1, "ExClass2.metatable");
            SH_this->self->declare(type, len);
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
    // splicer end class.ExClass2.method.declare
}

static int l_exclass2_destroyall(lua_State *L)
{
    // splicer begin class.ExClass2.method.destroyall
    l_ExClass2_Type * SH_this = (l_ExClass2_Type *)luaL_checkudata(L, 1, "ExClass2.metatable");
    SH_this->self->destroyall();
    return 0;
    // splicer end class.ExClass2.method.destroyall
}

static int l_exclass2_get_type_id(lua_State *L)
{
    // splicer begin class.ExClass2.method.getTypeID
    l_ExClass2_Type * SH_this = (l_ExClass2_Type *)luaL_checkudata(L, 1, "ExClass2.metatable");
    TypeID rv = SH_this->self->getTypeID();
    lua_pushinteger(L, static_cast<int>(rv));
    return 1;
    // splicer end class.ExClass2.method.getTypeID
}

// splicer begin class.ExClass2.additional_functions
// splicer end class.ExClass2.additional_functions

static const struct luaL_Reg l_ExClass2_Reg [] = {
    {"__gc", l_exclass2_dtor},
    {"getName", l_exclass2_get_name},
    {"getName2", l_exclass2_get_name2},
    {"getName3", l_exclass2_get_name3},
    {"getName4", l_exclass2_get_name4},
    {"GetNameLength", l_exclass2_get_name_length},
    {"get_class1", l_exclass2_get_class1},
    {"declare", l_exclass2_declare},
    {"destroyall", l_exclass2_destroyall},
    {"getTypeID", l_exclass2_get_type_id},
    // splicer begin class.ExClass2.register
    // splicer end class.ExClass2.register
    {NULL, NULL}   /*sentinel */
};

static int l_exclass3_exfunc(lua_State *L)
{
    // splicer begin class.ExClass3.method.exfunc
    l_ExClass3_Type * SH_this = (l_ExClass3_Type *)luaL_checkudata(L, 1, "ExClass3.metatable");
    SH_this->self->exfunc();
    return 0;
    // splicer end class.ExClass3.method.exfunc
}

// splicer begin class.ExClass3.additional_functions
// splicer end class.ExClass3.additional_functions

static const struct luaL_Reg l_ExClass3_Reg [] = {
    {"exfunc", l_exclass3_exfunc},
    // splicer begin class.ExClass3.register
    // splicer end class.ExClass3.register
    {NULL, NULL}   /*sentinel */
};

static int l_local_function1(lua_State *)
{
    // splicer begin function.local_function1
    local_function1();
    return 0;
    // splicer end function.local_function1
}

static int l_is_name_valid(lua_State *L)
{
    // splicer begin function.isNameValid
    const char * name = lua_tostring(L, 1);
    bool rv = isNameValid(name);
    lua_pushboolean(L, rv);
    return 1;
    // splicer end function.isNameValid
}

static int l_is_initialized(lua_State *L)
{
    // splicer begin function.isInitialized
    bool rv = isInitialized();
    lua_pushboolean(L, rv);
    return 1;
    // splicer end function.isInitialized
}

static int l_check_bool(lua_State *L)
{
    // splicer begin function.checkBool
    bool arg1 = lua_toboolean(L, 1);
    bool * arg2;
    bool * arg3 = lua_toboolean(L, 2);
    checkBool(arg1, arg2, arg3);
    lua_pushboolean(L, arg2);
    lua_pushboolean(L, lua_toboolean(L, 2));
    return 0;
    // splicer end function.checkBool
}

static int l_test_names(lua_State *L)
{
    // splicer begin function.test_names
    int SH_nresult = 0;
    int SH_nargs = lua_gettop(L);
    int SH_itype1 = lua_type(L, 1);
    int SH_itype2 = lua_type(L, 2);
    switch (SH_nargs) {
    case 1:
        if (SH_itype1 == LUA_TSTRING) {
            const char * name = lua_tostring(L, 1);
            test_names(name);
            SH_nresult = 0;
        }
        else {
            luaL_error(L, "error with arguments");
        }
        break;
    case 2:
        if (SH_itype1 == LUA_TSTRING &&
            SH_itype2 == LUA_TNUMBER) {
            const char * name = lua_tostring(L, 1);
            int flag = lua_tointeger(L, 2);
            test_names(name, flag);
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
    // splicer end function.test_names
}

static int l_testoptional(lua_State *L)
{
    // splicer begin function.testoptional
    int SH_nresult = 0;
    int SH_nargs = lua_gettop(L);
    int SH_itype1 = lua_type(L, 1);
    int SH_itype2 = lua_type(L, 2);
    switch (SH_nargs) {
    case 0:
        {
            testoptional();
            SH_nresult = 0;
        }
        break;
    case 1:
        if (SH_itype1 == LUA_TNUMBER) {
            int i = lua_tointeger(L, 1);
            testoptional(i);
            SH_nresult = 0;
        }
        else {
            luaL_error(L, "error with arguments");
        }
        break;
    case 2:
        if (SH_itype1 == LUA_TNUMBER &&
            SH_itype2 == LUA_TNUMBER) {
            int i = lua_tointeger(L, 1);
            long j = lua_tointeger(L, 2);
            testoptional(i, j);
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
    // splicer end function.testoptional
}

static int l_test_size_t(lua_State *L)
{
    // splicer begin function.test_size_t
    size_t rv = test_size_t();
    lua_pushinteger(L, rv);
    return 1;
    // splicer end function.test_size_t
}

static int l_testmpi(lua_State *L)
{
    // splicer begin function.testmpi
    MPI_Comm comm = MPI_Comm_f2c(POP);
    testmpi(comm);
    return 0;
    // splicer end function.testmpi
}

static int l_testgroup1(lua_State *L)
{
    // splicer begin function.testgroup1
    axom::sidre::Group * grp = static_cast<axom::sidre::Group *>(static_cast<void *>((XXLUA_userdata_type *)luaL_checkudata(L, 1, "XXLUA_metadata")));
    testgroup1(grp);
    return 0;
    // splicer end function.testgroup1
}

static int l_testgroup2(lua_State *L)
{
    // splicer begin function.testgroup2
    const axom::sidre::Group * grp = static_cast<axom::sidre::Group *>(static_cast<void *>((XXLUA_userdata_type *)luaL_checkudata(L, 1, "XXLUA_metadata")));
    testgroup2(grp);
    return 0;
    // splicer end function.testgroup2
}

static int l_func1(lua_State *L)
{
    // splicer begin function.func1
    void ( * get)() = POP;
    func1(get);
    return 0;
    // splicer end function.func1
}

static int l_func2(lua_State *L)
{
    // splicer begin function.func2
    double * ( * get)() = lua_tonumber(L, 1);
    func2(get);
    return 0;
    // splicer end function.func2
}

static int l_func_ptr3(lua_State *L)
{
    // splicer begin function.FuncPtr3
    double ( * get)(int i, int) = lua_tonumber(L, 1);
    FuncPtr3(get);
    return 0;
    // splicer end function.FuncPtr3
}

static int l_func4(lua_State *L)
{
    // splicer begin function.func4
    void ( * get)(int verylongname1, int verylongname2, int verylongname3, int verylongname4, int verylongname5, int verylongname6, int verylongname7, int verylongname8, int verylongname9, int verylongname10) = POP;
    func4(get);
    return 0;
    // splicer end function.func4
}

static int l_verlongfunctionname1(lua_State *L)
{
    // splicer begin function.verlongfunctionname1
    int verylongname1 = lua_tointeger(L, 1);
    int verylongname2 = lua_tointeger(L, 2);
    int verylongname3 = lua_tointeger(L, 3);
    int verylongname4 = lua_tointeger(L, 4);
    int verylongname5 = lua_tointeger(L, 5);
    int verylongname6 = lua_tointeger(L, 6);
    int verylongname7 = lua_tointeger(L, 7);
    int verylongname8 = lua_tointeger(L, 8);
    int verylongname9 = lua_tointeger(L, 9);
    int verylongname10 = lua_tointeger(L, 10);
    verlongfunctionname1(verylongname1, verylongname2, verylongname3,
        verylongname4, verylongname5, verylongname6, verylongname7,
        verylongname8, verylongname9, verylongname10);
    return 0;
    // splicer end function.verlongfunctionname1
}

static int l_verlongfunctionname2(lua_State *L)
{
    // splicer begin function.verlongfunctionname2
    int verylongname1 = lua_tointeger(L, 1);
    int verylongname2 = lua_tointeger(L, 2);
    int verylongname3 = lua_tointeger(L, 3);
    int verylongname4 = lua_tointeger(L, 4);
    int verylongname5 = lua_tointeger(L, 5);
    int verylongname6 = lua_tointeger(L, 6);
    int verylongname7 = lua_tointeger(L, 7);
    int verylongname8 = lua_tointeger(L, 8);
    int verylongname9 = lua_tointeger(L, 9);
    int verylongname10 = lua_tointeger(L, 10);
    int rv = verlongfunctionname2(verylongname1, verylongname2,
        verylongname3, verylongname4, verylongname5, verylongname6,
        verylongname7, verylongname8, verylongname9, verylongname10);
    lua_pushinteger(L, rv);
    return 1;
    // splicer end function.verlongfunctionname2
}

// splicer begin additional_functions
// splicer end additional_functions

static const struct luaL_Reg l_UserLibrary_Reg [] = {
    {"ExClass1_0", l_exclass1_ctor},
    {"ExClass2", l_exclass2_ctor},
    {"local_function1", l_local_function1},
    {"isNameValid", l_is_name_valid},
    {"isInitialized", l_is_initialized},
    {"checkBool", l_check_bool},
    {"test_names", l_test_names},
    {"testoptional", l_testoptional},
    {"test_size_t", l_test_size_t},
    {"testmpi", l_testmpi},
    {"testgroup1", l_testgroup1},
    {"testgroup2", l_testgroup2},
    {"func1", l_func1},
    {"func2", l_func2},
    {"FuncPtr3", l_func_ptr3},
    {"func4", l_func4},
    {"verlongfunctionname1", l_verlongfunctionname1},
    {"verlongfunctionname2", l_verlongfunctionname2},
    // splicer begin register
    // splicer end register
    {NULL, NULL}   /*sentinel */
};

#ifdef __cplusplus
extern "C" {
#endif
int luaopen_userlibrary(lua_State *L) {

    /* Create the metatable and put it on the stack. */
    luaL_newmetatable(L, "ExClass1.metatable");
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
    luaL_register(L, NULL, l_ExClass1_Reg);
#else
    luaL_setfuncs(L, l_ExClass1_Reg, 0);
#endif


    /* Create the metatable and put it on the stack. */
    luaL_newmetatable(L, "ExClass2.metatable");
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
    luaL_register(L, NULL, l_ExClass2_Reg);
#else
    luaL_setfuncs(L, l_ExClass2_Reg, 0);
#endif


    /* Create the metatable and put it on the stack. */
    luaL_newmetatable(L, "ExClass3.metatable");
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
    luaL_register(L, NULL, l_ExClass3_Reg);
#else
    luaL_setfuncs(L, l_ExClass3_Reg, 0);
#endif


#if LUA_VERSION_NUM < 502
    luaL_register(L, "userlibrary", l_UserLibrary_Reg);
#else
    luaL_newlib(L, l_UserLibrary_Reg);
#endif
    return 1;
}
#ifdef __cplusplus
}
#endif

}  // namespace nested
}  // namespace example
