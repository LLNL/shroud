/*
 * Copyright (c) 2017-2018, Lawrence Livermore National Security, LLC. 
 * Produced at the Lawrence Livermore National Laboratory 
 *
 * LLNL-CODE-738041.
 * All rights reserved. 
 *
 * This file is part of Shroud.  For details, see
 * https://github.com/LLNL/shroud. Please also read shroud/LICENSE.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are
 * met:
 *
 * * Redistributions of source code must retain the above copyright
 *   notice, this list of conditions and the disclaimer below.
 * 
 * * Redistributions in binary form must reproduce the above copyright
 *   notice, this list of conditions and the disclaimer (as noted below)
 *   in the documentation and/or other materials provided with the
 *   distribution.
 *
 * * Neither the name of the LLNS/LLNL nor the names of its contributors
 *   may be used to endorse or promote products derived from this
 *   software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 * A PARTICULAR PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL LAWRENCE
 * LIVERMORE NATIONAL SECURITY, LLC, THE U.S. DEPARTMENT OF ENERGY OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
 * LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
 * NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 * #######################################################################
 */
#include <stdio.h>
#include <string.h>
#include <lua.h>
#include <lauxlib.h>
#include <lualib.h>

//#include <luaTutorialmodule.hpp>
int luaopen_tutorial(lua_State *L);

int main (void)
{
    lua_State *L = luaL_newstate();   /* opens Lua */
    luaL_openlibs(L);
    luaL_requiref(L, "tutorial", luaopen_tutorial, 1);    

#if 0
    char buff[256];
    int error;
    while (fgets(buff, sizeof(buff), stdin) != NULL) {
        error = luaL_loadbuffer(L, buff, strlen(buff), "line") ||
	    lua_pcall(L, 0, 0, 0);
        if (error) {
	    fprintf(stderr, "%s", lua_tostring(L, -1));
	    lua_pop(L, 1);  /* pop error message from the stack */
        }
    }
#else
    if (luaL_dofile(L, "test.lua")) {
	luaL_error(L, "error running script: %s", lua_tostring(L, -1));
    }
#endif
    
    lua_close(L);
    return 0;
}
