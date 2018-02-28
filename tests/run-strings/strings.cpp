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
//
// tutorial.hpp - wrapped routines
//

#include "strings.hpp"
#include <cstring>

static std::string last_function_called;

// These variables exist to avoid warning errors
static const char * static_char = "bird";
static std::string static_str = std::string("dog");
static std::string global_str;
static std::string static_str_empty;

//----------------------------------------

void passChar(char status)
{
    if (status == 'w') {
	global_str = "w";
    }
}

char returnChar()
{
    return 'w';
}

//----------------------------------------

void passCharPtr(char *dest, const char *src)
{
    std::strcpy(dest, src);
}

void passCharPtrInOut(char *s)
{
    size_t n = strlen(s);
    for (unsigned int i = 0; i < n; i++) {
        s[i] = toupper(s[i]);
    }
}

//----------------------------------------

const char * getCharPtr1()
{
    return static_char;
}

const char * getCharPtr2()
{
    return static_char;
}

const char * getCharPtr3()
{
    return static_char;
}

//----------------------------------------

const std::string& getConstStringRefPure()
{
    return static_str;
}

const std::string& getConstStringRefLen()
{
    return static_str;
}

const std::string& getConstStringRefAsArg()
{
    return static_str;
}

const std::string& getConstStringRefLenEmpty()
{
    return static_str_empty;
}

const std::string& getConstStringRefAlloc()
{
    return static_str;
}

// -----

const std::string getString4()
{
    return static_str;
}

const std::string getConstStringLen()
{
    return static_str;
}

const std::string getConstStringAsArg()
{
    return static_str;
}

const std::string getConstStringAlloc()
{
    return std::string("getConstStringAlloc");
}

// -----

const std::string * getString7()
{
    // Caller is responsible to free string
    std::string * rv = new std::string("Hello");
    return rv;
}

const std::string * getConstStringPtrAlloc()
{
    // Caller is responsible to free string
    std::string * rv = new std::string("getConstStringPtrAlloc");
    return rv;
}

//----------------------------------------

void acceptStringConstReference(const std::string & arg1)
{
    global_str = arg1;
}

void acceptStringReferenceOut(std::string & arg1)
{
    arg1 = "dog";
}

void acceptStringReference(std::string & arg1)
{
    arg1.append("dog");
}

void acceptStringPointer(std::string * arg1)
{
  arg1->append("dog");
}

void returnStrings(std::string & arg1, std::string & arg2)
{
  arg1 = "up";
  arg2 = "down";
}

char returnMany(int * arg1)
{
  *arg1 = 100;
  return 'a';
}

//----------------------------------------

char *keep_explicit1;
void explicit1(char * name)
{
  keep_explicit1 = name;
}

void explicit2(char * name)
{
  *name = 'a';
}

//----------------------------------------
//----------------------------------------

extern "C" void CpassChar(char status)
{
    if (status == 'w') {
        global_str = "w";
    }
}

extern "C" char CreturnChar()
{
    return 'w';
}

//----------------------------------------

extern "C" void CpassCharPtr(char *dest, const char *src)
{
    std::strcpy(dest, src);
}

//----------------------------------------
