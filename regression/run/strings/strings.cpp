// Copyright (c) 2017-2020, Lawrence Livermore National Security, LLC and
// other Shroud Project Developers.
// See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
// #######################################################################
//
// strings.cpp
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

// start passCharPtr
void passCharPtr(char *dest, const char *src)
{
    std::strcpy(dest, src);
}
// end passCharPtr

void passCharPtrInOut(char *s)
{
    size_t n = strlen(s);
    for (unsigned int i = 0; i < n; i++) {
        s[i] = toupper(s[i]);
    }
}

//----------------------------------------

// start getCharPtr1
const char * getCharPtr1()
{
    return static_char;
}
// end getCharPtr1

// start getCharPtr2
const char * getCharPtr2()
{
    return static_char;
}
// end getCharPtr2

// start getCharPtr3
const char * getCharPtr3()
{
    return static_char;
}
// end getCharPtr3

//----------------------------------------

// start getConstStringRefPure
const std::string& getConstStringRefPure()
{
    return static_str;
}
// end getConstStringRefPure

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

const std::string getConstStringResult()
{
    const std::string rv("getConstStringResult");
    return rv;
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

const std::string * getConstStringPtrLen()
{
    // caller_owns_return = True
    // C_finalize_buf: delete {cxx_var};
    std::string * rv = new std::string("getConstStringPtrLen");
    return rv;
}

const std::string * getConstStringPtrAlloc()
{
    // +owner(library)
    return &static_str;
}

const std::string * getConstStringPtrOwnsAlloc()
{
    // +owner(caller)
    std::string * rv = new std::string("getConstStringPtrOwnsAlloc");
    return rv;
}

const std::string * getConstStringPtrOwnsAllocPattern()
{
    // +owner(caller) +pattern
    std::string * rv = new std::string("getConstStringPtrOwnsAllocPattern");
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

// start acceptStringReference
void acceptStringReference(std::string & arg1)
{
    arg1.append("dog");
}
// end acceptStringReference

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

void PostDeclare(int *count, std::string &name)
{
}

//----------------------------------------
