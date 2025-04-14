// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
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
static std::string static_str = std::string("dog");
static std::string global_str;
static std::string static_str_empty;

static const int MAXSTRS = 4;
static std::string strs_array[MAXSTRS];

//----------------------------------------
// Initialize datastructures for test

void init_test(void)
{
    strs_array[0] = "apple";
    strs_array[1] = "pear";
    strs_array[2] = "peach";
    strs_array[3] = "cherry";
}

//----------------------------------------

const std::string getConstStringLen()
{
    return static_str;
}

const std::string getConstStringAlloc()
{
    return std::string("getConstStringAlloc");
}

const std::string getConstStringPointer()
{
    return std::string("getConstStringPointer");
}

const std::string getConstStringRaw()
{
    return std::string("bird");
}

const std::string getConstStringAsArg()
{
    return static_str;
}

// -----

const std::string& getConstStringRefLen()
{
    return static_str;
}

const std::string& getConstStringRefLenEmpty()
{
    return static_str_empty;
}

// start getConstStringRefAlloc
const std::string& getConstStringRefAlloc()
{
    return static_str;
}
// end getConstStringRefAlloc

const std::string& getConstStringRefAsArg()
{
    return static_str;
}

// -----

const std::string *getConstStringPtrLen()
{
    // +owner(caller)
    std::string * rv = new std::string("getConstStringPtrLen");
    return rv;
}

const std::string *getConstStringPtrAlloc()
{
    // +owner(library)
    return &static_str;
}

const std::string *getConstStringPtrOwnsAlloc()
{
    // +owner(caller)
    std::string * rv = new std::string("getConstStringPtrOwnsAlloc");
    return rv;
}

const std::string *getConstStringPtrOwnsAllocPattern()
{
    // +owner(caller) +pattern
    std::string * rv = new std::string("getConstStringPtrOwnsAllocPattern");
    return rv;
}

const std::string *getConstStringPtrPointer()
{
    // +deref(pointer) +owner(library)
    return &static_str;
}

const std::string *getConstStringPtrOwnsPointer()
{
    // +deref(pointer) +owner(caller)
    std::string * rv = new std::string("getConstStringPtrOwnsPointer");
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

void acceptStringPointerConst(const std::string * arg1)
{
    global_str = *arg1;
}

void acceptStringPointer(std::string * arg1)
{
    arg1->append("dog");
}

void fetchStringPointer(std::string * arg1)
{
    *arg1 = global_str;
}

void acceptStringPointerLen(std::string * arg1, int *len)
{
    arg1->append("dog");
    *len = arg1->size();
}

void fetchStringPointerLen(std::string * arg1, int *len)
{
    *arg1 = global_str;
    *len = arg1->size();
}

// Return length of string
int acceptStringInstance(std::string arg1)
{
    arg1[0] = 'X';
    return arg1.length();
}

void returnStrings(std::string & arg1, std::string & arg2)
{
    arg1 = "up";
    arg2 = "down";
}

//----------------------------------------

void fetchArrayStringArg(std::string **strs, int *nstrs)
{
    *strs = strs_array;
    *nstrs = MAXSTRS;
}

void fetchArrayStringAlloc(std::string **strs, int *nstrs)
{
    *strs = strs_array;
    *nstrs = MAXSTRS;
}

void fetchArrayStringAllocLen(std::string **strs, int *nstrs)
{
    *strs = strs_array;
    *nstrs = MAXSTRS;
}

//----------------------------------------

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

void PostDeclare(int *count, std::string &name)
{
}

//----------------------------------------
