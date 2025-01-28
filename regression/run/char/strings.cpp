// Copyright (c) 2017-2024, Lawrence Livermore National Security, LLC and
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
static char global_char = ' ';

// These variables exist to avoid warning errors
static const char * static_char = "bird";
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

void passChar(char status)
{
    global_char = status;
    if (status == 'w') {
	global_str = "w";
    }
}

void passCharForce(char status)
{
    global_char = status;
    if (status == 'w') {
	global_str = "w";
    }
}

char returnChar()
{
    return global_char;
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

// +deref(raw)
// start getCharPtr4
const char * getCharPtr4()
{
    return static_char;
}
// end getCharPtr4


const char * getCharPtr5()
{
    return static_char;
}

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
    // +owner(caller)
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

const std::string * getConstStringPtrPointer()
{
    // +deref(pointer) +owner(library)
    return &static_str;
}

const std::string * getConstStringPtrOwnsPointer()
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
// Check for NULL pointer
// dest is assumed to be long enough.
// attribute +blanknull

extern "C" void CpassCharPtr(char *dest, const char *src)
{
    if (src == NULL) {
        std::strcpy(dest, "NULL");
    } else {
        std::strcpy(dest, src);
    }
}

//----------------------------------------
// Check for NULL pointer
// dest is assumed to be long enough.
// option F_blanknull

void CpassCharPtrBlank(char *dest, const char *src)
{
    if (src == NULL) {
        std::strcpy(dest, "NULL");
    } else {
        std::strcpy(dest, src);
    }
}

//----------------------------------------

void PostDeclare(int *count, std::string &name)
{
}

//----------------------------------------

int CpassCharPtrNotrim(const char *src)
{
    return strlen(src);
}

//----------------------------------------

int CpassCharPtrCAPI(void *addr, const char *src)
{
    if (addr == const_cast<char *>(src)) {
        return 1;
    } else {
        return 0;
    }
}

//----------------------------------------
// Check if strings compare, but only 'in' is null terminated.

int CpassCharPtrCAPI2(const char *in, const char *src)
{
    size_t n = strlen(in);
    if (strncmp(in, src, n) == 0) {
        return 1;
    } else {
        return 0;
    }
}

