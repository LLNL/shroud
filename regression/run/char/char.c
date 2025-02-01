// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
// other Shroud Project Developers.
// See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
// #######################################################################
//
// char.c
//

#include "char.h"
#include <stdlib.h>
#include <string.h>
#include <ctype.h>   //  toupper

static char global_char = ' ';

// These variables exist to avoid warning errors
static const char * static_char = "bird";

//----------------------------------------
// Initialize datastructures for test

void init_test(void)
{
}

//----------------------------------------

void passChar(char status)
{
    global_char = status;
}

void passCharForce(char status)
{
    global_char = status;
}

char returnChar()
{
    return global_char;
}

//----------------------------------------

// start passCharPtr
void passCharPtr(char *dest, const char *src)
{
    strcpy(dest, src);
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
const char * getCharPtr1(void)
{
    return static_char;
}
// end getCharPtr1

// start getConstCharPtrLen
const char *getConstCharPtrLen(void)
{
    return static_char;
}
// end getConstCharPtrLen

// start getConstCharPtrAsArg
const char *getConstCharPtrAsArg(void)
{
    return static_char;
}
// end getConstCharPtrAsArg

// +deref(raw)
// start getCharPtr4
const char * getCharPtr4(void)
{
    return static_char;
}
// end getCharPtr4


const char * getCharPtr5(void)
{
    return static_char;
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

// Assigning to a char* without knowing the length can result in an overwrite.

void explicit2(char * name)
{
    *name = 'a';
}

//----------------------------------------
//----------------------------------------

void CpassChar(char status)
{
    if (status == 'w') {
        global_char = status;
    }
}

char CreturnChar()
{
    return 'w';
}

//----------------------------------------
// Check for NULL pointer
// dest is assumed to be long enough.
// attribute +blanknull

void CpassCharPtr(char *dest, const char *src)
{
    if (src == NULL) {
        strcpy(dest, "NULL");
    } else {
        strcpy(dest, src);
    }
}

//----------------------------------------
// Check for NULL pointer
// dest is assumed to be long enough.
// option F_blanknull

void CpassCharPtrBlank(char *dest, const char *src)
{
    if (src == NULL) {
        strcpy(dest, "NULL");
    } else {
        strcpy(dest, src);
    }
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

