// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
// other Shroud Project Developers.
// See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
// #######################################################################
//
// char.h - wrapped routines
//

#ifndef CHAR_HPP
#define CHAR_HPP

void init_test(void);

void passChar(char status);
void passCharForce(char status);
char returnChar();

void passCharPtr(char * dest, const char *src);
void passCharPtrInOut(char * s);

const char *getCharPtr1(void);
const char *getConstCharPtrLen(void);
const char *getCharPtr4(void);
const char *getCharPtr5(void);

const char *getConstCharPtrAsArg(void);
const char *getConstCharPtrAsAllocArg(void);
const char *getConstCharPtrAsPointerArg(void);
const char *getConstCharPtrAsRawArg(void);

char returnMany(int * arg1);

void explicit1(char * name);
void explicit2(char * name);

void CpassChar(char status);
char CreturnChar();

void CpassCharPtr(char * dest, const char *src);
void CpassCharPtrBlank(char *dest, const char *src);

int CpassCharPtrNotrim(const char *src);
int CpassCharPtrCAPI(void *addr, const char *src);
int CpassCharPtrCAPI2(const char *in, const char *src);

int acceptCharArrayIn(char **names);

void fetchCharPtrCopyLibrary(char **outstr);
void fetchCharPtrLibrary(char **outstr);
int fetchCharPtrLibraryNULL(char **outstr);

#endif // CHAR_HPP
