// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
// other Shroud Project Developers.
// See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
// #######################################################################
//
// strings.hpp - wrapped routines
//

#ifndef STRINGS_HPP
#define STRINGS_HPP

#include <string>

void init_test(void);

const std::string getConstStringLen();
const std::string getConstStringAlloc();
const std::string getConstStringPointer();

const std::string getConstStringAsArg();

const std::string &getConstStringRefLen();
const std::string &getConstStringRefLenEmpty();
const std::string &getConstStringRefAlloc();

const std::string &getConstStringRefAsArg();

const std::string *getConstStringPtrLen();
const std::string *getConstStringPtrAlloc();
const std::string *getConstStringPtrOwnsAlloc();
const std::string *getConstStringPtrOwnsAllocPattern();

const std::string *getConstStringPtrPointer();
const std::string *getConstStringPtrOwnsPointer();

void acceptName_instance(std::string arg1);

void acceptStringConstReference(const std::string & arg1);

void acceptStringReferenceOut(std::string & arg1);

void acceptStringReference(std::string & arg1);

void acceptStringPointerConst(const std::string * arg1);

void acceptStringPointer(std::string * arg1);

void fetchStringPointer(std::string * arg1);

void acceptStringPointerLen(std::string * arg1, int *len);

void fetchStringPointerLen(std::string * arg1, int *len);

int acceptStringInstance(std::string arg1);

void returnStrings(std::string & arg1, std::string & arg2);

void fetchArrayStringArg(std::string **strs, int *nstrs);
void fetchArrayStringAlloc(std::string **strs, int *nstrs);
void fetchArrayStringAllocLen(std::string **strs, int *nstrs);

char returnMany(int * arg1);

void explicit1(char * name);
void explicit2(char * name);

extern "C" {
  void CpassChar(char status);
  char CreturnChar();

}

void PostDeclare(int *count, std::string &name);


#endif // STRINGS_HPP
