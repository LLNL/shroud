// Copyright (c) 2017-2019, Lawrence Livermore National Security, LLC. 
//
// Produced at the Lawrence Livermore National Laboratory 
//
// LLNL-CODE-738041.
//
// All rights reserved. 
//
// This file is part of Shroud.
//
// For details about use and distribution, please read LICENSE.
//
// #######################################################################
//
// strings.hpp - wrapped routines
//

#ifndef STRINGS_HPP
#define STRINGS_HPP

#include <string>

void passChar(char status);
char returnChar();

void passCharPtr(char * dest, const char *src);
void passCharPtrInOut(char * s);

const char * getCharPtr1();
const char * getCharPtr2();
const char * getCharPtr3();

const std::string getConstStringResult();
const std::string getConstStringLen();
const std::string getConstStringAsArg();
const std::string getConstStringAlloc();

const std::string& getConstStringRefPure();
const std::string& getConstStringRefLen();
const std::string& getConstStringRefAsArg();
const std::string& getConstStringRefLenEmpty();
const std::string& getConstStringRefAlloc();

const std::string * getConstStringPtrLen();
const std::string * getConstStringPtrAlloc();
const std::string * getConstStringPtrOwnsAlloc();
const std::string * getConstStringPtrOwnsAllocPattern();

void acceptName_instance(std::string arg1);

void acceptStringConstReference(const std::string & arg1);

void acceptStringReferenceOut(std::string & arg1);

void acceptStringReference(std::string & arg1);

void acceptStringPointer(std::string * arg1);

void returnStrings(std::string & arg1, std::string & arg2);

char returnMany(int * arg1);

void explicit1(char * name);
void explicit2(char * name);

extern "C" {
  void CpassChar(char status);
  char CreturnChar();

  void CpassCharPtr(char * dest, const char *src);
}


#endif // STRINGS_HPP
