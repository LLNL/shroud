/* Copyright (c) 2017-2019, Lawrence Livermore National Security, LLC. 
 *
 * Produced at the Lawrence Livermore National Laboratory 
 *
 * LLNL-CODE-738041.
 *
 * All rights reserved. 
 *
 * This file is part of Shroud.
 *
 * For details about use and distribution, please read LICENSE.
 *
 * #######################################################################
 *
 * struct.c
 */

#include "struct.h"
#include <string.h>

#define MAXLAST 50
static char last_function_called[MAXLAST];

static Cstruct1 global_Cstruct1;

//----------------------------------------------------------------------

// return sum of fields as a check
int passStructByValue(Cstruct1 arg)
{
  int rv = arg.ifield * 2;
  // Caller will not see changes.
  arg.ifield += 1;
  return rv;
}

int passStruct1(Cstruct1 *s1)
{
    strncpy(last_function_called, "passStruct1", MAXLAST);
    return s1->ifield;
}

int passStruct2(Cstruct1 *s1, char *outbuf)
{
    strncpy(outbuf, "passStruct2", LENOUTBUF);
    strncpy(last_function_called, "passStruct2", MAXLAST);
    return s1->ifield;
}

// return sum of fields as a check
int acceptStructInPtr(Cstruct1 *arg)
{
  int rv = arg->ifield + arg->dfield;
  arg->ifield += 1;
  return rv;
}

void acceptStructOutPtr(Cstruct1 *arg, int i, double d)
{
  arg->ifield = i;
  arg->dfield = d;
  return;
}

void acceptStructInOutPtr(Cstruct1 *arg)
{
  arg->ifield += 1;
  arg->dfield += 1.0;
  return;
}

Cstruct1 returnStructByValue(int i, double d)
{
  Cstruct1 s = {i, d};
  return s;
}

Cstruct1 *returnStructPtr1(int i, double d)
{
    strncpy(last_function_called, "returnStructPtr1", MAXLAST);
    global_Cstruct1.ifield = i;
    global_Cstruct1.dfield = d;
    return &global_Cstruct1;
}

Cstruct1 *returnStructPtr2(int i, double d, char *outbuf)
{
    strncpy(outbuf, "returnStructPtr2", LENOUTBUF);
    strncpy(last_function_called, "returnStructPtr2", MAXLAST);
    global_Cstruct1.ifield = i;
    global_Cstruct1.dfield = d;
    return &global_Cstruct1;
}

#if 0
Cstruct1 *returnStructPtrNew(int i, double d)
{
  Cstruct1 *s = new Cstruct1;
  s->ifield = i;
  s->dfield = d;
  return s;
}

void freeStruct(Cstruct1 *arg1)
{
  delete arg1;
}
#endif

