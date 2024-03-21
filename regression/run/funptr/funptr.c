/*
 * Copyright (c) 2017-2024, Lawrence Livermore National Security, LLC and
 * other Shroud Project Developers.
 * See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (BSD-3-Clause)
 *
 * funptr.c
 */

#include "funptr.h"

#include <string.h>

//----------------------------------------------------------------------
// Uses a Fortran abstract interface
// start callback1
void callback1(void (*incr)(void))
{
    incr();
}
// end callback1

// F_force_wrapper=True, abstract interface
void callback1_wrap(void (*incr)(void))
{
    incr();
}

// incr +external
void callback1_external(void (*incr)(void))
{
    incr();
}

// incr +funptr
void callback1_funptr(void (*incr)(void))
{
    incr();
}

//----------------------------------------------------------------------
// Uses a Fortran abstract interface
// start callback2
void callback2(const char *name, int ival, void (*incr)(int i))
{
    incr(ival);
}
// end callback2

// incr +external
void callback2_external(const char *name, int ival, void (*incr)(int i))
{
    incr(ival);
}

// incr +funptr
void callback2_funptr(const char *name, int ival, void (*incr)(int))
{
    incr(ival);
}

//----------------------------------------------------------------------

#if 0
void callback1a(int type, void (*incr)(void))
{
  // Use type to decide how to call incr
}

void callback2(int type, void * in, void (*incr)(int *))
{
  if (type == 1) {
    // default function pointer from prototype
    incr(in);
  } else if (type == 2) {
    void (*incr2)(double *) = (void(*)(double *)) incr;
    incr2(in);
  }
}

void callback3(const char *type, void * in, void (*incr)(int *),
               char *outbuf)
{
  if (strcmp(type, "int") == 0) {
    // default function pointer from prototype
    incr(in);
  } else if (strcmp(type, "double") == 0) {
    void (*incr2)(double *) = (void(*)(double *)) incr;
    incr2(in);
  }
  //  strncpy(outbuf, "callback3", LENOUTBUF);
}

void callback_set_alloc(int tc, array_info *arr, void (*alloc)(int tc, array_info *arr))
{
  alloc(tc, arr);
}
#endif

//----------------------------------------------------------------------
