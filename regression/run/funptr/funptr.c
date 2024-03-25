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
void callback1(void (*incr)(void))
{
    incr();
}

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
// start callback1_funptr
void callback1_funptr(void (*incr)(void))
{
    incr();
}
// end callback1_funptr

//----------------------------------------------------------------------
// Uses a Fortran abstract interface
// start callback2
void callback2(const char *name, int ival, incrtype incr)
{
    incr(ival);
}
// end callback2

// incr +external
void callback2_external(const char *name, int ival, incrtype incr)
{
    if (strcmp(name, "double") == 0) {
        incrtype_d incr_d = (incrtype_d) incr;
        incr_d( (double) ival);
    }
    else if (strcmp(name, "function") == 0) {
        incrtype_fun incr_fun = (incrtype_fun) incr;
        (void) incr_fun(ival);
    }
    else {
        incr(ival);
    }
}

// incr +funptr
void callback2_funptr(const char *name, int ival, incrtype incr)
{
    if (strcmp(name, "double") == 0) {
        incrtype_d incr_d = (incrtype_d) incr;
        incr_d( (double) ival);
    }
    else if (strcmp(name, "function") == 0) {
        incrtype_fun incr_fun = (incrtype_fun) incr;
        (void) incr_fun(ival);
    }
    else {
        incr(ival);
    }
}

//----------------------------------------------------------------------

void callback3(int type, void * in, void (*incr)(void))
{
    switch(type) {
    case 1: {
        void (*incr2)(int) = (void(*)(int)) incr;
        incr2(*(int *) in);
        break;
    }
    case 2: {
        void (*incr2)(double) = (void(*)(double)) incr;
        incr2(*(double *) in);
        break;
    }
    }
}

//----------------------------------------------------------------------

int callback4(int *ilow, int nargs,
              int (*actor)(int *ilow, int nargs))
{
    return actor(ilow, nargs);
}

//----------------------------------------------------------------------
