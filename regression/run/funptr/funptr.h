/*
 * Copyright (c) 2017-2024, Lawrence Livermore National Security, LLC and
 * other Shroud Project Developers.
 * See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (BSD-3-Clause)
 *
 * funptr.h - wrapped routines
 */

#ifndef FUNPTR_H
#define FUNPTR_H

void callback1(void (*incr)(void));
void callback1_wrap(void (*incr)(void));
void callback1_external(void (*incr)(void));
void callback1_funptr(void (*incr)(void));

typedef void (*incrtype)(int i);
typedef void (*incrtype_d)(double i);
typedef int (*incrtype_fun)(int i);

void callback2(const char *name, int ival, incrtype incr);
void callback2_external(const char *name, int ival, incrtype incr);
void callback2_funptr(const char *name, int ival, incrtype incr);

void callback3(int type, void * in, void (*incr)(void));

int callback4(int *ilow, int nargs,
              int (*actor)(int *ilow, int nargs));

#endif // FUNPTR_H
