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

#ifndef __cplusplus
#include <stdbool.h>
#endif

void callback1(void (*incr)(void));
void callback1_wrap(void (*incr)(void));
void callback1_external(void (*incr)(void));
void callback1_funptr(void (*incr)(void));

typedef int TypeID;
typedef void (*pfvoid)(void);
typedef void (*incrtype)(int i, TypeID j);
typedef void (*incrtype_d)(double i);
typedef int (*incrtype_fun)(int i);

void callback2(const char *name, int ival, incrtype incr);
void callback2_external(const char *name, int ival, incrtype incr);
void callback2_funptr(const char *name, int ival, incrtype incr);

void callback3(int type, void * in, void (*incr)(void));

int callback4(int *ilow, int nargs,
              int (*actor)(int *ilow, int nargs));

void callback_ptr(int *(*get)(void));
void callback_double(double (*get)(int i, int));

int abstract1(int input, int (*get)(double, int));

void callback_void_ptr(void (*void_ptr_arg)(void *));

void callback_all_types(void (*all_types)(int, int *, char, char *, bool, bool *));

void get_void_ptr(pfvoid *func);

#endif // FUNPTR_H
