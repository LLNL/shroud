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

void callback1a(int type, void (*incr)(void));
void callback2(int type, void * in, void (*incr)(int *));
void callback3(const char *type, void * in, void (*incr)(int *), char *outbuf);
//void callback_set_alloc(int tc, array_info *arr, void (*alloc)(int tc, array_info *arr));


#endif // FUNPTR_H
