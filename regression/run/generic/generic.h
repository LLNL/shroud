/*
 * Copyright (c) 2017-2021, Lawrence Livermore National Security, LLC and
 * other Shroud Project Developers.
 * See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (BSD-3-Clause)
 *
 * generic.h
 */

#ifndef GENERIC_H
#define GENERIC_H

#include <stddef.h>

#define T_INT     1
#define T_LONG    2
#define T_FLOAT   3
#define T_DOUBLE  4

double GetGlobalDouble(void);

void GenericReal(double arg);

long GenericReal2(long arg1, long arg2);

int SumValues(int *values, int nvalues);

void SavePointer(void *addr, int type, size_t size);
void SavePointer2(void *addr, int type, size_t size);
void GetPointer(void **addr, int *type, size_t *size);
void GetPointerAsPointer(void **addr, int *type, size_t *size);

#endif // GENERIC_HPP
