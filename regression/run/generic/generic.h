/*
 * Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and
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

struct s_StructAsClass {
    long nfield;
};
typedef struct s_StructAsClass StructAsClass;

double GetGlobalDouble(void);

void GenericReal(double arg);

long GenericReal2(long arg1, long arg2);

int SumValues(const int *values, int nvalues);

void AssignValues(const int *from, int nfrom, int *to, int nto);

void SavePointer(void *addr, int type, size_t size);
void SavePointer2(void *addr, int type, size_t size);
void GetPointer(void **addr, int *type, size_t *size);
void GetPointerAsPointer(void **addr, int *type, size_t *size);

StructAsClass *CreateStructAsClass(void);
long UpdateStructAsClass(StructAsClass *arg, long inew);

#endif // GENERIC_HPP
