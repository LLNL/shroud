/*
 * Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and
 * other Shroud Project Developers.
 * See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (BSD-3-Clause)
 */
// generic.c

#include <stdlib.h>

#include "generic.h"

double global_double;
void *global_void;
int global_type;
size_t global_size;

double GetGlobalDouble(void)
{
  return global_double;
}

void UpdateAsFloat(float arg)
{
    global_double = arg;
}

void UpdateAsDouble(double arg)
{
    global_double = arg;
}

// start GenericReal
void GenericReal(double arg)
{
    global_double = arg;
    return;
}
// end GenericReal

long GenericReal2(long arg1, long arg2)
{
  return arg1 + arg2;
}

int SumValues(const int *values, int nvalues)
{
    int sum = 0;
    for (int i=0; i < nvalues; i++) {
        sum += values[i];
    }
    return sum;
}

// Broadcast if nfrom == 1
// Copy if nfrom == nto
void AssignValues(const int *from, int nfrom, int *to, int nto)
{
    if (nfrom == 1) {
        for (int i=0; i < nto; i++) {
            to[i] = from[0];
        }
    }
    else if (nfrom == nto) {
        for (int i=0; i < nto; i++) {
            to[i] = from[i];
        }
    }
}

void SavePointer(void *addr, int type, size_t size)
{
  global_void = addr;
  global_type = type;
  global_size = size;
}

void SavePointer2(void *addr, int type, size_t size)
{
  global_void = addr;
  global_type = type;
  global_size = size;
}

void GetPointer(void **addr, int *type, size_t *size)
{
  *addr = global_void;
  *type = global_type;
  *size = global_size;
}

void GetPointerAsPointer(void **addr, int *type, size_t *size)
{
  *addr = global_void;
  *type = global_type;
  *size = global_size;
}

StructAsClass *CreateStructAsClass(void)
{
    StructAsClass *rv = malloc(sizeof(StructAsClass));
    rv->nfield = 5;
    return rv;
}

long UpdateStructAsClass(StructAsClass *arg, long inew)
{
    arg->nfield = inew;
    return inew;
}
