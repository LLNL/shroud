/*
 * Copyright (c) 2017-2021, Lawrence Livermore National Security, LLC and
 * other Shroud Project Developers.
 * See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (BSD-3-Clause)
 */
// generic.c

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

int SumValues(int *values, int nvalues)
{
    int sum = 0;
    for (int i=0; i < nvalues; i++) {
        sum += values[i];
    }
    return sum;
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
