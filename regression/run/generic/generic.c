// Copyright (c) 2017-2019, Lawrence Livermore National Security, LLC and
// other Shroud Project Developers.
// See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//
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

! start GenericReal
void GenericReal(double arg)
{
    global_double = arg;
    return;
}
! end GenericReal

long GenericReal2(long arg1, long arg2)
{
  return arg1 + arg2;
}

void SavePointer(void *addr, int type, size_t size)
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
