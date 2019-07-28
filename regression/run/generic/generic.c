// Copyright (c) 2017-2019, Lawrence Livermore National Security, LLC and
// other Shroud Project Developers.
// See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//
// generic.c

#include "generic.h"

double global_double;

double GetGlobalDouble(void)
{
  return global_double;
}

void GenericReal(double arg)
{
    global_double = arg;
    return;
}

long GenericReal2(long arg1, long arg2)
{
  return arg1 + arg2;
}

void SavePointer(void *addr, int type, size_t size)
{
}
