// Copyright (c) 2017-2019, Lawrence Livermore National Security, LLC and
// other Shroud Project Developers.
// See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//
// generic.h

#include <stddef.h>

double GetGlobalDouble(void);

void GenericReal(double arg);

long GenericReal2(long arg1, long arg2);

void SavePointer(void *addr, int type, size_t size);
