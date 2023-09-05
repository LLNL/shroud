/*
 * Copyright (c) 2017-2023, Lawrence Livermore National Security, LLC and
 * other Shroud Project Developers.
 * See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (BSD-3-Clause)
 *
 * Test C API generated from types.yaml.
 */

#include <math.h>
#include <wraptypemap.h>

#include <assert.h>

void test_passIndex(void)
{
    IndexType indx;

#if defined(USE_64BIT_INDEXTYPE)
    int64_t indx64 = pow(2,34);
    assert(sizeof(IndexType) == sizeof(int64_t) && "TYPEMAP_size64");
    assert(! TYP_passIndex(indx64 - 1, &indx));
    assert(  TYP_passIndex(indx64, &indx));
    assert(  indx == indx64);
#else
    int32_t indx32 = 2;
    assert(sizeof(IndexType) == sizeof(int32_t) && "TYPEMAP_size32");
    assert(! TYP_passIndex(indx32 - 1, &indx));
    assert(  TYP_passIndex(indx32, &indx));
    assert(  indx == indx32);
#endif
}

int main(int argc, char *argv[])
{
  test_passIndex();

  return 0;
}

