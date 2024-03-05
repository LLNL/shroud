/*
 * Copyright (c) 2017-2024, Lawrence Livermore National Security, LLC and
 * other Shroud Project Developers.
 * See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (BSD-3-Clause)
 *
 * Test C API generated from typedefs.yaml.
 */

#include <wraptypedefs.h>

#include <assert.h>

void test_alias(void)
{
    TYP_TypeID arg1, rv;

    assert(sizeof(int)  == sizeof(TYP_TypeID) && "test_alias - sizeof");

    arg1 = 10;
    rv = TYP_typefunc(arg1);
    assert(rv == arg1 + 1 && "TYP_typefunc");
}

int main(int argc, char *argv[])
{
  test_alias();

  return 0;
}

