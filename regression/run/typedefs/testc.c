/*
 * Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
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
    TYP_Alias arg1, rv;

    assert(sizeof(int)  == sizeof(TYP_Alias) && "test_alias - sizeof");

    arg1 = 10;
    rv = TYP_typefunc(arg1);
    assert(rv == arg1 + 1 && "TYP_typefunc");
}

void test_enum(void)
{
    TYP_TypeID type1, type2;

    type1 = TYP_INT_ID;
    type2 = TYP_returnTypeID(type1);
    assert(type1 == type2 && "returnTypeID");
}

int main(int argc, char *argv[])
{
  test_alias();
  test_enum();

  return 0;
}

