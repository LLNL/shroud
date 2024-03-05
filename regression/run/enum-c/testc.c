/*
 * Copyright (c) 2017-2024, Lawrence Livermore National Security, LLC and
 * other Shroud Project Developers.
 * See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (BSD-3-Clause)
 *
 * Test C API generated from enum.yaml.
 */

#include <wrapenum.h>

#include <assert.h>

void test_enum(void)
{
  assert(ENU_a1 == 0 && "enum a1");
  assert(ENU_b1 == 3 && "enum b1");
  assert(ENU_c1 == 4 && "enum c1");
  assert(ENU_d1 == 3 && "enum d1");
  assert(ENU_e1 == 3 && "enum e1");
  assert(ENU_f1 == 4 && "enum f1");
  assert(ENU_g1 == 5 && "enum g1");
  assert(ENU_h1 == 100 && "enum h1");

}

int main(int argc, char *argv[])
{
  test_enum();

  return 0;
}

