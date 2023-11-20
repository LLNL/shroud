/*
 * Copyright (c) 2017-2023, Lawrence Livermore National Security, LLC and
 * other Shroud Project Developers.
 * See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (BSD-3-Clause)
 *
 * Test C API generated from vectors.yaml.
 */

#include <wrapvectors.h>

#include <assert.h>

void test_vector_int(void)
{
  int intv[5] = { 1, 2, 3, 4, 5 };
  int irv;

  irv = VEC_vector_sum(intv, 5);
  assert(irv == 15 && "VEC_vector_sum");

}

int main(int argc, char *argv[])
{
  test_vector_int();

  return 0;
}

