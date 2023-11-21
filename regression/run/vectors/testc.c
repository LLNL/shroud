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
  int intv2[10];
  int irv;
  size_t irsize;
  long num;

  irv = VEC_vector_sum(intv, 5);
  assert(irv == 15 && "VEC_vector_sum");

  irsize = 5;  // size of input array and number of returned values
  for (int i=0; i < 5; i++) {
      intv[i] = 0;
  }
  VEC_vector_iota_out(intv, &irsize);
  assert(irsize == 5 && "VEC_vector_iota_out  size");
  assert(intv[0] == 1 && "VEC_vector_iota_out  intv[0]");
  assert(intv[1] == 2 && "VEC_vector_iota_out  intv[1]");
  assert(intv[2] == 3 && "VEC_vector_iota_out  intv[2]");
  assert(intv[3] == 4 && "VEC_vector_iota_out  intv[3]");
  assert(intv[4] == 5 && "VEC_vector_iota_out  intv[4]");

  // Fortran and C wrappers have custom statements.
  for (int i=0; i < 10; i++) {
      intv2[i] = 0;
  }
  irsize = 5;  // size of input array and number of returned values
  num = VEC_vector_iota_out_with_num(intv2, &irsize);
  assert(num == 5 && "VEC_vector_iota_out_num  size");
  assert(intv2[0] == 1 && "VEC_vector_iota_out_num  intv2[0]");
  assert(intv2[1] == 2 && "VEC_vector_iota_out_num  intv2[1]");
  assert(intv2[2] == 3 && "VEC_vector_iota_out_num  intv2[2]");
  assert(intv2[3] == 4 && "VEC_vector_iota_out_num  intv2[3]");
  assert(intv2[4] == 5 && "VEC_vector_iota_out_num  intv2[4]");
  
}

int main(int argc, char *argv[])
{
  test_vector_int();

  return 0;
}

