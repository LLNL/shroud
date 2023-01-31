/*
 * Copyright (c) 2017-2023, Lawrence Livermore National Security, LLC and
 * other Shroud Project Developers.
 * See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (BSD-3-Clause)
 *
 * Test C API generated from pointers.yaml.
 */

#include <wrappointers.h>

#include <assert.h>

void test_out_ptrs(void)
{
  int *count, ncount;
  
  POI_getPtrToDynamicArray(&count, &ncount);
  assert(ncount == 10 && "CLA_class1_method1");
  
}

int main(int argc, char *argv[])
{
  test_out_ptrs();

  return 0;
}

