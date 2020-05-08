/*
 * Copyright (c) 2017-2020, Lawrence Livermore National Security, LLC and
 * other Shroud Project Developers.
 * See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (BSD-3-Clause)
 *
 * Test C API generated from classes.yaml.
 */

#include <wrapclasses.h>
#include <wrapClass1.h>

#include <assert.h>

void test_class(void)
{
  int flag;
  CLA_Class1 c1_buf, *c1;

  c1 = CLA_Class1_new_default(&c1_buf);
  assert(c1 == &c1_buf && "CLA_class1_new_default");

  flag = CLA_Class1_method1(c1);
  assert(flag == 0 && "CLA_class1_method1");

}

int main(int argc, char *argv[])
{
  test_class();

  return 0;
}

