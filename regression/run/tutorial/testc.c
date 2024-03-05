/*
 * Copyright (c) 2017-2024, Lawrence Livermore National Security, LLC and
 * other Shroud Project Developers.
 * See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (BSD-3-Clause)
 *
 * Test C API generated from tutorial.yaml.
 */

#include <wrapTutorial.h>
#include <wrapClass1.h>

#include <assert.h>

void test_class(void)
{
  int flag;
  TUT_Class1 c1_buf, *c1;

  c1 = TUT_Class1_new_default(&c1_buf);
  assert(c1 == &c1_buf && "TUT_class1_new_default");

  flag = TUT_Class1_method1(c1);
  assert(flag == 0 && "TUT_class1_method1");

}

int main(int argc, char *argv[])
{
  test_class();

  return 0;
}

