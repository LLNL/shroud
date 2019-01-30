/*
 * Copyright (c) 2017-2019, Lawrence Livermore National Security, LLC.
 *
 * Produced at the Lawrence Livermore National Laboratory
 *
 * LLNL-CODE-738041.
 *
 * All rights reserved.
 *
 * This file is part of Shroud.
 *
 * For details about use and distribution, please read LICENSE.
 *
 * Test C interface for tutorial.yaml
 */

#include <stdbool.h>  // This should not be necessary
#include <wrapTutorial.h>
#include <wrapClass1.h>

#include <assert.h>

void test_class(void)
{
  int flag;
  TUT_class1 c1_buf, *c1;

  c1 = TUT_class1_new_default(&c1_buf);
  assert(c1 == &c1_buf && "TUT_class1_new_default");

  flag = TUT_class1_method1(c1);
  assert(flag == 0 && "TUT_class1_method1");

}

int main(int argc, char *argv[])
{
  test_class();

  return 0;
}

