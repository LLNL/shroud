/*
 * Copyright (c) 2017-2021, Lawrence Livermore National Security, LLC and
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

void *addr = NULL;

void test_class(void)
{
  int flag;
  CLA_Class1 c1;

  CLA_Class1_ctor_default(&c1);

  flag = CLA_Class1_method1(&c1);
  assert(flag == 0 && "CLA_class1_method1");

  // passClassByValue sets global_flag
  CLA_set_global_flag(-1);
  CLA_pass_class_by_value(&c1);
  flag = CLA_get_global_flag();
  assert(flag == 0 && "CLA_pass_class_by_value");
}

void test_class_setup(void)
{
    int flag;
    CLA_Class1 c1;

    // Create an instance and save to library
    CLA_Class1_ctor_default(&c1);
    flag = CLA_useclass(&c1);
    assert(flag == 0 && "CLA_useclass");

    addr = c1.addr;
}

// functions which return a pointer to the capsule

void test_class_func(void)
{
    // Fetch the instance
    CLA_Class1 c2, *pc2;
    pc2 = CLA_getclass2(&c2);
    assert(pc2 == &c2 && "CLA_getclass2");

    CLA_Class1 c3, *pc3;
    pc3 = CLA_getclass3(&c3);
    assert(pc3 == &c3 && "CLA_getclass3");
}

// functions which do not return a pointer to the capsule

void test_class_void(void)
{
    // Fetch the instance
    CLA_Class1 c2;
    CLA_getclass2_void(&c2);
    assert(c2.addr == addr && "CLA_getclass2_void");

    CLA_Class1 c3;
    CLA_getclass3(&c3);
    assert(c3.addr == addr && "CLA_getclass3_void");
}

int main(int argc, char *argv[])
{
  test_class();
  test_class_setup();
  test_class_func();
  test_class_void();

  return 0;
}

