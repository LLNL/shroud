/*
 * Copyright Shroud Project Developers. See LICENSE file for details.
 *
 * SPDX-License-Identifier: (BSD-3-Clause)
 *
 * Test C API generated from enum.yaml.
 */

#include <wrapenum.h>

#include <assert.h>

void test_enum(void)
{
  assert(ENU_RED   == 10 && "ENU_RED");
  assert(ENU_BLUE  == 11 && "ENU_BLUE");
  assert(ENU_WHITE == 12 && "ENU_WHITE");
  
  assert(ENU_a1 == 0 && "enum a1");
  assert(ENU_b1 == 3 && "enum b1");
  assert(ENU_c1 == 4 && "enum c1");
  assert(ENU_d1 == 3 && "enum d1");
  assert(ENU_e1 == 3 && "enum e1");
  assert(ENU_f1 == 4 && "enum f1");
  assert(ENU_g1 == 5 && "enum g1");
  assert(ENU_h1 == 100 && "enum h1");

}

void test_enum_functions(void)
{
    int icol;
    enum ENU_Color outcolor;

    icol = ENU_convert_to_int(ENU_RED);
    assert(ENU_RED == icol && "convert_to_int");

    outcolor = ENU_returnEnum(ENU_RED);
    assert(ENU_RED == outcolor && "returnEnum");
}

int main(int argc, char *argv[])
{
  test_enum();
  test_enum_functions();

  return 0;
}

