/*
 * Copyright (c) 2017-2023, Lawrence Livermore National Security, LLC and
 * other Shroud Project Developers.
 * See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (BSD-3-Clause)
 *
 * Test C API generated from struct.yaml.
 */

#include <wrapstruct.h>

#include <assert.h>

#if 0
static int callback_fcn(STR_cstruct1 *arg)
{
    return arg->ifield;
}
#endif

void test_Cstruct1(void)
{
    int rvi;
    char outbuf[100];
    STR_cstruct1 str1 = { 2, 2.0 };

    rvi = STR_passStructByValue(str1);
    assert(4 == rvi && "passStructByValue");

    str1.ifield = 12;
    str1.dfield = 12.6;
    assert(12 == STR_passStruct1(&str1) && "passStruct1");

    str1.ifield = 22;
    str1.dfield = 22.8;
    assert(22 == STR_passStruct2(&str1, outbuf) && "passStruct2");

    str1.ifield = 3;
    str1.dfield = 3.0;
    rvi = STR_acceptStructInPtr(&str1);
    assert(6 == rvi && "acceptStructInPtr");

    str1.ifield = 0;
    str1.dfield = 0.0;
    STR_acceptStructOutPtr(&str1, 4, 4.5);
    assert(4 == str1.ifield && "acceptStructOutPtr i field");
    assert(4.5 == str1.dfield && "acceptStructOutPtr d field");

    str1.ifield = 4;
    str1.dfield = 4.0;
    STR_acceptStructInOutPtr(&str1);
    assert(5 == str1.ifield && "acceptStructInOutPtr i field");
    assert(5.0 == str1.dfield && "acceptStructInOutPtr d field");

#if 0
    //typedef void ( *worker0 )(STR_cstruct1 * arg);
    //typedef void ( *worker )(Cstruct1 * arg);
    str1.ifield = 6;
    rvi = STR_callback1(&str1, callback_fcn);
    assert(6 == rvi && "callback1");
#endif
}

int main(int argc, char *argv[])
{
  test_Cstruct1();

  return 0;
}

