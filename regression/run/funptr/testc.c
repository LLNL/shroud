/*
 * Copyright Shroud Project Developers. See LICENSE file for details.
 *
 * SPDX-License-Identifier: (BSD-3-Clause)
 * #######################################################################
 *
 * Test C API generated from funptr.yaml.
 */

// XXX - Using funptr.h here to get stdbool.h.
#include "funptr.h"

#include "wrapfunptr.h"

#include <assert.h>

static int counter;

void incr1(void)
{
    counter++;
}

void incr2(int i, FUN_TypeID j)
{
    if (j == 1) counter = i;
}

void test_callback1(void)
{
    counter = 0;

    FUN_callback1(incr1);
    assert(counter == 1 && "callback1");
}

void test_callback2(void)
{
    counter = 0;

    FUN_callback2("one", 2, incr2);
    assert(counter == 2 && "callback2");
}

int main(int argc, char *argv[])
{
    test_callback1();
    test_callback2();

}
