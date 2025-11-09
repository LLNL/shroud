/*
 * Copyright Shroud Project Developers. See LICENSE file for details.
 *
 * SPDX-License-Identifier: (BSD-3-Clause)
 */

#include "enum.h"

enum Color global1;

int convert_to_int(enum Color in)
{
    return in;
}

enum Color returnEnum(enum Color in)
{
    return in;
}

void returnEnumOutArg(enum Color *out)
{
    *out = BLUE;
}

enum Color returnEnumInOutArg(enum Color *inout)
{
    enum Color old = *inout;
    *inout = BLUE;
    return old;
}
