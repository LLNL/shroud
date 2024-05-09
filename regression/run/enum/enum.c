/*
 * Copyright (c) 2017-2024, Lawrence Livermore National Security, LLC and
 * other Shroud Project Developers.
 * See the top-level COPYRIGHT file for details.
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

void returnEnumAsArg(enum Color *in)
{
    *in = BLUE;
}
