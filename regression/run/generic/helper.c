/*
 * Copyright (c) 2017-2021, Lawrence Livermore National Security, LLC and
 * other Shroud Project Developers.
 * See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (BSD-3-Clause)
 */

#include <generic.h>
#include <typesgeneric.h>

// Convert Shroud type into Library type
int convert_type(int type)
{
    switch (type) {
    case SH_TYPE_INT:
        type = T_INT;
        break;
    case SH_TYPE_LONG:
        type = T_LONG;
        break;
    case SH_TYPE_FLOAT:
        type = T_FLOAT;
        break;
    case SH_TYPE_DOUBLE:
        type = T_DOUBLE;
        break;
    default:
       type = -1;
    }
    return type;
}
