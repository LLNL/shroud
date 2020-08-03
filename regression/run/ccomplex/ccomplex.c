/*
 * Copyright (c) 2017-2020, Lawrence Livermore National Security, LLC and
 * other Shroud Project Developers.
 * See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (BSD-3-Clause)
 *
 * clibrary.c
 */

#include <complex.h>

#define MAXLAST 50
static char last_function_called[MAXLAST];

//----------------------------------------------------------------------

void accept_float_complex(float complex *arg1)
{
    *arg1 = 3.0 + 4.0 * I;
}
void accept_double_complex(double complex *arg1)
{
    *arg1 = 3.0 + 4.0 * I;
}

//----------------------------------------------------------------------
const char *LastFunctionCalled(void)
{
    return last_function_called;
}
