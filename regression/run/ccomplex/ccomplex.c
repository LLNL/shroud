/*
 * Copyright (c) 2017-2024, Lawrence Livermore National Security, LLC and
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

void acceptFloatComplexInoutPtr(float complex *arg1)
{
    *arg1 = 3.0 + 4.0 * I;
}
void acceptDoubleComplexInoutPtr(double complex *arg1)
{
    *arg1 = 3.0 + 4.0 * I;
}

void acceptDoubleComplexOutPtr(double complex *arg1)
{
    *arg1 = 3.0 + 4.0 * I;
}

// Return two values so Py_BuildValue is used.
void acceptDoubleComplexInoutPtrFlag(double complex *arg1, int *flag)
{
    *arg1 = 3.0 + 4.0 * I;
    *flag = 0;
}
void acceptDoubleComplexOutPtrFlag(double complex *arg1, int *flag)
{
    *arg1 = 3.0 + 4.0 * I;
    *flag = 0;
}

//----------------------------------------------------------------------

void acceptDoubleComplexInoutArrayList(double complex *arg1, int narg)
{
    *arg1 = 3.0 + 4.0 * I;
}

//----------------------------------------------------------------------
const char *LastFunctionCalled(void)
{
    return last_function_called;
}
