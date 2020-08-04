/*
 * Copyright (c) 2017-2020, Lawrence Livermore National Security, LLC and
 * other Shroud Project Developers.
 * See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (BSD-3-Clause)
 *
 * ccomplex.h - wrapped routines
 */

#ifndef CCOMPLEX_H
#define CCOMPLEX_H

#include <complex.h>

void acceptFloatComplexInoutPtr(float complex *arg1);

void acceptDoubleComplexInoutPtr(double complex *arg1);
void acceptDoubleComplexOutPtr(double complex *arg1);

void acceptDoubleComplexInoutPtrFlag(double complex *arg1, int *flag);
void acceptDoubleComplexOutPtrFlag(double complex *arg1, int *flag);

//----------------------------------------------------------------------

void acceptDoubleComplexInoutArrayList(double complex *arg1, int narg)

#endif // CCOMPLEX_H
