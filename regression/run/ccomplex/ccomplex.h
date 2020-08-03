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

void accept_float_complex(float complex *arg1);
void accept_double_complex(double complex *arg1);

#endif // CCOMPLEX_H
