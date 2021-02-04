// Copyright (c) 2017-2021, Lawrence Livermore National Security, LLC and
// other Shroud Project Developers.
// See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <stdio.h>

#include "ISO_Fortran_binding.h"

void Demo_CFI(const CFI_cdesc_t * arg)
{
  printf("Rank: %d\n", arg->rank);
}
