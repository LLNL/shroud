/*
 * Copyright (c) 2017-2020, Lawrence Livermore National Security, LLC and
 * other Shroud Project Developers.
 * See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (BSD-3-Clause)
 * #######################################################################
 *
 * cxxlibrary.cpp
 */

#include "cxxlibrary.hpp"

static Cstruct1 global_Cstruct1;

//----------------------------------------------------------------------
// Test Fortran.
// Test Python struct as numpy.

int passStructByReference(Cstruct1 &arg)
{
  int rv = arg.ifield * 2;
  arg.ifield += 1;
  global_Cstruct1 = arg;
  return rv;
}

int passStructByReferenceIn(const Cstruct1 &arg)
{
  int rv = arg.ifield * 2;
  global_Cstruct1 = arg;
  return rv;
}

void passStructByReferenceInout(Cstruct1 &arg)
{
  arg.ifield += 1;
}

void passStructByReferenceOut(Cstruct1 &arg)
{
    arg = global_Cstruct1;
}

//----------------------------------------------------------------------
// Test Python struct as class.

int passStructByReferenceCls(Cstruct1_cls &arg)
{
  int rv = arg.ifield * 2;
  arg.ifield += 1;
  return rv;
}
