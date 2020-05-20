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

//----------------------------------------------------------------------
// Python struct as numpy.

int passStructByReference(Cstruct1 &arg)
{
  int rv = arg.ifield * 2;
  arg.ifield += 1;
  return rv;
}

//----------------------------------------------------------------------
// Python struct as class.

int passStructByReferenceCls(Cstruct1_cls &arg)
{
  int rv = arg.ifield * 2;
  arg.ifield += 1;
  return rv;
}
