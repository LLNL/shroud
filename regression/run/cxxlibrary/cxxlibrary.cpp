/*
 * Copyright (c) 2017-2021, Lawrence Livermore National Security, LLC and
 * other Shroud Project Developers.
 * See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (BSD-3-Clause)
 * #######################################################################
 *
 * cxxlibrary.cpp
 */

#include "cxxlibrary.hpp"

static Cstruct1_cls global_Cstruct1_cls;

//----------------------------------------------------------------------
namespace structns {
static Cstruct1 global_Cstruct1;
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
};  // namespace cxxlibrary

//----------------------------------------------------------------------
// Test Python struct as class.

int passStructByReferenceCls(Cstruct1_cls &arg)
{
  int rv = arg.ifield * 2;
  arg.ifield += 1;
  return rv;
}

int passStructByReferenceInCls(const Cstruct1_cls &arg)
{
  int rv = arg.ifield * 2;
  global_Cstruct1_cls = arg;
  return rv;
}

void passStructByReferenceInoutCls(Cstruct1_cls &arg)
{
  arg.ifield += 1;
}

void passStructByReferenceOutCls(Cstruct1_cls &arg)
{
    arg = global_Cstruct1_cls;
}

//----------------------------------------------------------------------
// pointers
// default value

bool defaultPtrIsNULL(double *data)
{
    if (data == nullptr)
        return true;
    return false;
}

//----------------------------------------------------------------------

void defaultArgsInOut(int in1, int *out1, int *out2, bool flag)
{
    *out1 = 1;
    if (flag) {
        *out2 = 20;
    } else {
        *out2 = 2;
    }
}

//----------------------------------------------------------------------

void accept_complex(std::complex<double> *arg1)
{
}
