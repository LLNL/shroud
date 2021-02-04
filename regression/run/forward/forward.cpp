// Copyright (c) 2017-2021, Lawrence Livermore National Security, LLC and
// other Shroud Project Developers.
// See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
// #######################################################################
//
// forward.cpp - wrapped routines
//

#include "forward.hpp"

namespace forward
{

  void Class2::func1(tutorial::Class1 *arg)
  {
  };

  void Class2::acceptClass3(Class3 *arg)
  {
  };

  // Use a struct defined in another wrapped library.
  int passStruct1(const Cstruct1 *s1)
  {
    return s1->ifield;
  }
  
} /* end namespace forward */
