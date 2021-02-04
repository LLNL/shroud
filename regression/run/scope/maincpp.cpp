// Copyright (c) 2017-2021, Lawrence Livermore National Security, LLC and
// other Shroud Project Developers.
// See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//
//
#include "scope.hpp"
#include "wrapscope.h"
#include "wrapClass1.h"
#include "wrapClass2.h"
#include "wrapscope_ns1.h"
#include "wrapscope_ns2.h"
#include <string>
#include <iostream>

void check(bool expr, const std::string &msg)
{
  if (! expr) {
    std::cout << msg << std::endl;
  }
}

int main(int argc, char *argv[])
{

  check(static_cast<int>(SCO_RED) == static_cast<int>(RED),
        "top level enum");

  check(static_cast<int>(SCO_ns1_RED) ==
        static_cast<int>(ns1::RED),
        "namespace ns1 enum");
  check(static_cast<int>(SCO_ns2_RED) ==
        static_cast<int>(ns2::RED),
        "namespace ns2 enum");
  check(static_cast<int>(SCO_ns3_RED) ==
        static_cast<int>(ns3::RED),
        "namespace ns3 enum");

  // enum in a class
  check(static_cast<int>(SCO_Class1_RED) ==
        static_cast<int>(Class1::RED),
        "class1 enum");
  check(static_cast<int>(SCO_Class2_RED) ==
        static_cast<int>(Class2::RED),
        "class2 enum");

  // class enum
  check(static_cast<int>(SCO_ColorEnum_RED) ==
        static_cast<int>(ColorEnum::RED),
        "class enum");

  return 0;
}
