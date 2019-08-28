//
//
#include "scope.hpp"
#include "wrapscope.h"
#include "wrapcls1Enum.h"
#include "wrapcls2Enum.h"
#include "wrapscope_ns1Enum.h"
#include "wrapscope_ns2Enum.h"
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

  check(static_cast<int>(SCO_ns1Enum_RED) ==
        static_cast<int>(ns1Enum::RED),
        "namespace ns1 enum");
  check(static_cast<int>(SCO_ns2Enum_RED) ==
        static_cast<int>(ns2Enum::RED),
        "namespace ns2 enum");
  check(static_cast<int>(SCO_ns3Enum_RED) ==
        static_cast<int>(ns3Enum::RED),
        "namespace ns3 enum");

  // enum in a class
  check(static_cast<int>(SCO_cls1Enum_RED) ==
        static_cast<int>(cls1Enum::RED),
        "class1 enum");
  check(static_cast<int>(SCO_cls2Enum_RED) ==
        static_cast<int>(cls2Enum::RED),
        "class2 enum");

  // class enum
  check(static_cast<int>(SCO_ColorEnum_RED) ==
        static_cast<int>(ColorEnum::RED),
        "class enum");

  return 0;
}
