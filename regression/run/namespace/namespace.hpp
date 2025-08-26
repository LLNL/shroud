// Copyright Shroud Project Developers. See LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <string>

const std::string& LastFunctionCalled();

namespace outer {
  struct Cstruct1 {
    int ifield;
    double dfield;
  };
  void One();
};

void One();

namespace nswork {
  class ClassWork {
  };
};
