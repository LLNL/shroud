// Copyright (c) 2017-2024, Lawrence Livermore National Security, LLC and
// other Shroud Project Developers.
// See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
// #######################################################################
//
// scope.hpp
//

enum Color {
  RED = 10,
  BLUE,
  WHITE
};

namespace ns1 {
  enum Color {
    RED = 20,
    BLUE,
    WHITE
  };
  struct DataPointer {
    int nitems;
    int *items;
  };
};

namespace ns2 {
  enum Color {
    RED = 30,
    BLUE,
    WHITE
  };
  struct DataPointer {
    int nitems;
    int *items;
  };
};

namespace ns3 {
  enum Color {
    RED = 70,
    BLUE,
    WHITE
  };
  struct DataPointer {
    int nitems;
    int *items;
  };
};

class Class1 {
public:
  enum Color {
    RED = 40,
    BLUE,
    WHITE
  };
};

class Class2 {
public:
  enum Color {
    RED = 50,
    BLUE,
    WHITE
  };
};

enum class ColorEnum {
  RED = 60,
  BLUE,
  WHITE
};
