// Copyright (c) 2017-2019, Lawrence Livermore National Security, LLC. 
//
// Produced at the Lawrence Livermore National Laboratory 
//
// LLNL-CODE-738041.
//
// All rights reserved. 
//
// This file is part of Shroud.
//
// For details about use and distribution, please read LICENSE.
//
// #######################################################################
//
// scope.hpp
//

enum Color {
  RED,
  BLUE,
  WHITE
};

namespace ns1_enum {
  enum Color {
    RED,
    BLUE,
    WHITE
  };
};

namespace ns2_enum {
  enum Color {
    RED,
    BLUE,
    WHITE
  };
};




class cls1_enum {
  enum Color {
    RED,
    BLUE,
    WHITE
  };
};

class cls2_enum {
  enum Color {
    RED,
    BLUE,
    WHITE
  };
};
