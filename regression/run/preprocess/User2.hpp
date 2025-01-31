// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
// other Shroud Project Developers.
// See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
// #######################################################################
//
// User2.hpp
//

class User2 {
  public:
#ifdef USE_USER2_A
    void exfunc()
#endif
#ifdef USE_USER2_A
    void exfunc(int flag) {};
#endif
};

