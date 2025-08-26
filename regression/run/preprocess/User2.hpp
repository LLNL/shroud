// Copyright Shroud Project Developers. See LICENSE file for details.
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

