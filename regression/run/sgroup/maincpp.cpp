// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
// other Shroud Project Developers.
// See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
// #######################################################################
//
// Tests for sgroup.yaml
//

#include <sgroup.hpp>

#include <cassert>

void test_twostruct(void)
{
    double data[] = {1, 2, 3, 4, 5};
    twostruct<double, int> value{data, 5};

    process_twostruct(value);
}

int main(int argc, char *argv[])
{
  test_twostruct();
}
