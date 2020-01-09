/*
 * Copyright (c) 2017-2020, Lawrence Livermore National Security, LLC and
 * other Shroud Project Developers.
 * See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (BSD-3-Clause)
 */

#include <wrapns.h>
#include <wrapns_outer.h>

#include <assert.h>

void test_ns(void)
{
  NS_one();
}

void test_ns_outer(void)
{
  NS_outer_one();
}

int main(int argc, char *argv[])
{
  test_ns();
  test_ns_outer();

  return 0;
}

