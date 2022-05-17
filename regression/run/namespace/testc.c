/*
 * Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and
 * other Shroud Project Developers.
 * See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (BSD-3-Clause)
 *
 * Test C API generated from namespace.yaml.
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

