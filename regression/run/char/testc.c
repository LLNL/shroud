/*
 * Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
 * other Shroud Project Developers.
 * See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (BSD-3-Clause)
 *
 * Test C API generated from strings.yaml.
 */

#include <wrapstrings.h>

#include <assert.h>
#include <string.h>

void test_functions(void)
{
    const char *aaa = STR_getConstStringPtrLen();
    assert(strcmp(aaa, "getConstStringPtrLen") == 0 && "getConstStringPtrLen");
    // XXX - This will leak the std::string return by the C++ function.
}

int main(int argc, char *argv[])
{
  test_functions();

  return 0;
}

