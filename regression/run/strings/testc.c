/*
 * Copyright Shroud Project Developers. See LICENSE file for details.
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
    {
        STR_SHROUD_capsule_data capsule = {0};
        const char *rv = STR_getConstStringAlloc(&capsule);
        assert(strcmp(rv, "getConstStringAlloc") == 0 && "getConstStringAlloc");
        STR_SHROUD_memory_destructor(&capsule);
    }

    {
        const char *rv = STR_getConstStringPtrLen();
        assert(strcmp(rv, "getConstStringPtrLen") == 0 && "getConstStringPtrLen");
        // XXX - This will leak the std::string return by the C++ function.
    }
}

int main(int argc, char *argv[])
{
  test_functions();

  return 0;
}

