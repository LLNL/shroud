/*
 * Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
 * other Shroud Project Developers.
 * See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (BSD-3-Clause)
 *
 * Test C API generated from strings.yaml.
 */

#include <wrapchar.h>

#include <assert.h>
#include <string.h>

void test_char_ptr_out(void)
{
    char *outptr;
    
    CHA_fetchCharPtrLibrary(&outptr);
    assert(strcmp(outptr, "static_char_array") == 0 && "fetchCharPtrLibrary");
}

int main(int argc, char *argv[])
{
    CHA_init_test();
    
    test_char_ptr_out();

    return 0;
}

