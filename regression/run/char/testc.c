/*
 * Copyright Shroud Project Developers. See LICENSE file for details.
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
    int irv;
    char *outptr;

    outptr = NULL;
    CHA_fetchCharPtrLibrary(&outptr);
    assert(strcmp(outptr, "static_char_array") == 0 && "fetchCharPtrLibrary");

    outptr = NULL;
    irv = CHA_fetchCharPtrLibraryNULL(&outptr);
    assert(irv == 0 && "fetchCharPtrLibraryNULL");
    assert(outptr == NULL && "fetchCharPtrLibraryNULL");
}

int main(int argc, char *argv[])
{
    CHA_init_test();
    
    test_char_ptr_out();

    return 0;
}

