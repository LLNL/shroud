/*
 * Copyright (c) 2017-2023, Lawrence Livermore National Security, LLC and
 * other Shroud Project Developers.
 * See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (BSD-3-Clause)
 *
 * Test C API generated from statement.yaml.
 */

#include <wrapstatement.h>

#include <assert.h>
#include <string.h>

void test_statement(void)
{
  const char * name = STMT_getNameErrorPattern();
  assert(strcmp(name, "the-name") == 0 && "STMT_get_name_error_pattern");
}

int main(int argc, char *argv[])
{
  test_statement();

  return 0;
}

