/*
 * Copyright (c) 2017-2019, Lawrence Livermore National Security, LLC.
 *
 * Produced at the Lawrence Livermore National Laboratory
 *
 * LLNL-CODE-738041.
 *
 * All rights reserved.
 *
 * This file is part of Shroud.
 *
 * For details about use and distribution, please read LICENSE.
 *
 * Test C interface for templates.yaml
 */

#include <wrapvector_int.h>

#include <assert.h>

void test_int_vector(void)
{
  TEM_vector_int v1_buf;
  TEM_vector_int *v1 = TEM_vector_int_ctor(&v1_buf);
  int value = 1;
  int * out;

  TEM_vector_int_push_back(v1, &value);

  out = TEM_vector_int_at(v1, 0);
  assert(*out == 1 && "TEM_vector_int_at");

  /* XXX - need to catch std::out_of_range */
  // out = TEM_vector_int_at(v1, 10);

}

int main(int argc, char *argv[])
{
  test_int_vector();

  return 0;
}




