// Copyright (c) 2017-2021, Lawrence Livermore National Security, LLC and
// other Shroud Project Developers.
// See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
// #######################################################################
//
// Tests for ownership.cpp
//

#include "ownership.hpp"

#include <stdlib.h>   // exit


/* Test when a class implements reference counting */
void zbr()
{
  Zbr *bill = new Zbr("bill");
  if (bill->GetReferenceCount() != 1) exit(1);

  delete_stored_zbr();

  store_zbr(bill);
  if (bill->GetReferenceCount() != 1) exit(1);

  Zbr *other_bill = bill;
  other_bill->Ref();
  if (bill->GetReferenceCount() != 2) exit(1);

  delete_stored_zbr();
  if (bill->GetReferenceCount() != 1) exit(1);

  //  other_bill->Unref();
}


int main(int argc, char *argv[])
{
  zbr();
}
