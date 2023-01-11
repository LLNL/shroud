// Copyright (c) 2017-2023, Lawrence Livermore National Security, LLC and
// other Shroud Project Developers.
// See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//
// typdefs.h - wrapped routines
//

#include "typedefs.h"

TypeID typefunc(TypeID arg)
{
    return arg + 1;
}

void typestruct(Struct1 *arg1)
{
    arg1->d = arg1->i;
}
