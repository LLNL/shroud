// Copyright (c) 2017-2023, Lawrence Livermore National Security, LLC and
// other Shroud Project Developers.
// See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//
// typdefs.h - wrapped routines
//

#ifndef TYPEDEFS_HPP
#define TYPEDEFS_HPP

typedef int TypeID;
TypeID typefunc(TypeID arg);

struct s_Struct1 {
    int i;
    double d;
};
typedef struct s_Struct1 Struct1Rename;

void typestruct(Struct1Rename *arg1);

#endif // TYPEDEFS_HPP
