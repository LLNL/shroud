/*
 * Copyright (c) 2017-2020, Lawrence Livermore National Security, LLC and
 * other Shroud Project Developers.
 * See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (BSD-3-Clause)
 * #######################################################################
 *
 * cxxlibrary.hpp
 */

#ifndef CXXLIBRARY_H
#define CXXLIBRARY_H

struct Cstruct1 {
    int ifield;
    double dfield;
};

int passStructByReference(Cstruct1 &arg);

//----------------------------------------------------------------------

struct Cstruct1_cls {
    int ifield;
    double dfield;
};

int passStructByReferenceCls(Cstruct1_cls &arg);

#endif // CXXLIBRARY_H

