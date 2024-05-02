/*
 * Copyright (c) 2017-2024, Lawrence Livermore National Security, LLC and
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

#include <complex>

namespace structns {
    struct Cstruct1 {
        int ifield;
        double dfield;
    };

    int passStructByReference(Cstruct1 &arg);
    int passStructByReferenceIn(const Cstruct1 &arg);
    void passStructByReferenceInout(Cstruct1 &arg);
    void passStructByReferenceOut(Cstruct1 &arg);
};  // namespace cxxlibrary


//----------------------------------------------------------------------

struct Cstruct1_cls {
    int ifield;
    double dfield;
};

int passStructByReferenceCls(Cstruct1_cls &arg);
int passStructByReferenceInCls(const Cstruct1_cls &arg);
void passStructByReferenceInoutCls(Cstruct1_cls &arg);
void passStructByReferenceOutCls(Cstruct1_cls &arg);

//----------------------------------------------------------------------
// pointers
// default value

bool defaultPtrIsNULL(double *data = nullptr);

//----------------------------------------------------------------------

void defaultArgsInOut(int in1, int *out1, int *out2, bool flag = false);

//----------------------------------------------------------------------

void accept_complex(std::complex<double> *arg1);

//----------------------------------------------------------------------

const std::string& getGroupName(long idx);

//----------------------------------------------------------------------

struct nested {
    int index;
    int sublevels;
    nested *parent;
    nested **child;   // An array of pointers to children
    nested *array;    // An array of sublevels nested children
};

//----------------------------------------------------------------------

typedef long LengthType;

class Class1
{
public:
    int m_length;
    Class1() : m_length(99) {};

    // Test fortran_generic with default arguments.
    int check_length(int length = 1)
    {
        return length;
    };

    // test return_this
    Class1* declare(int flag, LengthType length = 1)
    {
        m_length = length;
        return this;
    };
};

#endif // CXXLIBRARY_H

