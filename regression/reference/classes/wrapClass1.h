// wrapClass1.h
// This is generated code, do not edit
// Copyright (c) 2017-2020, Lawrence Livermore National Security, LLC and
// other Shroud Project Developers.
// See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//
/**
 * \file wrapClass1.h
 * \brief Shroud generated wrapper for Class1 class
 */
// For C users and C++ implementation

#ifndef WRAPCLASS1_H
#define WRAPCLASS1_H

#include "typesclasses.h"
#ifndef __cplusplus
#include <stdbool.h>
#endif

// splicer begin class.Class1.CXX_declarations
// splicer end class.Class1.CXX_declarations

#ifdef __cplusplus
extern "C" {
#endif

//  classes::Class1::DIRECTION
enum CLA_Class1_DIRECTION {
    CLA_Class1_UP = 2,
    CLA_Class1_DOWN,
    CLA_Class1_LEFT = 100,
    CLA_Class1_RIGHT
};

// splicer begin class.Class1.C_declarations
// splicer end class.Class1.C_declarations

CLA_Class1 * CLA_Class1_new_default(CLA_Class1 * SHC_rv);

CLA_Class1 * CLA_Class1_new_flag(int flag, CLA_Class1 * SHC_rv);

void CLA_Class1_delete(CLA_Class1 * self);

int CLA_Class1_method1(CLA_Class1 * self);

bool CLA_Class1_equivalent(const CLA_Class1 * self, CLA_Class1 * obj2);

void CLA_Class1_return_this(CLA_Class1 * self);

CLA_Class1 * CLA_Class1_return_this_buffer(CLA_Class1 * self,
    char * name, bool flag, CLA_Class1 * SHC_rv);

CLA_Class1 * CLA_Class1_return_this_buffer_bufferify(CLA_Class1 * self,
    char * name, int Lname, bool flag, CLA_Class1 * SHC_rv);

CLA_Class1 * CLA_Class1_getclass3(const CLA_Class1 * self,
    CLA_Class1 * SHC_rv);

int CLA_Class1_direction_func(CLA_Class1 * self, int arg);

int CLA_Class1_get_m_flag(CLA_Class1 * self);

int CLA_Class1_get_test(CLA_Class1 * self);

void CLA_Class1_set_test(CLA_Class1 * self, int val);

#ifdef __cplusplus
}
#endif

#endif  // WRAPCLASS1_H