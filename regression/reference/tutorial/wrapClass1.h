// wrapClass1.h
// This is generated code, do not edit
// Copyright (c) 2017-2019, Lawrence Livermore National Security, LLC and
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

#include "typesTutorial.h"
#ifndef __cplusplus
#include <stdbool.h>
#endif

// splicer begin class.Class1.CXX_declarations
// splicer end class.Class1.CXX_declarations

#ifdef __cplusplus
extern "C" {
#endif

//  tutorial::Class1::DIRECTION
enum TUT_Class1_DIRECTION {
    TUT_Class1_UP = 2,
    TUT_Class1_DOWN,
    TUT_Class1_LEFT = 100,
    TUT_Class1_RIGHT
};

// splicer begin class.Class1.C_declarations
// splicer end class.Class1.C_declarations

TUT_Class1 * TUT_Class1_new_default(TUT_Class1 * SHC_rv);

TUT_Class1 * TUT_Class1_new_flag(int flag, TUT_Class1 * SHC_rv);

void TUT_Class1_delete(TUT_Class1 * self);

int TUT_Class1_method1(TUT_Class1 * self);

bool TUT_Class1_equivalent(const TUT_Class1 * self,
    const TUT_Class1 * obj2);

void TUT_Class1_return_this(TUT_Class1 * self);

TUT_Class1 * TUT_Class1_return_this_buffer(TUT_Class1 * self,
    char * name, bool flag, TUT_Class1 * SHC_rv);

TUT_Class1 * TUT_Class1_return_this_buffer_bufferify(TUT_Class1 * self,
    char * name, int Lname, bool flag, TUT_Class1 * SHC_rv);

TUT_Class1 * TUT_Class1_getclass3(const TUT_Class1 * self,
    TUT_Class1 * SHC_rv);

int TUT_Class1_direction_func(TUT_Class1 * self, int arg);

int TUT_Class1_get_m_flag(TUT_Class1 * self);

int TUT_Class1_get_test(TUT_Class1 * self);

void TUT_Class1_set_test(TUT_Class1 * self, int val);

#ifdef __cplusplus
}
#endif

#endif  // WRAPCLASS1_H
