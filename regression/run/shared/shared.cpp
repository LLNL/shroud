// Copyright Shroud Project Developers. See LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//
// shared.cpp - wrapped routines
//

#include "shared.hpp"

int global_id = 0;

// Reset global_id before starting a new group of tests.
void reset_id(void)
{
    global_id = 0;
}

// Function equivalent to use_count method.
// But allows a null pointer so it can be called after
// Fortran calls the destructor.
// It also works better with Fruit's assert_equals since it is an int.
// Fruit does not support the long returned by obj.use_count().

int use_count(const std::shared_ptr<Object> *f)
{
    if (!f) return 0;
    return f->use_count();
}

int use_count(const std::weak_ptr<Object> *f)
{
    if (!f) return 0;
    return f->use_count();
}
