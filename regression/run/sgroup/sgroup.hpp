// Copyright Shroud Project Developers. See LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

//----------------------------------------------------------------------

// template functions which take two arguments:
// a pointer to data and a length.

template<typename T, typename U>
struct twostruct
{
    T* values;
    U length;
};


template<typename T, typename U>
void process_twostruct(twostruct<T, U> arg)
{
}
