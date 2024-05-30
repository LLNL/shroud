// Copyright (c) 2017-2024, Lawrence Livermore National Security, LLC and
// other Shroud Project Developers.
// See the top-level COPYRIGHT file for details.
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
