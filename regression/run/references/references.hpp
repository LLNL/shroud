// Copyright (c) 2017-2020, Lawrence Livermore National Security, LLC and
// other Shroud Project Developers.
// See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//
// references.hpp - wrapped routines
//

#ifndef REFERENCES_HPP
#define REFERENCES_HPP

class ArrayWrapper
{
public:

    /// Default constructor
    ArrayWrapper() = default;

    /// Simple destructor
    ~ArrayWrapper() { delete[] arr; }

    /// Sets the size of the array
    void setSize(int size) { sz = size; }

    /// Sets the size of the array
    int getSize() const { return sz; }

    /// Allocates the array
    void allocate() 
    {
        delete[] arr;
        arr = nullptr;

        if(sz >=0 ) 
            { 
                arr = new double[sz]; 
            }
    }

    /// Access the array
    double* getArray() { return arr; }
    double* getArrayConst() const { return arr; }

private:
    double* arr {nullptr};   
    int sz{0};
};



#endif // REFERENCES_HPP
