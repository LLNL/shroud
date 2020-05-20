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

    void fillSize(int &size) { size = sz; }
    
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
    double *getArray() { return arr; }
    double *getArrayConst() const { return arr; }
    const double *getArrayC() { return arr; }
    const double *getArrayConstC() const { return arr; }
    void fetchArrayPtr(double **array, int *isize) { *array = arr; *isize = sz; }
    void fetchArrayRef(double *&array, int &isize) { array = arr; isize = sz; }
    void fetchArrayPtrConst(const double **array, int *isize) {
        *array = arr;
        *isize = sz;
    }
    void fetchArrayRefConst(const double *&array, int &isize) {
        array = arr;
        isize = sz;
    }
    void fetchVoidPtr(void **array) { *array = arr; }
    void fetchVoidRef(void *&array) { array = arr; }
    bool checkPtr(void *array) { return array == arr ? true : false; }
    double sumArray() {
        double sum = 0;
        for (int i=0; i < sz; ++i) {
            sum += arr[i];
        }
        return sum;
    }
private:
    double* arr {nullptr};   
    int sz{0};
};



#endif // REFERENCES_HPP
