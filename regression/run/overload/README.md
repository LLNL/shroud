

# Test overloading functions

    void apply(IndexType num_elems, IndexType offset, IndexType stride);
    void apply(TypeID type, IndexType num_elems, IndexType offset, IndexType stride);


In C++, the overload can be distinguished because TypeID is distinct from IndexType.
However, in Fortran the enum TypeID will convert to an integer.
This makes the generic function ambiguous if `sizeof(TypeID) == sizeof(IndexType)`.

The size of `IndexType` is controlled by the define `USE_64BIT_INDEXTYPE`.
