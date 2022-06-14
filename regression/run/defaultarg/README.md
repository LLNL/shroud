

# Test default arguments

    void apply(IndexType num_elems, IndexType offset = 0, IndexType stride = 1);
    void apply(TypeID type, IndexType num_elems, IndexType offset = 0, IndexType stride = 1);


In C++, the overload can be distinguished because TypeID is distinct from IndexType.
However, in Fortran the enum TypeID will convert to an integer.
This makes the generic function ambiguous if `sizeof(TypeID) == sizeof(IndexType)`.

The size of `IndexType` is controlled by the define `INDEXTYPE_SIZE`.
