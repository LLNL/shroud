
#cpdef object foo():
#    cdef intList li
#    li.value = 10

cimport ccstruct


cdef class Cstruct1:
    cdef ccstruct.Cstruct1 ob

    def __cinit__(self, int i, double d):
        self.ob.ifield = i
        self.ob.dfield = d
    property ifield:
        def __get__(self): return self.ob.ifield
        def __set__(self, value): self.ob.ifield = value
    property dfield:
        def __get__(self): return self.ob.dfield
        def __set__(self, value): self.ob.dfield = value

def acceptStructIn(Cstruct1 arg not None):
    # Transform python strings into c strings.
#    cdef bytes query_bytes = query.encode();
#    cdef char* cquery = query_bytes;
    rv = ccstruct.acceptStructIn(arg.ob)
    return rv
