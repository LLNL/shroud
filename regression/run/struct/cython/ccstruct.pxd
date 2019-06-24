
cdef extern from "struct.h":

    cdef struct s_Cstruct1:
        int    ifield
        double dfield
    ctypedef s_Cstruct1 Cstruct1

    double acceptStructIn(Cstruct1 arg)
