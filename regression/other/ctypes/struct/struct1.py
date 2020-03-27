#https://stackoverflow.com/questions/23329663/access-np-array-in-ctypes-struct

import numpy as np
import ctypes as C

# allocate this as a normal numpy array with specified dtype
array_1d_double = np.array([1,2,3,4,5],dtype="float64")

# set the structure to contain a C double pointer
class Test(C.Structure):
    _fields_ = [("x", C.POINTER(C.c_double))]

# Instantiate the structure so it can be passed to the C code
test = Test(np.ctypeslib.as_ctypes(array_1d_double))

# You can also do:
# test = Test()
# test.x = np.ctypeslib.as_ctypes(array_1d_double)

print test.x
# outputs: <__main__.LP_c_double object at 0x1014aa320>
