%module ownership_mod
%{
#include "ownership.hpp"
%}


//%include "typemaps.i"

// swig manual page  
// Tell SWIG to use type(c_ptr) for int * arguments in Fortran
//%typemap(fortin) int * "type(c_ptr)"
//%typemap(fortout) int * "type(c_ptr)"
//%typemap(fortin, descriptor="c_ptr") int * {
//    $1 = (int *)$input;
//}

int *ReturnIntPtrDimDefault(int *len);


