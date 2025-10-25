%module shared_mod
%{
#include "shared.hpp"
%}

// Tell SWIG to use std::shared_ptr for memory management
%include <std_shared_ptr.i>
%shared_ptr(Object);

// Expose shared_ptr<Object> as Object_Shared
//%template(Object_Shared) std::shared_ptr<Object>;

// Include the header file
%include "shared.hpp"
