// Using Swig to wrap the tutorial
//

%module tutorial
%{
    /* the resulting C file should be built as a python extension */
    #define SWIG_FILE_WITH_INIT
    /*  Includes the header in the wrapper code */
    #include "tutorial.hpp"
%}

/*  include the numpy typemaps */
//%include "numpy.i"
/*  need this for correct module initialization */
//%init %{
//    import_array();
//%}

namespace tutorial
{

double UseDefaultArguments(double arg1 = 3.1415, bool arg2 = true);

} /* end namespace tutorial */
