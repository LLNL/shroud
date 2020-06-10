%module classes_mod
%{
#include "classes.hpp"
%}

namespace classes
{

class Class1
{
 public:
    int m_flag;
    int m_test;
    Class1();
    Class1(int flag);
    ~Class1();
    int Method1();
};

} /* end namespace classes */


