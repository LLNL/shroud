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
    bool m_bool;
    Class1();
    Class1(int flag);
    ~Class1();
    int Method1();
};

Class1 getClass1Copy(int flag);
 
//----------------------------------------------------------------------
//----------------------------------------------------------------------
// Test inheritance

class Shape {
private:
    int m_ivar;
public:
    int get_ivar() const { return m_ivar; }
};

class Circle : public Shape {
public:
    double m_radius;
};
 
} /* end namespace classes */


