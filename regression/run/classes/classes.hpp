// Copyright (c) 2017-2021, Lawrence Livermore National Security, LLC and
// other Shroud Project Developers.
// See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//
// classes.hpp - wrapped routines
//

#ifndef CLASSES_HPP
#define CLASSES_HPP

#include <string>

extern int global_flag;

namespace classes
{

class Class1
{
public:
    int m_flag;
    int m_test;
    Class1()         : m_flag(0), m_test(0)    {};
    Class1(int flag) : m_flag(flag), m_test(0) {};
    int Method1();
    bool equivalent(Class1 const &obj2) const;
    Class1 * returnThis();
    Class1 * returnThisBuffer(std::string & name, bool flag);
    Class1 * getclass3() const;
    const std::string& getName();

    enum DIRECTION { UP = 2, DOWN, LEFT= 100, RIGHT };

    DIRECTION directionFunc(Class1::DIRECTION arg);
};

class Class2
{
public:
    const std::string& getName();
};
    
// Note that this function has the same name as a function in Class1
Class1::DIRECTION directionFunc(Class1::DIRECTION arg);

void passClassByValue(Class1 arg);
int useclass(const Class1 *arg);
void getclass(const Class1 **arg);
const Class1 * getclass2();
Class1 * getclass3();
const Class1 &getConstClassReference();
Class1 &getClassReference();
Class1 getClassCopy(int flag);

#if 0
class Singleton {
public:
    static Singleton* instancePtr() {
      //        if (Singleton::m_InstancePtr == 0) {
      //      Singleton::m_InstancePtr = new Singleton;
        if (m_InstancePtr == 0) {
            m_InstancePtr = new Singleton;
        }
        return m_InstancePtr;
    }
private:
    Singleton();
    Singleton(const Singleton& rhs);
    Singleton& operator=(const Singleton& rhs);
    static Singleton* m_InstancePtr;
};
#else
class Singleton
{
    public:
        static Singleton& getReference()
        {
            static Singleton    instance; // Guaranteed to be destroyed.
                                          // Instantiated on first use.
            return instance;
        }
    private:
        Singleton() {}                    // Constructor? (the {} brackets) are needed here.

        // C++ 03
        // ========
        // Don't forget to declare these two. You want to make sure they
        // are unacceptable otherwise you may accidentally get copies of
        // your singleton appearing.
        Singleton(Singleton const&);           // Don't Implement
        void operator=(Singleton const&);      // Don't implement
};
#endif

void set_global_flag(int arg);
int get_global_flag();
const std::string& LastFunctionCalled();

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


#endif // CLASSES_HPP
