// Copyright (c) 2017-2019, Lawrence Livermore National Security, LLC. 
//
// Produced at the Lawrence Livermore National Laboratory 
//
// LLNL-CODE-738041.
//
// All rights reserved. 
//
// This file is part of Shroud.
//
// For details about use and distribution, please read LICENSE.
//
// #######################################################################
//
// tutorial.hpp - wrapped routines
//

#ifndef TUTORIAL_HPP
#define TUTORIAL_HPP

#include <string>

extern int global_flag;

namespace tutorial
{

enum EnumTypeID {
    ENUM0,
    ENUM1,
    ENUM2
};

enum Color {
    RED,
    BLUE,
    WHITE,
};

typedef int TypeID;

struct struct1 {
  int ifield;
  double dfield;
};

extern int tutorial_flag;

void Function1();

double Function2(double arg1, int arg2);

bool Function3(bool arg);
void Function3b(const bool arg1, bool *arg2, bool *arg3);

const std::string  Function4a(const std::string& arg1, const std::string& arg2);
const std::string& Function4b(const std::string& arg1, const std::string& arg2);
const std::string  Function4c(const std::string& arg1, const std::string& arg2);
const std::string * Function4d();

double Function5(double arg1 = 3.1415, bool arg2 = true);

void Function6(const std::string& name);
void Function6(int indx);

// specialize for int and double in tutorial.cpp
template<typename ArgType>
void Function7(ArgType arg);

// specialize for int and double in tutorial.cpp
template<typename RetType>
RetType Function8();

void Function9(double arg);

void Function10();
void Function10(const std::string &name, double arg2);

void Sum(size_t len, int * values, int *result);

long long TypeLongLong(long long arg1);

int overload1(int num, int offset = 0, int stride = 1);
int overload1(double type, int num, int offset = 0, int stride = 1);

TypeID typefunc(TypeID arg);

EnumTypeID enumfunc(EnumTypeID arg);

Color colorfunc(Color arg);

void getMinMax(int &min, int &max);

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

    enum DIRECTION { UP = 2, DOWN, LEFT= 100, RIGHT };

    DIRECTION directionFunc(Class1::DIRECTION arg);
};

// Note that this function has the same name as a function in Class1
Class1::DIRECTION directionFunc(Class1::DIRECTION arg);

int useclass(const Class1 *arg);
void getclass(const Class1 **arg);
const Class1 * getclass2();
Class1 * getclass3();
Class1 getClassCopy(int flag);

int callback1(int in, int (*incr)(int));

struct1 returnStruct(int i, double d);
struct1 *returnStructPtr(int i, double d);
struct1 *returnStructPtrNew(int i, double d);
void freeStruct(struct1 *arg1);
double acceptStructIn(struct1 arg);
double acceptStructInPtr(struct1 *arg);
void acceptStructOutPtr(struct1 *arg, int i, double d);
void acceptStructInOutPtr(struct1 *arg);

const std::string& LastFunctionCalled();

} /* end namespace tutorial */


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


#endif // TUTORIAL_HPP
