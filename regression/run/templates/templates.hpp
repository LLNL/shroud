// Copyright Shroud Project Developers. See LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

// example from http://www.cplusplus.com/doc/tutorial/templates/
template <class T>
class mypair {
    T values [2];
  public:
    mypair (T first, T second)
    {
      values[0]=first;
      values[1]=second;
    }
    T getmax ();
};

template <class T>
T mypair<T>::getmax ()
{
  T retval = values[0] > values[1] ? values[0] : values[1];
  return retval;
}

//----------------------------------------------------------------------

class Worker
{
};

// Function template with two template parameters.
template<typename T, typename U> void FunctionTU(T arg1, U arg2)
{
}

template<typename T>
T ReturnT()
{
  T arg;
  return arg;
}

//----------------------------------------------------------------------

// Function which uses a templated T in the implemetation.
// templates.yaml contains:
//  cxx_template:
//  - instantiation: <internal::ImplWorker1>
//  - instantiation: <internal::ImplWorker2>

template<typename T>
int UseImplWorker()
{
  return T::getValue();
}

//----------------------------------------------------------------------

template<typename T>
class user {
public:
  template<typename U> void nested(T arg1, U arg2)
  { }
};

user<int> returnUserType(void);

//----------------------------------------------------------------------

template<typename T>
struct structAsClass {
    int npts;
    T value;
    void set_npts(int n) { npts=n; };
    int get_npts() { return npts; };
    void set_value(T v) { value = v; };
    T get_value() { return value; };
};


template<typename T>
struct userStruct {
    int npts;
    T value;
    void set_npts(int n) { npts=n; };
    int get_npts() { return npts; };
    void set_value(T v) { value = v; };
    T get_value() { return value; };
};

