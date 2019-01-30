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

class Worker
{
};

namespace internal
{
class ImplWorker1
{
  public:
  static int getValue() {
    return 1;
  }
};
}  // namespace internal

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


// Function which uses a templated T in the implemetation.
template<typename T>
int UseImplWorker()
{
  return T::getValue();
}
