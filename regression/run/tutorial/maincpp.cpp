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
// Tests for tutorial.cpp
//

#include "tutorial.hpp"

int tester1()
{
  return tutorial::Class1::DIRECTION::UP;
}
int tester2()
{
  return tutorial::Class1::UP;
}



int main(int argc, char *argv[])
{
    tutorial::Class1 * obj = new tutorial::Class1;

    obj->Method1();

    delete obj;
}
