// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
// other Shroud Project Developers.
// See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
// #######################################################################
//
// Tests for scope.cpp
//

#include "scope.hpp"

#include <stdio.h>

typedef enum Color (*fptr1)(void);

enum Color global1 = RED;
enum Color *global2;
static enum Color *global3 = &global1;

//int Color = 1;

Color decl(Color top)
{
    top = RED;
    return top;
}

enum Color decl2(enum Color top)
{
    top = RED;
    return top;
}

int main(int argc, char *argv[])
{
    int i;
    enum Color local1 = RED;

    printf("Value %d\n", local1);

    i = ns1::Color::RED;
    printf("Value %d\n", i);

    i = ns2::Color::RED;
    printf("Value %d\n", i);

    i = ns3::Color::RED;
    printf("Value %d\n", i);

    i = Class2::RED;
    printf("Value %d\n", i);

    i = static_cast<int>(ColorEnum::RED);
    printf("Value %d\n", i);

    i = decl(RED);
    printf("Value %d\n", i);
    i = decl2(RED);
    printf("Value %d\n", i);

    i = *global3;
    printf("Value %d\n", i);
    
}
