// helloworld.cpp
// Copyright (c) 2021, Lawrence Livermore National Security, LLC and
// other Shroud Project Developers.
// See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)


#include <iostream>
#include <string>
#include "helloworld.hpp"


namespace helloworld {
    
    int Person::SayHello() {
        std::cout << "Hello!\n";
        return 0;
    }

    int Person::NamedHello(std::string name) {
        std::cout << "Hello " << name << "!\n";
        return 0;
    }
      
}


int main() {

    helloworld::Person person;
    std::string name = "Dinosaur";
    person.NamedHello(name);
    return 0; 
}
