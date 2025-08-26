// helloworld.cpp
// Copyright Shroud Project Developers. See LICENSE file for details.
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
