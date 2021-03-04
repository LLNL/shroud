// helloworld.cpp
// Functions for spack to interact with libabigail (or other ABI libraries)

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
