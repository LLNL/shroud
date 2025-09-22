// Copyright Shroud Project Developers. See LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//
// maincpp.cpp - Test the shared.hpp file from C++.
//

#include "shared.hpp"

#include <iostream>
#include <vector>

// Similar to the generated Fortran derived types.
using ObjectShared = std::shared_ptr<Object>;
using ObjectWeak   = std::weak_ptr<Object>;

int main() {

    ObjectShared objectSharedPtr(new Object);

    ObjectShared* childA = objectSharedPtr->createChildA();
    ObjectShared* childB = objectSharedPtr->createChildB();

    std::cout << "shared A: " << childA->use_count() << std::endl;
    std::cout << "shared B: " << childB->use_count() << std::endl;

    ObjectWeak wpA = *childA;
    ObjectWeak wpB = *childB;

    std::cout << "weak A: " << wpA.use_count() << std::endl;
    std::cout << "weak B: " << wpB.use_count() << std::endl;

    objectSharedPtr->replaceChildB(childA);
    std::cout << "weak A: " << wpA.use_count() << std::endl;
    std::cout << "weak B: " << wpB.use_count() << std::endl;

    std::cout << "shared A: " << childA->use_count() << std::endl;
    std::cout << "shared B: " << childB->use_count() << std::endl;
    
    return 0;
}
