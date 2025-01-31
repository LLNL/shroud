// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
// other Shroud Project Developers.
// See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//
// shared.hpp - wrapped routines
//

#include "shared.hpp"

#include <iostream>
#include <vector>

int main() {

    std::shared_ptr<Object> objectSharedPtr(new Object);

    std::shared_ptr<Object>* childA = objectSharedPtr->createChildA();
    std::shared_ptr<Object>* childB = objectSharedPtr->createChildB();

    std::cout << "shared A: " << childA->use_count() << std::endl;
    std::cout << "shared B: " << childB->use_count() << std::endl;

    std::weak_ptr<Object> wpA = *childA;
    std::weak_ptr<Object> wpB = *childB;

    std::cout << "weak A: " << wpA.use_count() << std::endl;
    std::cout << "weak B: " << wpB.use_count() << std::endl;

    objectSharedPtr->replaceChildB(childA);
    std::cout << "weak A: " << wpA.use_count() << std::endl;
    std::cout << "weak B: " << wpB.use_count() << std::endl;

    std::cout << "shared A: " << childA->use_count() << std::endl;
    std::cout << "shared B: " << childB->use_count() << std::endl;
    
    return 0;
}
