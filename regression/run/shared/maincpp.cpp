// Copyright (c) 2017-2024, Lawrence Livermore National Security, LLC and
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

    std::weak_ptr<Object> wpA = *childA;
    std::weak_ptr<Object> wpB = *childB;

    std::cout << wpA.use_count() << std::endl;
    std::cout << wpB.use_count() << std::endl;

    objectSharedPtr->replaceChildB(childA);
    std::cout << wpA.use_count() << std::endl;
    std::cout << wpB.use_count() << std::endl;

    return 0;
}
