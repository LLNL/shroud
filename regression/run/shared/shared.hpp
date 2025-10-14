// Copyright Shroud Project Developers. See LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//
// shared.hpp - wrapped routines
//

#ifndef SHARED_HPP
#define SHARED_HPP

#include <memory>

class Object
{
public:
    Object() 
    {
        count = 0;
    }

    ~Object()
    {
    }

    std::shared_ptr<Object>* createChildA(void)
    {
        childA = std::shared_ptr<Object>(new Object);
        return &childA;
    }
    std::shared_ptr<Object>* createChildB(void)
    {
        childB = std::shared_ptr<Object>(new Object);
        return &childB;
    }
    void replaceChildB(std::shared_ptr<Object>* child)
    {
        childB = *child;
    }

    int count;

private:
    std::shared_ptr<Object> childA;
    std::shared_ptr<Object> childB;
};

#endif // SHARED_HPP
