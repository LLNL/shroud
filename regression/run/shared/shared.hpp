// Copyright Shroud Project Developers. See LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//
// shared.hpp - wrapped routines
//

#ifndef SHARED_HPP
#define SHARED_HPP

#include <memory>

extern int global_id;

class Object
{
public:
    Object()  : m_id(global_id)
    {
        global_id += 1;
    }

    ~Object()
    {
    }

    int get_id(void)
    {
        return m_id;
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

    // Used to identify Object instances.
    int m_id;

private:
    std::shared_ptr<Object> childA;
    std::shared_ptr<Object> childB;
};

void reset_id(void);
int use_count(const std::shared_ptr<Object> *f);

#endif // SHARED_HPP
