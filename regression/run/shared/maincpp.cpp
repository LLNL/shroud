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

//----------------------------------------------------------------------

void test_make_shared(void)
{
    std::shared_ptr<Object> *obj1 = new std::shared_ptr<Object>;
    std::shared_ptr<Object> *obj2 = new std::shared_ptr<Object>;

    *obj1 = std::make_shared<Object>();
    *obj2 = *obj1;
    std::cout << "test_make_shared: " << obj2->use_count() << std::endl;

    std::cout << "test_make_shared: " << obj1 << std::endl;
    std::cout << "test_make_shared: " << obj1->get() << std::endl;
    std::cout << "test_make_shared: " << obj2 << std::endl;
    std::cout << "test_make_shared: " << obj2->get() << std::endl;

    delete obj1;
    obj2->reset();
    std::cout << "test_make_shared: " << obj2->use_count() << std::endl;
}

//----------------------------------------------------------------------

ObjectShared *object_shared(void)
{
    std::shared_ptr<Object> *rv = new ObjectShared;
    *rv =  std::make_shared<Object>();
    return rv;
}

void test_object_assign(void)
{
    ObjectShared *objectPtr;

    objectPtr = object_shared();
    std::cout << "test_object_assign: " << objectPtr->use_count() << std::endl;

    delete objectPtr;
}      

void test_object_alias(void)
{
    ObjectShared *objectPtr, objectPtr2;
      
    objectPtr = object_shared();
    std::cout << "test_object_alias: " << objectPtr->use_count() << std::endl;
      
    // Create an alias.
    objectPtr2 = *objectPtr;
    std::cout << "test_object_alias: " << objectPtr->use_count() << std::endl;
    std::cout << "test_object_alias: " << objectPtr2.use_count() << std::endl;
      
    // A no-op since the same.
    //objectPtr = objectPtr2;

    // reference count will be decremented.
    // alias will not be deleted, it has no ownership.
    //delete objectPtr2;

    // Delete original object.
    delete objectPtr;
    std::cout << "test_object_alias: " << objectPtr2.use_count() << std::endl;
}      
    
#if 0
void test_object_assign_null(void)
{
    ObjectShared *objectPtr, *objectNULL = nullptr;
      
    objectPtr = object_shared();
      
    // Assign empty object will delete LHS.
    objectPtr = objectNULL;
}      
#endif

void test_object_move_alias(void)
{
    ObjectShared *objectPtr;
      
    objectPtr = object_shared();
    std::cout << "test_object_move_alias: " << objectPtr->use_count() << std::endl;
      
    objectPtr = object_shared();
    std::cout << "test_object_move_alias: " << objectPtr->use_count() << std::endl;
}
    
void test_object_copy_alias(void)
{
    ObjectShared *objectPtr, *objectPtr2;
      
    objectPtr = object_shared();
    std::cout << "test_object_copy_alias: " << objectPtr->use_count() << std::endl;
    std::cout << "  " << objectPtr->get() << std::endl;
      
    objectPtr2 = object_shared();
    std::cout << "test_object_copy_alias: " << objectPtr2->use_count() << std::endl;
    std::cout << "  " << objectPtr2->get() << std::endl;
      
    *objectPtr = *objectPtr2;
    std::cout << "test_object_copy_alias: " << objectPtr->use_count() << std::endl;
    std::cout << "test_object_copy_alias: " << objectPtr2->use_count() << std::endl;
    std::cout << "  " << objectPtr->get() << std::endl;
    std::cout << "  " << objectPtr2->get() << std::endl;
}

void test_shared(void)
{
    test_object_assign();
    test_object_alias();
    //test_object_assign_null();
    test_object_move_alias();
    test_object_copy_alias();
}

//----------------------------------------------------------------------

void test_object_methods(void) {

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
}

int main(int argc, char *argv[])
{
    test_make_shared();
    test_shared();
    test_object_methods();
    return 0;
}
