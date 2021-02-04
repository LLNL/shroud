-- Copyright (c) 2017-2021, Lawrence Livermore National Security, LLC and
-- other Shroud Project Developers.
-- See the top-level COPYRIGHT file for details.
--
-- SPDX-License-Identifier: (BSD-3-Clause)
--
-- #######################################################################
-- test classes module

local classes = require "classes"
local rv_int, rv_double, rv_logical, rv_char

function class1_final()
    -- call a class
    local obj = classes.Class1()
    obj:Method1()
    print(classes.LastFunctionCalled())

    -- __gc method
    --  obj:delete()  -- l_Class1_delete
end

--function class1_new_by_value()
--    local obj = classes.getClassCopy(5)
--end

function test_class1()
    local obj = classes.Class1()

--[[
    mtest = obj0%get_test()
    call assert_equals(0, mtest, "get_test 1")

    call obj0%set_test(4)
    mtest = obj0%get_test()
    call assert_equals(4, mtest, "get_test 2")

    obj1 = class1(1)
    ptr = obj1%get_instance()
    call assert_true(c_associated(ptr), "class1_new obj1")

    iflag = obj0%method1()
    call assert_equals(iflag, 0, "method1 0")

    iflag = obj1%method1()
    call assert_equals(iflag, 1, "method1 1")

    call assert_true(obj0 .eq. obj0, "obj0 .eq obj0")
    call assert_true(obj0 .ne. obj1, "obj0 .ne. obj1")

    call assert_true(obj0%equivalent(obj0), "equivalent 1")
    call assert_false(obj0%equivalent(obj1), "equivalent 2")

    ! This function has return_this=True, so it returns nothing
    call obj0%return_this()

    ! This function has return_this=False, so it returns obj0
    obj2 = obj0%return_this_buffer("bufferify", .true.)
    call assert_true(obj0 .eq. obj2, "return_this_buffer equal")

    direction = -1
    direction = obj0%direction_func(class1_left)
    call assert_equals(class1_left, direction, "obj0.directionFunc")

    direction = -1
    direction = direction_func(class1_left)
    call assert_equals(class1_right, direction, "directionFunc")

    ! Since obj0 is passed by value, save flag in global_flag
    
    call set_global_flag(0)
    call obj0%set_test(13)
    call pass_class_by_value(obj0)
    iflag = get_global_flag()
    call assert_equals(iflag, 13, "passClassByValue")

    ! use class assigns global_class1 returned by getclass
    call obj0%set_test(0)
    iflag = useclass(obj0)
    call assert_equals(iflag, 0, "useclass")

    obj0a = getclass2()
    ptr = obj0a%get_instance()
    call assert_true(c_associated(ptr), "getclass2 obj0a")
    call assert_true(obj0 .eq. obj0a, "getclass2 - obj0 .eq. obj0a")

    obj0a = getclass3()
    ptr = obj0a%get_instance()
    call assert_true(c_associated(ptr), "getclass3 obj0a")
    call assert_true(obj0 .eq. obj0a, "getclass3 - obj0 .eq. obj0a")

    obj0a = get_const_class_reference()
    ptr = obj0a%get_instance()
    call assert_true(c_associated(ptr), "getConstClassReference obj0a")
    call assert_true(obj0 .eq. obj0a, "getConstClassReference - obj0 .eq. obj0a")

    obj0a = get_class_reference()
    ptr = obj0a%get_instance()
    call assert_true(c_associated(ptr), "getClassReference obj0a")
    call assert_true(obj0 .eq. obj0a, "getClassReference - obj0 .eq. obj0a")

    call obj0%delete
    ptr = obj0%get_instance()
    call assert_true(.not. c_associated(ptr), "class1_delete obj0")

    call obj1%delete
    ptr = obj1%get_instance()
    call assert_true(.not. c_associated(ptr), "class1_delete obj1")
--]]

end

function test_subclass()
end

class1_final()
--class1_new_by_value()
test_class1()

