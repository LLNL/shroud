! Copyright (c) 2017-2023, Lawrence Livermore National Security, LLC and
! other Shroud Project Developers.
! See the top-level COPYRIGHT file for details.
!
! SPDX-License-Identifier: (BSD-3-Clause)
! #######################################################################
!
! Test Fortran API generated from classes.i.
!
program tester
  use fruit
  use iso_c_binding
  use classes_mod
  real(C_DOUBLE), parameter :: pi = 3.1415926_C_DOUBLE
  logical ok

  call init_fruit

  call test_class1_final
  call test_class1
  call test_subclass

  call fruit_summary
  call fruit_finalize

  call is_all_successful(ok)
  if (.not. ok) then
     call exit(1)
  endif

contains

  subroutine test_class1_final
    type(class1) obj0, obj1

    call set_case_name("test_class1_final")

    ! Test generic constructor
!!    obj0 = class1_new()
    obj0 = class1()
!    call assert_equals(1, obj0%cxxmem%refcount, "reference count after new")

    obj1 = obj0
!    call assert_equals(2, obj0%cxxmem%refcount, "rhs reference count after assign")
!    call assert_equals(2, obj1%cxxmem%refcount, "lhs reference count after assign")

    call obj0%release
!    call assert_equals(1, obj1%cxxmem%refcount, "reference count after delete")

    ! should call TUT_SHROUD_array_destructor_function as part of 
    ! FINAL of capsule_data.
  end subroutine test_class1_final

  subroutine test_class1
!--    integer iflag, mtest
!--    integer direction
!--    type(class1) obj0, obj1, obj2
!--    type(class1) obj0a
!--    type(c_ptr) ptr

    call set_case_name("test_class1")

    ! Test generic constructor
!--    obj0 = class1_new()
!--    ptr = obj0%get_instance()
!--    call assert_true(c_associated(ptr), "class1_new obj0")
!--
!--    mtest = obj0%get_test()
!--    call assert_equals(0, mtest, "get_test 1")
!--
!--    call obj0%set_test(4)
!--    mtest = obj0%get_test()
!--    call assert_equals(4, mtest, "get_test 2")
!--
!--    obj1 = class1_new(1)
!--    ptr = obj1%get_instance()
!--    call assert_true(c_associated(ptr), "class1_new obj1")
!--
!--    iflag = obj0%method1()
!--    call assert_equals(iflag, 0, "method1 0")
!--
!--    iflag = obj1%method1()
!--    call assert_equals(iflag, 1, "method1 1")
!--
!--    call assert_true(obj0 .eq. obj0, "obj0 .eq obj0")
!--    call assert_true(obj0 .ne. obj1, "obj0 .ne. obj1")
!--
!--    call assert_true(obj0%equivalent(obj0), "equivalent 1")
!--    call assert_false(obj0%equivalent(obj1), "equivalent 2")
!--
!--    ! This function has return_this=True, so it returns nothing
!--    call obj0%return_this()
!--
!--    ! This function has return_this=False, so it returns obj0
!--    obj2 = obj0%return_this_buffer("bufferify", .true.)
!--    call assert_true(obj0 .eq. obj2, "return_this_buffer equal")
!--
!--    direction = -1
!--    direction = obj0%direction_func(class1_left)
!--    call assert_equals(class1_left, direction, "obj0.directionFunc")
!--
!--    direction = -1
!--    direction = direction_func(class1_left)
!--    call assert_equals(class1_right, direction, "directionFunc")
!--
!--    ! Since obj0 is passed by value, save flag in global_flag
!--    
!--    call set_global_flag(0)
!--    call obj0%set_test(13)
!--    call pass_class_by_value(obj0)
!--    iflag = get_global_flag()
!--    call assert_equals(iflag, 13, "passClassByValue")
!--
!--    ! use class assigns global_class1 returned by getclass
!--    call obj0%set_test(0)
!--    iflag = useclass(obj0)
!--    call assert_equals(iflag, 0, "useclass")
!--
!--    obj0a = getclass2()
!--    ptr = obj0a%get_instance()
!--    call assert_true(c_associated(ptr), "getclass2 obj0a")
!--    call assert_true(obj0 .eq. obj0a, "getclass2 - obj0 .eq. obj0a")
!--
!--    obj0a = getclass3()
!--    ptr = obj0a%get_instance()
!--    call assert_true(c_associated(ptr), "getclass3 obj0a")
!--    call assert_true(obj0 .eq. obj0a, "getclass3 - obj0 .eq. obj0a")
!--
!--    obj0a = get_const_class_reference()
!--    ptr = obj0a%get_instance()
!--    call assert_true(c_associated(ptr), "getConstClassReference obj0a")
!--    call assert_true(obj0 .eq. obj0a, "getConstClassReference - obj0 .eq. obj0a")
!--
!--    obj0a = get_class_reference()
!--    ptr = obj0a%get_instance()
!--    call assert_true(c_associated(ptr), "getClassReference obj0a")
!--    call assert_true(obj0 .eq. obj0a, "getClassReference - obj0 .eq. obj0a")
!--
!--    call obj0%delete
!--    ptr = obj0%get_instance()
!--    call assert_true(.not. c_associated(ptr), "class1_delete obj0")
!--
!--    call obj1%delete
!--    ptr = obj1%get_instance()
!--    call assert_true(.not. c_associated(ptr), "class1_delete obj1")
!--
    ! obj0a has a dangling reference to a deleted object
  end subroutine test_class1

  subroutine test_subclass
    type(Shape) base
    type(Circle) circle1
    integer ivar

    base = Shape()
    ivar = base%get_ivar()

    circle1 = Circle()
    ivar = circle1%get_ivar()
  end subroutine test_subclass

end program tester
