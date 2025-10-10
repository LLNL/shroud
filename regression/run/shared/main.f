! Copyright Shroud Project Developers. See LICENSE file for details.
!
! SPDX-License-Identifier: (BSD-3-Clause)
! #######################################################################
!
! Test Fortran API generated from shared.yaml.
!

module test_object_mod
  use fruit
  use shared_mod
contains
  subroutine test_object
    call test_object_assign
    call test_object_alias
    call test_object_assign_null
    call test_object_move_alias
    call test_object_copy_alias
  contains
    subroutine test_object_assign
      type(object) objectPtr

      call set_case_name("test_object")

      objectPtr = object()
      call assert_true(objectPtr%associated(), "objectPtr associated after ctor")

      call objectPtr%dtor
      call assert_false(objectPtr%associated(), "objectPtr associated after dtor")
      
    end subroutine test_object_assign

    subroutine test_object_alias
      type(object) objectPtr, objectPtr2

      call set_case_name("test_object_alias")
      
      objectPtr = object()
      call assert_true(objectPtr%associated(), "objectPtr associated after ctor")
      
      ! Create an alias.
      objectPtr2 = objectPtr 
      call assert_true(objectPtr2%associated(), "objectPtr2 associated after assignment")
      call assert_true(objectPtr .eq. objectPtr2, "Aliased object")
      
      ! A no-op since the same.
      objectPtr = objectPtr2
      
      ! alias will not be deleted, it has no ownership.
      call objectPtr2%dtor
      call assert_false(objectPtr2%associated(), "objectPtr2 associated after dtor")
      
      ! Delete original object.
      call objectPtr%dtor
      call assert_false(objectPtr%associated(), "objectPtr associated after dtor")
      
    end subroutine test_object_alias
    
    subroutine test_object_assign_null
      type(object) objectPtr, objectNULL
      
      call set_case_name("test_object_assign_null")
      
      objectPtr = object()
      call assert_true(objectPtr%associated(), "objectPtr associated after ctor")
      
      ! Assign empty object will delete LHS.
      objectPtr = objectNULL
      call assert_false(objectPtr%associated(), "objectPtr associated after assignment")
      
    end subroutine test_object_assign_null
    
    subroutine test_object_move_alias
      type(object) objectPtr
      
      call set_case_name("test_object_move_alias")
      
      objectPtr = object()
      call assert_true(objectPtr%associated(), "objectPtr associated after ctor")
      
      objectPtr = object()
      call assert_true(objectPtr%associated(), "objectPtr associated after second ctor")
      
    end subroutine test_object_move_alias
    
    subroutine test_object_copy_alias
      type(object) objectPtr, objectPtr2
      
      call set_case_name("test_object_copy_alias")
      
      objectPtr = object()
      call assert_true(objectPtr%associated(), "objectPtr associated after ctor")
      
      objectPtr2 = object()
      call assert_true(objectPtr%associated(), "objectPtr associated after second ctor")
      
      objectPtr = objectPtr2
      
    end subroutine test_object_copy_alias
  end subroutine test_object

end module test_object_mod

!-----------------------------------------------------------------------
!-----------------------------------------------------------------------

program tester
  use fruit
  use iso_c_binding
  use shared_mod
  use test_object_mod
  implicit none
  logical ok

  call init_fruit

  call test_object
  call test_object_methods

  call fruit_summary
  call fruit_finalize

  call is_all_successful(ok)
  if (.not. ok) then
     call exit(1)
  endif

contains

  subroutine test_object_methods
    type(object_shared) objectSharedPtr
    type(object_shared) childA, childB
    type(object_weak) wpA, wpB
    integer(C_LONG) count
    ! use_count returns a LONG but assert_equals does not have generic for LONG.
    ! convert with int(count).

    call set_case_name("test_object_methods")

    objectSharedPtr = object_shared()
    call assert_true(objectSharedPtr%associated())

    childA = objectSharedPtr%create_child_a()
    call assert_true(childA%associated(), "create ChildA")

    childB = objectSharedPtr%create_child_b()
    call assert_true(childB%associated(), "create ChildB")

    count = childA%use_count()
    call assert_equals(1, int(count), "childA use_count")

!    call wpA%assign_weak(childA)
    wpA = childA
    count = wpA%use_count()
    call assert_equals(1, int(count), "wpA use_count before")

!    call wpB%assign_weak(childB)
    wpB = childB
    count = wpB%use_count()
    call assert_equals(1, int(count), "wpB use_count before")
    
    count = childB%use_count()
    call assert_equals(1, int(count), "childB use_count")

    call objectSharedPtr%replace_child_b(childA)

    count = wpA%use_count()
    call assert_equals(2, int(count), "wpA use_count after")

    count = wpB%use_count()
    call assert_equals(0, int(count), "wpB use_count after")

    count = childA%use_count()
    call assert_equals(2, int(count), "childA use_count post replace")

    count = childB%use_count()
    call assert_equals(2, int(count), "childB use_count post replace")
    
  end subroutine test_object_methods

end program tester
