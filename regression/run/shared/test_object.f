! Copyright Shroud Project Developers. See LICENSE file for details.
!
! SPDX-License-Identifier: (BSD-3-Clause)
! #######################################################################
!
! Test Fortran API generated from shared.yaml.
! Test Object assignment.
!

module test_object_mod
  use fruit
  use shared_mod
  implicit none
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
      call reset_id

      objectPtr = object()
      call assert_true(objectPtr%associated(), "objectPtr associated after ctor")
      call assert_equals(0, objectPtr%get_id(), "objectPtr id A")

      call objectPtr%dtor
      call assert_false(objectPtr%associated(), "objectPtr associated after dtor")
      
    end subroutine test_object_assign

    subroutine test_object_alias
      type(object) objectPtr, objectPtr2

      call set_case_name("test_object_alias")
      call reset_id
      
      objectPtr = object()
      call assert_true(objectPtr%associated(), "objectPtr associated after ctor")
      call assert_equals(0, objectPtr%get_id(), "objectPtr id A")
      
      ! Create an alias.
      objectPtr2 = objectPtr 
      call assert_true(objectPtr2%associated(), "objectPtr2 associated after assignment")
      call assert_true(objectPtr .eq. objectPtr2, "Aliased object")
      call assert_equals(0, objectPtr2%get_id(), "objectPtr id B")
      
      ! A no-op since the same.
      objectPtr = objectPtr2
      call assert_equals(0, objectPtr%get_id(), "objectPtr id C")
      call assert_equals(0, objectPtr2%get_id(), "objectPtr id D")
      
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
      call reset_id
      
      objectPtr = object()
      call assert_true(objectPtr%associated(), "objectPtr associated after ctor")
      
      ! Assign empty object will delete LHS.
      objectPtr = objectNULL
      call assert_false(objectPtr%associated(), "objectPtr associated after assignment")
      
    end subroutine test_object_assign_null
    
    subroutine test_object_move_alias
      type(object) objectPtr
      
      call set_case_name("test_object_move_alias")
      call reset_id
      
      objectPtr = object()
      call assert_true(objectPtr%associated(), "objectPtr associated after ctor")
      call assert_equals(0, objectPtr%get_id(), "objectPtr id A")
      
      objectPtr = object()
      call assert_true(objectPtr%associated(), "objectPtr associated after second ctor")
      call assert_equals(1, objectPtr%get_id(), "objectPtr id B")
      
    end subroutine test_object_move_alias
    
    subroutine test_object_copy_alias
      type(object) objectPtr, objectPtr2
      
      call set_case_name("test_object_copy_alias")
      call reset_id
      
      objectPtr = object()
      call assert_true(objectPtr%associated(), "objectPtr associated after ctor")
      call assert_equals(0, objectPtr%get_id(), "objectPtr id A")
      
      objectPtr2 = object()
      call assert_true(objectPtr%associated(), "objectPtr associated after second ctor")
      call assert_equals(1, objectPtr2%get_id(), "objectPtr id B")
      
      objectPtr = objectPtr2
      call assert_equals(1, objectPtr%get_id(), "objectPtr id C")
      call assert_equals(1, objectPtr2%get_id(), "objectPtr id D")
      
    end subroutine test_object_copy_alias
  end subroutine test_object

end module test_object_mod
