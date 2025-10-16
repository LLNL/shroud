! Copyright Shroud Project Developers. See LICENSE file for details.
!
! SPDX-License-Identifier: (BSD-3-Clause)
! #######################################################################
!
! Test Fortran API generated from shared.yaml.
! Test std::shared_ptr assignment.
!

module test_shared_mod
  use iso_c_binding
  use fruit
  use shared_mod
  implicit none
contains
  subroutine test_shared
    call test_object_assign
    call test_object_alias
    call test_object_assign_null
    call test_object_move_alias
    call test_object_copy_alias
    call test_shared_from_object
  contains
    subroutine test_object_assign
      type(object_shared) objectPtr

      call set_case_name("test_shared")
      call reset_id

      objectPtr = object_shared()
      call assert_true(objectPtr%associated(), "objectPtr associated after ctor")
      call assert_equals(0, objectPtr%get_id(), "objectPtr id A")
      call assert_equals(1, use_count(objectPtr), "objectPtr use_count A")

      call objectPtr%dtor
      call assert_false(objectPtr%associated(), "objectPtr associated after dtor")
      call assert_equals(0, use_count(objectPtr), "objectPtr use_count B")
      
    end subroutine test_object_assign

    subroutine test_object_alias
      type(object_shared) objectPtr, objectPtr2

      call set_case_name("test_shared_alias")
      call reset_id
      
      objectPtr = object_shared()
      call assert_true(objectPtr%associated(), "objectPtr associated after ctor")
      call assert_equals(0, objectPtr%get_id(), "objectPtr id A")
      call assert_equals(1, use_count(objectPtr), "objectPtr use_count A")
      
      ! Create an alias.
      objectPtr2 = objectPtr 
      call assert_true(objectPtr2%associated(), "objectPtr2 associated after assignment")
      call assert_equals(0, objectPtr2%get_id(), "objectPtr id B")
      call assert_equals(2, use_count(objectPtr2), "objectPtr use_count B")
      ! The shared_ptr are different, but the Object pointers are the same.
      !call assert_true(objectPtr .eq. objectPtr2, "Aliased object")
      
      ! A no-op since the same.
      objectPtr = objectPtr2
      call assert_equals(0, objectPtr%get_id(), "objectPtr id C")
      call assert_equals(0, objectPtr2%get_id(), "objectPtr id D")
      call assert_equals(2, use_count(objectPtr), "objectPtr use_count C")
      call assert_equals(2, use_count(objectPtr2), "objectPtr use_count D")

      ! reference count will be decremented.
      ! alias will not be deleted, it has no ownership.
      call objectPtr2%dtor
      call assert_false(objectPtr2%associated(), "objectPtr2 associated after dtor")
      call assert_equals(1, use_count(objectPtr), "objectPtr use_count E")
      
      ! Delete original object.
      call objectPtr%dtor
      call assert_false(objectPtr%associated(), "objectPtr associated after dtor")
      call assert_equals(0, use_count(objectPtr), "objectPtr use_count F")
      
    end subroutine test_object_alias
    
    subroutine test_object_assign_null
      type(object_shared) objectPtr, objectNULL
      
      call set_case_name("test_shared_assign_null")
      call reset_id
      
      objectPtr = object_shared()
      call assert_true(objectPtr%associated(), "objectPtr associated after ctor")
      call assert_equals(1, use_count(objectPtr), "objectPtr use_count A")
      
      ! Assign empty object will delete LHS.
      objectPtr = objectNULL
      call assert_false(objectPtr%associated(), "objectPtr associated after assignment")
      call assert_equals(0, use_count(objectPtr), "objectPtr use_count B")
      
    end subroutine test_object_assign_null
    
    subroutine test_object_move_alias
      type(object_shared) objectPtr
      
      call set_case_name("test_shared_move_alias")
      call reset_id
      
      objectPtr = object_shared()
      call assert_true(objectPtr%associated(), "objectPtr associated after ctor")
      call assert_equals(0, objectPtr%get_id(), "objectPtr id A")
      call assert_equals(1, use_count(objectPtr), "objectPtr use_count A")
      
      objectPtr = object_shared()
      call assert_true(objectPtr%associated(), "objectPtr associated after second ctor")
      call assert_equals(1, objectPtr%get_id(), "objectPtr id B")
      call assert_equals(1, use_count(objectPtr), "objectPtr use_count B")
      
    end subroutine test_object_move_alias
    
    subroutine test_object_copy_alias
      type(object_shared) objectPtr, objectPtr2
      
      call set_case_name("test_shared_copy_alias")
      call reset_id
      
      objectPtr = object_shared()
      call assert_true(objectPtr%associated(), "objectPtr associated after ctor")
      call assert_equals(0, objectPtr%get_id(), "objectPtr id A")
      call assert_equals(1, use_count(objectPtr), "objectPtr use_count A")
     
      objectPtr2 = object_shared()
      call assert_true(objectPtr2%associated(), "objectPtr associated after second ctor")
      call assert_equals(1, objectPtr2%get_id(), "objectPtr id B")
      call assert_equals(1, use_count(objectPtr2), "objectPtr use_count A")
      
      objectPtr = objectPtr2
      call assert_equals(1, objectPtr%get_id(), "objectPtr id C")
      call assert_equals(1, objectPtr2%get_id(), "objectPtr id D")
      call assert_equals(2, use_count(objectPtr), "objectPtr use_count A")
      
    end subroutine test_object_copy_alias

    subroutine test_shared_from_object
!      type(object) obj1
      type(object_shared) sp1
      integer(C_LONG) count
      
      call set_case_name("test_shared_from_object")
      call reset_id

      sp1 = object()
      count = sp1%use_count()
      call assert_equals(1, int(count), "use_count after assignment")
      call assert_equals(1, use_count(sp1), "use_count after assignment")
      call assert_equals(0, sp1%get_id(), "objectPtr id A")

    end subroutine test_shared_from_object

  end subroutine test_shared

end module test_shared_mod
