! Copyright Shroud Project Developers. See LICENSE file for details.
!
! SPDX-License-Identifier: (BSD-3-Clause)
! #######################################################################
!
! Test Fortran API generated from shared.yaml.
! Test std::weak_ptr assignment.
!
! A object_weak may not have an object associated with it.
! In that case the use_count will be 0.

module test_weak_mod
  use fruit
  use shared_mod
  implicit none
contains
  subroutine test_weak
    call test_object_assign
    call test_object_alias
    call test_object_assign_null
    call test_object_move_alias
    call test_object_copy_alias
    call test_weak_from_shared
  contains
    subroutine test_object_assign
      type(object_weak) objectPtr

      call set_case_name("test_weak")
      call reset_id

      objectPtr = object_weak()
      call assert_true(objectPtr%associated(), "objectPtr associated after ctor")
      call assert_equals(0, int(objectPtr%use_count()), "objectPtr use_count A")

      call objectPtr%dtor
      call assert_false(objectPtr%associated(), "objectPtr associated after dtor")
      call assert_equals(0, use_count(objectPtr), "objectPtr use_count B")
      
    end subroutine test_object_assign

    subroutine test_object_alias
      type(object_weak) objectPtr, objectPtr2

      call set_case_name("test_weak_alias")
      call reset_id
      
      objectPtr = object_weak()
      call assert_true(objectPtr%associated(), "objectPtr associated after ctor")
      call assert_equals(0, use_count(objectPtr), "objectPtr use_count A")
      call assert_equals(0, int(objectPtr%use_count()), "objectPtr use_count A")
      
      ! Create an alias.
      objectPtr2 = objectPtr 
      call assert_true(objectPtr2%associated(), "objectPtr2 associated after assignment")
      call assert_equals(0, int(objectPtr2%use_count()), "objectPtr2 use_count B")
      ! The weak_ptr are different, but the Object pointers are the same.
      !call assert_true(objectPtr .eq. objectPtr2, "Aliased object")
      
      ! A no-op since the same.
      objectPtr = objectPtr2
      call assert_equals(0, int(objectPtr%use_count()), "objectPtr use_count C")
      call assert_equals(0, int(objectPtr2%use_count()), "objectPtr use_count D")

      ! reference count will be decremented.
      ! alias will not be deleted, it has no ownership.
      call objectPtr2%dtor
      call assert_false(objectPtr2%associated(), "objectPtr2 associated after dtor")
      
      ! Delete original object.
      call objectPtr%dtor
      call assert_false(objectPtr%associated(), "objectPtr associated after dtor")
      
    end subroutine test_object_alias
    
    subroutine test_object_assign_null
      type(object_weak) objectPtr, objectNULL
      
      call set_case_name("test_weak_assign_null")
      call reset_id
      
      objectPtr = object_weak()
      call assert_true(objectPtr%associated(), "objectPtr associated after ctor")
      
      ! Assign empty object will delete LHS.
      objectPtr = objectNULL
      call assert_false(objectPtr%associated(), "objectPtr associated after assignment")
      
    end subroutine test_object_assign_null
    
    subroutine test_object_move_alias
      type(object_weak) objectPtr
      
      call set_case_name("test_weak_move_alias")
      call reset_id
      
      objectPtr = object_weak()
      call assert_true(objectPtr%associated(), "objectPtr associated after ctor")
      call assert_equals(0, int(objectPtr%use_count()), "objectPtr use_count A")
      
      objectPtr = object_weak()
      call assert_true(objectPtr%associated(), "objectPtr associated after second ctor")
      call assert_equals(0, int(objectPtr%use_count()), "objectPtr use_count B")
      
    end subroutine test_object_move_alias
    
    subroutine test_object_copy_alias
      type(object_weak) objectPtr, objectPtr2
      
      call set_case_name("test_weak_copy_alias")
      call reset_id
      
      objectPtr = object_weak()
      call assert_true(objectPtr%associated(), "objectPtr associated after ctor")
      call assert_equals(0, int(objectPtr%use_count()), "objectPtr use_count A")
      
      objectPtr2 = object_weak()
      call assert_true(objectPtr%associated(), "objectPtr associated after second ctor")
      call assert_equals(0, int(objectPtr2%use_count()), "objectPtr use_count B")
      
      objectPtr = objectPtr2
      call assert_equals(0, int(objectPtr%use_count()), "objectPtr use_count C")
      call assert_equals(0, int(objectPtr2%use_count()), "objectPtr use_count D")
      
    end subroutine test_object_copy_alias

    subroutine test_weak_from_shared
      type(object_shared) sp1
      type(object_weak) wp1
      
      call set_case_name("test_shared_from_object")
      call reset_id

      sp1 = object_shared()
      call assert_equals(1, int(sp1%use_count()), "use_count A")
      call assert_equals(1, use_count(sp1), "use_count after assignment")
      call assert_equals(0, sp1%get_id(), "objectPtr id A")

      wp1 = sp1
      call assert_equals(1, int(sp1%use_count()), "use_count B")
      call assert_equals(1, int(wp1%use_count()), "use_count B")

    end subroutine test_weak_from_shared
    
  end subroutine test_weak

end module test_weak_mod
