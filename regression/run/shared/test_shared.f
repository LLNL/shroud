! Copyright Shroud Project Developers. See LICENSE file for details.
!
! SPDX-License-Identifier: (BSD-3-Clause)
! #######################################################################
!
! Test Fortran API generated from shared.yaml.
! Test std::shared_ptr assignment.
!

module test_shared_mod
  use fruit
  use shared_mod
contains
  subroutine test_shared
    call test_object_assign
    call test_object_alias
    call test_object_assign_null
    call test_object_move_alias
    call test_object_copy_alias
  contains
    subroutine test_object_assign
      type(object_shared) objectPtr

      call set_case_name("test_shared")

      objectPtr = object_shared()
      call assert_true(objectPtr%associated(), "objectPtr associated after ctor")

      call objectPtr%dtor
      call assert_false(objectPtr%associated(), "objectPtr associated after dtor")
      
    end subroutine test_object_assign

    subroutine test_object_alias
      type(object_shared) objectPtr, objectPtr2

      call set_case_name("test_shared_alias")
      
      objectPtr = object_shared()
      call assert_true(objectPtr%associated(), "objectPtr associated after ctor")
      
      ! Create an alias.
      objectPtr2 = objectPtr 
      call assert_true(objectPtr2%associated(), "objectPtr2 associated after assignment")
      ! The shared_ptr are different, but the Object pointers are the same.
      !call assert_true(objectPtr .eq. objectPtr2, "Aliased object")
      
      ! A no-op since the same.
      objectPtr = objectPtr2

      ! reference count will be decremented.
      ! alias will not be deleted, it has no ownership.
      call objectPtr2%dtor
      call assert_false(objectPtr2%associated(), "objectPtr2 associated after dtor")
      
      ! Delete original object.
      call objectPtr%dtor
      call assert_false(objectPtr%associated(), "objectPtr associated after dtor")
      
    end subroutine test_object_alias
    
    subroutine test_object_assign_null
      type(object_shared) objectPtr, objectNULL
      
      call set_case_name("test_shared_assign_null")
      
      objectPtr = object_shared()
      call assert_true(objectPtr%associated(), "objectPtr associated after ctor")
      
      ! Assign empty object will delete LHS.
      objectPtr = objectNULL
      call assert_false(objectPtr%associated(), "objectPtr associated after assignment")
      
    end subroutine test_object_assign_null
    
    subroutine test_object_move_alias
      type(object_shared) objectPtr
      
      call set_case_name("test_shared_move_alias")
      
      objectPtr = object_shared()
      call assert_true(objectPtr%associated(), "objectPtr associated after ctor")
      
      objectPtr = object_shared()
      call assert_true(objectPtr%associated(), "objectPtr associated after second ctor")
      
    end subroutine test_object_move_alias
    
    subroutine test_object_copy_alias
      type(object_shared) objectPtr, objectPtr2
      
      call set_case_name("test_shared_copy_alias")
      
      objectPtr = object_shared()
      call assert_true(objectPtr%associated(), "objectPtr associated after ctor")
      
      objectPtr2 = object_shared()
      call assert_true(objectPtr%associated(), "objectPtr associated after second ctor")
      
      objectPtr = objectPtr2
      
    end subroutine test_object_copy_alias
  end subroutine test_shared

end module test_shared_mod
