! Copyright Shroud Project Developers. See LICENSE file for details.
!
! SPDX-License-Identifier: (BSD-3-Clause)
! #######################################################################
!
! Test Fortran API generated from defaultarg.yaml.
!

program tester
  use fruit
  use iso_c_binding
  use defaultarg_mod
  implicit none
  logical ok

  call init_fruit

  call test_defaultarg
  call test_class

  call fruit_summary
  call fruit_finalize

  call is_all_successful(ok)
  if (.not. ok) then
     call exit(1)
  endif

contains

  subroutine test_defaultarg
    call set_case_name("defaultarg")
  end subroutine test_defaultarg

  subroutine test_class
    type(class1) obj
    
    call set_case_name("class")

    obj = class1(10)
    call assert_equals(10, obj%get_field1(), "get_field1  #1")
    call assert_equals( 1, obj%get_field2(), "get_field2  #1")
    call assert_equals( 2, obj%get_field3(), "get_field3  #1")

    call obj%default_arguments(20, 21, 22)
    call assert_equals(20, obj%get_field1(), "get_field1  #2")
    call assert_equals(21, obj%get_field2(), "get_field2  #2")
    call assert_equals(22, obj%get_field3(), "get_field3  #2")
    
    call obj%default_arguments(30)
    call assert_equals(30, obj%get_field1(), "get_field1  #3")
    call assert_equals( 1, obj%get_field2(), "get_field2  #3")
    call assert_equals( 2, obj%get_field3(), "get_field3  #3")

    ! Call the specific function
    call class1_default_arguments_2(obj, 40,41,42)
    call assert_equals(40, obj%get_field1(), "get_field1  #4")
    call assert_equals(41, obj%get_field2(), "get_field2  #4")
    call assert_equals(42, obj%get_field3(), "get_field3  #4")

    ! Call generic function
    call class1_default_arguments(obj, 43,44,45)
    call assert_equals(43, obj%get_field1(), "get_field1  #5")
    call assert_equals(44, obj%get_field2(), "get_field2  #5")
    call assert_equals(45, obj%get_field3(), "get_field3  #5")
    
  end subroutine test_class

end program tester
