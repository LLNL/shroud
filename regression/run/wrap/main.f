! Copyright Shroud Project Developers. See LICENSE file for details.
!
! SPDX-License-Identifier: (BSD-3-Clause)
! #######################################################################
!
! Test Fortran API generated from wrap.yaml.
!
program tester
  use fruit
  use iso_c_binding
  use wrap_mod
  implicit none
  logical ok

  call init_fruit

  call test_fortran

  call fruit_summary
  call fruit_finalize

  call is_all_successful(ok)
  if (.not. ok) then
     call exit(1)
  endif

contains

  subroutine test_fortran
    integer(C_INT) rv
    type(class1) obj
    
    call set_case_name("test_fortran")

    obj = class1()
    
    rv = obj%func_in_class()
    call assert_equals(0, rv, "FuncInClass");

  end subroutine test_fortran

  
end program tester
