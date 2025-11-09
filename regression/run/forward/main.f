! Copyright Shroud Project Developers. See LICENSE file for details.
!
! SPDX-License-Identifier: (BSD-3-Clause)
! #######################################################################
!
! Test Fortran API generated from forward.yaml.
!
program tester
  use fruit
  use iso_c_binding
  use forward_mod
  implicit none
  logical ok

  call init_fruit

  call test_struct

  call fruit_summary
  call fruit_finalize

  call is_all_successful(ok)
  if (.not. ok) then
     call exit(1)
  endif

contains

  subroutine test_struct
    use struct_mod

    type(cstruct1) s1

    call set_case_name("test_struct")

    s1 = cstruct1(4, 5)
    call assert_equals(4, pass_struct1(s1), "pass_struct1")

  end subroutine test_struct

end program tester
