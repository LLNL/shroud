! Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and
! other Shroud Project Developers.
! See the top-level COPYRIGHT file for details.
!
! SPDX-License-Identifier: (BSD-3-Clause)
! #######################################################################
!
! Test Fortran API generated from pointers.yaml.
! Used with pointers-c and pointers-cxx.
!
program tester
  use fruit
  use iso_c_binding
  use strings_mod
  logical ok

  call init_fruit

  call test_charargs

  call fruit_summary
  call fruit_finalize

  call is_all_successful(ok)
  if (.not. ok) then
     call exit(1)
  endif

contains

  subroutine test_charargs
    ! test C++ functions

!    character(30) str
    character ch

    call set_case_name("test_charargs")

    call passChar("w")  ! pass_char
    ch = returnChar()  ! returnChar
    call assert_equals("w", ch, "passChar/returnChar")

!    call pass_char_force("x")
!    ch = return_char()
!    call assert_equals("x", ch, "passCharForce/returnChar")

  end subroutine test_charargs
  
end program tester
