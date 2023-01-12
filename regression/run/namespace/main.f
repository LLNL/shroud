! Copyright (c) 2017-2023, Lawrence Livermore National Security, LLC and
! other Shroud Project Developers.
! See the top-level COPYRIGHT file for details.
!
! SPDX-License-Identifier: (BSD-3-Clause)
! #######################################################################
!
! Test Fortran API generated from namespace.yaml.
!
program tester
  use fruit
  use iso_c_binding
  implicit none
  logical ok

  call init_fruit

  call test_ns
  call test_ns_outer

  call fruit_summary
  call fruit_finalize

  call is_all_successful(ok)
  if (.not. ok) then
     call exit(1)
  endif

contains

  subroutine test_ns
    use ns_mod
    character(:), allocatable :: last

    call set_case_name("test_ns")

    call one
    last = last_function_called()
    call assert_equals("One", last, "One")

  end subroutine test_ns

  subroutine test_ns_outer
    use ns_mod, only : last_function_called
    use ns_outer_mod
    character(:), allocatable :: last

    call set_case_name("test_ns_outer")

    call one
    last = last_function_called()
    call assert_equals("outer::One", last, "outer::One")

  end subroutine test_ns_outer

end program tester
