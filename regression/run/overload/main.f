! Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and
! other Shroud Project Developers.
! See the top-level COPYRIGHT file for details.
!
! SPDX-License-Identifier: (BSD-3-Clause)
! #######################################################################
!
! Test Fortran API generated from overload.yaml.
!

program tester
  use fruit
  use iso_c_binding
  use overload_mod
  implicit none
  logical ok

  call init_fruit

  call test_overload

  call fruit_summary
  call fruit_finalize

  call is_all_successful(ok)
  if (.not. ok) then
     call exit(1)
  endif

contains

  subroutine test_overload

  end subroutine test_overload

end program tester
