! Copyright (c) 2017-2023, Lawrence Livermore National Security, LLC and
! other Shroud Project Developers.
! See the top-level COPYRIGHT file for details.
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

  call fruit_summary
  call fruit_finalize

  call is_all_successful(ok)
  if (.not. ok) then
     call exit(1)
  endif

contains

  subroutine test_defaultarg

  end subroutine test_defaultarg

end program tester
