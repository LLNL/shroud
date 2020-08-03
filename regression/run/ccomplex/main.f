! Copyright (c) 2017-2020, Lawrence Livermore National Security, LLC and
! other Shroud Project Developers.
! See the top-level COPYRIGHT file for details.
!
! SPDX-License-Identifier: (BSD-3-Clause)
! #######################################################################
!
! Test Fortran API generated from ccomplex.yaml.
!

program tester
  use fruit
  use iso_c_binding
  use ccomplex_mod
  implicit none
  logical ok

  call init_fruit

  call test_complex

  call fruit_summary
  call fruit_finalize

  call is_all_successful(ok)
  if (.not. ok) then
     call exit(1)
  endif

contains

  subroutine test_complex
    complex(C_FLOAT_COMPLEX) c4
    complex(C_DOUBLE_COMPLEX) c8

    ! intent(INOUT) argument
    c4 = (1.0, 2.0)
    call accept_float_complex(c4)
    call assert_equals(3.0, real(c4), "accept_float_complex")
    call assert_equals(4.0, imag(c4), "accept_float_complex")

    ! intent(INOUT) argument
    c8 = (1.0d0, 2.0d0)
    call accept_double_complex(c8)
    call assert_equals(3.0d0, real(c8), "accept_double_complex")
    call assert_equals(4.0d0, imag(c8), "accept_double_complex")
  end subroutine test_complex

end program tester
