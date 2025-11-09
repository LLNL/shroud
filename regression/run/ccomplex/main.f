! Copyright Shroud Project Developers. See LICENSE file for details.
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
    integer(C_INT) flag
    complex(C_FLOAT_COMPLEX) c4
    complex(C_DOUBLE_COMPLEX) c8

    ! intent(INOUT) argument
    c4 = (1.0, 2.0)
    call accept_float_complex_inout_ptr(c4)
    call assert_equals(3.0, real(c4), "acceptFloatComplexInoutPtr")
    call assert_equals(4.0, imag(c4), "acceptFloatComplexInoutPtr")

    ! intent(INOUT) argument
    c8 = (1.0d0, 2.0d0)
    call accept_double_complex_inout_ptr(c8)
    call assert_equals(3.0d0, real(c8), "acceptDoubleComplexInoutPtr")
    call assert_equals(4.0d0, imag(c8), "acceptDoubleComplexInoutPtr")

    call accept_double_complex_out_ptr(c8)
    call assert_equals(3.0d0, real(c8), "acceptDoubleComplexOutPtr")
    call assert_equals(4.0d0, imag(c8), "acceptDoubleComplexOutPtr")

    call accept_double_complex_out_ptr_flag(c8, flag)
    call assert_equals(3.0d0, real(c8), "acceptDoubleComplexOutPtr")
    call assert_equals(4.0d0, imag(c8), "acceptDoubleComplexOutPtr")
    call assert_equals(0, flag, "acceptDoubleComplexOutPtr")
    
  end subroutine test_complex

end program tester
