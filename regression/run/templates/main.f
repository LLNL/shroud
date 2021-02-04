! Copyright (c) 2017-2021, Lawrence Livermore National Security, LLC and
! other Shroud Project Developers.
! See the top-level COPYRIGHT file for details.
!
! SPDX-License-Identifier: (BSD-3-Clause)
! #######################################################################
!
! Test Fortran API generated from templates.yaml.
!
program tester
  use fruit
  use iso_c_binding
  use templates_mod
  use templates_internal_mod
  use templates_std_mod
  implicit none
  logical ok

  call init_fruit

  call test_vector_int
  call test_vector_double
  call function_generic
  call function_templates

  call fruit_summary
  call fruit_finalize

  call is_all_successful(ok)
  if (.not. ok) then
     call exit(1)
  endif

contains

  subroutine test_vector_int

    type(vector_int) v1
    integer(C_INT), pointer :: ivalue

    call set_case_name("test_vector_int")

    v1 = vector_int()

    call v1%push_back(1)

    ivalue => v1%at(0_C_SIZE_T)
    call assert_equals(1, ivalue)

! XXX - need to catch std::out_of_range
!    ivalue => v1%at(10_C_SIZE_T)

  end subroutine test_vector_int

  subroutine test_vector_double

    type(vector_double) v1
    real(C_DOUBLE), pointer :: ivalue

    call set_case_name("test_vector_double")

    v1 = vector_double()

    call v1%push_back(1.5_C_DOUBLE)

    ivalue => v1%at(0_C_SIZE_T)
    call assert_equals(1.5_C_DOUBLE, ivalue)

! XXX - need to catch std::out_of_range
!    ivalue => v1%at(10_C_SIZE_T)

  end subroutine test_vector_double

  subroutine function_generic
    call function_tu(1_C_INT, 2_C_LONG)
    call function_tu(1.5_C_FLOAT, 2.5_C_DOUBLE)
  end subroutine function_generic
  
  subroutine function_templates

    integer(C_INT) rv_int
!    type(worker) w1, w2

    call set_case_name("function_templates")

    call function_tu(1_C_INT, 2_C_LONG)
    call function_tu(1.2_C_FLOAT, 2.2_C_DOUBLE)
!    call function_tu(w1, w2)

    rv_int = use_impl_worker_internal_implworker1()
    call assert_equals(1, rv_int)

    rv_int = use_impl_worker_internal_implworker2()
    call assert_equals(2, rv_int)

  end subroutine function_templates

end program tester
