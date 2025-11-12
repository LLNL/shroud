! Copyright Shroud Project Developers. See LICENSE file for details.
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
  call user_class
  call struct_templates

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

    call v1%dtor
    
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

    call v1%dtor

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

  subroutine user_class
    type(user_int) uvar1

    uvar1 = return_user_type()

    call uvar1%nested_double(12, 45.5d0)
    
  end subroutine user_class

  subroutine struct_templates

    type(struct_as_class_int) s_int
    type(struct_as_class_double) s_double

    call set_case_name("struct_templates")

    ! int
    s_int = struct_as_class_int()

    call s_int%set_npts(5_C_INT)
    call s_int%set_value(2_C_INT)

    call assert_equals(5_C_INT, s_int%get_npts())
    call assert_equals(2_C_INT, s_int%get_value())

    ! double
    s_double = struct_as_class_double()

    call s_double%set_npts(5_C_INT)
    call s_double%set_value(2.5_C_DOUBLE)

    call assert_equals(5_C_INT, s_double%get_npts())
    call assert_equals(2.5_C_DOUBLE, s_double%get_value())

  end subroutine struct_templates

end program tester
