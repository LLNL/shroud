! Copyright (c) 2017-2023, Lawrence Livermore National Security, LLC and
! other Shroud Project Developers.
! See the top-level COPYRIGHT file for details.
!
! SPDX-License-Identifier: (BSD-3-Clause)
! #######################################################################
!
! Test Fortran API generated from generic.yaml.
!

program tester
  use fruit
  use iso_c_binding
  use generic_mod
  implicit none
  logical ok

  call init_fruit

  call test_generic_group
  call test_functions
  call test_assumed_rank
  call test_scalar_array
  call test_database
  call test_struct

  call fruit_summary
  call fruit_finalize

  call is_all_successful(ok)
  if (.not. ok) then
     call exit(1)
  endif

contains

  ! Create a generic interface for two functions.
  subroutine test_generic_group
    call set_case_name("test_generic_group")

    call update_as_float(12.0_C_FLOAT)
    call assert_equals(12.0d0, get_global_double(), "update_as_float")

    call update_as_double(13.0_C_DOUBLE)
    call assert_equals(13.0d0, get_global_double(), "update_as_double")
    
    call update_real(22.0_C_FLOAT)
    call assert_equals(22.0d0, get_global_double(), "update_as_float")

    call update_real(23.0_C_DOUBLE)
    call assert_equals(23.0d0, get_global_double(), "update_as_double")
    
  end subroutine test_generic_group

  subroutine test_functions
    integer(C_LONG) rv

    call set_case_name("test_functions")

    call generic_real(1.0)
    call assert_equals(1.0d0, get_global_double(), "generic_real real")

    call generic_real(2.0d0)
    call assert_equals(2.0d0, get_global_double(), "generic_real double")

    rv = generic_real2(1_C_INT, 2_C_INT)
    call assert_true(3_C_LONG == rv, "generic_real2 int")

    rv = generic_real2(10_C_LONG, 20_C_LONG)
    call assert_true(30_C_LONG == rv, "generic_real2 long")

  end subroutine test_functions

  subroutine test_assumed_rank
    integer scalar
    integer array(5), array2d(2,3)
    
    call set_case_name("test_assumed_rank")

    scalar = 5
    call assert_equals(5, sum_values(scalar, 1), "sum_values scalar")
    call assert_equals(6, sum_values(6, 1), "sum_values scalar constant")

    array = [1,2,3,4,5]
    call assert_equals(15, sum_values(array, 5), "sum_values 1d")
    call assert_equals(9, sum_values([3, 3, 3], 3), "sum_values 1d constant")

    array2d = reshape([1, 2, 3, 4, 5, 6], shape(array2d))
    call assert_equals(21, sum_values(array2d, size(array2d)), "sum_values 2d")
  end subroutine test_assumed_rank

  subroutine test_scalar_array
    integer sfrom, sto
    integer from(5), to(5)
    
    call set_case_name("test_scalar_array")

    ! assign
    sfrom = 5
    sto = -1
    call assign_values(sfrom, 1, sto, 1)
    call assert_equals(sfrom, sto, "assign_values assign")

    ! broadcast
    to = -1
    call assign_values(5, 1, to, size(to))
    call assert_true( all(to == [5,5,5,5,5]), "assign_values broadcast")

    ! copy
    from = 7
    to = -1
    call assign_values(from, size(from), to, size(to))
    call assert_true( all(to == from), "assign_values copy")
    
  end subroutine test_scalar_array

  subroutine test_database
    real(C_FLOAT) var1(10)
    integer(C_INT)  itype
    integer(C_SIZE_T) isize
    type(C_PTR) fwa

    call set_case_name("test_database")

    call save_pointer(var1)

    fwa = C_NULL_PTR
    call get_pointer(fwa, itype, isize)
    call assert_true(c_associated(fwa), "fwa is not none")
    call assert_true(T_FLOAT == itype, "type of var1")
    call assert_true(size(var1) == isize, "size of var1")

    call save_pointer2(var1)

    fwa = C_NULL_PTR
    call get_pointer(fwa, itype, isize)
    call assert_true(c_associated(fwa), "fwa is not none")
    call assert_true(T_FLOAT == itype, "type of var1")
    call assert_true(size(var1) == isize, "size of var1")

  end subroutine test_database

  subroutine test_struct
    type(struct_as_class) stru1
    integer(C_LONG) ll
    
    call set_case_name("test_struct")

    stru1 = struct_as_class()
    ll = update_struct_as_class(stru1, 10_C_INT)
    call assert_true(ll .eq. 10_C_INT, "update_struct_as_class int")
    ll = update_struct_as_class(stru1, 20_C_LONG)
    call assert_true(ll .eq. 20_C_INT, "update_struct_as_class long")
  end subroutine test_struct

end program tester
