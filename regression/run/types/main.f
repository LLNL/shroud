! Copyright Shroud Project Developers. See LICENSE file for details.
!
! SPDX-License-Identifier: (BSD-3-Clause)
! #######################################################################
!
! Test Fortran API generated from types.yaml.
!
program tester
  use fruit
  use iso_c_binding
  use types_mod
  implicit none
  logical ok

  call init_fruit

  call test_native_types
  call test_unsigned_native_types
  call test_intsize_types
  call test_stddef
  call test_bool

  call fruit_summary
  call fruit_finalize

  call is_all_successful(ok)
  if (.not. ok) then
     call exit(1)
  endif

contains

  subroutine test_native_types
    integer(C_SHORT) rv_short
    integer(C_INT) rv_int
    integer(C_LONG) rv_long
    integer(C_LONG_LONG) rv_long_long

    call set_case_name("test_native_types")

    rv_short = -1_C_SHORT
    rv_short = short_func(1_C_SHORT)
    call assert_true(rv_short .eq. 1_C_SHORT, "short_func")

    rv_int = -1_C_INT
    rv_int = int_func(1_C_INT)
    call assert_true(rv_int .eq. 1_C_INT, "int_func")

    rv_long = -1_C_LONG
    rv_long = long_func(1_C_LONG)
    call assert_true(rv_long .eq. 1_C_LONG, "long_func")

    rv_long_long = -1_C_LONG_LONG
    rv_long_long = long_long_func(1_C_LONG_LONG)
    call assert_true(rv_long_long .eq. 1_C_LONG_LONG, "long_long_func")

    ! explicit int
    rv_short = -1_C_SHORT
    rv_short = short_int_func(1_C_SHORT)
    call assert_true(rv_short .eq. 1_C_SHORT, "short_int_func")

    rv_long = -1_C_LONG
    rv_long = long_int_func(1_C_LONG)
    call assert_true(rv_long .eq. 1_C_LONG, "long_int_func")

    rv_long_long = -1_C_LONG_LONG
    rv_long_long = long_long_int_func(1_C_LONG_LONG)
    call assert_true(rv_long_long .eq. 1_C_LONG_LONG, "long_long_int_func")

  end subroutine test_native_types

  subroutine test_unsigned_native_types
    integer(C_SHORT) rv_short
    integer(C_INT) rv_int
    integer(C_LONG) rv_long
    integer(C_LONG_LONG) rv_long_long

    call set_case_name("test_native_types")

    rv_int = -1_C_INT
    rv_int = unsigned_func(1_C_INT)
    call assert_true(rv_int .eq. 1_C_INT, "unsigned_func")

    rv_short = -1_C_SHORT
    rv_short = ushort_func(1_C_SHORT)
    call assert_true(rv_short .eq. 1_C_SHORT, "ushort_func")

    rv_int = -1_C_INT
    rv_int = uint_func(1_C_INT)
    call assert_true(rv_int .eq. 1_C_INT, "uint_func")

    rv_long = -1_C_LONG
    rv_long = ulong_func(1_C_LONG)
    call assert_true(rv_long .eq. 1_C_LONG, "ulong_func")

    rv_long_long = -1_C_LONG_LONG
    rv_long_long = ulong_long_func(1_C_LONG_LONG)
    call assert_true(rv_long_long .eq. 1_C_LONG_LONG, "ulong_long_func")

    ! implied int
    rv_long = -1_C_LONG
    rv_long = ulong_int_func(1_C_LONG)
    call assert_true(rv_long .eq. 1_C_LONG, "ulong_int_func")

    ! test negative number, C treats as large unsigned number.
    rv_int = -1_C_INT
    rv_int = uint_func(rv_int)
    call assert_true(rv_int .eq. -1_C_INT, "uint_func")

  end subroutine test_unsigned_native_types

  subroutine test_intsize_types
    integer(C_INT8_T) rv_int8
    integer(C_INT16_T) rv_int16
    integer(C_INT32_T) rv_int32
    integer(C_INT64_T) rv_int64

    call set_case_name("test_intsize_types")

    rv_int8 = -1_C_INT8_T
    rv_int8 = int8_func(1_C_INT8_T)
    call assert_true(rv_int8 .eq. 1_C_INT8_T, "int8_func")

    rv_int16 = -1_C_INT16_T
    rv_int16 = int16_func(1_C_INT16_T)
    call assert_true(rv_int16 .eq. 1_C_INT16_T, "int16_func")

    rv_int32 = -1_C_INT32_T
    rv_int32 = int32_func(1_C_INT32_T)
    call assert_true(rv_int32 .eq. 1_C_INT32_T, "int32_func")

    rv_int64 = -1_C_INT64_T
    rv_int64 = int64_func(1_C_INT64_T)
    call assert_true(rv_int64 .eq. 1_C_INT64_T, "int64_func")

    ! unsigned
    rv_int8 = -1_C_INT8_T
    rv_int8 = uint8_func(1_C_INT8_T)
    call assert_true(rv_int8 .eq. 1_C_INT8_T, "int8_func")

    rv_int16 = -1_C_INT16_T
    rv_int16 = uint16_func(1_C_INT16_T)
    call assert_true(rv_int16 .eq. 1_C_INT16_T, "int16_func")

    rv_int32 = -1_C_INT32_T
    rv_int32 = uint32_func(1_C_INT32_T)
    call assert_true(rv_int32 .eq. 1_C_INT32_T, "int32_func")

    rv_int64 = -1_C_INT64_T
    rv_int64 = uint64_func(1_C_INT64_T)
    call assert_true(rv_int64 .eq. 1_C_INT64_T, "int64_func")

  end subroutine test_intsize_types

  subroutine test_stddef
    integer(C_SIZE_T) rv_size

    call set_case_name("test_stddef")

    rv_size = size_func(-1_C_SIZE_T)
    call assert_true(rv_size .eq. -1_C_SIZE_T, "neg_size_func")

    rv_size = size_func(1_C_SIZE_T)
    call assert_true(rv_size .eq. 1_C_SIZE_T, "size_func")

  end subroutine test_stddef

  subroutine test_bool
    logical rv

    call set_case_name("test_bool")

    rv = bool_func(.true.)
    call assert_true(rv, "bool true")

    rv = bool_func(.false.)
    call assert_false(rv, "bool false")

  end subroutine test_bool

end program tester
