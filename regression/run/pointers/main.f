! Copyright (c) 2017-2021, Lawrence Livermore National Security, LLC and
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
  use pointers_mod
  implicit none
  real(C_DOUBLE), parameter :: pi = 3.1415926_C_DOUBLE
  logical ok

  call init_fruit

  call test_functions
  call test_functions2
  call test_swig
  call test_char_arrays
  call test_out_ptrs
  call test_nested_ptrs
  call test_dimension

  call fruit_summary
  call fruit_finalize

  call is_all_successful(ok)
  if (.not. ok) then
     call exit(1)
  endif

contains

  subroutine test_functions
    integer(c_int) iargin, iarginout, iargout
    real(c_double) :: in_double(5)
    real(c_double), allocatable :: out_double(:)
    integer(c_int), allocatable :: out_int(:)
    integer(c_int) :: nvalues, values1(3), values2(3)

    call set_case_name("test_functions")

    ! set global_int.
    call intargs_in(5)

    iargout = 0
    call intargs_out(iargout)
    call assert_equals(5, iargout) ! get global_int
    
    iarginout = 6
    call intargs_inout(iarginout)  ! set global_int
    call intargs_out(iargout)      ! get global_int
    call assert_equals(6, iargout)
    call assert_equals(7, iarginout) ! incremented iarginout
    
    iargin    = 1
    iarginout = 2
    iargout   = -1
    call intargs(iargin, iarginout, iargout)
    call assert_true(iarginout == 1)
    call assert_true(iargout   == 2)

    call assert_false(allocated(out_double))
    in_double = [0.0*pi, 0.5*pi, pi, 1.5*pi, 2.0*pi]
    call cos_doubles(in_double, out_double)
    call assert_true(allocated(out_double))
    call assert_true(all(abs(out_double - cos(in_double)) < 1.e-08 ))

    call assert_false(allocated(out_int))
    call truncate_to_int([1.2d0, 2.3d0, 3.4d0, 4.5d0], out_int)
    call assert_true(allocated(out_int))
    call assert_true(all(out_int == [1, 2, 3, 4]))
    deallocate(out_int)

    values1 = 0
    call get_values(nvalues, values1)
    call assert_equals(3, nvalues)
    call assert_true(all(values1(1:3) == [1, 2, 3]))

    values1 = 0
    values2 = 0
    call get_values2(values1, values2)
    call assert_true(all(values1(1:3) == [1, 2, 3]))
    call assert_true(all(values2(1:3) == [11, 12, 13]))

    call assert_false(allocated(out_int), "iot_allocatable")
    call iota_allocatable(nvalues, out_int)
    call assert_equals(3, nvalues, "iota_allocatable")
    call assert_true(allocated(out_int), "iota_allocatable")
    call assert_true(all(out_int(1:3) == [1, 2, 3]), "iota_allocatable")
    deallocate(out_int)

    values1 = 0
    call iota_dimension(nvalues, values1)
    call assert_equals(3, nvalues)
    call assert_true(all(values1(1:3) == [1, 2, 3]))

  end subroutine test_functions

  subroutine test_functions2
    integer(c_int) rv_int, out(3), values(5)

    call set_case_name("test_functions2")

    call sum([1,2,3,4,5], rv_int)
    call assert_true(rv_int .eq. 15, "sum")

    out = 0
    call fill_int_array(out)
    call assert_true(all(out(1:3) == [1, 2, 3]), "fillIntArray")

    values = [1, 2, 3, 4, 5]
    call increment_int_array(values)
    call assert_true(all(values(1:5) == [2, 3, 4, 5, 6]), "incrementIntArray")

  end subroutine test_functions2

  subroutine test_swig

    real(C_DOUBLE) zero(10)
    integer(C_INT) sum, count(5)

    call fill_with_zeros(zero)

    count = [1, 2, 3, 4, 5]
    sum = accumulate(count)
    call assert_equals(15, sum)
    
  end subroutine test_swig
  
  subroutine test_char_arrays
    integer nchar
    character(10) :: in(3) = [ &
         "dog       ", &
         "cat       ", &
         "monkey    "  &
         ]
    character(10), target :: word1, word2, word3
    type(C_PTR)  cin(4)

    call set_case_name("test_char_arrays")

    ! Call the bufferify function.
    ! It will copy strings to create char ** variable.
    nchar = accept_char_array_in(in)
    call assert_equals(len_trim(in(1)), nchar, "acceptCharArrayIn")

    ! Build up a native char ** variable and pass to C.
    ! Caller is responsibile for explicilty NULL terminating.
    word1 = "word1" // C_NULL_CHAR
    word2 = "word2+" // C_NULL_CHAR
    word3 = "word3long" // C_NULL_CHAR
    cin(1) = c_loc(word1)
    cin(2) = c_loc(word2)
    cin(3) = c_loc(word3)
    cin(4) = C_NULL_PTR
    nchar = c_accept_char_array_in(cin)
    call assert_equals(5, nchar, "acceptCharArrayIn") ! 5 = len(word1) - trailing NULL
    
  end subroutine test_char_arrays

  subroutine test_out_ptrs
    integer(C_INT) :: ivalue
    integer(C_INT), target :: ivalue1, ivalue2
    integer(C_INT), pointer :: iscalar, irvscalar
    integer(C_INT), pointer :: iarray(:), irvarray(:)
    type(C_PTR) :: cptr_scalar, cptr_array
    type(C_PTR) :: void
    type(C_PTR) :: cptr_arrays(2)

    call set_global_int(0)
    call get_ptr_to_scalar(iscalar)
    call assert_equals(0, iscalar)

    ! iscalar points to global_int in pointers.c.
    call set_global_int(5)
    call assert_equals(5, iscalar)

    nullify(iarray)
    call get_ptr_to_fixed_array(iarray)
    call assert_equals(10, size(iarray))
    iarray = 0
    call assert_equals(0, sum_fixed_array())
    ! Make sure we're assigning to global_array.
    iarray(1) = 1
    iarray(10) = 2
    call assert_equals(3, sum_fixed_array())

    ! Returns global_array in pointers.c.
    nullify(iarray)
    call get_ptr_to_dynamic_array(iarray)
    call assert_true(associated(iarray))
    call assert_true(size(iarray) == 10)

    ! Returns global_array in pointers.c.
    nullify(iarray)
    call get_ptr_to_func_array(iarray)
    call assert_true(associated(iarray))
    call assert_true(size(iarray) == 10)

    call get_raw_ptr_to_scalar(cptr_scalar)
    call assert_true(c_associated(cptr_scalar), "getRawPtrToScalar")
    ! associated with global_int in pointers.c
    call assert_true(c_associated(cptr_scalar, c_loc(iscalar)), "getRawPtrToScalar")

    call get_raw_ptr_to_scalar_force(cptr_scalar)
    call assert_true(c_associated(cptr_scalar), "getRawPtrToScalarForce")
    ! associated with global_int in pointers.c
    call assert_true(c_associated(cptr_scalar, c_loc(iscalar)), "getRawPtrToScalarForce")

    call get_raw_ptr_to_fixed_array(cptr_array)
    call assert_true(c_associated(cptr_array), "getRawPtrToFixedArray")
    ! associated with global_fixed_array in pointers.c
    call assert_true(c_associated(cptr_array, c_loc(iarray)), "getRawPtrToFixedArray")

    call get_raw_ptr_to_fixed_array_force(cptr_array)
    call assert_true(c_associated(cptr_array), "getRawPtrToFixedArrayForce")
    ! associated with global_fixed_array in pointers.c
    call assert_true(c_associated(cptr_array, c_loc(iarray)), "getRawPtrToFixedArrayForce")

    ! Return pointer to global_int as a type(C_PTR).
    ! via interface
    void = C_NULL_PTR
    void = return_address1(1)
    call assert_true(c_associated(void, cptr_scalar))
    ! via wrapper
    void = C_NULL_PTR
    void = return_address2(1)
    call assert_true(c_associated(void, cptr_scalar))
    ! via argument
    void = C_NULL_PTR
    call fetch_void_ptr(void)
    call assert_true(c_associated(void, cptr_scalar))

    ! Pass array of pointers  (void **)
    ivalue1 = 10
    ivalue2 = 4
    cptr_arrays(1) = c_loc(ivalue1)
    cptr_arrays(2) = c_loc(ivalue2)
    call assert_equals(14, void_ptr_array(cptr_arrays))

    ! ***** Non-const results
    ! Return pointer to global_int as a fortran pointer.
    nullify(irvscalar)
    irvscalar => return_int_ptr_to_scalar()
    call assert_true(associated(irvscalar, iscalar))
    call set_global_int(7)
    call assert_equals(7, irvscalar)

    ! ivalue is not a pointer.
    call set_global_int(8)
    ivalue = return_int_scalar()
    call assert_equals(8, ivalue)

    ! Return pointer to global_fixed_int as a fortran pointer.
    nullify(irvarray)
    irvarray => return_int_ptr_to_fixed_array()
    call assert_true(associated(irvarray))
    call assert_true(size(irvarray) == 10)
    call assert_true(associated(irvscalar, iscalar))

    ! ***** const results
    ! Return pointer to global_int as a fortran pointer.
    nullify(irvscalar)
    irvscalar => return_int_ptr_to_const_scalar()
    call assert_true(associated(irvscalar, iscalar))

    ! Return pointer to global_fixed_int as a fortran pointer.
    nullify(irvarray)
    irvarray => return_int_ptr_to_fixed_const_array()
    call assert_true(associated(irvarray))
    call assert_true(size(irvarray) == 10)
    call assert_true(associated(irvscalar, iscalar))

    ! +deref(scalar)
    ivalue = return_int_scalar()

    ! Return pointer to global_int.
    void = return_int_raw()
    call assert_true(c_associated(void, c_loc(irvscalar)))
    call assert_true(c_associated(void, c_loc(iscalar)))

    void = return_int_raw_with_args("with args")
    call assert_true(c_associated(void, c_loc(irvscalar)))
    call assert_true(c_associated(void, c_loc(iscalar)))
    
  end subroutine test_out_ptrs

  subroutine test_nested_ptrs
    type(C_PTR) addr, rvaddr
    type(C_PTR), pointer :: array2d(:)
    integer(C_INT), pointer :: row1(:), row2(:)
    integer total
    
    addr = C_NULL_PTR
    call get_raw_ptr_to_int2d(addr)
    call assert_equals(15, check_int2d(addr), "getRawPtrToInt2d")

    call c_f_pointer(addr, array2d, [2])
    call c_f_pointer(array2d(1), row1, [3])
    call c_f_pointer(array2d(2), row2, [2])

    total = row1(1) + row1(2) + row1(3) + row2(1) + row2(2)
    call assert_equals(15, total)

    ! function result
    rvaddr = return_raw_ptr_to_int2d()
    call assert_true(c_associated(rvaddr, addr), "returnRawPtrToInt2d")

  end subroutine test_nested_ptrs

  subroutine test_dimension
    ! Test +dimension(10,20) +intent(in)  together.
!    integer(C_INT) arg(2,3)
    integer(C_INT) arg2(20,30)

    ! gcc Warning: Actual argument contains too few elements for dummy argument 'arg'
    ! intel error #7983: The storage extent of the dummy argument exceeds that of the actual argument.   [ARG]

!    call dimension_in(arg)

    ! compilers seem ok with too much space
    call dimension_in(arg2)

  end subroutine test_dimension
  
end program tester
