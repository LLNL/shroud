! wrapfvectors.f
! This file is generated by Shroud nowrite-version. Do not edit.
! Copyright (c) 2017-2023, Lawrence Livermore National Security, LLC and
! other Shroud Project Developers.
! See the top-level COPYRIGHT file for details.
!
! SPDX-License-Identifier: (BSD-3-Clause)
!
!>
!! \file wrapfvectors.f
!! \brief Shroud generated wrapper for vectors library
!<
! splicer begin file_top
! splicer end file_top
module vectors_mod
    use iso_c_binding, only : C_INT, C_LONG, C_NULL_PTR, C_PTR, C_SIZE_T
    ! splicer begin module_use
    ! splicer end module_use
    implicit none

    ! splicer begin module_top
    ! splicer end module_top

    ! helper type_defines
    ! Shroud type defines from helper type_defines
    integer, parameter, private :: &
        SH_TYPE_SIGNED_CHAR= 1, &
        SH_TYPE_SHORT      = 2, &
        SH_TYPE_INT        = 3, &
        SH_TYPE_LONG       = 4, &
        SH_TYPE_LONG_LONG  = 5, &
        SH_TYPE_SIZE_T     = 6, &
        SH_TYPE_UNSIGNED_SHORT      = SH_TYPE_SHORT + 100, &
        SH_TYPE_UNSIGNED_INT        = SH_TYPE_INT + 100, &
        SH_TYPE_UNSIGNED_LONG       = SH_TYPE_LONG + 100, &
        SH_TYPE_UNSIGNED_LONG_LONG  = SH_TYPE_LONG_LONG + 100, &
        SH_TYPE_INT8_T    =  7, &
        SH_TYPE_INT16_T   =  8, &
        SH_TYPE_INT32_T   =  9, &
        SH_TYPE_INT64_T   = 10, &
        SH_TYPE_UINT8_T  =  SH_TYPE_INT8_T + 100, &
        SH_TYPE_UINT16_T =  SH_TYPE_INT16_T + 100, &
        SH_TYPE_UINT32_T =  SH_TYPE_INT32_T + 100, &
        SH_TYPE_UINT64_T =  SH_TYPE_INT64_T + 100, &
        SH_TYPE_FLOAT       = 22, &
        SH_TYPE_DOUBLE      = 23, &
        SH_TYPE_LONG_DOUBLE = 24, &
        SH_TYPE_FLOAT_COMPLEX      = 25, &
        SH_TYPE_DOUBLE_COMPLEX     = 26, &
        SH_TYPE_LONG_DOUBLE_COMPLEX= 27, &
        SH_TYPE_BOOL      = 28, &
        SH_TYPE_CHAR      = 29, &
        SH_TYPE_CPTR      = 30, &
        SH_TYPE_STRUCT    = 31, &
        SH_TYPE_OTHER     = 32

    ! start array_context
    ! helper array_context
    type, bind(C) :: VEC_SHROUD_array
        ! address of data
        type(C_PTR) :: base_addr = C_NULL_PTR
        ! type of element
        integer(C_INT) :: type
        ! bytes-per-item or character len of data in cxx
        integer(C_SIZE_T) :: elem_len = 0_C_SIZE_T
        ! size of data in cxx
        integer(C_SIZE_T) :: size = 0_C_SIZE_T
        ! number of dimensions
        integer(C_INT) :: rank = -1
        integer(C_LONG) :: shape(7) = 0
    end type VEC_SHROUD_array
    ! end array_context

    ! start helper capsule_data_helper
    ! helper capsule_data_helper
    type, bind(C) :: VEC_SHROUD_capsule_data
        type(C_PTR) :: addr = C_NULL_PTR  ! address of C++ memory
        integer(C_INT) :: idtor = 0       ! index of destructor
    end type VEC_SHROUD_capsule_data
    ! end helper capsule_data_helper

    ! Generated by arg_to_buffer
    ! ----------------------------------------
    ! Function:  int vector_sum
    ! Statement: f_function_native_scalar
    ! ----------------------------------------
    ! Argument:  const std::vector<int> & arg +rank(1)
    ! Statement: f_in_vector_&_buf_targ_native_scalar
    ! start c_vector_sum_bufferify
    interface
        function c_vector_sum_bufferify(arg, SHT_arg_size) &
                result(SHT_rv) &
                bind(C, name="VEC_vector_sum_bufferify")
            use iso_c_binding, only : C_INT, C_SIZE_T
            implicit none
            integer(C_INT), intent(IN) :: arg(*)
            integer(C_SIZE_T), intent(IN), value :: SHT_arg_size
            integer(C_INT) :: SHT_rv
        end function c_vector_sum_bufferify
    end interface
    ! end c_vector_sum_bufferify

    ! Generated by arg_to_buffer
    ! ----------------------------------------
    ! Function:  void vector_iota_out
    ! Statement: f_subroutine
    ! ----------------------------------------
    ! Argument:  std::vector<int> & arg +intent(out)+rank(1)
    ! Statement: f_out_vector_&_cdesc_targ_native_scalar
    ! start c_vector_iota_out_bufferify
    interface
        subroutine c_vector_iota_out_bufferify(SHT_arg_cdesc) &
                bind(C, name="VEC_vector_iota_out_bufferify")
            import :: VEC_SHROUD_array
            implicit none
            type(VEC_SHROUD_array), intent(OUT) :: SHT_arg_cdesc
        end subroutine c_vector_iota_out_bufferify
    end interface
    ! end c_vector_iota_out_bufferify

    ! Generated by arg_to_buffer
    ! ----------------------------------------
    ! Function:  void vector_iota_out_with_num
    ! Statement: f_subroutine
    ! ----------------------------------------
    ! Argument:  std::vector<int> & arg +intent(out)+rank(1)
    ! Statement: f_out_vector_&_cdesc_targ_native_scalar
    ! start c_vector_iota_out_with_num_bufferify
    interface
        function c_vector_iota_out_with_num_bufferify(SHT_arg_cdesc) &
                result(num) &
                bind(C, name="VEC_vector_iota_out_with_num_bufferify")
            use iso_c_binding, only : C_LONG
            import :: VEC_SHROUD_array
            implicit none
            type(VEC_SHROUD_array), intent(OUT) :: SHT_arg_cdesc
            integer(C_LONG) :: num
        end function c_vector_iota_out_with_num_bufferify
    end interface
    ! end c_vector_iota_out_with_num_bufferify

    ! Generated by arg_to_buffer
    ! ----------------------------------------
    ! Function:  void vector_iota_out_with_num2
    ! Statement: f_subroutine
    ! ----------------------------------------
    ! Argument:  std::vector<int> & arg +intent(out)+rank(1)
    ! Statement: f_out_vector_&_cdesc_targ_native_scalar
    ! start c_vector_iota_out_with_num2_bufferify
    interface
        subroutine c_vector_iota_out_with_num2_bufferify(SHT_arg_cdesc) &
                bind(C, name="VEC_vector_iota_out_with_num2_bufferify")
            import :: VEC_SHROUD_array
            implicit none
            type(VEC_SHROUD_array), intent(OUT) :: SHT_arg_cdesc
        end subroutine c_vector_iota_out_with_num2_bufferify
    end interface
    ! end c_vector_iota_out_with_num2_bufferify

    ! Generated by arg_to_buffer
    ! ----------------------------------------
    ! Function:  void vector_iota_out_alloc
    ! Statement: f_subroutine
    ! ----------------------------------------
    ! Argument:  std::vector<int> & arg +deref(allocatable)+intent(out)+rank(1)
    ! Statement: f_out_vector_&_cdesc_allocatable_targ_native_scalar
    ! start c_vector_iota_out_alloc_bufferify
    interface
        subroutine c_vector_iota_out_alloc_bufferify(SHT_arg_cdesc) &
                bind(C, name="VEC_vector_iota_out_alloc_bufferify")
            import :: VEC_SHROUD_array
            implicit none
            type(VEC_SHROUD_array), intent(OUT) :: SHT_arg_cdesc
        end subroutine c_vector_iota_out_alloc_bufferify
    end interface
    ! end c_vector_iota_out_alloc_bufferify

    ! Generated by arg_to_buffer
    ! ----------------------------------------
    ! Function:  void vector_iota_inout_alloc
    ! Statement: f_subroutine
    ! ----------------------------------------
    ! Argument:  std::vector<int> & arg +deref(allocatable)+intent(inout)+rank(1)
    ! Statement: f_inout_vector_&_cdesc_allocatable_targ_native_scalar
    ! start c_vector_iota_inout_alloc_bufferify
    interface
        subroutine c_vector_iota_inout_alloc_bufferify(arg, &
                SHT_arg_size, SHT_arg_cdesc) &
                bind(C, name="VEC_vector_iota_inout_alloc_bufferify")
            use iso_c_binding, only : C_INT, C_SIZE_T
            import :: VEC_SHROUD_array
            implicit none
            integer(C_INT), intent(IN) :: arg(*)
            integer(C_SIZE_T), intent(IN), value :: SHT_arg_size
            type(VEC_SHROUD_array), intent(OUT) :: SHT_arg_cdesc
        end subroutine c_vector_iota_inout_alloc_bufferify
    end interface
    ! end c_vector_iota_inout_alloc_bufferify

    ! Generated by arg_to_buffer
    ! ----------------------------------------
    ! Function:  void vector_increment
    ! Statement: f_subroutine
    ! ----------------------------------------
    ! Argument:  std::vector<int> & arg +rank(1)
    ! Statement: f_inout_vector_&_cdesc_targ_native_scalar
    interface
        subroutine c_vector_increment_bufferify(arg, SHT_arg_size, &
                SHT_arg_cdesc) &
                bind(C, name="VEC_vector_increment_bufferify")
            use iso_c_binding, only : C_INT, C_SIZE_T
            import :: VEC_SHROUD_array
            implicit none
            integer(C_INT), intent(IN) :: arg(*)
            integer(C_SIZE_T), intent(IN), value :: SHT_arg_size
            type(VEC_SHROUD_array), intent(OUT) :: SHT_arg_cdesc
        end subroutine c_vector_increment_bufferify
    end interface

    ! Generated by arg_to_buffer
    ! ----------------------------------------
    ! Function:  void vector_iota_out_d
    ! Statement: f_subroutine
    ! ----------------------------------------
    ! Argument:  std::vector<double> & arg +intent(out)+rank(1)
    ! Statement: f_out_vector_&_cdesc_targ_native_scalar
    interface
        subroutine c_vector_iota_out_d_bufferify(SHT_arg_cdesc) &
                bind(C, name="VEC_vector_iota_out_d_bufferify")
            import :: VEC_SHROUD_array
            implicit none
            type(VEC_SHROUD_array), intent(OUT) :: SHT_arg_cdesc
        end subroutine c_vector_iota_out_d_bufferify
    end interface

    ! Generated by arg_to_buffer
    ! ----------------------------------------
    ! Function:  int vector_of_pointers
    ! Statement: f_function_native_scalar
    ! ----------------------------------------
    ! Argument:  std::vector<const double * > & arg1 +intent(in)+rank(1)
    ! Statement: f_in_vector_&_buf_targ_native_*
    ! ----------------------------------------
    ! Argument:  int num +value
    ! Statement: f_in_native_scalar
    interface
        function c_vector_of_pointers_bufferify(arg1, SHT_arg1_len, &
                SHT_arg1_size, num) &
                result(SHT_rv) &
                bind(C, name="VEC_vector_of_pointers_bufferify")
            use iso_c_binding, only : C_DOUBLE, C_INT, C_SIZE_T
            implicit none
            real(C_DOUBLE), intent(IN) :: arg1(*)
            integer(C_SIZE_T), intent(IN), value :: SHT_arg1_len
            integer(C_SIZE_T), intent(IN), value :: SHT_arg1_size
            integer(C_INT), value, intent(IN) :: num
            integer(C_INT) :: SHT_rv
        end function c_vector_of_pointers_bufferify
    end interface

    ! Generated by arg_to_buffer
    ! ----------------------------------------
    ! Function:  int vector_string_count
    ! Statement: f_function_native_scalar
    ! ----------------------------------------
    ! Argument:  const std::vector<std::string> & arg +rank(1)
    ! Statement: f_in_vector_&_buf_targ_string_scalar
    interface
        function c_vector_string_count_bufferify(arg, SHT_arg_size, &
                SHT_arg_len) &
                result(SHT_rv) &
                bind(C, name="VEC_vector_string_count_bufferify")
            use iso_c_binding, only : C_CHAR, C_INT, C_SIZE_T
            implicit none
            character(kind=C_CHAR), intent(IN) :: arg(*)
            integer(C_SIZE_T), intent(IN), value :: SHT_arg_size
            integer(C_INT), intent(IN), value :: SHT_arg_len
            integer(C_INT) :: SHT_rv
        end function c_vector_string_count_bufferify
    end interface

    ! Generated by arg_to_buffer
    ! ----------------------------------------
    ! Function:  void vector_string_fill
    ! Statement: f_subroutine
    ! ----------------------------------------
    ! Argument:  std::vector<std::string> & arg +intent(out)+rank(1)
    ! Statement: f_out_vector_&_cdesc_targ_string_scalar
    interface
        subroutine c_vector_string_fill_bufferify(SHT_arg_cdesc) &
                bind(C, name="VEC_vector_string_fill_bufferify")
            import :: VEC_SHROUD_array
            implicit none
            type(VEC_SHROUD_array), intent(OUT) :: SHT_arg_cdesc
        end subroutine c_vector_string_fill_bufferify
    end interface

    ! Generated by arg_to_buffer
    ! ----------------------------------------
    ! Function:  void vector_string_fill_allocatable
    ! Statement: f_subroutine
    ! ----------------------------------------
    ! Argument:  std::vector<std::string> & arg +deref(allocatable)+intent(out)+rank(1)
    ! Statement: f_out_vector_&_cdesc_allocatable_targ_string_scalar
    interface
        subroutine c_vector_string_fill_allocatable_bufferify( &
                SHT_arg_cdesc, SHT_arg_capsule) &
                bind(C, name="VEC_vector_string_fill_allocatable_bufferify")
            import :: VEC_SHROUD_array, VEC_SHROUD_capsule_data
            implicit none
            type(VEC_SHROUD_array), intent(OUT) :: SHT_arg_cdesc
            type(VEC_SHROUD_capsule_data), intent(OUT) :: SHT_arg_capsule
        end subroutine c_vector_string_fill_allocatable_bufferify
    end interface

    ! Generated by arg_to_buffer
    ! ----------------------------------------
    ! Function:  void vector_string_fill_allocatable_len
    ! Statement: f_subroutine
    ! ----------------------------------------
    ! Argument:  std::vector<std::string> & arg +deref(allocatable)+intent(out)+len(20)+rank(1)
    ! Statement: f_out_vector_&_cdesc_allocatable_targ_string_scalar
    interface
        subroutine c_vector_string_fill_allocatable_len_bufferify( &
                SHT_arg_cdesc, SHT_arg_capsule) &
                bind(C, name="VEC_vector_string_fill_allocatable_len_bufferify")
            import :: VEC_SHROUD_array, VEC_SHROUD_capsule_data
            implicit none
            type(VEC_SHROUD_array), intent(OUT) :: SHT_arg_cdesc
            type(VEC_SHROUD_capsule_data), intent(OUT) :: SHT_arg_capsule
        end subroutine c_vector_string_fill_allocatable_len_bufferify
    end interface

#if 0
    ! Not Implemented
    ! ----------------------------------------
    ! Function:  std::vector<int> ReturnVectorAlloc +rank(1)
    ! Statement: c_function_vector_scalar_targ_native_scalar
    ! ----------------------------------------
    ! Argument:  int n +value
    ! Statement: c_in_native_scalar
    interface
        function c_return_vector_alloc(n) &
                result(SHT_rv) &
                bind(C, name="VEC_ReturnVectorAlloc")
            use iso_c_binding, only : C_INT
            implicit none
            integer(C_INT), value, intent(IN) :: n
            integer(C_INT) :: SHT_rv(*)
        end function c_return_vector_alloc
    end interface
#endif

    ! Generated by arg_to_buffer
    ! ----------------------------------------
    ! Function:  std::vector<int> ReturnVectorAlloc +rank(1)
    ! Statement: f_function_vector_scalar_cdesc_allocatable_targ_native_scalar
    ! ----------------------------------------
    ! Argument:  int n +value
    ! Statement: f_in_native_scalar
    interface
        subroutine c_return_vector_alloc_bufferify(n, SHT_rv_cdesc) &
                bind(C, name="VEC_ReturnVectorAlloc_bufferify")
            use iso_c_binding, only : C_INT
            import :: VEC_SHROUD_array
            implicit none
            integer(C_INT), value, intent(IN) :: n
            type(VEC_SHROUD_array), intent(OUT) :: SHT_rv_cdesc
        end subroutine c_return_vector_alloc_bufferify
    end interface

    ! ----------------------------------------
    ! Function:  int returnDim2
    ! Statement: f_function_native_scalar
    ! ----------------------------------------
    ! Argument:  int * arg +intent(in)+rank(2)
    ! Statement: f_in_native_*
    ! ----------------------------------------
    ! Argument:  int len +implied(size(arg,2))+value
    ! Statement: f_in_native_scalar
    interface
        function c_return_dim2(arg, len) &
                result(SHT_rv) &
                bind(C, name="VEC_returnDim2")
            use iso_c_binding, only : C_INT
            implicit none
            integer(C_INT), intent(IN) :: arg(*)
            integer(C_INT), value, intent(IN) :: len
            integer(C_INT) :: SHT_rv
        end function c_return_dim2
    end interface

    interface
        ! helper capsule_dtor
        ! Delete memory in a capsule.
        subroutine VEC_SHROUD_capsule_dtor(ptr) &
            bind(C, name="VEC_SHROUD_memory_destructor")
            import VEC_SHROUD_capsule_data
            implicit none
            type(VEC_SHROUD_capsule_data), intent(INOUT) :: ptr
        end subroutine VEC_SHROUD_capsule_dtor
    end interface

    interface
        ! helper copy_array
        ! Copy contents of context into c_var.
        subroutine VEC_SHROUD_copy_array(context, c_var, c_var_size) &
            bind(C, name="VEC_ShroudCopyArray")
            use iso_c_binding, only : C_PTR, C_SIZE_T
            import VEC_SHROUD_array
            type(VEC_SHROUD_array), intent(IN) :: context
            type(C_PTR), intent(IN), value :: c_var
            integer(C_SIZE_T), value :: c_var_size
        end subroutine VEC_SHROUD_copy_array
    end interface

    interface
        ! helper vector_string_allocatable
        ! Copy the char* or std::string in context into c_var.
        subroutine VEC_SHROUD_vector_string_allocatable(dest, src) &
             bind(c,name="VEC_ShroudVectorStringAllocatable")
            import VEC_SHROUD_capsule_data, VEC_SHROUD_array
            type(VEC_SHROUD_array), intent(IN) :: dest
            type(VEC_SHROUD_capsule_data), intent(IN) :: src
        end subroutine VEC_SHROUD_vector_string_allocatable
    end interface

    ! splicer begin additional_declarations
    ! splicer end additional_declarations

contains

    ! Generated by arg_to_buffer
    ! ----------------------------------------
    ! Function:  int vector_sum
    ! Statement: f_function_native_scalar
    ! ----------------------------------------
    ! Argument:  const std::vector<int> & arg +rank(1)
    ! Statement: f_in_vector_&_buf_targ_native_scalar
    ! start vector_sum
    function vector_sum(arg) &
            result(SHT_rv)
        use iso_c_binding, only : C_INT, C_SIZE_T
        integer(C_INT), intent(IN) :: arg(:)
        integer(C_INT) :: SHT_rv
        ! splicer begin function.vector_sum
        SHT_rv = c_vector_sum_bufferify(arg, size(arg, kind=C_SIZE_T))
        ! splicer end function.vector_sum
    end function vector_sum
    ! end vector_sum

    ! Generated by arg_to_buffer
    ! ----------------------------------------
    ! Function:  void vector_iota_out
    ! Statement: f_subroutine
    ! ----------------------------------------
    ! Argument:  std::vector<int> & arg +intent(out)+rank(1)
    ! Statement: f_out_vector_&_cdesc_targ_native_scalar
    !>
    !! \brief Copy vector into Fortran input array
    !!
    !<
    ! start vector_iota_out
    subroutine vector_iota_out(arg)
        use iso_c_binding, only : C_INT, C_LOC, C_SIZE_T
        integer(C_INT), intent(OUT), target :: arg(:)
        ! splicer begin function.vector_iota_out
        type(VEC_SHROUD_array) :: SHT_arg_cdesc
        call c_vector_iota_out_bufferify(SHT_arg_cdesc)
        call VEC_SHROUD_copy_array(SHT_arg_cdesc, C_LOC(arg), &
            size(arg,kind=C_SIZE_T))
        ! splicer end function.vector_iota_out
    end subroutine vector_iota_out
    ! end vector_iota_out

    ! Generated by arg_to_buffer
    ! ----------------------------------------
    ! Function:  void vector_iota_out_with_num
    ! Statement: f_subroutine
    ! ----------------------------------------
    ! Argument:  std::vector<int> & arg +intent(out)+rank(1)
    ! Statement: f_out_vector_&_cdesc_targ_native_scalar
    !>
    !! \brief Copy vector into Fortran input array
    !!
    !! Convert subroutine in to a function and
    !! return the number of items copied into argument
    !! by setting fstatements for both C and Fortran.
    !<
    ! start vector_iota_out_with_num
    function vector_iota_out_with_num(arg) &
            result(num)
        use iso_c_binding, only : C_INT, C_LOC, C_LONG, C_SIZE_T
        integer(C_INT), intent(OUT), target :: arg(:)
        integer(C_LONG) :: num
        ! splicer begin function.vector_iota_out_with_num
        type(VEC_SHROUD_array) :: SHT_arg_cdesc
        num = c_vector_iota_out_with_num_bufferify(SHT_arg_cdesc)
        call VEC_SHROUD_copy_array(SHT_arg_cdesc, C_LOC(arg), &
            size(arg,kind=C_SIZE_T))
        ! splicer end function.vector_iota_out_with_num
    end function vector_iota_out_with_num
    ! end vector_iota_out_with_num

    ! Generated by arg_to_buffer
    ! ----------------------------------------
    ! Function:  void vector_iota_out_with_num2
    ! Statement: f_subroutine
    ! ----------------------------------------
    ! Argument:  std::vector<int> & arg +intent(out)+rank(1)
    ! Statement: f_out_vector_&_cdesc_targ_native_scalar
    !>
    !! \brief Copy vector into Fortran input array
    !!
    !! Convert subroutine in to a function.
    !! Return the number of items copied into argument
    !! by setting fstatements for the Fortran wrapper only.
    !<
    ! start vector_iota_out_with_num2
    function vector_iota_out_with_num2(arg) &
            result(num)
        use iso_c_binding, only : C_INT, C_LOC, C_LONG, C_SIZE_T
        integer(C_INT), intent(OUT), target :: arg(:)
        integer(C_LONG) :: num
        ! splicer begin function.vector_iota_out_with_num2
        type(VEC_SHROUD_array) :: SHT_arg_cdesc
        call c_vector_iota_out_with_num2_bufferify(SHT_arg_cdesc)
        call VEC_SHROUD_copy_array(SHT_arg_cdesc, C_LOC(arg), &
            size(arg,kind=C_SIZE_T))
        num = SHT_arg_cdesc%size
        ! splicer end function.vector_iota_out_with_num2
    end function vector_iota_out_with_num2
    ! end vector_iota_out_with_num2

    ! Generated by arg_to_buffer
    ! ----------------------------------------
    ! Function:  void vector_iota_out_alloc
    ! Statement: f_subroutine
    ! ----------------------------------------
    ! Argument:  std::vector<int> & arg +deref(allocatable)+intent(out)+rank(1)
    ! Statement: f_out_vector_&_cdesc_allocatable_targ_native_scalar
    !>
    !! \brief Copy vector into Fortran allocatable array
    !!
    !<
    ! start vector_iota_out_alloc
    subroutine vector_iota_out_alloc(arg)
        use iso_c_binding, only : C_INT, C_LOC, C_SIZE_T
        integer(C_INT), intent(OUT), allocatable, target :: arg(:)
        ! splicer begin function.vector_iota_out_alloc
        type(VEC_SHROUD_array) :: SHT_arg_cdesc
        call c_vector_iota_out_alloc_bufferify(SHT_arg_cdesc)
        allocate(arg(SHT_arg_cdesc%size))
        call VEC_SHROUD_copy_array(SHT_arg_cdesc, C_LOC(arg), &
            size(arg,kind=C_SIZE_T))
        ! splicer end function.vector_iota_out_alloc
    end subroutine vector_iota_out_alloc
    ! end vector_iota_out_alloc

    ! Generated by arg_to_buffer
    ! ----------------------------------------
    ! Function:  void vector_iota_inout_alloc
    ! Statement: f_subroutine
    ! ----------------------------------------
    ! Argument:  std::vector<int> & arg +deref(allocatable)+intent(inout)+rank(1)
    ! Statement: f_inout_vector_&_cdesc_allocatable_targ_native_scalar
    !>
    !! \brief Copy vector into Fortran allocatable array
    !!
    !<
    ! start vector_iota_inout_alloc
    subroutine vector_iota_inout_alloc(arg)
        use iso_c_binding, only : C_INT, C_LOC, C_SIZE_T
        integer(C_INT), intent(INOUT), allocatable, target :: arg(:)
        ! splicer begin function.vector_iota_inout_alloc
        type(VEC_SHROUD_array) :: SHT_arg_cdesc
        call c_vector_iota_inout_alloc_bufferify(arg, &
            size(arg, kind=C_SIZE_T), SHT_arg_cdesc)
        if (allocated(arg)) deallocate(arg)
        allocate(arg(SHT_arg_cdesc%size))
        call VEC_SHROUD_copy_array(SHT_arg_cdesc, C_LOC(arg), &
            size(arg,kind=C_SIZE_T))
        ! splicer end function.vector_iota_inout_alloc
    end subroutine vector_iota_inout_alloc
    ! end vector_iota_inout_alloc

    ! Generated by arg_to_buffer
    ! ----------------------------------------
    ! Function:  void vector_increment
    ! Statement: f_subroutine
    ! ----------------------------------------
    ! Argument:  std::vector<int> & arg +rank(1)
    ! Statement: f_inout_vector_&_cdesc_targ_native_scalar
    subroutine vector_increment(arg)
        use iso_c_binding, only : C_INT, C_LOC, C_SIZE_T
        integer(C_INT), intent(INOUT), target :: arg(:)
        ! splicer begin function.vector_increment
        type(VEC_SHROUD_array) :: SHT_arg_cdesc
        call c_vector_increment_bufferify(arg, size(arg, kind=C_SIZE_T), &
            SHT_arg_cdesc)
        call VEC_SHROUD_copy_array(SHT_arg_cdesc, C_LOC(arg), &
            size(arg,kind=C_SIZE_T))
        ! splicer end function.vector_increment
    end subroutine vector_increment

    ! Generated by arg_to_buffer
    ! ----------------------------------------
    ! Function:  void vector_iota_out_d
    ! Statement: f_subroutine
    ! ----------------------------------------
    ! Argument:  std::vector<double> & arg +intent(out)+rank(1)
    ! Statement: f_out_vector_&_cdesc_targ_native_scalar
    !>
    !! \brief Copy vector into Fortran input array
    !!
    !<
    subroutine vector_iota_out_d(arg)
        use iso_c_binding, only : C_DOUBLE, C_LOC, C_SIZE_T
        real(C_DOUBLE), intent(OUT), target :: arg(:)
        ! splicer begin function.vector_iota_out_d
        type(VEC_SHROUD_array) :: SHT_arg_cdesc
        call c_vector_iota_out_d_bufferify(SHT_arg_cdesc)
        call VEC_SHROUD_copy_array(SHT_arg_cdesc, C_LOC(arg), &
            size(arg,kind=C_SIZE_T))
        ! splicer end function.vector_iota_out_d
    end subroutine vector_iota_out_d

    ! Generated by arg_to_buffer
    ! ----------------------------------------
    ! Function:  int vector_of_pointers
    ! Statement: f_function_native_scalar
    ! ----------------------------------------
    ! Argument:  std::vector<const double * > & arg1 +intent(in)+rank(1)
    ! Statement: f_in_vector_&_buf_targ_native_*
    ! ----------------------------------------
    ! Argument:  int num +value
    ! Statement: f_in_native_scalar
    !>
    !! \brief Fortran 2-d array to vector<const double *>
    !!
    !<
    function vector_of_pointers(arg1, num) &
            result(SHT_rv)
        use iso_c_binding, only : C_DOUBLE, C_INT, C_SIZE_T
        real(C_DOUBLE), intent(IN) :: arg1(:,:)
        integer(C_INT), value, intent(IN) :: num
        integer(C_INT) :: SHT_rv
        ! splicer begin function.vector_of_pointers
        SHT_rv = c_vector_of_pointers_bufferify(arg1, &
            size(arg1, 1, kind=C_SIZE_T), size(arg1, 2, kind=C_SIZE_T), &
            num)
        ! splicer end function.vector_of_pointers
    end function vector_of_pointers

    ! Generated by arg_to_buffer
    ! ----------------------------------------
    ! Function:  int vector_string_count
    ! Statement: f_function_native_scalar
    ! ----------------------------------------
    ! Argument:  const std::vector<std::string> & arg +rank(1)
    ! Statement: f_in_vector_&_buf_targ_string_scalar
    !>
    !! \brief count number of underscore in vector of strings
    !!
    !<
    function vector_string_count(arg) &
            result(SHT_rv)
        use iso_c_binding, only : C_INT, C_SIZE_T
        character(len=*), intent(IN) :: arg(:)
        integer(C_INT) :: SHT_rv
        ! splicer begin function.vector_string_count
        SHT_rv = c_vector_string_count_bufferify(arg, &
            size(arg, kind=C_SIZE_T), len(arg, kind=C_INT))
        ! splicer end function.vector_string_count
    end function vector_string_count

    ! Generated by arg_to_buffer
    ! ----------------------------------------
    ! Function:  void vector_string_fill
    ! Statement: f_subroutine
    ! ----------------------------------------
    ! Argument:  std::vector<std::string> & arg +intent(out)+rank(1)
    ! Statement: f_out_vector_&_cdesc_targ_string_scalar
    !>
    !! \brief Fill in arg with some animal names
    !!
    !! The C++ function returns void. But the C and Fortran wrappers return
    !! an int with the number of items added to arg.
    !<
    subroutine vector_string_fill(arg)
        use iso_c_binding, only : C_LOC
        character(*), intent(OUT), target :: arg(:)
        ! splicer begin function.vector_string_fill
        type(VEC_SHROUD_array) :: SHT_arg_cdesc
        SHT_arg_cdesc%base_addr = C_LOC(arg)
        SHT_arg_cdesc%type = SH_TYPE_CHAR
        SHT_arg_cdesc%elem_len = len(arg)
        SHT_arg_cdesc%size = size(arg)
        SHT_arg_cdesc%rank = rank(arg)
        SHT_arg_cdesc%shape(1:1) = shape(arg)
        call c_vector_string_fill_bufferify(SHT_arg_cdesc)
        ! splicer end function.vector_string_fill
    end subroutine vector_string_fill

    ! Generated by arg_to_buffer
    ! ----------------------------------------
    ! Function:  void vector_string_fill_allocatable
    ! Statement: f_subroutine
    ! ----------------------------------------
    ! Argument:  std::vector<std::string> & arg +deref(allocatable)+intent(out)+rank(1)
    ! Statement: f_out_vector_&_cdesc_allocatable_targ_string_scalar
    subroutine vector_string_fill_allocatable(arg)
        use iso_c_binding, only : C_LOC
        character(:), intent(OUT), allocatable, target :: arg(:)
        ! splicer begin function.vector_string_fill_allocatable
        type(VEC_SHROUD_array) :: SHT_arg_cdesc
        type(VEC_SHROUD_capsule_data) :: SHT_arg_capsule
        call c_vector_string_fill_allocatable_bufferify(SHT_arg_cdesc, &
            SHT_arg_capsule)
        allocate(character(len=SHT_arg_cdesc%elem_len) :: &
            arg(SHT_arg_cdesc%size))
        SHT_arg_cdesc%base_addr = C_LOC(arg)
        call VEC_SHROUD_vector_string_allocatable(SHT_arg_cdesc, SHT_arg_capsule)
        call VEC_SHROUD_capsule_dtor(SHT_arg_capsule)
        ! splicer end function.vector_string_fill_allocatable
    end subroutine vector_string_fill_allocatable

    ! Generated by arg_to_buffer
    ! ----------------------------------------
    ! Function:  void vector_string_fill_allocatable_len
    ! Statement: f_subroutine
    ! ----------------------------------------
    ! Argument:  std::vector<std::string> & arg +deref(allocatable)+intent(out)+len(20)+rank(1)
    ! Statement: f_out_vector_&_cdesc_allocatable_targ_string_scalar
    subroutine vector_string_fill_allocatable_len(arg)
        use iso_c_binding, only : C_LOC
        character(len=20), intent(OUT), allocatable, target :: arg(:)
        ! splicer begin function.vector_string_fill_allocatable_len
        type(VEC_SHROUD_array) :: SHT_arg_cdesc
        type(VEC_SHROUD_capsule_data) :: SHT_arg_capsule
        call c_vector_string_fill_allocatable_len_bufferify(SHT_arg_cdesc, &
            SHT_arg_capsule)
        allocate(arg(SHT_arg_cdesc%size))
        SHT_arg_cdesc%base_addr = C_LOC(arg)
        call VEC_SHROUD_vector_string_allocatable(SHT_arg_cdesc, SHT_arg_capsule)
        call VEC_SHROUD_capsule_dtor(SHT_arg_capsule)
        ! splicer end function.vector_string_fill_allocatable_len
    end subroutine vector_string_fill_allocatable_len

    ! Generated by arg_to_buffer
    ! ----------------------------------------
    ! Function:  std::vector<int> ReturnVectorAlloc +rank(1)
    ! Statement: f_function_vector_scalar_cdesc_allocatable_targ_native_scalar
    ! ----------------------------------------
    ! Argument:  int n +value
    ! Statement: f_in_native_scalar
    !>
    !! Implement iota function.
    !! Return a vector as an ALLOCATABLE array.
    !! Copy results into the new array.
    !<
    function return_vector_alloc(n) &
            result(SHT_rv)
        use iso_c_binding, only : C_INT, C_LOC, C_SIZE_T
        integer(C_INT), value, intent(IN) :: n
        integer(C_INT), allocatable, target :: SHT_rv(:)
        ! splicer begin function.return_vector_alloc
        type(VEC_SHROUD_array) :: SHT_rv_cdesc
        call c_return_vector_alloc_bufferify(n, SHT_rv_cdesc)
        allocate(SHT_rv(SHT_rv_cdesc%size))
        call VEC_SHROUD_copy_array(SHT_rv_cdesc, C_LOC(SHT_rv), &
            size(SHT_rv,kind=C_SIZE_T))
        ! splicer end function.return_vector_alloc
    end function return_vector_alloc

    ! ----------------------------------------
    ! Function:  int returnDim2
    ! Statement: f_function_native_scalar
    ! ----------------------------------------
    ! Argument:  int * arg +intent(in)+rank(2)
    ! Statement: f_in_native_*
    function return_dim2(arg) &
            result(SHT_rv)
        use iso_c_binding, only : C_INT
        integer(C_INT), intent(IN) :: arg(:,:)
        integer(C_INT) :: SH_len
        integer(C_INT) :: SHT_rv
        ! splicer begin function.return_dim2
        SH_len = size(arg,2,kind=C_INT)
        SHT_rv = c_return_dim2(arg, SH_len)
        ! splicer end function.return_dim2
    end function return_dim2

    ! splicer begin additional_functions
    ! splicer end additional_functions

end module vectors_mod
