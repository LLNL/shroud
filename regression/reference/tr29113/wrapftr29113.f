! wrapftr29113.f
! This file is generated by Shroud 0.11.0. Do not edit.
! Copyright (c) 2017-2024, Lawrence Livermore National Security, LLC and
! other Shroud Project Developers.
! See the top-level COPYRIGHT file for details.
!
! SPDX-License-Identifier: (BSD-3-Clause)
!
!>
!! \file wrapftr29113.f
!! \brief Shroud generated wrapper for tr29113 library
!<
! splicer begin file_top
! splicer end file_top
module tr29113_mod
    use iso_c_binding, only : C_INT, C_LONG, C_NULL_PTR, C_PTR, C_SIZE_T
    ! splicer begin module_use
    ! splicer end module_use
    implicit none

    ! splicer begin module_top
    ! splicer end module_top

    ! helper capsule_data_helper
    type, bind(C) :: SHROUD_capsule_data
        type(C_PTR) :: addr = C_NULL_PTR  ! address of C++ memory
        integer(C_INT) :: idtor = 0       ! index of destructor
    end type SHROUD_capsule_data

    ! helper array_context
    type, bind(C) :: SHROUD_array
        ! address of C++ memory
        type(SHROUD_capsule_data) :: cxx
        ! address of data in cxx
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
    end type SHROUD_array

    interface

        ! ----------------------------------------
        ! Function:  const std::string * getConstStringPtrAlloc +deref(allocatable)
        ! Requested: c_string_*_result
        ! Match:     c_string_result
        function c_get_const_string_ptr_alloc() &
                result(SHT_rv) &
                bind(C, name="TR2_get_const_string_ptr_alloc")
            use iso_c_binding, only : C_PTR
            implicit none
            type(C_PTR) SHT_rv
        end function c_get_const_string_ptr_alloc

        ! ----------------------------------------
        ! Function:  void getConstStringPtrAlloc
        ! Requested: c_void_scalar_result_buf
        ! Match:     c_default
        ! ----------------------------------------
        ! Argument:  const std::string * SHF_rv +context(DSHF_rv)+deref(allocatable)+intent(out)
        ! Requested: c_string_*_result_buf_allocatable
        ! Match:     c_string_result_buf_allocatable
        subroutine c_get_const_string_ptr_alloc_bufferify(DSHF_rv) &
                bind(C, name="TR2_get_const_string_ptr_alloc_bufferify")
            import :: SHROUD_array
            implicit none
            type(SHROUD_array), intent(OUT) :: DSHF_rv
        end subroutine c_get_const_string_ptr_alloc_bufferify

        ! splicer begin additional_interfaces

        subroutine c_get_const_string_ptr_alloc_bufferify_tr(SHT_rv) &
             bind(C, name="TR2_get_const_string_ptr_alloc_tr_bufferify")
            implicit none
            character(len=:), allocatable :: SHT_rv
        end subroutine c_get_const_string_ptr_alloc_bufferify_tr

        subroutine get_const_string_ptr_alloc_bufferify_tr_zerolength(SHT_rv) &
             bind(C, name="TR2_get_const_string_ptr_alloc_tr_bufferify_zerolength")
            implicit none
            character(len=:), allocatable :: SHT_rv
        end subroutine get_const_string_ptr_alloc_bufferify_tr_zerolength

        ! splicer end additional_interfaces
    end interface

    interface
        ! helper copy_string
        ! Copy the char* or std::string in context into c_var.
        subroutine SHROUD_copy_string_and_free(context, c_var, c_var_size) &
             bind(c,name="TR2_ShroudCopyStringAndFree")
            use, intrinsic :: iso_c_binding, only : C_CHAR, C_SIZE_T
            import SHROUD_array
            type(SHROUD_array), intent(IN) :: context
            character(kind=C_CHAR), intent(OUT) :: c_var(*)
            integer(C_SIZE_T), value :: c_var_size
        end subroutine SHROUD_copy_string_and_free
    end interface

contains

    ! Generated by arg_to_buffer
    ! ----------------------------------------
    ! Function:  const std::string * getConstStringPtrAlloc +deref(allocatable)
    ! const std::string * getConstStringPtrAlloc +deref(allocatable)
    ! Requested: f_string_scalar_result_allocatable
    ! Match:     f_string_result_allocatable
    ! Function:  void getConstStringPtrAlloc
    ! Exact:     c_string_scalar_result_buf
    ! ----------------------------------------
    ! Argument:  const std::string * SHF_rv +context(DSHF_rv)+deref(allocatable)+intent(out)
    ! Requested: f_string_*_result_allocatable
    ! Match:     f_string_result_allocatable
    ! Requested: c_string_*_result_buf_allocatable
    ! Match:     c_string_result_buf_allocatable
    function get_const_string_ptr_alloc() &
            result(SHT_rv)
        type(SHROUD_array) :: DSHF_rv
        character(len=:), allocatable :: SHT_rv
        ! splicer begin function.get_const_string_ptr_alloc
        call c_get_const_string_ptr_alloc_bufferify(DSHF_rv)
        allocate(character(len=DSHF_rv%elem_len):: SHT_rv)
        call SHROUD_copy_string_and_free(DSHF_rv, SHT_rv, DSHF_rv%elem_len)
        ! splicer end function.get_const_string_ptr_alloc
    end function get_const_string_ptr_alloc

    ! splicer begin additional_functions
    function get_const_string_ptr_alloc_tr() &
            result(SHT_rv)
    !    type(SHROUD_array) :: DSHF_rv
        character(len=:), allocatable :: SHT_rv
        call c_get_const_string_ptr_alloc_bufferify_tr(SHT_rv)
    !    allocate(character(len=DSHF_rv%elem_len):: SHT_rv)
    !    call SHROUD_copy_string_and_free(DSHF_rv, SHT_rv, DSHF_rv%elem_len)
    end function get_const_string_ptr_alloc_tr

    ! splicer end additional_functions

end module tr29113_mod
