! wrapfscope_ns2.f
! This file is generated by Shroud nowrite-version. Do not edit.
! Copyright (c) 2017-2023, Lawrence Livermore National Security, LLC and
! other Shroud Project Developers.
! See the top-level COPYRIGHT file for details.
!
! SPDX-License-Identifier: (BSD-3-Clause)
!
!>
!! \file wrapfscope_ns2.f
!! \brief Shroud generated wrapper for ns2 namespace
!<
! splicer begin namespace.ns2.file_top
! splicer end namespace.ns2.file_top
module scope_ns2_mod
    use iso_c_binding, only : C_INT, C_LONG, C_NULL_PTR, C_PTR, C_SIZE_T
    ! splicer begin namespace.ns2.module_use
    ! splicer end namespace.ns2.module_use
    implicit none

    ! splicer begin namespace.ns2.module_top
    ! splicer end namespace.ns2.module_top

    ! helper capsule_data_helper
    type, bind(C) :: SCO_SHROUD_capsule_data
        type(C_PTR) :: addr = C_NULL_PTR  ! address of C++ memory
        integer(C_INT) :: idtor = 0       ! index of destructor
    end type SCO_SHROUD_capsule_data

    ! helper array_context
    type, bind(C) :: SCO_SHROUD_array
        ! address of C++ memory
        type(SCO_SHROUD_capsule_data) :: cxx
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
    end type SCO_SHROUD_array

    !  enum ns2::Color
    integer(C_INT), parameter :: red = 30
    integer(C_INT), parameter :: blue = 31
    integer(C_INT), parameter :: white = 32


    type, bind(C) :: data_pointer
        integer(C_INT) :: nitems
        type(C_PTR) :: items
    end type data_pointer

    interface

        ! ----------------------------------------
        ! Function:  int * DataPointer_get_items
        ! Attrs:     +api(cdesc)+deref(pointer)+intent(getter)+struct(ns2_DataPointer)
        ! Exact:     c_getter_native_*_cdesc_pointer
        ! ----------------------------------------
        ! Argument:  ns2::DataPointer * SH_this
        ! Attrs:     +intent(in)+struct(ns2_DataPointer)
        ! Exact:     c_in_struct_*
        subroutine c_data_pointer_get_items_bufferify(SH_this, SHT_rv) &
                bind(C, name="SCO_ns2_DataPointer_get_items_bufferify")
            import :: SCO_SHROUD_array, data_pointer
            implicit none
            type(data_pointer), intent(IN) :: SH_this
            type(SCO_SHROUD_array), intent(OUT) :: SHT_rv
        end subroutine c_data_pointer_get_items_bufferify

        ! ----------------------------------------
        ! Function:  void DataPointer_set_items
        ! Attrs:     +intent(setter)
        ! Requested: c_setter_void_scalar
        ! Match:     c_setter
        ! ----------------------------------------
        ! Argument:  ns2::DataPointer * SH_this
        ! Attrs:     +intent(inout)+struct(ns2_DataPointer)
        ! Exact:     c_inout_struct_*
        ! ----------------------------------------
        ! Argument:  int * val +intent(in)+rank(1)
        ! Attrs:     +intent(setter)
        ! Exact:     c_setter_native_*
        subroutine data_pointer_set_items(SH_this, val) &
                bind(C, name="SCO_ns2_DataPointer_set_items")
            use iso_c_binding, only : C_INT
            import :: data_pointer
            implicit none
            type(data_pointer), intent(INOUT) :: SH_this
            integer(C_INT), intent(IN) :: val(*)
        end subroutine data_pointer_set_items
    end interface

    ! splicer begin namespace.ns2.additional_declarations
    ! splicer end namespace.ns2.additional_declarations

contains

    ! Generated by getter/setter - arg_to_buffer
    ! ----------------------------------------
    ! Function:  int * DataPointer_get_items
    ! Attrs:     +deref(pointer)+intent(getter)+struct(ns2_DataPointer)
    ! Exact:     f_getter_native_*_cdesc_pointer
    ! Attrs:     +api(cdesc)+deref(pointer)+intent(getter)+struct(ns2_DataPointer)
    ! Exact:     c_getter_native_*_cdesc_pointer
    ! ----------------------------------------
    ! Argument:  ns2::DataPointer * SH_this
    ! Attrs:     +intent(in)+struct(ns2_DataPointer)
    ! Exact:     f_in_struct_*
    ! Attrs:     +intent(in)+struct(ns2_DataPointer)
    ! Exact:     c_in_struct_*
    function data_pointer_get_items(SH_this) &
            result(SHT_rv)
        use iso_c_binding, only : C_INT, c_f_pointer
        type(data_pointer), intent(IN) :: SH_this
        integer(C_INT), pointer :: SHT_rv(:)
        ! splicer begin namespace.ns2.function.data_pointer_get_items
        type(SCO_SHROUD_array) :: SHT_rv_cdesc
        call c_data_pointer_get_items_bufferify(SH_this, SHT_rv_cdesc)
        call c_f_pointer(SHT_rv_cdesc%base_addr, SHT_rv, &
            SHT_rv_cdesc%shape(1:1))
        ! splicer end namespace.ns2.function.data_pointer_get_items
    end function data_pointer_get_items

#if 0
    ! Only the interface is needed
    ! Generated by getter/setter
    ! ----------------------------------------
    ! Function:  void DataPointer_set_items
    ! Attrs:     +intent(setter)
    ! Exact:     f_setter
    ! Attrs:     +intent(setter)
    ! Exact:     c_setter
    ! ----------------------------------------
    ! Argument:  ns2::DataPointer * SH_this
    ! Attrs:     +intent(inout)+struct(ns2_DataPointer)
    ! Exact:     f_inout_struct_*
    ! Attrs:     +intent(inout)+struct(ns2_DataPointer)
    ! Exact:     c_inout_struct_*
    ! ----------------------------------------
    ! Argument:  int * val +intent(in)+rank(1)
    ! Attrs:     +intent(setter)
    ! Exact:     f_setter_native_*
    ! Attrs:     +intent(setter)
    ! Exact:     c_setter_native_*
    subroutine data_pointer_set_items(SH_this, val)
        use iso_c_binding, only : C_INT
        type(data_pointer), intent(INOUT) :: SH_this
        integer(C_INT), intent(IN) :: val(:)
        ! splicer begin namespace.ns2.function.data_pointer_set_items
        call c_data_pointer_set_items(SH_this, val)
        ! splicer end namespace.ns2.function.data_pointer_set_items
    end subroutine data_pointer_set_items
#endif

    ! splicer begin namespace.ns2.additional_functions
    ! splicer end namespace.ns2.additional_functions

end module scope_ns2_mod
