! wrapfscope_ns1.f
! This file is generated by Shroud nowrite-version. Do not edit.
! Copyright (c) 2017-2023, Lawrence Livermore National Security, LLC and
! other Shroud Project Developers.
! See the top-level COPYRIGHT file for details.
!
! SPDX-License-Identifier: (BSD-3-Clause)
!
!>
!! \file wrapfscope_ns1.f
!! \brief Shroud generated wrapper for ns1 namespace
!<
! splicer begin namespace.ns1.file_top
! splicer end namespace.ns1.file_top
module scope_ns1_mod
    use iso_c_binding, only : C_INT, C_LONG, C_NULL_PTR, C_PTR, C_SIZE_T
    ! splicer begin namespace.ns1.module_use
    ! splicer end namespace.ns1.module_use
    implicit none

    ! splicer begin namespace.ns1.module_top
    ! splicer end namespace.ns1.module_top

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

    ! helper array_context
    type, bind(C) :: SCO_SHROUD_array
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
    end type SCO_SHROUD_array

    !  enum ns1::Color
    integer(C_INT), parameter :: red = 20
    integer(C_INT), parameter :: blue = 21
    integer(C_INT), parameter :: white = 22


    type, bind(C) :: data_pointer
        integer(C_INT) :: nitems
        type(C_PTR) :: items
    end type data_pointer

    interface

        ! Generated by getter/setter
        ! ----------------------------------------
        ! Function:  int * DataPointer_get_items +dimension(nitems)
        ! Statement: f_getter_native_*_cdesc_pointer
        ! ----------------------------------------
        ! Argument:  ns1::DataPointer * SH_this +intent(in)
        ! Statement: f_in_struct_*
        subroutine c_data_pointer_get_items(SH_this, SHT_rv_cdesc) &
                bind(C, name="SCO_ns1_DataPointer_get_items")
            import :: SCO_SHROUD_array, data_pointer
            implicit none
            type(data_pointer), intent(IN) :: SH_this
            type(SCO_SHROUD_array), intent(OUT) :: SHT_rv_cdesc
        end subroutine c_data_pointer_get_items

        ! Generated by getter/setter
        ! ----------------------------------------
        ! Function:  void DataPointer_set_items
        ! Statement: f_setter
        ! ----------------------------------------
        ! Argument:  ns1::DataPointer * SH_this
        ! Statement: f_inout_struct_*
        ! ----------------------------------------
        ! Argument:  int * val +intent(in)+rank(1)
        ! Statement: f_setter_native_*
        subroutine data_pointer_set_items(SH_this, val) &
                bind(C, name="SCO_ns1_DataPointer_set_items")
            use iso_c_binding, only : C_INT
            import :: data_pointer
            implicit none
            type(data_pointer), intent(INOUT) :: SH_this
            integer(C_INT), intent(IN) :: val(*)
        end subroutine data_pointer_set_items
    end interface

    ! splicer begin namespace.ns1.additional_declarations
    ! splicer end namespace.ns1.additional_declarations

contains

    ! Generated by getter/setter
    ! ----------------------------------------
    ! Function:  int * DataPointer_get_items +dimension(nitems)
    ! Statement: f_getter_native_*_cdesc_pointer
    ! ----------------------------------------
    ! Argument:  ns1::DataPointer * SH_this +intent(in)
    ! Statement: f_in_struct_*
    function data_pointer_get_items(SH_this) &
            result(SHT_rv)
        use iso_c_binding, only : C_INT, c_f_pointer
        type(data_pointer), intent(IN) :: SH_this
        integer(C_INT), pointer :: SHT_rv(:)
        ! splicer begin namespace.ns1.function.data_pointer_get_items
        type(SCO_SHROUD_array) :: SHT_rv_cdesc
        call c_data_pointer_get_items(SH_this, SHT_rv_cdesc)
        call c_f_pointer(SHT_rv_cdesc%base_addr, SHT_rv, &
            SHT_rv_cdesc%shape(1:1))
        ! splicer end namespace.ns1.function.data_pointer_get_items
    end function data_pointer_get_items

#if 0
    ! Only the interface is needed
    ! Generated by getter/setter
    ! ----------------------------------------
    ! Function:  void DataPointer_set_items
    ! Statement: f_setter
    ! ----------------------------------------
    ! Argument:  ns1::DataPointer * SH_this
    ! Statement: f_inout_struct_*
    ! ----------------------------------------
    ! Argument:  int * val +intent(in)+rank(1)
    ! Statement: f_setter_native_*
    subroutine data_pointer_set_items(SH_this, val)
        use iso_c_binding, only : C_INT
        type(data_pointer), intent(INOUT) :: SH_this
        integer(C_INT), intent(IN) :: val(:)
        ! splicer begin namespace.ns1.function.data_pointer_set_items
        call c_data_pointer_set_items(SH_this, val)
        ! splicer end namespace.ns1.function.data_pointer_set_items
    end subroutine data_pointer_set_items
#endif

    ! splicer begin namespace.ns1.additional_functions
    ! splicer end namespace.ns1.additional_functions

end module scope_ns1_mod
