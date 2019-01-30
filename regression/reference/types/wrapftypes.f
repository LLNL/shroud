! wrapftypes.f
! This is generated code, do not edit
! #######################################################################
! Copyright (c) 2018-2019, Lawrence Livermore National Security, LLC.
!
! Produced at the Lawrence Livermore National Laboratory
!
! LLNL-CODE-738041.
!
! All rights reserved.
!
! This file is part of Shroud.
!
! For details about use and distribution, please read LICENSE.
!
! #######################################################################
!>
!! \file wrapftypes.f
!! \brief Shroud generated wrapper for types library
!<
! splicer begin file_top
! splicer end file_top
module types_mod
    ! splicer begin module_use
    ! splicer end module_use
    implicit none

    ! splicer begin module_top
    ! splicer end module_top

    interface

        function short_func(arg1) &
                result(SHT_rv) &
                bind(C, name="TYP_short_func")
            use iso_c_binding, only : C_SHORT
            implicit none
            integer(C_SHORT), value, intent(IN) :: arg1
            integer(C_SHORT) :: SHT_rv
        end function short_func

        function int_func(arg1) &
                result(SHT_rv) &
                bind(C, name="TYP_int_func")
            use iso_c_binding, only : C_INT
            implicit none
            integer(C_INT), value, intent(IN) :: arg1
            integer(C_INT) :: SHT_rv
        end function int_func

        function long_func(arg1) &
                result(SHT_rv) &
                bind(C, name="TYP_long_func")
            use iso_c_binding, only : C_LONG
            implicit none
            integer(C_LONG), value, intent(IN) :: arg1
            integer(C_LONG) :: SHT_rv
        end function long_func

        function long_long_func(arg1) &
                result(SHT_rv) &
                bind(C, name="TYP_long_long_func")
            use iso_c_binding, only : C_LONG_LONG
            implicit none
            integer(C_LONG_LONG), value, intent(IN) :: arg1
            integer(C_LONG_LONG) :: SHT_rv
        end function long_long_func

        function short_int_func(arg1) &
                result(SHT_rv) &
                bind(C, name="TYP_short_int_func")
            use iso_c_binding, only : C_SHORT
            implicit none
            integer(C_SHORT), value, intent(IN) :: arg1
            integer(C_SHORT) :: SHT_rv
        end function short_int_func

        function long_int_func(arg1) &
                result(SHT_rv) &
                bind(C, name="TYP_long_int_func")
            use iso_c_binding, only : C_LONG
            implicit none
            integer(C_LONG), value, intent(IN) :: arg1
            integer(C_LONG) :: SHT_rv
        end function long_int_func

        function long_long_int_func(arg1) &
                result(SHT_rv) &
                bind(C, name="TYP_long_long_int_func")
            use iso_c_binding, only : C_LONG_LONG
            implicit none
            integer(C_LONG_LONG), value, intent(IN) :: arg1
            integer(C_LONG_LONG) :: SHT_rv
        end function long_long_int_func

        function unsigned_func(arg1) &
                result(SHT_rv) &
                bind(C, name="TYP_unsigned_func")
            use iso_c_binding, only : C_INT
            implicit none
            integer(C_INT), value, intent(IN) :: arg1
            integer(C_INT) :: SHT_rv
        end function unsigned_func

        function ushort_func(arg1) &
                result(SHT_rv) &
                bind(C, name="TYP_ushort_func")
            use iso_c_binding, only : C_SHORT
            implicit none
            integer(C_SHORT), value, intent(IN) :: arg1
            integer(C_SHORT) :: SHT_rv
        end function ushort_func

        function uint_func(arg1) &
                result(SHT_rv) &
                bind(C, name="TYP_uint_func")
            use iso_c_binding, only : C_INT
            implicit none
            integer(C_INT), value, intent(IN) :: arg1
            integer(C_INT) :: SHT_rv
        end function uint_func

        function ulong_func(arg1) &
                result(SHT_rv) &
                bind(C, name="TYP_ulong_func")
            use iso_c_binding, only : C_LONG
            implicit none
            integer(C_LONG), value, intent(IN) :: arg1
            integer(C_LONG) :: SHT_rv
        end function ulong_func

        function ulong_long_func(arg1) &
                result(SHT_rv) &
                bind(C, name="TYP_ulong_long_func")
            use iso_c_binding, only : C_LONG_LONG
            implicit none
            integer(C_LONG_LONG), value, intent(IN) :: arg1
            integer(C_LONG_LONG) :: SHT_rv
        end function ulong_long_func

        function ulong_int_func(arg1) &
                result(SHT_rv) &
                bind(C, name="TYP_ulong_int_func")
            use iso_c_binding, only : C_LONG
            implicit none
            integer(C_LONG), value, intent(IN) :: arg1
            integer(C_LONG) :: SHT_rv
        end function ulong_int_func

        function int8_func(arg1) &
                result(SHT_rv) &
                bind(C, name="TYP_int8_func")
            use iso_c_binding, only : C_INT8_T
            implicit none
            integer(C_INT8_T), value, intent(IN) :: arg1
            integer(C_INT8_T) :: SHT_rv
        end function int8_func

        function int16_func(arg1) &
                result(SHT_rv) &
                bind(C, name="TYP_int16_func")
            use iso_c_binding, only : C_INT16_T
            implicit none
            integer(C_INT16_T), value, intent(IN) :: arg1
            integer(C_INT16_T) :: SHT_rv
        end function int16_func

        function int32_func(arg1) &
                result(SHT_rv) &
                bind(C, name="TYP_int32_func")
            use iso_c_binding, only : C_INT32_T
            implicit none
            integer(C_INT32_T), value, intent(IN) :: arg1
            integer(C_INT32_T) :: SHT_rv
        end function int32_func

        function int64_func(arg1) &
                result(SHT_rv) &
                bind(C, name="TYP_int64_func")
            use iso_c_binding, only : C_INT64_T
            implicit none
            integer(C_INT64_T), value, intent(IN) :: arg1
            integer(C_INT64_T) :: SHT_rv
        end function int64_func

        function uint8_func(arg1) &
                result(SHT_rv) &
                bind(C, name="TYP_uint8_func")
            use iso_c_binding, only : C_INT8_T
            implicit none
            integer(C_INT8_T), value, intent(IN) :: arg1
            integer(C_INT8_T) :: SHT_rv
        end function uint8_func

        function uint16_func(arg1) &
                result(SHT_rv) &
                bind(C, name="TYP_uint16_func")
            use iso_c_binding, only : C_INT16_T
            implicit none
            integer(C_INT16_T), value, intent(IN) :: arg1
            integer(C_INT16_T) :: SHT_rv
        end function uint16_func

        function uint32_func(arg1) &
                result(SHT_rv) &
                bind(C, name="TYP_uint32_func")
            use iso_c_binding, only : C_INT32_T
            implicit none
            integer(C_INT32_T), value, intent(IN) :: arg1
            integer(C_INT32_T) :: SHT_rv
        end function uint32_func

        function uint64_func(arg1) &
                result(SHT_rv) &
                bind(C, name="TYP_uint64_func")
            use iso_c_binding, only : C_INT64_T
            implicit none
            integer(C_INT64_T), value, intent(IN) :: arg1
            integer(C_INT64_T) :: SHT_rv
        end function uint64_func

        function size_func(arg1) &
                result(SHT_rv) &
                bind(C, name="TYP_size_func")
            use iso_c_binding, only : C_SIZE_T
            implicit none
            integer(C_SIZE_T), value, intent(IN) :: arg1
            integer(C_SIZE_T) :: SHT_rv
        end function size_func

        ! splicer begin additional_interfaces
        ! splicer end additional_interfaces
    end interface

contains

    ! splicer begin additional_functions
    ! splicer end additional_functions

end module types_mod
