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

        function c_bool_func(arg) &
                result(SHT_rv) &
                bind(C, name="TYP_bool_func")
            use iso_c_binding, only : C_BOOL
            implicit none
            logical(C_BOOL), value, intent(IN) :: arg
            logical(C_BOOL) :: SHT_rv
        end function c_bool_func

        function c_return_bool_and_others(flag) &
                result(SHT_rv) &
                bind(C, name="TYP_return_bool_and_others")
            use iso_c_binding, only : C_BOOL, C_INT
            implicit none
            integer(C_INT), intent(OUT) :: flag
            logical(C_BOOL) :: SHT_rv
        end function c_return_bool_and_others

        ! splicer begin additional_interfaces
        ! splicer end additional_interfaces
    end interface

contains

    ! bool bool_func(bool arg +intent(in)+value)
    function bool_func(arg) &
            result(SHT_rv)
        use iso_c_binding, only : C_BOOL
        logical, value, intent(IN) :: arg
        logical(C_BOOL) SH_arg
        logical :: SHT_rv
        SH_arg = arg  ! coerce to C_BOOL
        ! splicer begin function.bool_func
        SHT_rv = c_bool_func(SH_arg)
        ! splicer end function.bool_func
    end function bool_func

    ! bool returnBoolAndOthers(int * flag +intent(out))
    !>
    !! \brief Function which returns bool with other intent(out) arguments
    !!
    !! Python treats bool differently since Py_BuildValue does not support
    !! bool until Python 3.3.
    !! Must create a PyObject with PyBool_FromLong then include that object
    !! in call to Py_BuildValue as type 'O'.  But since two return values
    !! are being created, function return and argument flag, rename first
    !! local C variable to avoid duplicate names in wrapper.
    !<
    function return_bool_and_others(flag) &
            result(SHT_rv)
        use iso_c_binding, only : C_BOOL, C_INT
        integer(C_INT), intent(OUT) :: flag
        logical :: SHT_rv
        ! splicer begin function.return_bool_and_others
        SHT_rv = c_return_bool_and_others(flag)
        ! splicer end function.return_bool_and_others
    end function return_bool_and_others

    ! splicer begin additional_functions
    ! splicer end additional_functions

end module types_mod
