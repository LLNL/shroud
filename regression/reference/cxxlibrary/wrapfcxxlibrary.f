! wrapfcxxlibrary.f
! This file is generated by Shroud nowrite-version. Do not edit.
! Copyright (c) 2017-2021, Lawrence Livermore National Security, LLC and
! other Shroud Project Developers.
! See the top-level COPYRIGHT file for details.
!
! SPDX-License-Identifier: (BSD-3-Clause)
!
!>
!! \file wrapfcxxlibrary.f
!! \brief Shroud generated wrapper for cxxlibrary library
!<
! splicer begin file_top
! splicer end file_top
module cxxlibrary_mod
    ! splicer begin module_use
    ! splicer end module_use
    implicit none

    ! splicer begin module_top
    ! splicer end module_top

    interface

        ! ----------------------------------------
        ! Function:  bool defaultPtrIsNULL
        ! Attrs:     +intent(function)
        ! Requested: c_function_bool_scalar
        ! Match:     c_function
        function c_default_ptr_is_null_0() &
                result(SHT_rv) &
                bind(C, name="CXX_default_ptr_is_null_0")
            use iso_c_binding, only : C_BOOL
            implicit none
            logical(C_BOOL) :: SHT_rv
        end function c_default_ptr_is_null_0

        ! ----------------------------------------
        ! Function:  bool defaultPtrIsNULL
        ! Attrs:     +intent(function)
        ! Requested: c_function_bool_scalar
        ! Match:     c_function
        ! ----------------------------------------
        ! Argument:  double * data=nullptr +intent(IN)+rank(1)
        ! Attrs:     +intent(in)
        ! Requested: c_in_native_*
        ! Match:     c_default
        function c_default_ptr_is_null_1(data) &
                result(SHT_rv) &
                bind(C, name="CXX_default_ptr_is_null_1")
            use iso_c_binding, only : C_BOOL, C_DOUBLE
            implicit none
            real(C_DOUBLE), intent(IN) :: data(*)
            logical(C_BOOL) :: SHT_rv
        end function c_default_ptr_is_null_1

        ! ----------------------------------------
        ! Function:  void defaultArgsInOut
        ! Attrs:     +intent(subroutine)
        ! Requested: c_subroutine_void_scalar
        ! Match:     c_subroutine
        ! ----------------------------------------
        ! Argument:  int in1 +value
        ! Attrs:     +intent(in)
        ! Requested: c_in_native_scalar
        ! Match:     c_default
        ! ----------------------------------------
        ! Argument:  int * out1 +intent(out)
        ! Attrs:     +intent(out)
        ! Requested: c_out_native_*
        ! Match:     c_default
        ! ----------------------------------------
        ! Argument:  int * out2 +intent(out)
        ! Attrs:     +intent(out)
        ! Requested: c_out_native_*
        ! Match:     c_default
        subroutine c_default_args_in_out_0(in1, out1, out2) &
                bind(C, name="CXX_default_args_in_out_0")
            use iso_c_binding, only : C_INT
            implicit none
            integer(C_INT), value, intent(IN) :: in1
            integer(C_INT), intent(OUT) :: out1
            integer(C_INT), intent(OUT) :: out2
        end subroutine c_default_args_in_out_0

        ! ----------------------------------------
        ! Function:  void defaultArgsInOut
        ! Attrs:     +intent(subroutine)
        ! Requested: c_subroutine_void_scalar
        ! Match:     c_subroutine
        ! ----------------------------------------
        ! Argument:  int in1 +value
        ! Attrs:     +intent(in)
        ! Requested: c_in_native_scalar
        ! Match:     c_default
        ! ----------------------------------------
        ! Argument:  int * out1 +intent(out)
        ! Attrs:     +intent(out)
        ! Requested: c_out_native_*
        ! Match:     c_default
        ! ----------------------------------------
        ! Argument:  int * out2 +intent(out)
        ! Attrs:     +intent(out)
        ! Requested: c_out_native_*
        ! Match:     c_default
        ! ----------------------------------------
        ! Argument:  bool flag=false +value
        ! Attrs:     +intent(in)
        ! Requested: c_in_bool_scalar
        ! Match:     c_default
        subroutine c_default_args_in_out_1(in1, out1, out2, flag) &
                bind(C, name="CXX_default_args_in_out_1")
            use iso_c_binding, only : C_BOOL, C_INT
            implicit none
            integer(C_INT), value, intent(IN) :: in1
            integer(C_INT), intent(OUT) :: out1
            integer(C_INT), intent(OUT) :: out2
            logical(C_BOOL), value, intent(IN) :: flag
        end subroutine c_default_args_in_out_1

        ! splicer begin additional_interfaces
        ! splicer end additional_interfaces
    end interface

    interface default_args_in_out
        module procedure default_args_in_out_0
        module procedure default_args_in_out_1
    end interface default_args_in_out

    interface default_ptr_is_null
        module procedure default_ptr_is_null_0
        module procedure default_ptr_is_null_1
    end interface default_ptr_is_null

contains

    ! Generated by has_default_arg
    ! ----------------------------------------
    ! Function:  bool defaultPtrIsNULL
    ! Attrs:     +intent(function)
    ! Requested: f_function_bool_scalar
    ! Match:     f_function_bool
    ! Attrs:     +intent(function)
    ! Requested: c_function_bool_scalar
    ! Match:     c_function
    function default_ptr_is_null_0() &
            result(SHT_rv)
        use iso_c_binding, only : C_BOOL
        logical :: SHT_rv
        ! splicer begin function.default_ptr_is_null_0
        SHT_rv = c_default_ptr_is_null_0()
        ! splicer end function.default_ptr_is_null_0
    end function default_ptr_is_null_0

    ! ----------------------------------------
    ! Function:  bool defaultPtrIsNULL
    ! Attrs:     +intent(function)
    ! Requested: f_function_bool_scalar
    ! Match:     f_function_bool
    ! Attrs:     +intent(function)
    ! Requested: c_function_bool_scalar
    ! Match:     c_function
    ! ----------------------------------------
    ! Argument:  double * data=nullptr +intent(IN)+rank(1)
    ! Attrs:     +intent(in)
    ! Requested: f_in_native_*
    ! Match:     f_default
    ! Attrs:     +intent(in)
    ! Requested: c_in_native_*
    ! Match:     c_default
    function default_ptr_is_null_1(data) &
            result(SHT_rv)
        use iso_c_binding, only : C_BOOL, C_DOUBLE
        real(C_DOUBLE), intent(IN) :: data(:)
        logical :: SHT_rv
        ! splicer begin function.default_ptr_is_null_1
        SHT_rv = c_default_ptr_is_null_1(data)
        ! splicer end function.default_ptr_is_null_1
    end function default_ptr_is_null_1

    ! Generated by has_default_arg
    ! ----------------------------------------
    ! Function:  void defaultArgsInOut
    ! Attrs:     +intent(subroutine)
    ! Exact:     f_subroutine
    ! Attrs:     +intent(subroutine)
    ! Exact:     c_subroutine
    ! ----------------------------------------
    ! Argument:  int in1 +value
    ! Attrs:     +intent(in)
    ! Requested: f_in_native_scalar
    ! Match:     f_default
    ! Attrs:     +intent(in)
    ! Requested: c_in_native_scalar
    ! Match:     c_default
    ! ----------------------------------------
    ! Argument:  int * out1 +intent(out)
    ! Attrs:     +intent(out)
    ! Requested: f_out_native_*
    ! Match:     f_default
    ! Attrs:     +intent(out)
    ! Requested: c_out_native_*
    ! Match:     c_default
    ! ----------------------------------------
    ! Argument:  int * out2 +intent(out)
    ! Attrs:     +intent(out)
    ! Requested: f_out_native_*
    ! Match:     f_default
    ! Attrs:     +intent(out)
    ! Requested: c_out_native_*
    ! Match:     c_default
    subroutine default_args_in_out_0(in1, out1, out2)
        use iso_c_binding, only : C_INT
        integer(C_INT), value, intent(IN) :: in1
        integer(C_INT), intent(OUT) :: out1
        integer(C_INT), intent(OUT) :: out2
        ! splicer begin function.default_args_in_out_0
        call c_default_args_in_out_0(in1, out1, out2)
        ! splicer end function.default_args_in_out_0
    end subroutine default_args_in_out_0

    ! ----------------------------------------
    ! Function:  void defaultArgsInOut
    ! Attrs:     +intent(subroutine)
    ! Exact:     f_subroutine
    ! Attrs:     +intent(subroutine)
    ! Exact:     c_subroutine
    ! ----------------------------------------
    ! Argument:  int in1 +value
    ! Attrs:     +intent(in)
    ! Requested: f_in_native_scalar
    ! Match:     f_default
    ! Attrs:     +intent(in)
    ! Requested: c_in_native_scalar
    ! Match:     c_default
    ! ----------------------------------------
    ! Argument:  int * out1 +intent(out)
    ! Attrs:     +intent(out)
    ! Requested: f_out_native_*
    ! Match:     f_default
    ! Attrs:     +intent(out)
    ! Requested: c_out_native_*
    ! Match:     c_default
    ! ----------------------------------------
    ! Argument:  int * out2 +intent(out)
    ! Attrs:     +intent(out)
    ! Requested: f_out_native_*
    ! Match:     f_default
    ! Attrs:     +intent(out)
    ! Requested: c_out_native_*
    ! Match:     c_default
    ! ----------------------------------------
    ! Argument:  bool flag=false +value
    ! Attrs:     +intent(in)
    ! Requested: f_in_bool_scalar
    ! Match:     f_in_bool
    ! Attrs:     +intent(in)
    ! Requested: c_in_bool_scalar
    ! Match:     c_default
    subroutine default_args_in_out_1(in1, out1, out2, flag)
        use iso_c_binding, only : C_BOOL, C_INT
        integer(C_INT), value, intent(IN) :: in1
        integer(C_INT), intent(OUT) :: out1
        integer(C_INT), intent(OUT) :: out2
        logical, value, intent(IN) :: flag
        ! splicer begin function.default_args_in_out_1
        logical(C_BOOL) SH_flag
        SH_flag = flag  ! coerce to C_BOOL
        call c_default_args_in_out_1(in1, out1, out2, SH_flag)
        ! splicer end function.default_args_in_out_1
    end subroutine default_args_in_out_1

    ! splicer begin additional_functions
    ! splicer end additional_functions

end module cxxlibrary_mod
