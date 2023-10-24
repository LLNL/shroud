! wrapfccomplex.f
! This file is generated by Shroud nowrite-version. Do not edit.
! Copyright (c) 2017-2023, Lawrence Livermore National Security, LLC and
! other Shroud Project Developers.
! See the top-level COPYRIGHT file for details.
!
! SPDX-License-Identifier: (BSD-3-Clause)
!
!>
!! \file wrapfccomplex.f
!! \brief Shroud generated wrapper for ccomplex library
!<
! splicer begin file_top
! splicer end file_top
module ccomplex_mod
    ! splicer begin module_use
    ! splicer end module_use
    implicit none

    ! splicer begin module_top
    ! splicer end module_top

    ! ----------------------------------------
    ! Function:  void acceptFloatComplexInoutPtr
    ! Attrs:     +intent(subroutine)
    ! Statement: f_subroutine
    ! ----------------------------------------
    ! Argument:  float complex * arg1
    ! Attrs:     +intent(inout)
    ! Statement: f_inout_native_*
    interface
        subroutine accept_float_complex_inout_ptr(arg1) &
                bind(C, name="acceptFloatComplexInoutPtr")
            use iso_c_binding, only : C_FLOAT_COMPLEX
            implicit none
            complex(C_FLOAT_COMPLEX), intent(INOUT) :: arg1
        end subroutine accept_float_complex_inout_ptr
    end interface

    ! ----------------------------------------
    ! Function:  void acceptDoubleComplexInoutPtr
    ! Attrs:     +intent(subroutine)
    ! Statement: f_subroutine
    ! ----------------------------------------
    ! Argument:  double complex * arg1
    ! Attrs:     +intent(inout)
    ! Statement: f_inout_native_*
    interface
        subroutine accept_double_complex_inout_ptr(arg1) &
                bind(C, name="acceptDoubleComplexInoutPtr")
            use iso_c_binding, only : C_DOUBLE_COMPLEX
            implicit none
            complex(C_DOUBLE_COMPLEX), intent(INOUT) :: arg1
        end subroutine accept_double_complex_inout_ptr
    end interface

    ! ----------------------------------------
    ! Function:  void acceptDoubleComplexOutPtr
    ! Attrs:     +intent(subroutine)
    ! Statement: f_subroutine
    ! ----------------------------------------
    ! Argument:  double complex * arg1 +intent(out)
    ! Attrs:     +intent(out)
    ! Statement: f_out_native_*
    interface
        subroutine accept_double_complex_out_ptr(arg1) &
                bind(C, name="acceptDoubleComplexOutPtr")
            use iso_c_binding, only : C_DOUBLE_COMPLEX
            implicit none
            complex(C_DOUBLE_COMPLEX), intent(OUT) :: arg1
        end subroutine accept_double_complex_out_ptr
    end interface

    ! ----------------------------------------
    ! Function:  void acceptDoubleComplexOutPtrFlag
    ! Attrs:     +intent(subroutine)
    ! Statement: f_subroutine
    ! ----------------------------------------
    ! Argument:  double complex * arg1 +intent(out)
    ! Attrs:     +intent(out)
    ! Statement: f_out_native_*
    ! ----------------------------------------
    ! Argument:  int * flag +intent(out)
    ! Attrs:     +intent(out)
    ! Statement: f_out_native_*
    interface
        subroutine accept_double_complex_out_ptr_flag(arg1, flag) &
                bind(C, name="acceptDoubleComplexOutPtrFlag")
            use iso_c_binding, only : C_DOUBLE_COMPLEX, C_INT
            implicit none
            complex(C_DOUBLE_COMPLEX), intent(OUT) :: arg1
            integer(C_INT), intent(OUT) :: flag
        end subroutine accept_double_complex_out_ptr_flag
    end interface

    ! splicer begin additional_declarations
    ! splicer end additional_declarations

contains

#if 0
    ! Only the interface is needed
    ! ----------------------------------------
    ! Function:  void acceptFloatComplexInoutPtr
    ! Attrs:     +intent(subroutine)
    ! Statement: f_subroutine
    ! ----------------------------------------
    ! Argument:  float complex * arg1
    ! Attrs:     +intent(inout)
    ! Statement: f_inout_native_*
    subroutine accept_float_complex_inout_ptr(arg1)
        use iso_c_binding, only : C_FLOAT_COMPLEX
        complex(C_FLOAT_COMPLEX), intent(INOUT) :: arg1
        ! splicer begin function.accept_float_complex_inout_ptr
        call c_accept_float_complex_inout_ptr(arg1)
        ! splicer end function.accept_float_complex_inout_ptr
    end subroutine accept_float_complex_inout_ptr
#endif

#if 0
    ! Only the interface is needed
    ! ----------------------------------------
    ! Function:  void acceptDoubleComplexInoutPtr
    ! Attrs:     +intent(subroutine)
    ! Statement: f_subroutine
    ! ----------------------------------------
    ! Argument:  double complex * arg1
    ! Attrs:     +intent(inout)
    ! Statement: f_inout_native_*
    subroutine accept_double_complex_inout_ptr(arg1)
        use iso_c_binding, only : C_DOUBLE_COMPLEX
        complex(C_DOUBLE_COMPLEX), intent(INOUT) :: arg1
        ! splicer begin function.accept_double_complex_inout_ptr
        call c_accept_double_complex_inout_ptr(arg1)
        ! splicer end function.accept_double_complex_inout_ptr
    end subroutine accept_double_complex_inout_ptr
#endif

#if 0
    ! Only the interface is needed
    ! ----------------------------------------
    ! Function:  void acceptDoubleComplexOutPtr
    ! Attrs:     +intent(subroutine)
    ! Statement: f_subroutine
    ! ----------------------------------------
    ! Argument:  double complex * arg1 +intent(out)
    ! Attrs:     +intent(out)
    ! Statement: f_out_native_*
    subroutine accept_double_complex_out_ptr(arg1)
        use iso_c_binding, only : C_DOUBLE_COMPLEX
        complex(C_DOUBLE_COMPLEX), intent(OUT) :: arg1
        ! splicer begin function.accept_double_complex_out_ptr
        call c_accept_double_complex_out_ptr(arg1)
        ! splicer end function.accept_double_complex_out_ptr
    end subroutine accept_double_complex_out_ptr
#endif

#if 0
    ! Only the interface is needed
    ! ----------------------------------------
    ! Function:  void acceptDoubleComplexOutPtrFlag
    ! Attrs:     +intent(subroutine)
    ! Statement: f_subroutine
    ! ----------------------------------------
    ! Argument:  double complex * arg1 +intent(out)
    ! Attrs:     +intent(out)
    ! Statement: f_out_native_*
    ! ----------------------------------------
    ! Argument:  int * flag +intent(out)
    ! Attrs:     +intent(out)
    ! Statement: f_out_native_*
    !>
    !! Return two values so Py_BuildValue is used.
    !! Creates a Py_complex for intent(out)
    !<
    subroutine accept_double_complex_out_ptr_flag(arg1, flag)
        use iso_c_binding, only : C_DOUBLE_COMPLEX, C_INT
        complex(C_DOUBLE_COMPLEX), intent(OUT) :: arg1
        integer(C_INT), intent(OUT) :: flag
        ! splicer begin function.accept_double_complex_out_ptr_flag
        call c_accept_double_complex_out_ptr_flag(arg1, flag)
        ! splicer end function.accept_double_complex_out_ptr_flag
    end subroutine accept_double_complex_out_ptr_flag
#endif

    ! splicer begin additional_functions
    ! splicer end additional_functions

end module ccomplex_mod
