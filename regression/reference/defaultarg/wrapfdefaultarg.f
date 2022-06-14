! wrapfdefaultarg.f
! This file is generated by Shroud nowrite-version. Do not edit.
! Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and
! other Shroud Project Developers.
! See the top-level COPYRIGHT file for details.
!
! SPDX-License-Identifier: (BSD-3-Clause)
!
!>
!! \file wrapfdefaultarg.f
!! \brief Shroud generated wrapper for defaultarg library
!<
! splicer begin file_top
! splicer end file_top
module defaultarg_mod
    use iso_c_binding, only : C_INT32_T, C_INT64_T
    ! splicer begin module_use
    ! splicer end module_use
    implicit none

    ! splicer begin module_top
#if INDEXTYPE_SIZE == 64
    integer, parameter :: INDEXTYPE = C_INT64_T
#else
    integer, parameter :: INDEXTYPE = C_INT32_T
#endif
    ! splicer end module_top

    ! ----------------------------------------
    ! Function:  void apply
    ! Attrs:     +intent(subroutine)
    ! Requested: c_subroutine_void_scalar
    ! Match:     c_subroutine
    ! ----------------------------------------
    ! Argument:  IndexType num_elems +value
    ! Attrs:     +intent(in)
    ! Requested: c_in_native_scalar
    ! Match:     c_default
    interface
        subroutine c_apply_nelems(num_elems) &
                bind(C, name="DEF_apply_nelems")
            import :: INDEXTYPE
            implicit none
            integer(INDEXTYPE), value, intent(IN) :: num_elems
        end subroutine c_apply_nelems
    end interface

    ! ----------------------------------------
    ! Function:  void apply
    ! Attrs:     +intent(subroutine)
    ! Requested: c_subroutine_void_scalar
    ! Match:     c_subroutine
    ! ----------------------------------------
    ! Argument:  IndexType num_elems +value
    ! Attrs:     +intent(in)
    ! Requested: c_in_native_scalar
    ! Match:     c_default
    ! ----------------------------------------
    ! Argument:  IndexType offset=0 +value
    ! Attrs:     +intent(in)
    ! Requested: c_in_native_scalar
    ! Match:     c_default
    interface
        subroutine c_apply_nelems_offset(num_elems, offset) &
                bind(C, name="DEF_apply_nelems_offset")
            import :: INDEXTYPE
            implicit none
            integer(INDEXTYPE), value, intent(IN) :: num_elems
            integer(INDEXTYPE), value, intent(IN) :: offset
        end subroutine c_apply_nelems_offset
    end interface

    ! ----------------------------------------
    ! Function:  void apply
    ! Attrs:     +intent(subroutine)
    ! Requested: c_subroutine_void_scalar
    ! Match:     c_subroutine
    ! ----------------------------------------
    ! Argument:  IndexType num_elems +value
    ! Attrs:     +intent(in)
    ! Requested: c_in_native_scalar
    ! Match:     c_default
    ! ----------------------------------------
    ! Argument:  IndexType offset=0 +value
    ! Attrs:     +intent(in)
    ! Requested: c_in_native_scalar
    ! Match:     c_default
    ! ----------------------------------------
    ! Argument:  IndexType stride=1 +value
    ! Attrs:     +intent(in)
    ! Requested: c_in_native_scalar
    ! Match:     c_default
    interface
        subroutine c_apply_nelems_offset_stride(num_elems, offset, &
                stride) &
                bind(C, name="DEF_apply_nelems_offset_stride")
            import :: INDEXTYPE
            implicit none
            integer(INDEXTYPE), value, intent(IN) :: num_elems
            integer(INDEXTYPE), value, intent(IN) :: offset
            integer(INDEXTYPE), value, intent(IN) :: stride
        end subroutine c_apply_nelems_offset_stride
    end interface

#if INDETYPE_SIZE == 64
    ! ----------------------------------------
    ! Function:  void apply
    ! Attrs:     +intent(subroutine)
    ! Requested: c_subroutine_void_scalar
    ! Match:     c_subroutine
    ! ----------------------------------------
    ! Argument:  TypeID type +value
    ! Attrs:     +intent(in)
    ! Requested: c_in_native_scalar
    ! Match:     c_default
    ! ----------------------------------------
    ! Argument:  IndexType num_elems +value
    ! Attrs:     +intent(in)
    ! Requested: c_in_native_scalar
    ! Match:     c_default
    interface
        subroutine c_apply_type_nelems(type, num_elems) &
                bind(C, name="DEF_apply_type_nelems")
            use iso_c_binding, only : C_INT
            import :: INDEXTYPE
            implicit none
            integer(C_INT), value, intent(IN) :: type
            integer(INDEXTYPE), value, intent(IN) :: num_elems
        end subroutine c_apply_type_nelems
    end interface
#endif

#if INDETYPE_SIZE == 64
    ! ----------------------------------------
    ! Function:  void apply
    ! Attrs:     +intent(subroutine)
    ! Requested: c_subroutine_void_scalar
    ! Match:     c_subroutine
    ! ----------------------------------------
    ! Argument:  TypeID type +value
    ! Attrs:     +intent(in)
    ! Requested: c_in_native_scalar
    ! Match:     c_default
    ! ----------------------------------------
    ! Argument:  IndexType num_elems +value
    ! Attrs:     +intent(in)
    ! Requested: c_in_native_scalar
    ! Match:     c_default
    ! ----------------------------------------
    ! Argument:  IndexType offset=0 +value
    ! Attrs:     +intent(in)
    ! Requested: c_in_native_scalar
    ! Match:     c_default
    interface
        subroutine c_apply_type_nelems_offset(type, num_elems, offset) &
                bind(C, name="DEF_apply_type_nelems_offset")
            use iso_c_binding, only : C_INT
            import :: INDEXTYPE
            implicit none
            integer(C_INT), value, intent(IN) :: type
            integer(INDEXTYPE), value, intent(IN) :: num_elems
            integer(INDEXTYPE), value, intent(IN) :: offset
        end subroutine c_apply_type_nelems_offset
    end interface
#endif

#if INDETYPE_SIZE == 64
    ! ----------------------------------------
    ! Function:  void apply
    ! Attrs:     +intent(subroutine)
    ! Requested: c_subroutine_void_scalar
    ! Match:     c_subroutine
    ! ----------------------------------------
    ! Argument:  TypeID type +value
    ! Attrs:     +intent(in)
    ! Requested: c_in_native_scalar
    ! Match:     c_default
    ! ----------------------------------------
    ! Argument:  IndexType num_elems +value
    ! Attrs:     +intent(in)
    ! Requested: c_in_native_scalar
    ! Match:     c_default
    ! ----------------------------------------
    ! Argument:  IndexType offset=0 +value
    ! Attrs:     +intent(in)
    ! Requested: c_in_native_scalar
    ! Match:     c_default
    ! ----------------------------------------
    ! Argument:  IndexType stride=1 +value
    ! Attrs:     +intent(in)
    ! Requested: c_in_native_scalar
    ! Match:     c_default
    interface
        subroutine c_apply_type_nelems_offset_stride(type, num_elems, &
                offset, stride) &
                bind(C, name="DEF_apply_type_nelems_offset_stride")
            use iso_c_binding, only : C_INT
            import :: INDEXTYPE
            implicit none
            integer(C_INT), value, intent(IN) :: type
            integer(INDEXTYPE), value, intent(IN) :: num_elems
            integer(INDEXTYPE), value, intent(IN) :: offset
            integer(INDEXTYPE), value, intent(IN) :: stride
        end subroutine c_apply_type_nelems_offset_stride
    end interface
#endif

    interface
        ! splicer begin additional_interfaces
        ! splicer end additional_interfaces
    end interface

    interface apply
        module procedure apply_nelems
        module procedure apply_nelems_offset
        module procedure apply_nelems_offset_stride
#if INDETYPE_SIZE == 64
        module procedure apply_type_nelems
#endif
#if INDETYPE_SIZE == 64
        module procedure apply_type_nelems_offset
#endif
#if INDETYPE_SIZE == 64
        module procedure apply_type_nelems_offset_stride
#endif
    end interface apply

contains

    ! Generated by has_default_arg
    ! ----------------------------------------
    ! Function:  void apply
    ! Attrs:     +intent(subroutine)
    ! Exact:     f_subroutine
    ! Attrs:     +intent(subroutine)
    ! Exact:     c_subroutine
    ! ----------------------------------------
    ! Argument:  IndexType num_elems +value
    ! Attrs:     +intent(in)
    ! Requested: f_in_native_scalar
    ! Match:     f_default
    ! Attrs:     +intent(in)
    ! Requested: c_in_native_scalar
    ! Match:     c_default
    subroutine apply_nelems(num_elems)
        integer(INDEXTYPE), value, intent(IN) :: num_elems
        ! splicer begin function.apply_nelems
        call c_apply_nelems(num_elems)
        ! splicer end function.apply_nelems
    end subroutine apply_nelems

    ! Generated by has_default_arg
    ! ----------------------------------------
    ! Function:  void apply
    ! Attrs:     +intent(subroutine)
    ! Exact:     f_subroutine
    ! Attrs:     +intent(subroutine)
    ! Exact:     c_subroutine
    ! ----------------------------------------
    ! Argument:  IndexType num_elems +value
    ! Attrs:     +intent(in)
    ! Requested: f_in_native_scalar
    ! Match:     f_default
    ! Attrs:     +intent(in)
    ! Requested: c_in_native_scalar
    ! Match:     c_default
    ! ----------------------------------------
    ! Argument:  IndexType offset=0 +value
    ! Attrs:     +intent(in)
    ! Requested: f_in_native_scalar
    ! Match:     f_default
    ! Attrs:     +intent(in)
    ! Requested: c_in_native_scalar
    ! Match:     c_default
    subroutine apply_nelems_offset(num_elems, offset)
        integer(INDEXTYPE), value, intent(IN) :: num_elems
        integer(INDEXTYPE), value, intent(IN) :: offset
        ! splicer begin function.apply_nelems_offset
        call c_apply_nelems_offset(num_elems, offset)
        ! splicer end function.apply_nelems_offset
    end subroutine apply_nelems_offset

    ! ----------------------------------------
    ! Function:  void apply
    ! Attrs:     +intent(subroutine)
    ! Exact:     f_subroutine
    ! Attrs:     +intent(subroutine)
    ! Exact:     c_subroutine
    ! ----------------------------------------
    ! Argument:  IndexType num_elems +value
    ! Attrs:     +intent(in)
    ! Requested: f_in_native_scalar
    ! Match:     f_default
    ! Attrs:     +intent(in)
    ! Requested: c_in_native_scalar
    ! Match:     c_default
    ! ----------------------------------------
    ! Argument:  IndexType offset=0 +value
    ! Attrs:     +intent(in)
    ! Requested: f_in_native_scalar
    ! Match:     f_default
    ! Attrs:     +intent(in)
    ! Requested: c_in_native_scalar
    ! Match:     c_default
    ! ----------------------------------------
    ! Argument:  IndexType stride=1 +value
    ! Attrs:     +intent(in)
    ! Requested: f_in_native_scalar
    ! Match:     f_default
    ! Attrs:     +intent(in)
    ! Requested: c_in_native_scalar
    ! Match:     c_default
    subroutine apply_nelems_offset_stride(num_elems, offset, stride)
        integer(INDEXTYPE), value, intent(IN) :: num_elems
        integer(INDEXTYPE), value, intent(IN) :: offset
        integer(INDEXTYPE), value, intent(IN) :: stride
        ! splicer begin function.apply_nelems_offset_stride
        call c_apply_nelems_offset_stride(num_elems, offset, stride)
        ! splicer end function.apply_nelems_offset_stride
    end subroutine apply_nelems_offset_stride

#if INDETYPE_SIZE == 64
    ! Generated by has_default_arg
    ! ----------------------------------------
    ! Function:  void apply
    ! Attrs:     +intent(subroutine)
    ! Exact:     f_subroutine
    ! Attrs:     +intent(subroutine)
    ! Exact:     c_subroutine
    ! ----------------------------------------
    ! Argument:  TypeID type +value
    ! Attrs:     +intent(in)
    ! Requested: f_in_native_scalar
    ! Match:     f_default
    ! Attrs:     +intent(in)
    ! Requested: c_in_native_scalar
    ! Match:     c_default
    ! ----------------------------------------
    ! Argument:  IndexType num_elems +value
    ! Attrs:     +intent(in)
    ! Requested: f_in_native_scalar
    ! Match:     f_default
    ! Attrs:     +intent(in)
    ! Requested: c_in_native_scalar
    ! Match:     c_default
    subroutine apply_type_nelems(type, num_elems)
        use iso_c_binding, only : C_INT
        integer(C_INT), value, intent(IN) :: type
        integer(INDEXTYPE), value, intent(IN) :: num_elems
        ! splicer begin function.apply_type_nelems
        call c_apply_type_nelems(type, num_elems)
        ! splicer end function.apply_type_nelems
    end subroutine apply_type_nelems
#endif

#if INDETYPE_SIZE == 64
    ! Generated by has_default_arg
    ! ----------------------------------------
    ! Function:  void apply
    ! Attrs:     +intent(subroutine)
    ! Exact:     f_subroutine
    ! Attrs:     +intent(subroutine)
    ! Exact:     c_subroutine
    ! ----------------------------------------
    ! Argument:  TypeID type +value
    ! Attrs:     +intent(in)
    ! Requested: f_in_native_scalar
    ! Match:     f_default
    ! Attrs:     +intent(in)
    ! Requested: c_in_native_scalar
    ! Match:     c_default
    ! ----------------------------------------
    ! Argument:  IndexType num_elems +value
    ! Attrs:     +intent(in)
    ! Requested: f_in_native_scalar
    ! Match:     f_default
    ! Attrs:     +intent(in)
    ! Requested: c_in_native_scalar
    ! Match:     c_default
    ! ----------------------------------------
    ! Argument:  IndexType offset=0 +value
    ! Attrs:     +intent(in)
    ! Requested: f_in_native_scalar
    ! Match:     f_default
    ! Attrs:     +intent(in)
    ! Requested: c_in_native_scalar
    ! Match:     c_default
    subroutine apply_type_nelems_offset(type, num_elems, offset)
        use iso_c_binding, only : C_INT
        integer(C_INT), value, intent(IN) :: type
        integer(INDEXTYPE), value, intent(IN) :: num_elems
        integer(INDEXTYPE), value, intent(IN) :: offset
        ! splicer begin function.apply_type_nelems_offset
        call c_apply_type_nelems_offset(type, num_elems, offset)
        ! splicer end function.apply_type_nelems_offset
    end subroutine apply_type_nelems_offset
#endif

#if INDETYPE_SIZE == 64
    ! ----------------------------------------
    ! Function:  void apply
    ! Attrs:     +intent(subroutine)
    ! Exact:     f_subroutine
    ! Attrs:     +intent(subroutine)
    ! Exact:     c_subroutine
    ! ----------------------------------------
    ! Argument:  TypeID type +value
    ! Attrs:     +intent(in)
    ! Requested: f_in_native_scalar
    ! Match:     f_default
    ! Attrs:     +intent(in)
    ! Requested: c_in_native_scalar
    ! Match:     c_default
    ! ----------------------------------------
    ! Argument:  IndexType num_elems +value
    ! Attrs:     +intent(in)
    ! Requested: f_in_native_scalar
    ! Match:     f_default
    ! Attrs:     +intent(in)
    ! Requested: c_in_native_scalar
    ! Match:     c_default
    ! ----------------------------------------
    ! Argument:  IndexType offset=0 +value
    ! Attrs:     +intent(in)
    ! Requested: f_in_native_scalar
    ! Match:     f_default
    ! Attrs:     +intent(in)
    ! Requested: c_in_native_scalar
    ! Match:     c_default
    ! ----------------------------------------
    ! Argument:  IndexType stride=1 +value
    ! Attrs:     +intent(in)
    ! Requested: f_in_native_scalar
    ! Match:     f_default
    ! Attrs:     +intent(in)
    ! Requested: c_in_native_scalar
    ! Match:     c_default
    subroutine apply_type_nelems_offset_stride(type, num_elems, offset, &
            stride)
        use iso_c_binding, only : C_INT
        integer(C_INT), value, intent(IN) :: type
        integer(INDEXTYPE), value, intent(IN) :: num_elems
        integer(INDEXTYPE), value, intent(IN) :: offset
        integer(INDEXTYPE), value, intent(IN) :: stride
        ! splicer begin function.apply_type_nelems_offset_stride
        call c_apply_type_nelems_offset_stride(type, num_elems, offset, &
            stride)
        ! splicer end function.apply_type_nelems_offset_stride
    end subroutine apply_type_nelems_offset_stride
#endif

    ! splicer begin additional_functions
    ! splicer end additional_functions

end module defaultarg_mod
