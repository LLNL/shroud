! wrapferror.f
! This file is generated by Shroud nowrite-version. Do not edit.
! Copyright (c) 2017-2023, Lawrence Livermore National Security, LLC and
! other Shroud Project Developers.
! See the top-level COPYRIGHT file for details.
!
! SPDX-License-Identifier: (BSD-3-Clause)
!
!>
!! \file wrapferror.f
!! \brief Shroud generated wrapper for error library
!<
! splicer begin file_top
! splicer end file_top
module error_mod
    use iso_c_binding, only : C_INT, C_NULL_PTR, C_PTR
    ! splicer begin module_use
    ! splicer end module_use
    implicit none

    ! splicer begin module_top
    ! splicer end module_top

    ! helper capsule_data_helper
    type, bind(C) :: ERR_SHROUD_capsule_data
        type(C_PTR) :: addr = C_NULL_PTR  ! address of C++ memory
        integer(C_INT) :: idtor = 0       ! index of destructor
    end type ERR_SHROUD_capsule_data

    type, extends(===>F_derived_member_base<===) :: cstruct_as_subclass
        ! splicer begin class.Cstruct_as_subclass.component_part
        ! splicer end class.Cstruct_as_subclass.component_part
    contains
        procedure :: get_x1 => cstruct_as_subclass_get_x1
        procedure :: set_x1 => cstruct_as_subclass_set_x1
        procedure :: get_y1 => cstruct_as_subclass_get_y1
        procedure :: set_y1 => cstruct_as_subclass_set_y1
        procedure :: get_z1 => cstruct_as_subclass_get_z1
        procedure :: set_z1 => cstruct_as_subclass_set_z1
        procedure :: get_instance => cstruct_as_subclass_get_instance
        procedure :: set_instance => cstruct_as_subclass_set_instance
        procedure :: associated => cstruct_as_subclass_associated
        ! splicer begin class.Cstruct_as_subclass.type_bound_procedure_part
        ! splicer end class.Cstruct_as_subclass.type_bound_procedure_part
    end type cstruct_as_subclass

    interface operator (.eq.)
        module procedure cstruct_as_subclass_eq
    end interface

    interface operator (.ne.)
        module procedure cstruct_as_subclass_ne
    end interface

    interface

        ! Generated by getter/setter
        ! ----------------------------------------
        ! Function:  int get_x1
        ! Attrs:     +intent(getter)
        ! Statement: f_getter_native_scalar
        function c_cstruct_as_subclass_get_x1(self) &
                result(SHT_rv) &
                bind(C, name="ERR_Cstruct_as_subclass_get_x1")
            use iso_c_binding, only : C_INT
            import :: ERR_SHROUD_capsule_data
            implicit none
            type(ERR_SHROUD_capsule_data), intent(IN) :: self
            integer(C_INT) :: SHT_rv
        end function c_cstruct_as_subclass_get_x1

        ! Generated by getter/setter
        ! ----------------------------------------
        ! Function:  void set_x1
        ! Attrs:     +intent(setter)
        ! Statement: f_setter
        ! ----------------------------------------
        ! Argument:  int val +intent(in)+value
        ! Attrs:     +intent(setter)
        ! Statement: f_setter_native_scalar
        subroutine c_cstruct_as_subclass_set_x1(self, val) &
                bind(C, name="ERR_Cstruct_as_subclass_set_x1")
            use iso_c_binding, only : C_INT
            import :: ERR_SHROUD_capsule_data
            implicit none
            type(ERR_SHROUD_capsule_data), intent(IN) :: self
            integer(C_INT), value, intent(IN) :: val
        end subroutine c_cstruct_as_subclass_set_x1

        ! Generated by getter/setter
        ! ----------------------------------------
        ! Function:  int get_y1
        ! Attrs:     +intent(getter)
        ! Statement: f_getter_native_scalar
        function c_cstruct_as_subclass_get_y1(self) &
                result(SHT_rv) &
                bind(C, name="ERR_Cstruct_as_subclass_get_y1")
            use iso_c_binding, only : C_INT
            import :: ERR_SHROUD_capsule_data
            implicit none
            type(ERR_SHROUD_capsule_data), intent(IN) :: self
            integer(C_INT) :: SHT_rv
        end function c_cstruct_as_subclass_get_y1

        ! Generated by getter/setter
        ! ----------------------------------------
        ! Function:  void set_y1
        ! Attrs:     +intent(setter)
        ! Statement: f_setter
        ! ----------------------------------------
        ! Argument:  int val +intent(in)+value
        ! Attrs:     +intent(setter)
        ! Statement: f_setter_native_scalar
        subroutine c_cstruct_as_subclass_set_y1(self, val) &
                bind(C, name="ERR_Cstruct_as_subclass_set_y1")
            use iso_c_binding, only : C_INT
            import :: ERR_SHROUD_capsule_data
            implicit none
            type(ERR_SHROUD_capsule_data), intent(IN) :: self
            integer(C_INT), value, intent(IN) :: val
        end subroutine c_cstruct_as_subclass_set_y1

        ! Generated by getter/setter
        ! ----------------------------------------
        ! Function:  int get_z1
        ! Attrs:     +intent(getter)
        ! Statement: f_getter_native_scalar
        function c_cstruct_as_subclass_get_z1(self) &
                result(SHT_rv) &
                bind(C, name="ERR_Cstruct_as_subclass_get_z1")
            use iso_c_binding, only : C_INT
            import :: ERR_SHROUD_capsule_data
            implicit none
            type(ERR_SHROUD_capsule_data), intent(IN) :: self
            integer(C_INT) :: SHT_rv
        end function c_cstruct_as_subclass_get_z1

        ! Generated by getter/setter
        ! ----------------------------------------
        ! Function:  void set_z1
        ! Attrs:     +intent(setter)
        ! Statement: f_setter
        ! ----------------------------------------
        ! Argument:  int val +intent(in)+value
        ! Attrs:     +intent(setter)
        ! Statement: f_setter_native_scalar
        subroutine c_cstruct_as_subclass_set_z1(self, val) &
                bind(C, name="ERR_Cstruct_as_subclass_set_z1")
            use iso_c_binding, only : C_INT
            import :: ERR_SHROUD_capsule_data
            implicit none
            type(ERR_SHROUD_capsule_data), intent(IN) :: self
            integer(C_INT), value, intent(IN) :: val
        end subroutine c_cstruct_as_subclass_set_z1

        ! ----------------------------------------
        ! Function:  void BadFstatements
        ! Attrs:     +intent(subroutine)
        ! Statement: f_subroutine
        function c_bad_fstatements() &
                result(SHT_rv) &
                bind(C, name="ERR_BadFstatements")
            implicit none
        end function c_bad_fstatements

        ! ----------------------------------------
        ! Function:  void AssumedRank
        ! Attrs:     +intent(subroutine)
        ! Statement: f_subroutine
        ! ----------------------------------------
        ! Argument:  int * data
        ! Attrs:     +intent(inout)
        ! Statement: f_inout_native_*
        subroutine c_assumed_rank(data) &
                bind(C, name="ERR_AssumedRank")
            use iso_c_binding, only : C_INT
            implicit none
            integer(C_INT), intent(INOUT) :: data
        end subroutine c_assumed_rank

        ! Generated by fortran_generic
        ! ----------------------------------------
        ! Function:  void AssumedRank
        ! Attrs:     +intent(subroutine)
        ! Statement: f_subroutine
        ! ----------------------------------------
        ! Argument:  int * data +rank(0)
        ! Attrs:     +intent(inout)
        ! Statement: f_inout_native_*
        subroutine c_assumed_rank_0d(data) &
                bind(C, name="ERR_AssumedRank_0d")
            use iso_c_binding, only : C_INT
            implicit none
            integer(C_INT), intent(INOUT) :: data
        end subroutine c_assumed_rank_0d

        ! Generated by fortran_generic
        ! ----------------------------------------
        ! Function:  void AssumedRank
        ! Attrs:     +intent(subroutine)
        ! Statement: f_subroutine
        ! ----------------------------------------
        ! Argument:  int * data +rank(1)
        ! Attrs:     +intent(inout)
        ! Statement: f_inout_native_*
        subroutine c_assumed_rank_1d(data) &
                bind(C, name="ERR_AssumedRank_1d")
            use iso_c_binding, only : C_INT
            implicit none
            integer(C_INT), intent(INOUT) :: data(*)
        end subroutine c_assumed_rank_1d

        ! Generated by fortran_generic
        ! ----------------------------------------
        ! Function:  void AssumedRank
        ! Attrs:     +intent(subroutine)
        ! Statement: f_subroutine
        ! ----------------------------------------
        ! Argument:  int * data +rank(2)
        ! Attrs:     +intent(inout)
        ! Statement: f_inout_native_*
        subroutine c_assumed_rank_2d(data) &
                bind(C, name="ERR_AssumedRank_2d")
            use iso_c_binding, only : C_INT
            implicit none
            integer(C_INT), intent(INOUT) :: data(*)
        end subroutine c_assumed_rank_2d
    end interface

    interface assumed_rank
        module procedure assumed_rank_0d
        module procedure assumed_rank_1d
        module procedure assumed_rank_2d
    end interface assumed_rank

    ! splicer begin additional_declarations
    ! splicer end additional_declarations

contains

    ! Generated by getter/setter
    ! ----------------------------------------
    ! Function:  int get_x1
    ! Attrs:     +intent(getter)
    ! Statement: f_getter_native_scalar
    function cstruct_as_subclass_get_x1(obj) &
            result(SHT_rv)
        use iso_c_binding, only : C_INT
        class(cstruct_as_subclass) :: obj
        integer(C_INT) :: SHT_rv
        ! splicer begin class.Cstruct_as_subclass.method.get_x1
        SHT_rv = c_cstruct_as_subclass_get_x1(obj%cxxmem)
        ! splicer end class.Cstruct_as_subclass.method.get_x1
    end function cstruct_as_subclass_get_x1

    ! Generated by getter/setter
    ! ----------------------------------------
    ! Function:  void set_x1
    ! Attrs:     +intent(setter)
    ! Statement: f_setter
    ! ----------------------------------------
    ! Argument:  int val +intent(in)+value
    ! Attrs:     +intent(setter)
    ! Statement: f_setter_native_scalar
    subroutine cstruct_as_subclass_set_x1(obj, val)
        use iso_c_binding, only : C_INT
        class(cstruct_as_subclass) :: obj
        integer(C_INT), value, intent(IN) :: val
        ! splicer begin class.Cstruct_as_subclass.method.set_x1
        call c_cstruct_as_subclass_set_x1(obj%cxxmem, val)
        ! splicer end class.Cstruct_as_subclass.method.set_x1
    end subroutine cstruct_as_subclass_set_x1

    ! Generated by getter/setter
    ! ----------------------------------------
    ! Function:  int get_y1
    ! Attrs:     +intent(getter)
    ! Statement: f_getter_native_scalar
    function cstruct_as_subclass_get_y1(obj) &
            result(SHT_rv)
        use iso_c_binding, only : C_INT
        class(cstruct_as_subclass) :: obj
        integer(C_INT) :: SHT_rv
        ! splicer begin class.Cstruct_as_subclass.method.get_y1
        SHT_rv = c_cstruct_as_subclass_get_y1(obj%cxxmem)
        ! splicer end class.Cstruct_as_subclass.method.get_y1
    end function cstruct_as_subclass_get_y1

    ! Generated by getter/setter
    ! ----------------------------------------
    ! Function:  void set_y1
    ! Attrs:     +intent(setter)
    ! Statement: f_setter
    ! ----------------------------------------
    ! Argument:  int val +intent(in)+value
    ! Attrs:     +intent(setter)
    ! Statement: f_setter_native_scalar
    subroutine cstruct_as_subclass_set_y1(obj, val)
        use iso_c_binding, only : C_INT
        class(cstruct_as_subclass) :: obj
        integer(C_INT), value, intent(IN) :: val
        ! splicer begin class.Cstruct_as_subclass.method.set_y1
        call c_cstruct_as_subclass_set_y1(obj%cxxmem, val)
        ! splicer end class.Cstruct_as_subclass.method.set_y1
    end subroutine cstruct_as_subclass_set_y1

    ! Generated by getter/setter
    ! ----------------------------------------
    ! Function:  int get_z1
    ! Attrs:     +intent(getter)
    ! Statement: f_getter_native_scalar
    function cstruct_as_subclass_get_z1(obj) &
            result(SHT_rv)
        use iso_c_binding, only : C_INT
        class(cstruct_as_subclass) :: obj
        integer(C_INT) :: SHT_rv
        ! splicer begin class.Cstruct_as_subclass.method.get_z1
        SHT_rv = c_cstruct_as_subclass_get_z1(obj%cxxmem)
        ! splicer end class.Cstruct_as_subclass.method.get_z1
    end function cstruct_as_subclass_get_z1

    ! Generated by getter/setter
    ! ----------------------------------------
    ! Function:  void set_z1
    ! Attrs:     +intent(setter)
    ! Statement: f_setter
    ! ----------------------------------------
    ! Argument:  int val +intent(in)+value
    ! Attrs:     +intent(setter)
    ! Statement: f_setter_native_scalar
    subroutine cstruct_as_subclass_set_z1(obj, val)
        use iso_c_binding, only : C_INT
        class(cstruct_as_subclass) :: obj
        integer(C_INT), value, intent(IN) :: val
        ! splicer begin class.Cstruct_as_subclass.method.set_z1
        call c_cstruct_as_subclass_set_z1(obj%cxxmem, val)
        ! splicer end class.Cstruct_as_subclass.method.set_z1
    end subroutine cstruct_as_subclass_set_z1

    ! Return pointer to C++ memory.
    function cstruct_as_subclass_get_instance(obj) result (cxxptr)
        use iso_c_binding, only: C_PTR
        class(cstruct_as_subclass), intent(IN) :: obj
        type(C_PTR) :: cxxptr
        cxxptr = obj%cxxmem%addr
    end function cstruct_as_subclass_get_instance

    subroutine cstruct_as_subclass_set_instance(obj, cxxmem)
        use iso_c_binding, only: C_PTR
        class(cstruct_as_subclass), intent(INOUT) :: obj
        type(C_PTR), intent(IN) :: cxxmem
        obj%cxxmem%addr = cxxmem
        obj%cxxmem%idtor = 0
    end subroutine cstruct_as_subclass_set_instance

    function cstruct_as_subclass_associated(obj) result (rv)
        use iso_c_binding, only: c_associated
        class(cstruct_as_subclass), intent(IN) :: obj
        logical rv
        rv = c_associated(obj%cxxmem%addr)
    end function cstruct_as_subclass_associated

    ! splicer begin class.Cstruct_as_subclass.additional_functions
    ! splicer end class.Cstruct_as_subclass.additional_functions

    ! ----------------------------------------
    ! Function:  void BadFstatements
    ! Attrs:     +intent(subroutine)
    ! Statement: f_subroutine
    subroutine bad_fstatements()
        ! splicer begin function.bad_fstatements
        call c_bad_fstatements()
        ===>{no_such_var} = 10<===
        ! splicer end function.bad_fstatements
    end subroutine bad_fstatements

    ! Generated by fortran_generic
    ! ----------------------------------------
    ! Function:  void AssumedRank
    ! Attrs:     +intent(subroutine)
    ! Statement: f_subroutine
    ! ----------------------------------------
    ! Argument:  int * data +rank(0)
    ! Attrs:     +intent(inout)
    ! Statement: f_inout_native_*
    subroutine assumed_rank_0d(data)
        use iso_c_binding, only : C_INT
        integer(C_INT), intent(INOUT) :: data
        ! splicer begin function.assumed_rank_0d
        call c_assumed_rank_0d(data)
        ! splicer end function.assumed_rank_0d
    end subroutine assumed_rank_0d

    ! Generated by fortran_generic
    ! ----------------------------------------
    ! Function:  void AssumedRank
    ! Attrs:     +intent(subroutine)
    ! Statement: f_subroutine
    ! ----------------------------------------
    ! Argument:  int * data +rank(1)
    ! Attrs:     +intent(inout)
    ! Statement: f_inout_native_*
    subroutine assumed_rank_1d(data)
        use iso_c_binding, only : C_INT
        integer(C_INT), intent(INOUT) :: data(:)
        ! splicer begin function.assumed_rank_1d
        call c_assumed_rank_1d(data)
        ! splicer end function.assumed_rank_1d
    end subroutine assumed_rank_1d

    ! Generated by fortran_generic
    ! ----------------------------------------
    ! Function:  void AssumedRank
    ! Attrs:     +intent(subroutine)
    ! Statement: f_subroutine
    ! ----------------------------------------
    ! Argument:  int * data +rank(2)
    ! Attrs:     +intent(inout)
    ! Statement: f_inout_native_*
    subroutine assumed_rank_2d(data)
        use iso_c_binding, only : C_INT
        integer(C_INT), intent(INOUT) :: data(:,:)
        ! splicer begin function.assumed_rank_2d
        call c_assumed_rank_2d(data)
        ! splicer end function.assumed_rank_2d
    end subroutine assumed_rank_2d

    ! splicer begin additional_functions
    ! splicer end additional_functions

    function cstruct_as_subclass_eq(a,b) result (rv)
        use iso_c_binding, only: c_associated
        type(cstruct_as_subclass), intent(IN) ::a,b
        logical :: rv
        if (c_associated(a%cxxmem%addr, b%cxxmem%addr)) then
            rv = .true.
        else
            rv = .false.
        endif
    end function cstruct_as_subclass_eq

    function cstruct_as_subclass_ne(a,b) result (rv)
        use iso_c_binding, only: c_associated
        type(cstruct_as_subclass), intent(IN) ::a,b
        logical :: rv
        if (.not. c_associated(a%cxxmem%addr, b%cxxmem%addr)) then
            rv = .true.
        else
            rv = .false.
        endif
    end function cstruct_as_subclass_ne

end module error_mod
