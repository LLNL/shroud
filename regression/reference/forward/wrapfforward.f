! wrapfforward.f
! This file is generated by Shroud nowrite-version. Do not edit.
! Copyright (c) 2017-2023, Lawrence Livermore National Security, LLC and
! other Shroud Project Developers.
! See the top-level COPYRIGHT file for details.
!
! SPDX-License-Identifier: (BSD-3-Clause)
!
!>
!! \file wrapfforward.f
!! \brief Shroud generated wrapper for forward namespace
!<
! splicer begin file_top
! splicer end file_top
module forward_mod
    use iso_c_binding, only : C_INT, C_NULL_PTR, C_PTR
    ! splicer begin module_use
    ! splicer end module_use
    implicit none

    ! splicer begin module_top
    ! splicer end module_top

    ! helper capsule_data_helper
    type, bind(C) :: FOR_SHROUD_capsule_data
        type(C_PTR) :: addr = C_NULL_PTR  ! address of C++ memory
        integer(C_INT) :: idtor = 0       ! index of destructor
    end type FOR_SHROUD_capsule_data

    type class3
        type(FOR_SHROUD_capsule_data) :: cxxmem
        ! splicer begin class.Class3.component_part
        ! splicer end class.Class3.component_part
    contains
        procedure :: get_instance => class3_get_instance
        procedure :: set_instance => class3_set_instance
        procedure :: associated => class3_associated
        ! splicer begin class.Class3.type_bound_procedure_part
        ! splicer end class.Class3.type_bound_procedure_part
    end type class3

    type class2
        type(FOR_SHROUD_capsule_data) :: cxxmem
        ! splicer begin class.Class2.component_part
        ! splicer end class.Class2.component_part
    contains
        procedure :: dtor => class2_dtor
        procedure :: func1 => class2_func1
        procedure :: accept_class3 => class2_accept_class3
        procedure :: get_instance => class2_get_instance
        procedure :: set_instance => class2_set_instance
        procedure :: associated => class2_associated
        ! splicer begin class.Class2.type_bound_procedure_part
        ! splicer end class.Class2.type_bound_procedure_part
    end type class2

    interface operator (.eq.)
        module procedure class3_eq
        module procedure class2_eq
    end interface

    interface operator (.ne.)
        module procedure class3_ne
        module procedure class2_ne
    end interface

    interface

        ! ----------------------------------------
        ! Function:  Class2
        ! Attrs:     +api(capptr)+intent(ctor)
        ! Statement: f_ctor_shadow_scalar_capptr
        function c_class2_ctor(SHT_rv) &
                result(SHT_prv) &
                bind(C, name="FOR_Class2_ctor")
            use iso_c_binding, only : C_PTR
            import :: FOR_SHROUD_capsule_data
            implicit none
            type(FOR_SHROUD_capsule_data), intent(OUT) :: SHT_rv
            type(C_PTR) :: SHT_prv
        end function c_class2_ctor

        ! ----------------------------------------
        ! Function:  ~Class2
        ! Attrs:     +intent(dtor)
        ! Statement: f_dtor_void_scalar
        subroutine c_class2_dtor(self) &
                bind(C, name="FOR_Class2_dtor")
            import :: FOR_SHROUD_capsule_data
            implicit none
            type(FOR_SHROUD_capsule_data), intent(INOUT) :: self
        end subroutine c_class2_dtor

        ! ----------------------------------------
        ! Function:  void func1
        ! Attrs:     +intent(subroutine)
        ! Statement: f_subroutine_void_scalar
        ! ----------------------------------------
        ! Argument:  tutorial::Class1 * arg +intent(in)
        ! Attrs:     +intent(in)
        ! Statement: f_in_shadow_*
        subroutine c_class2_func1(self, arg) &
                bind(C, name="FOR_Class2_func1")
            use tutorial_mod, only : SHROUD_class1_capsule
            import :: FOR_SHROUD_capsule_data
            implicit none
            type(FOR_SHROUD_capsule_data), intent(IN) :: self
            type(SHROUD_class1_capsule), intent(IN) :: arg
        end subroutine c_class2_func1

        ! ----------------------------------------
        ! Function:  void acceptClass3
        ! Attrs:     +intent(subroutine)
        ! Statement: f_subroutine_void_scalar
        ! ----------------------------------------
        ! Argument:  Class3 * arg +intent(in)
        ! Attrs:     +intent(in)
        ! Statement: f_in_shadow_*
        subroutine c_class2_accept_class3(self, arg) &
                bind(C, name="FOR_Class2_acceptClass3")
            import :: FOR_SHROUD_capsule_data
            implicit none
            type(FOR_SHROUD_capsule_data), intent(IN) :: self
            type(FOR_SHROUD_capsule_data), intent(IN) :: arg
        end subroutine c_class2_accept_class3

        ! ----------------------------------------
        ! Function:  int passStruct1
        ! Attrs:     +intent(function)
        ! Statement: f_function_native_scalar
        ! ----------------------------------------
        ! Argument:  const Cstruct1 * arg
        ! Attrs:     +intent(in)
        ! Statement: f_in_struct_*
        function pass_struct1(arg) &
                result(SHT_rv) &
                bind(C, name="FOR_passStruct1")
            use iso_c_binding, only : C_INT
            use struct_mod, only : cstruct1
            implicit none
            type(cstruct1), intent(IN) :: arg
            integer(C_INT) :: SHT_rv
        end function pass_struct1
    end interface

    interface class2
        module procedure class2_ctor
    end interface class2

    ! splicer begin additional_declarations
    ! splicer end additional_declarations

contains

    ! Return pointer to C++ memory.
    function class3_get_instance(obj) result (cxxptr)
        use iso_c_binding, only: C_PTR
        class(class3), intent(IN) :: obj
        type(C_PTR) :: cxxptr
        cxxptr = obj%cxxmem%addr
    end function class3_get_instance

    subroutine class3_set_instance(obj, cxxmem)
        use iso_c_binding, only: C_PTR
        class(class3), intent(INOUT) :: obj
        type(C_PTR), intent(IN) :: cxxmem
        obj%cxxmem%addr = cxxmem
        obj%cxxmem%idtor = 0
    end subroutine class3_set_instance

    function class3_associated(obj) result (rv)
        use iso_c_binding, only: c_associated
        class(class3), intent(IN) :: obj
        logical rv
        rv = c_associated(obj%cxxmem%addr)
    end function class3_associated

    ! splicer begin class.Class3.additional_functions
    ! splicer end class.Class3.additional_functions

    ! ----------------------------------------
    ! Function:  Class2
    ! Attrs:     +api(capptr)+intent(ctor)
    ! Statement: f_ctor_shadow_scalar_capptr
    function class2_ctor() &
            result(SHT_rv)
        use iso_c_binding, only : C_PTR
        type(class2) :: SHT_rv
        type(C_PTR) :: SHT_prv
        ! splicer begin class.Class2.method.ctor
        SHT_prv = c_class2_ctor(SHT_rv%cxxmem)
        ! splicer end class.Class2.method.ctor
    end function class2_ctor

    ! ----------------------------------------
    ! Function:  ~Class2
    ! Attrs:     +intent(dtor)
    ! Statement: f_dtor
    subroutine class2_dtor(obj)
        class(class2) :: obj
        ! splicer begin class.Class2.method.dtor
        call c_class2_dtor(obj%cxxmem)
        ! splicer end class.Class2.method.dtor
    end subroutine class2_dtor

    ! ----------------------------------------
    ! Function:  void func1
    ! Attrs:     +intent(subroutine)
    ! Statement: f_subroutine
    ! ----------------------------------------
    ! Argument:  tutorial::Class1 * arg +intent(in)
    ! Attrs:     +intent(in)
    ! Statement: f_in_shadow_*
    subroutine class2_func1(obj, arg)
        use tutorial_mod, only : class1
        class(class2) :: obj
        type(class1), intent(IN) :: arg
        ! splicer begin class.Class2.method.func1
        call c_class2_func1(obj%cxxmem, arg%cxxmem)
        ! splicer end class.Class2.method.func1
    end subroutine class2_func1

    ! ----------------------------------------
    ! Function:  void acceptClass3
    ! Attrs:     +intent(subroutine)
    ! Statement: f_subroutine
    ! ----------------------------------------
    ! Argument:  Class3 * arg +intent(in)
    ! Attrs:     +intent(in)
    ! Statement: f_in_shadow_*
    subroutine class2_accept_class3(obj, arg)
        class(class2) :: obj
        type(class3), intent(IN) :: arg
        ! splicer begin class.Class2.method.accept_class3
        call c_class2_accept_class3(obj%cxxmem, arg%cxxmem)
        ! splicer end class.Class2.method.accept_class3
    end subroutine class2_accept_class3

    ! Return pointer to C++ memory.
    function class2_get_instance(obj) result (cxxptr)
        use iso_c_binding, only: C_PTR
        class(class2), intent(IN) :: obj
        type(C_PTR) :: cxxptr
        cxxptr = obj%cxxmem%addr
    end function class2_get_instance

    subroutine class2_set_instance(obj, cxxmem)
        use iso_c_binding, only: C_PTR
        class(class2), intent(INOUT) :: obj
        type(C_PTR), intent(IN) :: cxxmem
        obj%cxxmem%addr = cxxmem
        obj%cxxmem%idtor = 0
    end subroutine class2_set_instance

    function class2_associated(obj) result (rv)
        use iso_c_binding, only: c_associated
        class(class2), intent(IN) :: obj
        logical rv
        rv = c_associated(obj%cxxmem%addr)
    end function class2_associated

    ! splicer begin class.Class2.additional_functions
    ! splicer end class.Class2.additional_functions

#if 0
    ! Only the interface is needed
    ! ----------------------------------------
    ! Function:  int passStruct1
    ! Attrs:     +intent(function)
    ! Statement: f_function_native_scalar
    ! ----------------------------------------
    ! Argument:  const Cstruct1 * arg
    ! Attrs:     +intent(in)
    ! Statement: f_in_struct_*
    function pass_struct1(arg) &
            result(SHT_rv)
        use iso_c_binding, only : C_INT
        use struct_mod, only : cstruct1
        type(cstruct1), intent(IN) :: arg
        integer(C_INT) :: SHT_rv
        ! splicer begin function.pass_struct1
        SHT_rv = c_pass_struct1(arg)
        ! splicer end function.pass_struct1
    end function pass_struct1
#endif

    ! splicer begin additional_functions
    ! splicer end additional_functions

    function class3_eq(a,b) result (rv)
        use iso_c_binding, only: c_associated
        type(class3), intent(IN) ::a,b
        logical :: rv
        if (c_associated(a%cxxmem%addr, b%cxxmem%addr)) then
            rv = .true.
        else
            rv = .false.
        endif
    end function class3_eq

    function class3_ne(a,b) result (rv)
        use iso_c_binding, only: c_associated
        type(class3), intent(IN) ::a,b
        logical :: rv
        if (.not. c_associated(a%cxxmem%addr, b%cxxmem%addr)) then
            rv = .true.
        else
            rv = .false.
        endif
    end function class3_ne

    function class2_eq(a,b) result (rv)
        use iso_c_binding, only: c_associated
        type(class2), intent(IN) ::a,b
        logical :: rv
        if (c_associated(a%cxxmem%addr, b%cxxmem%addr)) then
            rv = .true.
        else
            rv = .false.
        endif
    end function class2_eq

    function class2_ne(a,b) result (rv)
        use iso_c_binding, only: c_associated
        type(class2), intent(IN) ::a,b
        logical :: rv
        if (.not. c_associated(a%cxxmem%addr, b%cxxmem%addr)) then
            rv = .true.
        else
            rv = .false.
        endif
    end function class2_ne

end module forward_mod
