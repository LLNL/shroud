! wrapfownership.f
! This is generated code, do not edit
! #######################################################################
! Copyright (c) 2017-2018, Lawrence Livermore National Security, LLC.
! Produced at the Lawrence Livermore National Laboratory
!
! LLNL-CODE-738041.
! All rights reserved.
!
! This file is part of Shroud.  For details, see
! https://github.com/LLNL/shroud. Please also read shroud/LICENSE.
!
! Redistribution and use in source and binary forms, with or without
! modification, are permitted provided that the following conditions are
! met:
!
! * Redistributions of source code must retain the above copyright
!   notice, this list of conditions and the disclaimer below.
!
! * Redistributions in binary form must reproduce the above copyright
!   notice, this list of conditions and the disclaimer (as noted below)
!   in the documentation and/or other materials provided with the
!   distribution.
!
! * Neither the name of the LLNS/LLNL nor the names of its contributors
!   may be used to endorse or promote products derived from this
!   software without specific prior written permission.
!
! THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
! "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
! LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
! A PARTICULAR PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL LAWRENCE
! LIVERMORE NATIONAL SECURITY, LLC, THE U.S. DEPARTMENT OF ENERGY OR
! CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
! EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
! PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
! PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
! LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
! NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
! SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
!
! #######################################################################
!>
!! \file wrapfownership.f
!! \brief Shroud generated wrapper for ownership library
!<
! splicer begin file_top
! splicer end file_top
module ownership_mod
    use iso_c_binding, only : C_INT, C_NULL_PTR, C_PTR, C_SIZE_T
    ! splicer begin module_use
    ! splicer end module_use
    implicit none

    ! splicer begin module_top
    ! splicer end module_top

    type, bind(C) :: SHROUD_capsule_data
        type(C_PTR) :: addr = C_NULL_PTR  ! address of C++ memory
        integer(C_INT) :: idtor = 0       ! index of destructor
    end type SHROUD_capsule_data

    type, bind(C) :: SHROUD_array
        type(SHROUD_capsule_data) :: cxx       ! address of C++ memory
        type(C_PTR) :: addr = C_NULL_PTR       ! address of data in cxx
        integer(C_SIZE_T) :: len = 0_C_SIZE_T  ! bytes-per-item or character len of data in cxx
        integer(C_SIZE_T) :: size = 0_C_SIZE_T ! size of data in cxx
    end type SHROUD_array

    ! splicer begin class.Class1.module_top
    ! splicer end class.Class1.module_top

    type class1
        type(SHROUD_capsule_data) :: cxxmem
        ! splicer begin class.Class1.component_part
        ! splicer end class.Class1.component_part
    contains
        procedure :: dtor => class1_dtor
        procedure :: get_flag => class1_get_flag
        procedure :: get_instance => class1_get_instance
        procedure :: set_instance => class1_set_instance
        procedure :: associated => class1_associated
        ! splicer begin class.Class1.type_bound_procedure_part
        ! splicer end class.Class1.type_bound_procedure_part
    end type class1

    interface operator (.eq.)
        module procedure class1_eq
    end interface

    interface operator (.ne.)
        module procedure class1_ne
    end interface

    interface

        subroutine c_class1_dtor(self) &
                bind(C, name="OWN_class1_dtor")
            import :: SHROUD_capsule_data
            implicit none
            type(SHROUD_capsule_data), intent(IN) :: self
        end subroutine c_class1_dtor

        function c_class1_get_flag(self) &
                result(SHT_rv) &
                bind(C, name="OWN_class1_get_flag")
            use iso_c_binding, only : C_INT
            import :: SHROUD_capsule_data
            implicit none
            type(SHROUD_capsule_data), intent(IN) :: self
            integer(C_INT) :: SHT_rv
        end function c_class1_get_flag

        ! splicer begin class.Class1.additional_interfaces
        ! splicer end class.Class1.additional_interfaces

        function c_return_int_ptr_raw() &
                result(SHT_rv) &
                bind(C, name="OWN_return_int_ptr_raw")
            use iso_c_binding, only : C_INT, C_PTR
            implicit none
            type(C_PTR) SHT_rv
        end function c_return_int_ptr_raw

        function return_int_ptr_scalar() &
                result(SHT_rv) &
                bind(C, name="OWN_return_int_ptr_scalar")
            use iso_c_binding, only : C_INT
            implicit none
            integer(C_INT) :: SHT_rv
        end function return_int_ptr_scalar

        function c_return_int_ptr_pointer() &
                result(SHT_rv) &
                bind(C, name="OWN_return_int_ptr_pointer")
            use iso_c_binding, only : C_INT, C_PTR
            implicit none
            type(C_PTR) SHT_rv
        end function c_return_int_ptr_pointer

        function c_return_int_ptr_dim_raw(len) &
                result(SHT_rv) &
                bind(C, name="OWN_return_int_ptr_dim_raw")
            use iso_c_binding, only : C_INT, C_PTR
            implicit none
            integer(C_INT), intent(OUT) :: len
            type(C_PTR) SHT_rv
        end function c_return_int_ptr_dim_raw

        function c_return_int_ptr_dim_pointer(len) &
                result(SHT_rv) &
                bind(C, name="OWN_return_int_ptr_dim_pointer")
            use iso_c_binding, only : C_INT, C_PTR
            implicit none
            integer(C_INT), intent(OUT) :: len
            type(C_PTR) SHT_rv
        end function c_return_int_ptr_dim_pointer

        function c_return_int_ptr_dim_alloc(len) &
                result(SHT_rv) &
                bind(C, name="OWN_return_int_ptr_dim_alloc")
            use iso_c_binding, only : C_INT, C_PTR
            implicit none
            integer(C_INT), intent(OUT) :: len
            type(C_PTR) SHT_rv
        end function c_return_int_ptr_dim_alloc

        function c_return_int_ptr_dim_alloc_bufferify(DSHC_rv, len) &
                result(SHT_rv) &
                bind(C, name="OWN_return_int_ptr_dim_alloc_bufferify")
            use iso_c_binding, only : C_INT, C_PTR
            import :: SHROUD_array
            implicit none
            type(SHROUD_array), intent(INOUT) :: DSHC_rv
            integer(C_INT), intent(OUT) :: len
            type(C_PTR) SHT_rv
        end function c_return_int_ptr_dim_alloc_bufferify

        function c_return_int_ptr_dim_default(len) &
                result(SHT_rv) &
                bind(C, name="OWN_return_int_ptr_dim_default")
            use iso_c_binding, only : C_INT, C_PTR
            implicit none
            integer(C_INT), intent(OUT) :: len
            type(C_PTR) SHT_rv
        end function c_return_int_ptr_dim_default

        function c_return_int_ptr_dim_raw_new(len) &
                result(SHT_rv) &
                bind(C, name="OWN_return_int_ptr_dim_raw_new")
            use iso_c_binding, only : C_INT, C_PTR
            implicit none
            integer(C_INT), intent(OUT) :: len
            type(C_PTR) SHT_rv
        end function c_return_int_ptr_dim_raw_new

        function c_return_int_ptr_dim_pointer_new(len) &
                result(SHT_rv) &
                bind(C, name="OWN_return_int_ptr_dim_pointer_new")
            use iso_c_binding, only : C_INT, C_PTR
            implicit none
            integer(C_INT), intent(OUT) :: len
            type(C_PTR) SHT_rv
        end function c_return_int_ptr_dim_pointer_new

        function c_return_int_ptr_dim_alloc_new(len) &
                result(SHT_rv) &
                bind(C, name="OWN_return_int_ptr_dim_alloc_new")
            use iso_c_binding, only : C_INT, C_PTR
            implicit none
            integer(C_INT), intent(OUT) :: len
            type(C_PTR) SHT_rv
        end function c_return_int_ptr_dim_alloc_new

        function c_return_int_ptr_dim_default_new(len) &
                result(SHT_rv) &
                bind(C, name="OWN_return_int_ptr_dim_default_new")
            use iso_c_binding, only : C_INT, C_PTR
            implicit none
            integer(C_INT), intent(OUT) :: len
            type(C_PTR) SHT_rv
        end function c_return_int_ptr_dim_default_new

        subroutine create_class_static(flag) &
                bind(C, name="OWN_create_class_static")
            use iso_c_binding, only : C_INT
            implicit none
            integer(C_INT), value, intent(IN) :: flag
        end subroutine create_class_static

        function c_get_class_static() &
                result(SHT_rv) &
                bind(C, name="OWN_get_class_static")
            import :: SHROUD_capsule_data
            implicit none
            type(SHROUD_capsule_data) :: SHT_rv
        end function c_get_class_static

        function c_get_class_new(flag) &
                result(SHT_rv) &
                bind(C, name="OWN_get_class_new")
            use iso_c_binding, only : C_INT
            import :: SHROUD_capsule_data
            implicit none
            integer(C_INT), value, intent(IN) :: flag
            type(SHROUD_capsule_data) :: SHT_rv
        end function c_get_class_new

        ! splicer begin additional_interfaces
        ! splicer end additional_interfaces
    end interface

    interface
        ! helper function
        ! Copy contents of context into c_var.
        subroutine SHROUD_copy_array_int(context, c_var, c_var_size) &
            bind(C, name="OWN_ShroudCopyArray")
            use iso_c_binding, only : C_INT, C_SIZE_T
            import SHROUD_array
            type(SHROUD_array), intent(IN) :: context
            integer(C_INT), intent(OUT) :: c_var(*)
            integer(C_SIZE_T), value :: c_var_size
        end subroutine SHROUD_copy_array_int
    end interface

contains

    ! ~Class1()
    subroutine class1_dtor(obj)
        class(class1) :: obj
        ! splicer begin class.Class1.method.dtor
        call c_class1_dtor(obj%cxxmem)
        ! splicer end class.Class1.method.dtor
    end subroutine class1_dtor

    ! int getFlag()
    function class1_get_flag(obj) &
            result(SHT_rv)
        use iso_c_binding, only : C_INT
        class(class1) :: obj
        integer(C_INT) :: SHT_rv
        ! splicer begin class.Class1.method.get_flag
        SHT_rv = c_class1_get_flag(obj%cxxmem)
        ! splicer end class.Class1.method.get_flag
    end function class1_get_flag

    ! Return pointer to C++ memory.
    function class1_get_instance(obj) result (cxxptr)
        use iso_c_binding, only: c_associated, C_NULL_PTR, C_PTR
        class(class1), intent(IN) :: obj
        type(C_PTR) :: cxxptr
        cxxptr = obj%cxxmem%addr
    end function class1_get_instance

    subroutine class1_set_instance(obj, cxxmem)
        use iso_c_binding, only: C_PTR
        class(class1), intent(INOUT) :: obj
        type(C_PTR), intent(IN) :: cxxmem
        obj%cxxmem%addr = cxxmem
        obj%cxxmem%idtor = 0
    end subroutine class1_set_instance

    function class1_associated(obj) result (rv)
        use iso_c_binding, only: c_associated
        class(class1), intent(IN) :: obj
        logical rv
        rv = c_associated(obj%cxxmem%addr)
    end function class1_associated

    ! splicer begin class.Class1.additional_functions
    ! splicer end class.Class1.additional_functions

    ! int * ReturnIntPtrPointer() +deref(pointer)
    function return_int_ptr_pointer() &
            result(SHT_rv)
        use iso_c_binding, only : C_INT, C_PTR, c_f_pointer
        integer(C_INT), pointer :: SHT_rv
        type(C_PTR) :: SHT_ptr
        ! splicer begin function.return_int_ptr_pointer
        SHT_ptr = c_return_int_ptr_pointer()
        call c_f_pointer(SHT_ptr, SHT_rv)
        ! splicer end function.return_int_ptr_pointer
    end function return_int_ptr_pointer

    ! int * ReturnIntPtrDimPointer(int * len +hidden+intent(out)) +deref(pointer)+dimension(len)
    function return_int_ptr_dim_pointer() &
            result(SHT_rv)
        use iso_c_binding, only : C_INT, C_PTR, c_f_pointer
        integer(C_INT) :: len
        integer(C_INT), pointer :: SHT_rv(:)
        type(C_PTR) :: SHT_ptr
        ! splicer begin function.return_int_ptr_dim_pointer
        SHT_ptr = c_return_int_ptr_dim_pointer(len)
        call c_f_pointer(SHT_ptr, SHT_rv, [len])
        ! splicer end function.return_int_ptr_dim_pointer
    end function return_int_ptr_dim_pointer

    ! int * ReturnIntPtrDimAlloc(int * len +hidden+intent(out)) +deref(allocatable)+dimension(len)
    ! arg_to_buffer
    function return_int_ptr_dim_alloc() &
            result(SHT_rv)
        use iso_c_binding, only : C_INT, C_PTR
        type(SHROUD_array) :: DSHC_rv
        integer(C_INT) :: len
        integer(C_INT), allocatable :: SHT_rv(:)
        type(C_PTR) :: SHT_ptr
        ! splicer begin function.return_int_ptr_dim_alloc
        SHT_ptr = c_return_int_ptr_dim_alloc_bufferify(DSHC_rv, len)
        ! splicer end function.return_int_ptr_dim_alloc
        allocate(SHT_rv(len))
        call SHROUD_copy_array_int(DSHC_rv, SHT_rv, size(SHT_rv, kind=C_SIZE_T))
    end function return_int_ptr_dim_alloc

    ! int * ReturnIntPtrDimDefault(int * len +hidden+intent(out)) +dimension(len)
    function return_int_ptr_dim_default() &
            result(SHT_rv)
        use iso_c_binding, only : C_INT, C_PTR, c_f_pointer
        integer(C_INT) :: len
        integer(C_INT), pointer :: SHT_rv(:)
        type(C_PTR) :: SHT_ptr
        ! splicer begin function.return_int_ptr_dim_default
        SHT_ptr = c_return_int_ptr_dim_default(len)
        call c_f_pointer(SHT_ptr, SHT_rv, [len])
        ! splicer end function.return_int_ptr_dim_default
    end function return_int_ptr_dim_default

    ! int * ReturnIntPtrDimPointerNew(int * len +hidden+intent(out)) +deref(pointer)+dimension(len)+owner(caller)
    function return_int_ptr_dim_pointer_new() &
            result(SHT_rv)
        use iso_c_binding, only : C_INT, C_PTR, c_f_pointer
        integer(C_INT) :: len
        integer(C_INT), pointer :: SHT_rv(:)
        type(C_PTR) :: SHT_ptr
        ! splicer begin function.return_int_ptr_dim_pointer_new
        SHT_ptr = c_return_int_ptr_dim_pointer_new(len)
        call c_f_pointer(SHT_ptr, SHT_rv, [len])
        ! splicer end function.return_int_ptr_dim_pointer_new
    end function return_int_ptr_dim_pointer_new

    ! int * ReturnIntPtrDimDefaultNew(int * len +hidden+intent(out)) +dimension(len)+owner(caller)
    function return_int_ptr_dim_default_new() &
            result(SHT_rv)
        use iso_c_binding, only : C_INT, C_PTR, c_f_pointer
        integer(C_INT) :: len
        integer(C_INT), pointer :: SHT_rv(:)
        type(C_PTR) :: SHT_ptr
        ! splicer begin function.return_int_ptr_dim_default_new
        SHT_ptr = c_return_int_ptr_dim_default_new(len)
        call c_f_pointer(SHT_ptr, SHT_rv, [len])
        ! splicer end function.return_int_ptr_dim_default_new
    end function return_int_ptr_dim_default_new

    ! Class1 * getClassStatic() +owner(library)
    function get_class_static() &
            result(SHT_rv)
        type(class1) :: SHT_rv
        ! splicer begin function.get_class_static
        SHT_rv%cxxmem = c_get_class_static()
        ! splicer end function.get_class_static
    end function get_class_static

    ! Class1 * getClassNew(int flag +intent(in)+value) +owner(caller)
    !>
    !! \brief Return pointer to new Class1 instance.
    !!
    !<
    function get_class_new(flag) &
            result(SHT_rv)
        use iso_c_binding, only : C_INT
        integer(C_INT), value, intent(IN) :: flag
        type(class1) :: SHT_rv
        ! splicer begin function.get_class_new
        SHT_rv%cxxmem = c_get_class_new(flag)
        ! splicer end function.get_class_new
    end function get_class_new

    ! splicer begin additional_functions
    ! splicer end additional_functions

    function class1_eq(a,b) result (rv)
        use iso_c_binding, only: c_associated
        type(class1), intent(IN) ::a,b
        logical :: rv
        if (c_associated(a%cxxmem%addr, b%cxxmem%addr)) then
            rv = .true.
        else
            rv = .false.
        endif
    end function class1_eq

    function class1_ne(a,b) result (rv)
        use iso_c_binding, only: c_associated
        type(class1), intent(IN) ::a,b
        logical :: rv
        if (.not. c_associated(a%cxxmem%addr, b%cxxmem%addr)) then
            rv = .true.
        else
            rv = .false.
        endif
    end function class1_ne

end module ownership_mod
