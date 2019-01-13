! wrapfWorker.f
! This is generated code, do not edit
!>
!! \file wrapfWorker.f
!! \brief Shroud generated wrapper for Worker class
!<
! splicer begin file_top
! splicer end file_top
module worker_mod
    use iso_c_binding, only : C_INT, C_NULL_PTR, C_PTR
    ! splicer begin class.Worker.module_use
    ! splicer end class.Worker.module_use
    implicit none


    ! splicer begin class.Worker.module_top
    ! splicer end class.Worker.module_top

    type, bind(C) :: SHROUD_worker_capsule
        type(C_PTR) :: addr = C_NULL_PTR  ! address of C++ memory
        integer(C_INT) :: idtor = 0       ! index of destructor
    end type SHROUD_worker_capsule

    type worker
        type(SHROUD_worker_capsule) :: cxxmem
        ! splicer begin class.Worker.component_part
        ! splicer end class.Worker.component_part
    contains
        procedure :: get_instance => worker_get_instance
        procedure :: set_instance => worker_set_instance
        procedure :: associated => worker_associated
        ! splicer begin class.Worker.type_bound_procedure_part
        ! splicer end class.Worker.type_bound_procedure_part
    end type worker

    interface operator (.eq.)
        module procedure worker_eq
    end interface

    interface operator (.ne.)
        module procedure worker_ne
    end interface

    interface

        ! splicer begin class.Worker.additional_interfaces
        ! splicer end class.Worker.additional_interfaces
    end interface

contains

    ! Return pointer to C++ memory.
    function worker_get_instance(obj) result (cxxptr)
        use iso_c_binding, only: C_PTR
        class(worker), intent(IN) :: obj
        type(C_PTR) :: cxxptr
        cxxptr = obj%cxxmem%addr
    end function worker_get_instance

    subroutine worker_set_instance(obj, cxxmem)
        use iso_c_binding, only: C_PTR
        class(worker), intent(INOUT) :: obj
        type(C_PTR), intent(IN) :: cxxmem
        obj%cxxmem%addr = cxxmem
        obj%cxxmem%idtor = 0
    end subroutine worker_set_instance

    function worker_associated(obj) result (rv)
        use iso_c_binding, only: c_associated
        class(worker), intent(IN) :: obj
        logical rv
        rv = c_associated(obj%cxxmem%addr)
    end function worker_associated

    ! splicer begin class.Worker.additional_functions
    ! splicer end class.Worker.additional_functions

    function worker_eq(a,b) result (rv)
        use iso_c_binding, only: c_associated
        type(worker), intent(IN) ::a,b
        logical :: rv
        if (c_associated(a%cxxmem%addr, b%cxxmem%addr)) then
            rv = .true.
        else
            rv = .false.
        endif
    end function worker_eq

    function worker_ne(a,b) result (rv)
        use iso_c_binding, only: c_associated
        type(worker), intent(IN) ::a,b
        logical :: rv
        if (.not. c_associated(a%cxxmem%addr, b%cxxmem%addr)) then
            rv = .true.
        else
            rv = .false.
        endif
    end function worker_ne

end module worker_mod
