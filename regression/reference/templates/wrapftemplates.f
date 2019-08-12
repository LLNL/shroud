! wrapftemplates.f
! This is generated code, do not edit
!>
!! \file wrapftemplates.f
!! \brief Shroud generated wrapper for templates library
!<
! splicer begin file_top
! splicer end file_top
module templates_mod
    use iso_c_binding, only : C_INT, C_NULL_PTR, C_PTR
    ! splicer begin module_use
    ! splicer end module_use
    implicit none

    ! splicer begin module_top
    ! splicer end module_top

    ! splicer begin class.vector_int.module_top
    ! splicer end class.vector_int.module_top

    type, bind(C) :: SHROUD_vector_int_capsule
        type(C_PTR) :: addr = C_NULL_PTR  ! address of C++ memory
        integer(C_INT) :: idtor = 0       ! index of destructor
    end type SHROUD_vector_int_capsule

    type vector_int
        type(SHROUD_vector_int_capsule) :: cxxmem
        ! splicer begin class.vector_int.component_part
        ! splicer end class.vector_int.component_part
    contains
        procedure :: dtor => vector_int_dtor
        procedure :: push_back => vector_int_push_back
        procedure :: at => vector_int_at
        procedure :: get_instance => vector_int_get_instance
        procedure :: set_instance => vector_int_set_instance
        procedure :: associated => vector_int_associated
        ! splicer begin class.vector_int.type_bound_procedure_part
        ! splicer end class.vector_int.type_bound_procedure_part
    end type vector_int

    ! splicer begin class.vector_double.module_top
    ! splicer end class.vector_double.module_top

    type, bind(C) :: SHROUD_vector_double_capsule
        type(C_PTR) :: addr = C_NULL_PTR  ! address of C++ memory
        integer(C_INT) :: idtor = 0       ! index of destructor
    end type SHROUD_vector_double_capsule

    type vector_double
        type(SHROUD_vector_double_capsule) :: cxxmem
        ! splicer begin class.vector_double.component_part
        ! splicer end class.vector_double.component_part
    contains
        procedure :: dtor => vector_double_dtor
        procedure :: push_back => vector_double_push_back
        procedure :: at => vector_double_at
        procedure :: get_instance => vector_double_get_instance
        procedure :: set_instance => vector_double_set_instance
        procedure :: associated => vector_double_associated
        ! splicer begin class.vector_double.type_bound_procedure_part
        ! splicer end class.vector_double.type_bound_procedure_part
    end type vector_double

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

    ! splicer begin class.user_int.module_top
    ! splicer end class.user_int.module_top

    type, bind(C) :: SHROUD_user_int_capsule
        type(C_PTR) :: addr = C_NULL_PTR  ! address of C++ memory
        integer(C_INT) :: idtor = 0       ! index of destructor
    end type SHROUD_user_int_capsule

    type user_int
        type(SHROUD_user_int_capsule) :: cxxmem
        ! splicer begin class.user_int.component_part
        ! splicer end class.user_int.component_part
    contains
        procedure :: nested_double => user_int_nested_double
        procedure :: get_instance => user_int_get_instance
        procedure :: set_instance => user_int_set_instance
        procedure :: associated => user_int_associated
        ! splicer begin class.user_int.type_bound_procedure_part
        ! splicer end class.user_int.type_bound_procedure_part
    end type user_int

    interface operator (.eq.)
        module procedure vector_int_eq
        module procedure vector_double_eq
        module procedure worker_eq
        module procedure user_int_eq
    end interface

    interface operator (.ne.)
        module procedure vector_int_ne
        module procedure vector_double_ne
        module procedure worker_ne
        module procedure user_int_ne
    end interface

    interface

        function c_vector_int_ctor(SHT_crv) &
                result(SHT_rv) &
                bind(C, name="TEM_vector_int_ctor")
            use iso_c_binding, only : C_PTR
            import :: SHROUD_vector_int_capsule
            implicit none
            type(SHROUD_vector_int_capsule), intent(OUT) :: SHT_crv
            type(C_PTR) SHT_rv
        end function c_vector_int_ctor

        subroutine c_vector_int_dtor(self) &
                bind(C, name="TEM_vector_int_dtor")
            import :: SHROUD_vector_int_capsule
            implicit none
            type(SHROUD_vector_int_capsule), intent(IN) :: self
        end subroutine c_vector_int_dtor

        subroutine c_vector_int_push_back(self, value) &
                bind(C, name="TEM_vector_int_push_back")
            use iso_c_binding, only : C_INT
            import :: SHROUD_vector_int_capsule
            implicit none
            type(SHROUD_vector_int_capsule), intent(IN) :: self
            integer(C_INT), intent(IN) :: value
        end subroutine c_vector_int_push_back

        function c_vector_int_at(self, n) &
                result(SHT_rv) &
                bind(C, name="TEM_vector_int_at")
            use iso_c_binding, only : C_INT, C_PTR, C_SIZE_T
            import :: SHROUD_vector_int_capsule
            implicit none
            type(SHROUD_vector_int_capsule), intent(IN) :: self
            integer(C_SIZE_T), value, intent(IN) :: n
            type(C_PTR) SHT_rv
        end function c_vector_int_at

        ! splicer begin class.vector_int.additional_interfaces
        ! splicer end class.vector_int.additional_interfaces

        function c_vector_double_ctor(SHT_crv) &
                result(SHT_rv) &
                bind(C, name="TEM_vector_double_ctor")
            use iso_c_binding, only : C_PTR
            import :: SHROUD_vector_double_capsule
            implicit none
            type(SHROUD_vector_double_capsule), intent(OUT) :: SHT_crv
            type(C_PTR) SHT_rv
        end function c_vector_double_ctor

        subroutine c_vector_double_dtor(self) &
                bind(C, name="TEM_vector_double_dtor")
            import :: SHROUD_vector_double_capsule
            implicit none
            type(SHROUD_vector_double_capsule), intent(IN) :: self
        end subroutine c_vector_double_dtor

        subroutine c_vector_double_push_back(self, value) &
                bind(C, name="TEM_vector_double_push_back")
            use iso_c_binding, only : C_DOUBLE
            import :: SHROUD_vector_double_capsule
            implicit none
            type(SHROUD_vector_double_capsule), intent(IN) :: self
            real(C_DOUBLE), intent(IN) :: value
        end subroutine c_vector_double_push_back

        function c_vector_double_at(self, n) &
                result(SHT_rv) &
                bind(C, name="TEM_vector_double_at")
            use iso_c_binding, only : C_DOUBLE, C_PTR, C_SIZE_T
            import :: SHROUD_vector_double_capsule
            implicit none
            type(SHROUD_vector_double_capsule), intent(IN) :: self
            integer(C_SIZE_T), value, intent(IN) :: n
            type(C_PTR) SHT_rv
        end function c_vector_double_at

        ! splicer begin class.vector_double.additional_interfaces
        ! splicer end class.vector_double.additional_interfaces

        ! splicer begin class.Worker.additional_interfaces
        ! splicer end class.Worker.additional_interfaces

        subroutine c_user_int_nested_double(self, arg1, arg2) &
                bind(C, name="TEM_user_int_nested_double")
            use iso_c_binding, only : C_DOUBLE, C_INT
            import :: SHROUD_user_int_capsule
            implicit none
            type(SHROUD_user_int_capsule), intent(IN) :: self
            integer(C_INT), value, intent(IN) :: arg1
            real(C_DOUBLE), value, intent(IN) :: arg2
        end subroutine c_user_int_nested_double

        ! splicer begin class.user_int.additional_interfaces
        ! splicer end class.user_int.additional_interfaces

        subroutine c_function_tu_0(arg1, arg2) &
                bind(C, name="TEM_function_tu_0")
            use iso_c_binding, only : C_INT, C_LONG
            implicit none
            integer(C_INT), value, intent(IN) :: arg1
            integer(C_LONG), value, intent(IN) :: arg2
        end subroutine c_function_tu_0

        subroutine c_function_tu_1(arg1, arg2) &
                bind(C, name="TEM_function_tu_1")
            use iso_c_binding, only : C_DOUBLE, C_FLOAT
            implicit none
            real(C_FLOAT), value, intent(IN) :: arg1
            real(C_DOUBLE), value, intent(IN) :: arg2
        end subroutine c_function_tu_1

        function c_use_impl_worker_internal_implworker1() &
                result(SHT_rv) &
                bind(C, name="TEM_use_impl_worker_internal_ImplWorker1")
            use iso_c_binding, only : C_INT
            implicit none
            integer(C_INT) :: SHT_rv
        end function c_use_impl_worker_internal_implworker1

        ! splicer begin additional_interfaces
        ! splicer end additional_interfaces
    end interface

    interface function_tu
        module procedure function_tu_0
        module procedure function_tu_1
    end interface function_tu

contains

    function vector_int_ctor() &
            result(SHT_rv)
        use iso_c_binding, only : C_PTR
        type(C_PTR) :: SHT_prv
        type(vector_int) :: SHT_rv
        ! splicer begin class.vector_int.method.ctor
        SHT_prv = c_vector_int_ctor(SHT_rv%cxxmem)
        ! splicer end class.vector_int.method.ctor
    end function vector_int_ctor

    subroutine vector_int_dtor(obj)
        class(vector_int) :: obj
        ! splicer begin class.vector_int.method.dtor
        call c_vector_int_dtor(obj%cxxmem)
        ! splicer end class.vector_int.method.dtor
    end subroutine vector_int_dtor

    subroutine vector_int_push_back(obj, value)
        use iso_c_binding, only : C_INT
        class(vector_int) :: obj
        integer(C_INT), intent(IN) :: value
        ! splicer begin class.vector_int.method.push_back
        call c_vector_int_push_back(obj%cxxmem, value)
        ! splicer end class.vector_int.method.push_back
    end subroutine vector_int_push_back

    function vector_int_at(obj, n) &
            result(SHT_rv)
        use iso_c_binding, only : C_INT, C_PTR, C_SIZE_T, c_f_pointer
        class(vector_int) :: obj
        integer(C_SIZE_T), value, intent(IN) :: n
        integer(C_INT), pointer :: SHT_rv
        type(C_PTR) :: SHT_ptr
        ! splicer begin class.vector_int.method.at
        SHT_ptr = c_vector_int_at(obj%cxxmem, n)
        call c_f_pointer(SHT_ptr, SHT_rv)
        ! splicer end class.vector_int.method.at
    end function vector_int_at

    ! Return pointer to C++ memory.
    function vector_int_get_instance(obj) result (cxxptr)
        use iso_c_binding, only: C_PTR
        class(vector_int), intent(IN) :: obj
        type(C_PTR) :: cxxptr
        cxxptr = obj%cxxmem%addr
    end function vector_int_get_instance

    subroutine vector_int_set_instance(obj, cxxmem)
        use iso_c_binding, only: C_PTR
        class(vector_int), intent(INOUT) :: obj
        type(C_PTR), intent(IN) :: cxxmem
        obj%cxxmem%addr = cxxmem
        obj%cxxmem%idtor = 0
    end subroutine vector_int_set_instance

    function vector_int_associated(obj) result (rv)
        use iso_c_binding, only: c_associated
        class(vector_int), intent(IN) :: obj
        logical rv
        rv = c_associated(obj%cxxmem%addr)
    end function vector_int_associated

    ! splicer begin class.vector_int.additional_functions
    ! splicer end class.vector_int.additional_functions

    function vector_double_ctor() &
            result(SHT_rv)
        use iso_c_binding, only : C_PTR
        type(C_PTR) :: SHT_prv
        type(vector_double) :: SHT_rv
        ! splicer begin class.vector_double.method.ctor
        SHT_prv = c_vector_double_ctor(SHT_rv%cxxmem)
        ! splicer end class.vector_double.method.ctor
    end function vector_double_ctor

    subroutine vector_double_dtor(obj)
        class(vector_double) :: obj
        ! splicer begin class.vector_double.method.dtor
        call c_vector_double_dtor(obj%cxxmem)
        ! splicer end class.vector_double.method.dtor
    end subroutine vector_double_dtor

    subroutine vector_double_push_back(obj, value)
        use iso_c_binding, only : C_DOUBLE
        class(vector_double) :: obj
        real(C_DOUBLE), intent(IN) :: value
        ! splicer begin class.vector_double.method.push_back
        call c_vector_double_push_back(obj%cxxmem, value)
        ! splicer end class.vector_double.method.push_back
    end subroutine vector_double_push_back

    function vector_double_at(obj, n) &
            result(SHT_rv)
        use iso_c_binding, only : C_DOUBLE, C_PTR, C_SIZE_T, c_f_pointer
        class(vector_double) :: obj
        integer(C_SIZE_T), value, intent(IN) :: n
        real(C_DOUBLE), pointer :: SHT_rv
        type(C_PTR) :: SHT_ptr
        ! splicer begin class.vector_double.method.at
        SHT_ptr = c_vector_double_at(obj%cxxmem, n)
        call c_f_pointer(SHT_ptr, SHT_rv)
        ! splicer end class.vector_double.method.at
    end function vector_double_at

    ! Return pointer to C++ memory.
    function vector_double_get_instance(obj) result (cxxptr)
        use iso_c_binding, only: C_PTR
        class(vector_double), intent(IN) :: obj
        type(C_PTR) :: cxxptr
        cxxptr = obj%cxxmem%addr
    end function vector_double_get_instance

    subroutine vector_double_set_instance(obj, cxxmem)
        use iso_c_binding, only: C_PTR
        class(vector_double), intent(INOUT) :: obj
        type(C_PTR), intent(IN) :: cxxmem
        obj%cxxmem%addr = cxxmem
        obj%cxxmem%idtor = 0
    end subroutine vector_double_set_instance

    function vector_double_associated(obj) result (rv)
        use iso_c_binding, only: c_associated
        class(vector_double), intent(IN) :: obj
        logical rv
        rv = c_associated(obj%cxxmem%addr)
    end function vector_double_associated

    ! splicer begin class.vector_double.additional_functions
    ! splicer end class.vector_double.additional_functions

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

    subroutine user_int_nested_double(obj, arg1, arg2)
        use iso_c_binding, only : C_DOUBLE, C_INT
        class(user_int) :: obj
        integer(C_INT), value, intent(IN) :: arg1
        real(C_DOUBLE), value, intent(IN) :: arg2
        ! splicer begin class.user_int.method.nested_double
        call c_user_int_nested_double(obj%cxxmem, arg1, arg2)
        ! splicer end class.user_int.method.nested_double
    end subroutine user_int_nested_double

    ! Return pointer to C++ memory.
    function user_int_get_instance(obj) result (cxxptr)
        use iso_c_binding, only: C_PTR
        class(user_int), intent(IN) :: obj
        type(C_PTR) :: cxxptr
        cxxptr = obj%cxxmem%addr
    end function user_int_get_instance

    subroutine user_int_set_instance(obj, cxxmem)
        use iso_c_binding, only: C_PTR
        class(user_int), intent(INOUT) :: obj
        type(C_PTR), intent(IN) :: cxxmem
        obj%cxxmem%addr = cxxmem
        obj%cxxmem%idtor = 0
    end subroutine user_int_set_instance

    function user_int_associated(obj) result (rv)
        use iso_c_binding, only: c_associated
        class(user_int), intent(IN) :: obj
        logical rv
        rv = c_associated(obj%cxxmem%addr)
    end function user_int_associated

    ! splicer begin class.user_int.additional_functions
    ! splicer end class.user_int.additional_functions

    !>
    !! \brief Function template with two template parameters.
    !!
    !<
    subroutine function_tu_0(arg1, arg2)
        use iso_c_binding, only : C_INT, C_LONG
        integer(C_INT), value, intent(IN) :: arg1
        integer(C_LONG), value, intent(IN) :: arg2
        ! splicer begin function.function_tu_0
        call c_function_tu_0(arg1, arg2)
        ! splicer end function.function_tu_0
    end subroutine function_tu_0

    !>
    !! \brief Function template with two template parameters.
    !!
    !<
    subroutine function_tu_1(arg1, arg2)
        use iso_c_binding, only : C_DOUBLE, C_FLOAT
        real(C_FLOAT), value, intent(IN) :: arg1
        real(C_DOUBLE), value, intent(IN) :: arg2
        ! splicer begin function.function_tu_1
        call c_function_tu_1(arg1, arg2)
        ! splicer end function.function_tu_1
    end subroutine function_tu_1

    !>
    !! \brief Function which uses a templated T in the implemetation.
    !!
    !<
    function use_impl_worker_internal_ImplWorker1() &
            result(SHT_rv)
        use iso_c_binding, only : C_INT
        integer(C_INT) :: SHT_rv
        ! splicer begin function.use_impl_worker_internal_ImplWorker1
        SHT_rv = c_use_impl_worker_internal_implworker1()
        ! splicer end function.use_impl_worker_internal_ImplWorker1
    end function use_impl_worker_internal_ImplWorker1

    ! splicer begin additional_functions
    ! splicer end additional_functions

    function vector_int_eq(a,b) result (rv)
        use iso_c_binding, only: c_associated
        type(vector_int), intent(IN) ::a,b
        logical :: rv
        if (c_associated(a%cxxmem%addr, b%cxxmem%addr)) then
            rv = .true.
        else
            rv = .false.
        endif
    end function vector_int_eq

    function vector_int_ne(a,b) result (rv)
        use iso_c_binding, only: c_associated
        type(vector_int), intent(IN) ::a,b
        logical :: rv
        if (.not. c_associated(a%cxxmem%addr, b%cxxmem%addr)) then
            rv = .true.
        else
            rv = .false.
        endif
    end function vector_int_ne

    function vector_double_eq(a,b) result (rv)
        use iso_c_binding, only: c_associated
        type(vector_double), intent(IN) ::a,b
        logical :: rv
        if (c_associated(a%cxxmem%addr, b%cxxmem%addr)) then
            rv = .true.
        else
            rv = .false.
        endif
    end function vector_double_eq

    function vector_double_ne(a,b) result (rv)
        use iso_c_binding, only: c_associated
        type(vector_double), intent(IN) ::a,b
        logical :: rv
        if (.not. c_associated(a%cxxmem%addr, b%cxxmem%addr)) then
            rv = .true.
        else
            rv = .false.
        endif
    end function vector_double_ne

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

    function user_int_eq(a,b) result (rv)
        use iso_c_binding, only: c_associated
        type(user_int), intent(IN) ::a,b
        logical :: rv
        if (c_associated(a%cxxmem%addr, b%cxxmem%addr)) then
            rv = .true.
        else
            rv = .false.
        endif
    end function user_int_eq

    function user_int_ne(a,b) result (rv)
        use iso_c_binding, only: c_associated
        type(user_int), intent(IN) ::a,b
        logical :: rv
        if (.not. c_associated(a%cxxmem%addr, b%cxxmem%addr)) then
            rv = .true.
        else
            rv = .false.
        endif
    end function user_int_ne

end module templates_mod
