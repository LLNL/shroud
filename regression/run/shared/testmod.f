!
! testmod.f
!
! Test for overloading assignment(=) with derived classes.
! Different compilers have different needs.
! The lhs argument is class, while the rhs is type to make
! it as specific as possible.
! 

module test_mod

    type object
    end type object

    type, extends(object) :: object_shared
    end type object_shared

    type, extends(object) :: object_weak
    end type object_weak

    interface assignment(=)
       module procedure object_assign
       module procedure object_shared_assign_shared
       module procedure object_shared_assign_weak
       module procedure object_weak_assign_shared
       module procedure object_weak_assign_weak
    end interface assignment(=)

contains

    ! Parent assignment: object = object
    subroutine object_assign(lhs, rhs)
        class(object), intent(INOUT) :: lhs
        type(object), intent(IN) :: rhs
        print *, "Called: object_assign"
    end subroutine object_assign

    ! shared = shared
    subroutine object_shared_assign_shared(lhs, rhs)
        class(object_shared), intent(INOUT) :: lhs
        type(object_shared), intent(IN) :: rhs
        print *, "Called: object_shared_assign_shared"
    end subroutine object_shared_assign_shared

    ! shared = weak
    subroutine object_shared_assign_weak(lhs, rhs)
        class(object_shared), intent(INOUT) :: lhs
        type(object_weak), intent(IN) :: rhs
        print *, "Called: object_shared_assign_weak"
    end subroutine object_shared_assign_weak

    ! object_weak = object_shared
    subroutine object_weak_assign_shared(lhs, rhs)
        class(object_weak), intent(INOUT) :: lhs
        type(object_shared), intent(IN) :: rhs
        print *, "Called: object_weak_assign_shared"
    end subroutine object_weak_assign_shared

    ! object_weak = object_weak
    subroutine object_weak_assign_weak(lhs, rhs)
        class(object_weak), intent(INOUT) :: lhs
        type(object_weak), intent(IN) :: rhs
        print *, "Called: object_weak_assign_weak"
    end subroutine object_weak_assign_weak

end module test_mod

program test_assign
    use test_mod
    implicit none

    type(object) :: o1, o2
    type(object_shared) :: s1, s2
    type(object_weak) :: w1, w2

    ! Test base assignment
    print *, "Test: object = object"
    o1 = o2

    ! Test assignment for object_shared (should use parent's assignment)
    print *, "Test: object_shared = object_shared"
    s1 = s2

    ! Test assignment for object_weak = object_weak (should use parent's assignment)
    print *, "Test: object_weak = object_weak"
    w1 = w2

    ! Test custom assignment: object_weak = object_shared
    print *, "Test: object_weak = object_shared"
    w1 = s1

end program test_assign
