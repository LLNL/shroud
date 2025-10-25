!
! Test compiler final and assign interaction
!

module test_mod

    use iso_c_binding

    type object
       character(20) :: msg = "none"
    contains
        final :: object_final
    end type object

    interface object
       module procedure object_create
    end interface object

    interface assignment (=)
        module procedure object_assign
    end interface
    
contains

  function object_create(msg) result(obj)
        character(*) msg
        type(object) :: obj
        obj%msg = msg
        print *, "Called object_create: ", obj%msg
    end function object_create

    subroutine object_assign(lhs, rhs)
        class(object), intent(INOUT) :: lhs
        type(object), intent(IN) :: rhs
        print *, "Called object_assign: ", lhs%msg, "=", rhs%msg
        lhs%msg = "assign "//rhs%msg
    end subroutine object_assign

    subroutine object_final(obj)
        type(object), intent(INOUT) :: obj
        print *, "Called object_final: ", obj%msg
    end subroutine object_final

end module test_mod

program test_final
    use iso_c_binding
    use test_mod
    implicit none

    call scope

contains

    ! Create a scope to cause o2 final to be called.
    subroutine scope
      type(object) :: o1, o2

      o1 = object("1")
      o2 = object("2")
      
      print *, "o1:", o1%msg
      print *, "o2:", o2%msg
   
      ! Test base assignment
      print *, "Test: object = object"
      o1 = o2
      print *, "Test: after assignment"
    end subroutine scope

end program test_final
