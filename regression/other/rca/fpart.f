#define DEBUG(msg) print *, msg
#if 1
#define DEBUG_UNIVERSAL(msg, obj) call obj%debug(msg)
#define DEBUG_HERMETIC(msg, obj) call obj%debug(msg)
#define DEBUG_REF(msg, obj) call obj%debug(msg)
#define DEBUG_VECTOR(msg, obj) print *, "vector - ", msg, obj%unum
#else
#define DEBUG_UNIVERSAL(msg, obj)
#define DEBUG_HERMETIC(msg, obj)
#define DEBUG_REF(msg, obj)
#define DEBUG_VECTOR(msg, obj)
#endif

!#define XLF

module hermetic_interface
  private
  public :: hermetic
  type, abstract :: hermetic
     integer :: num = 0
   contains
     procedure(free_memory), deferred :: cpp_delete
     procedure :: debug
  end type hermetic

  abstract interface
     subroutine free_memory (this)
       import :: hermetic
       class(hermetic), intent(inout) :: this
     end subroutine free_memory
  end interface

contains
  
  subroutine debug(this, msg)
    class(hermetic), intent(in) :: this
    character(*), intent(in) :: msg
    print *, "hermetic - ", msg, this%num
  end subroutine debug

end module hermetic_interface


module ref_counter_implementation
  use hermetic_interface, only : hermetic
  private
  public :: ref_counter
  type ref_counter
     private
     integer :: num = -100
     integer, pointer :: count => null()
     class(hermetic), pointer :: obj => null()
   contains
     procedure, non_overridable :: grab
     procedure, non_overridable :: release
     procedure :: assign
     generic :: assignment(=) => assign
     final :: finalize_ref_counter
     procedure :: debug
  end type ref_counter

  interface ref_counter
     module procedure new_ref_counter
  end interface ref_counter

  integer :: ref_counter_num = 10

contains
  function new_ref_counter(object)
    class(hermetic), intent(in) :: object
    type(ref_counter), allocatable :: new_ref_counter
    DEBUG_HERMETIC("new_ref_counter", object)
    allocate (new_ref_counter)
    new_ref_counter%num = ref_counter_num
    ref_counter_num = ref_counter_num + 1
    allocate (new_ref_counter%count, source=0)
    allocate (new_ref_counter%obj, source=object)
    DEBUG_REF("new_ref_counter", new_ref_counter)
    call new_ref_counter%grab
  end function new_ref_counter

  subroutine grab(this)
    class(ref_counter), intent(inout) :: this
    DEBUG_REF("grab", this)
    if (associated(this%count)) then
       this%count = this%count + 1
    else
       stop 'Error in grab: count not associated'
    end if
  end subroutine grab

  subroutine release(this)
    class (ref_counter), intent(inout) :: this
    if (associated(this%count)) then
       this%count = this%count - 1
       DEBUG_REF("decref", this)
       if (this%count == 0) then
          DEBUG_REF("release", this)
          call this%obj%cpp_delete
          deallocate (this%count, this%obj)
       else
          this%count => null()
          this%obj => null()
       end if
    else
       stop 'Error in release: count not associated'
    end if
  end subroutine release

  subroutine assign (lhs, rhs)
    class (ref_counter), intent(inout) :: lhs
    class (ref_counter), intent(in) :: rhs
    DEBUG_REF("assign lhs", lhs)
    DEBUG_REF("assign rhs", rhs)
    lhs%count =>rhs%count
    lhs%obj =>rhs%obj
    call lhs%grab
  end subroutine assign

  recursive subroutine finalize_ref_counter (this)
    type(ref_counter), intent(inout) :: this
    if (associated(this%count)) call this%release
  end subroutine finalize_ref_counter

  subroutine debug(this, msg)
    class(ref_counter), intent(in) :: this
    character(*), intent(in) :: msg
    if (associated(this%count)) then
       print *, "ref_count - ", msg, this%num, this%count
    else
       print *, "ref_count - ", msg, this%num
    endif
  end subroutine debug
  
end module ref_counter_implementation



module universal_interface
  use hermetic_interface, only: hermetic
  use ref_counter_implementation, only: ref_counter
  implicit none
  private
  public :: universal

  type, abstract, extends(hermetic) :: universal
     integer :: num1 = 0
     type(ref_counter) :: counter
   contains
     procedure, non_overridable :: force_finalize
     procedure, non_overridable :: register_self
     procedure :: debug
  end type universal
  
contains

  subroutine force_finalize (this)
    class(universal), intent(inout) :: this
    DEBUG_UNIVERSAL("force_finalize", this)
    call this%counter%release
  end subroutine force_finalize

  subroutine register_self (this)
    class(universal), intent(inout) :: this
    DEBUG_UNIVERSAL("register_self", this)
    this%counter = ref_counter(this)
    DEBUG_REF("register_self", this%counter)
  end subroutine register_self

  subroutine debug(this, msg)
    class(universal), intent(in) :: this
    character(*), intent(in) :: msg
    print *, "universal - ", msg, this%num, this%num1
  end subroutine debug
  
end module universal_interface


module faux_cpp_server
  implicit none
  private
#ifdef XLF
  public specific_new_vector, specific_duplicate_vector, cpp_delete_vector
#else
  public cpp_new_vector, cpp_delete_vector
#endif

  interface
     subroutine cpp_delete_vector(id) bind(C, name="VEC_delete_vector")
       use iso_c_binding
       implicit none
       integer(c_int), value, intent(IN) :: id
     end subroutine cpp_delete_vector

#ifdef XLF
     function specific_new_vector(x,y,z) bind(C, name="VEC_new_vector") result(id)
       use iso_c_binding
       implicit none
       real(c_double), value, intent(IN) :: x, y, z
       integer(c_int) :: id
     end function specific_new_vector

     function specific_duplicate_vector(id_in) bind(C, name="VEC_duplicate_vector") result(id)
       use iso_c_binding
       integer(c_int), value :: id_in
       integer(c_int) :: id
     end function specific_duplicate_vector
#endif
  end interface

#ifndef XLF
  interface cpp_new_vector
     function local_new_vector(x,y,z) bind(C, name="VEC_new_vector") result(id)
       use iso_c_binding
       real(c_double), value, intent(IN) :: x, y, z
       integer(c_int) :: id
     end function local_new_vector

     function local_duplicate_vector(id_in) bind(C, name="VEC_duplicate_vector") result(id)
       use iso_c_binding
       integer(c_int), value :: id_in
       integer(c_int) :: id
     end function local_duplicate_vector
  end interface cpp_new_vector
#endif

end module faux_cpp_server



module vector_implementation
  use iso_c_binding, only : c_int,c_double
  use universal_interface, only: universal
  use faux_cpp_server
  
  implicit none ! Prevent implicit typing

  private ! Hide everything by default
  public :: vector ! Expose type, constructors, type-bound procedures

  ! Shadow object
  type, extends(universal) :: vector
     private
     integer(c_int) :: id ! C++ object identification tag
     integer :: unum = 0
   contains
!     procedure :: sum
!     procedure :: difference
!     procedure :: product
!     procedure :: ratio
!     procedure :: integral
!     generic :: operator(+) => sum
!     generic :: operator(-) => difference
!     generic :: operator(*) => product
!     generic :: operator(/) => ratio
!     generic :: operator(.integral.) => integral
     procedure :: cpp_delete => call_cpp_delete_vector
  end type vector

  ! Constructors
  interface vector
     module procedure new_vector,default_vector,duplicate
  end interface vector

  integer :: vector_num = 100
  
contains

  type(vector) function default_vector(id)
    integer(c_int), intent(in) :: id
    default_vector%id = id
    default_vector%unum = vector_num
    vector_num = vector_num + 1
    DEBUG_VECTOR("default_vector", default_vector)
    call default_vector%register_self
  end function default_vector

  type(vector) function new_vector(vec)
    real(c_double), dimension(3) :: vec
    DEBUG("new_vector")
#ifndef XLF
    new_vector = vector(cpp_new_vector(vec(1),vec(2),vec(3)))
#else
    ! 1516-044 (S) A conversion from type INTEGER is not permitted.
    new_vector%id = specific_new_vector(vec(1),vec(2),vec(3))
    call new_vector%register_self
#endif
  end function new_vector

  type(vector) function duplicate(original)
    type(vector), intent(in) :: original
    DEBUG("duplicate")
#ifndef XLF
    duplicate = vector(cpp_new_vector(original%id))
#else
    ! 1516-044 (S) A conversion from type INTEGER is not permitted.
    duplicate%id = specific_duplicate_vector(original%id)
    call duplicate%register_self
#endif
  end function duplicate

!  type(vector) function sum(lhs,rhs)
!    class(vector), intent(in) :: lhs,rhs
!    sum = vector(cpp_add_vectors(lhs%id,rhs%id))
!  end function sum
  
!  type(vector) function difference(lhs,rhs)
!    class(vector), intent(in) :: lhs,rhs
!    difference = vector(cpp_subtract_vectors(lhs%id,rhs%id))
!  end function difference

!  type(vector) function product(lhs,rhs)
!    class(vector), intent(in) :: lhs
!    real(c_double), intent(in) :: rhs
!    product = vector(cpp_rescale_vector(lhs%id,rhs))
!  end function product

!  type(vector) function ratio(lhs,rhs)
!    class(vector), intent(in) :: lhs
!    real(c_double), intent(in) :: rhs
!    ratio = vector(cpp_rescale_vector(lhs%id,1._c_double/rhs))
!  end function ratio

!  type(vector) function integral(rhs) ! Explicit Euler quadrature
!    class(vector) ,intent(in) :: rhs
!    integral = vector(rhs)
!  end function integral

  subroutine call_cpp_delete_vector(this)
    class(vector),intent(inout) :: this
    call cpp_delete_vector(this%id)
  end subroutine call_cpp_delete_vector

end module vector_implementation
