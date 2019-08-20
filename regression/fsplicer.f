! Copyright (c) 2017-2019, Lawrence Livermore National Security, LLC. 
!
! Produced at the Lawrence Livermore National Laboratory 
!
! LLNL-CODE-738041.
!
! All rights reserved. 
!
! This file is part of Shroud.
!
! For details about use and distribution, please read LICENSE.
!
! #######################################################################


! splicer begin module_top
top of module library splicer  1
! splicer end module_top

! splicer begin class.ExClass1.component_part
  component part 1a
  component part 1b
! splicer end   class.ExClass1.component_part

! splicer begin class.ExClass1.type_bound_procedure_part
  type bound procedure part 1
! splicer end   class.ExClass1.type_bound_procedure_part

! splicer begin class.ExClass1.method.splicer_special
blah blah blah
! splicer end class.ExClass1.method.splicer_special

! splicer begin class.ExClass1.extra_methods
  insert extra methods here
! splicer end   class.ExClass1.extra_methods






# test a full path
! splicer begin  class.ExClass1.method.extra_method2
  ! extra method 2
! splicer end    class.ExClass1.method.extra_method2
