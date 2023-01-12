! Copyright (c) 2017-2023, Lawrence Livermore National Security, LLC and
! other Shroud Project Developers.
! See the top-level COPYRIGHT file for details.
!
! SPDX-License-Identifier: (BSD-3-Clause)
! #######################################################################


! splicer begin module_top
top of module library splicer  1
! splicer end module_top

! splicer begin namespace.example.module_top
top of module namespace example splicer  2
! splicer end namespace.example.module_top

! splicer begin namespace.example::nested.module_top
top of module namespace example splicer  3
! splicer end namespace.example::nested.module_top


! splicer begin namespace.example::nested.class.ExClass1.component_part
  component part 1a
  component part 1b
! splicer end namespace.example::nested.class.ExClass1.component_part

! splicer begin namespace.example::nested.class.ExClass1.type_bound_procedure_part
  type bound procedure part 1
! splicer end   namespace.example::nested.class.ExClass1.type_bound_procedure_part

! splicer begin namespace.example::nested.class.ExClass1.method.splicer_special
blah blah blah
! splicer end namespace.example::nested.class.ExClass1.method.splicer_special

! splicer begin namespace.example::nested.class.ExClass1.additional_functions
  insert extra functions here
! splicer end   namespace.example::nested.class.ExClass1.additional_functions






# test a full path
! splicer begin  namespace.example::nested.class.ExClass1.method.extra_method2
  ! extra method 2
! splicer end    namespace.example::nested.class.ExClass1.method.extra_method2
