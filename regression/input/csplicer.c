// Copyright (c) 2017-2024, Lawrence Livermore National Security, LLC and
// other Shroud Project Developers.
// See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
// ########################################################################
//
// code to test C splicers
//

# XXX this splicer is not used.  It has the C++ name, not the C wrapper name.
# XXX That used to work...
#// splicer begin  namespace.example::nested.class.ExClass1.method.SplicerSpecial
#//   splicer for SplicerSpecial
#// splicer end    namespace.example::nested.class.ExClass1.method.SplicerSpecial

// splicer begin  namespace.example::nested.class.ExClass1.method.splicer_special
//   splicer for SplicerSpecial
// splicer end    namespace.example::nested.class.ExClass1.method.splicer_special


// splicer begin CXX_definitions
//   CXX_definitions
// splicer end   CXX_definitions

// splicer begin namespace.example::nested.class.ExClass1.CXX_definitions
//   namespace.example::nested.class.ExClass1.CXX_definitions
// splicer end   namespace.example::nested.class.ExClass1.CXX_definitions

// splicer begin namespace.example::nested.class.ExClass2.CXX_definitions
//   namespace.example::nested.class.ExClass2.CXX_definitions
// splicer end   namespace.example::nested.class.ExClass2.CXX_definitions
