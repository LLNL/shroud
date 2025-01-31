.. Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
   other Shroud Project Developers.
   See the top-level COPYRIGHT file for details.

   SPDX-License-Identifier: (BSD-3-Clause)

Expand
======

The wrapping features of Shroud are controlled by data files which are
read in upon startup.  This allows Shroud to wrap many types of
arguments without any additional input from the user.  However, there
will always be situations that require some additional ability.  This
section describes the input data files and how a user may add to them.

Shroud can be thought of as a fancy macro processor.  It reads the
input and performs lots of redundant, tedious replacements. One
function argument causes several layers of code to be generated which
involve transformations of the input. To help manage the redundency,
several layers of abstraction are provided.

The lowest layer is the :ref:`typemap <TypemapsAnchor>`.
It factors out some common fields for individual types.

The next layer is the format dictionary.  This is created for each
argument for each language. Some values are computed from the argument
description and attribute. Other values can be added directly by the
user in the YAML file.

Finally, :ref:`statements <StatementsAnchor>` use format entries to
generate code at the many locations required to create the wrappers.
