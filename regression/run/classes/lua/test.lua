-- Copyright (c) 2017-2020, Lawrence Livermore National Security, LLC and
-- other Shroud Project Developers.
-- See the top-level COPYRIGHT file for details.
--
-- SPDX-License-Identifier: (BSD-3-Clause)
--
-- #######################################################################
-- test classes module

local classes = require "classes"
local rv_int, rv_double, rv_logical, rv_char

# moved to classes.yaml
-- call a class
local obj = classes.Class1()
obj:Method1()
print(classes.LastFunctionCalled())

--XXX    classes.useclass(obj)
