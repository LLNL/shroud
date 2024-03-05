-- Copyright (c) 2017-2024, Lawrence Livermore National Security, LLC and
-- other Shroud Project Developers.
-- See the top-level COPYRIGHT file for details.
--
-- SPDX-License-Identifier: (BSD-3-Clause)
--
-- #######################################################################
-- test tutorial module

local tutorial = require "tutorial"
local rv_int, rv_double, rv_logical, rv_char

tutorial.NoReturnNoArguments()
print(tutorial.LastFunctionCalled())

rv_double = tutorial.PassByValue(1.0, 4)
print(tutorial.LastFunctionCalled(), rv_double)

rv_char = tutorial.ConcatenateStrings("dog", "cat")
print(tutorial.LastFunctionCalled(), rv_char)

rv_double = tutorial.UseDefaultArguments()
-- 13.1415
print(tutorial.LastFunctionCalled(), rv_double)
rv_double = tutorial.UseDefaultArguments(11.0)
print(tutorial.LastFunctionCalled(), rv_double)
-- 11.0
rv_double = tutorial.UseDefaultArguments(11.0, false)
print(tutorial.LastFunctionCalled(), rv_double)
-- 1.0

tutorial.OverloadedFunction("name")
print(tutorial.LastFunctionCalled())
tutorial.OverloadedFunction(1)
print(tutorial.LastFunctionCalled())

--[[
tutorial.TemplateArgument(1)
print(tutorial.LastFunctionCalled())
tutorial.TemplateArgument(10.0)
print(tutorial.LastFunctionCalled())
--]]


tutorial.FortranGenericOverloaded()
print(tutorial.LastFunctionCalled())
tutorial.FortranGenericOverloaded("foo", 1.0)
print(tutorial.LastFunctionCalled())


rv_int = tutorial.UseDefaultOverload(10)
print(tutorial.LastFunctionCalled(), rv_int)
-- This should call overload (double type, int num)
-- but instead calls (int num, int offset)
-- since there is only one number type
rv_int = tutorial.UseDefaultOverload(1.0, 10)
print(tutorial.LastFunctionCalled(), rv_int)

rv_int = tutorial.UseDefaultOverload(10, 11, 12)
print(tutorial.LastFunctionCalled(), rv_int)
rv_int = tutorial.UseDefaultOverload(1.0, 10, 11, 12)
print(tutorial.LastFunctionCalled(), rv_int)

-- rv_int = tutorial.UseDefaultOverload("no such overload")
