# Copyright Shroud Project Developers. See LICENSE file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
########################################################################

import unittest

from shroud import ast, error, fcfmt, statements, util

error.get_cursor()

class WFormat(unittest.TestCase):
    def test_func_void(self):
        library = ast.LibraryNode()
        func = library.add_function("void func1(void)")

        bind = statements.fetch_func_bind(func, "c")
        statements.set_bind_fmtdict(bind, None)
        fmtdict = bind.fmtdict

        fmtdict.typemap = func.ast.typemap
        fmtdict.c_var = "rv"
        fmtdict.cxx_var = "rv_cxx"
        fmtdict.other = "other_name"

        gen = fcfmt.FormatGen(func, func.ast, bind, "c")
        self.assertEqual("func1", str(gen))
        self.assertEqual("func1", gen.name)
        self.assertEqual("cxx", gen.language)

        self.assertEqual("void(\tvoid)",
                         str(gen.cdecl))
        self.assertEqual("void rv(\tvoid)",
                         str(gen.cdecl.c_var))
        self.assertEqual("void other_name(\tvoid)",
                         str(gen.cdecl.other))

        self.assertEqual("void (void)",
                         str(gen.cxxdecl))
        self.assertEqual("void other_name(\tvoid)",
                         str(gen.cxxdecl.other))
        self.assertEqual("void ===>xxx<===(\tvoid)",
                         gen.cxxdecl.xxx)

        self.assertEqual("void (void)",
                         str(gen.cxxresult))
        self.assertEqual("void other_name",
                         str(gen.cxxresult.other))

    def test_func_voidptr(self):
        library = ast.LibraryNode()
        func = library.add_function("void *func1(void)")

        bind = statements.fetch_func_bind(func, "c")
        statements.set_bind_fmtdict(bind, None)
        fmtdict = bind.fmtdict

        fmtdict.typemap = func.ast.typemap
        fmtdict.c_var = "rv"
        fmtdict.cxx_var = "rv_cxx"
        fmtdict.other = "other_name"

        gen = fcfmt.FormatGen(func, func.ast, bind, "c")

        self.assertEqual("void *(\tvoid)",
                         str(gen.cdecl))
        self.assertEqual("void *rv(\tvoid)",
                         str(gen.cdecl.c_var))
        self.assertEqual("void *other_name(\tvoid)",
                         str(gen.cdecl.other))

        self.assertEqual("void *(void)",
                         str(gen.cxxdecl))
        self.assertEqual("void *other_name(\tvoid)",
                         str(gen.cxxdecl.other))
        self.assertEqual("void *===>xxx<===(\tvoid)",
                         gen.cxxdecl.xxx)

        self.assertEqual("void *(void)",
                         str(gen.cxxresult))
        self.assertEqual("void *other_name",
                         str(gen.cxxresult.other))
        
    def test_arg_cxx_int(self):
        library = ast.LibraryNode()
        func = library.add_function("void func1(int *array)")

        arg = func.ast.declarator.params[0]
        bind = statements.fetch_arg_bind(func, arg, "c")
        statements.set_bind_fmtdict(bind, None)
        fmtdict = bind.fmtdict

        fmtdict.typemap = arg.typemap
        fmtdict.c_var = "arg1"
        fmtdict.cxx_var = "cxx_var_name"
        fmtdict.other = "other_name"

        fmtarg = fcfmt.FormatGen(func, arg, bind, "c")
        self.assertEqual("array", str(fmtarg))
        self.assertEqual("array", fmtarg.name)
        self.assertEqual("cxx", fmtarg.language)
#        print(4, fmtarg.nonconst_addr)
#        print(5, fmtarg.nonconst_addr.cxx_var)
#        print(6, fmtarg.nonconst_addr.cxx_other)
#        print(6, fmtarg.nonconst_addr.cxx_local)

#        print(7, fmtarg.no_such_attr)

#        print(8, fmtarg.tester)
#        print(10, fmtarg.__name)

        self.assertEqual("int *",
                         str(fmtarg.cdecl))
        self.assertEqual("int *arg1",
                         str(fmtarg.cdecl.c_var))
        self.assertEqual("int *other_name",
                         str(fmtarg.cdecl.other))
        self.assertEqual("int *",
                         str(fmtarg.cxxdecl))
        self.assertEqual("int *other_name",
                         str(fmtarg.cxxdecl.other))
        self.assertEqual("int *===>xxx<===",
                         fmtarg.cxxdecl.xxx)

        # cidecl
        self.assertEqual("int *array", fmtarg.cidecl.c_var)
#        print(11, fmtarg.cxxdecl.cxx_var)

        fmtarg = fcfmt.FormatGen(func, arg, bind, "f")

    def test_arg_cxx_const_int(self):
        library = ast.LibraryNode()
        func = library.add_function("void func1(const int *array)")

        arg = func.ast.declarator.params[0]
        bind = statements.fetch_arg_bind(func, arg, "c")
        statements.set_bind_fmtdict(bind, None)
        fmtdict = bind.fmtdict

        fmtdict.typemap = arg.typemap
        fmtdict.c_var = "arg1"
        fmtdict.cxx_var = "cxx_var_name"
        fmtdict.other = "other_name"

        fmtarg = fcfmt.FormatGen(func, arg, bind, "c")
        self.assertEqual("array", str(fmtarg))
        self.assertEqual("array", fmtarg.name)
        self.assertEqual("const int *",
                         str(fmtarg.cdecl))
        self.assertEqual("const int *arg1",
                         str(fmtarg.cdecl.c_var))
        self.assertEqual("const int *other_name",
                         str(fmtarg.cdecl.other))
        self.assertEqual("const int *",
                         str(fmtarg.cxxdecl))
        self.assertEqual("const int *other_name",
                         fmtarg.cxxdecl.other)
        self.assertEqual("const int *===>xxx<===",
                         fmtarg.cxxdecl.xxx)

    def test_arg_funptr(self):
        library = ast.LibraryNode()
        func = library.add_function("int callback1(int (*incr)(int))")

        arg = func.ast.declarator.params[0]
        bind = statements.fetch_arg_bind(func, arg, "c")
        statements.set_bind_fmtdict(bind, None)
        fmtdict = bind.fmtdict

        fmtdict.typemap = arg.typemap
        fmtdict.c_var = "arg1"
        fmtdict.cxx_var = "cxx_var_name"
        fmtdict.other = "other_name"

        gen = fcfmt.FormatGen(func, arg, bind, "c")
        self.assertEqual("incr", str(gen))
        self.assertEqual("incr", gen.name)

        self.assertEqual("int (*)(\tint)",
                         str(gen.cdecl))
        self.assertEqual("int (*arg1)(\tint)",
                         str(gen.cdecl.c_var))
        self.assertEqual("int (*other_name)(\tint)",
                         str(gen.cdecl.other))

        self.assertEqual("int (*)(int)",
                         str(gen.cxxdecl))
        self.assertEqual("int (*cxx_var_name)(\tint)",
                         gen.cxxdecl.cxx_var)
        self.assertEqual("int (*===>xxx<===)(\tint)",
                         gen.cxxdecl.xxx)

        self.assertEqual("int (*)(int)",
                         str(gen.cxxresult))
        self.assertEqual("int (*cxx_var_name)",
                         gen.cxxresult.cxx_var)
        
    def test_arg_cxx_enum(self):
        library = ast.LibraryNode()
        enum = library.add_enum("enum Color {RED}",
                                options=dict(F_enum_type="short"))
        func = library.add_function("void func1(enum Color arg1)")

        arg = func.ast.declarator.params[0]
        bind = statements.fetch_arg_bind(func, arg, "c")
        statements.set_bind_fmtdict(bind, None)
        fmtdict = bind.fmtdict

        fmtdict.typemap = arg.typemap
        fmtdict.c_var = "arg1"
        fmtdict.cxx_var = "cxx_var_name"
        fmtdict.other = "other_name"

        fmtarg = fcfmt.FormatGen(func, arg, bind, "c")
        self.assertEqual("arg1", str(fmtarg))
        self.assertEqual("arg1", fmtarg.name)
        self.assertEqual("enum LIB_Color",
                         str(fmtarg.cdecl))
        self.assertEqual("enum LIB_Color arg1",
                         str(fmtarg.cdecl.c_var))
        self.assertEqual("enum LIB_Color other_name",
                         str(fmtarg.cdecl.other))
        self.assertEqual("enum Color",
                         str(fmtarg.cxxdecl))
        self.assertEqual("Color cxx_var_name",
                         fmtarg.cxxdecl.cxx_var)
        self.assertEqual("Color ===>xxx<===",
                         fmtarg.cxxdecl.xxx)
    
        # cidecl
        self.assertEqual("enum LIB_Color arg1", fmtarg.cidecl.c_var)

        fmtarg = fcfmt.FormatGen(func, arg, bind, "f")
        self.assertEqual("short arg1", fmtarg.cidecl.c_var)

    def test_arg_cxx_vector(self):
        library = ast.LibraryNode()
        func = library.add_function("void func1(vector<int> *array)")

        arg = func.ast.declarator.params[0]
        bind = statements.fetch_arg_bind(func, arg, "c")
        statements.set_bind_fmtdict(bind, None)
        fmtdict = bind.fmtdict

        fmtdict.typemap = arg.typemap
        fmtdict.c_var = "arg1"
        fmtdict.cxx_var = "cxx_var_name"
        fmtdict.other = "other_name"

        fmtarg = fcfmt.FormatGen(func, arg, bind, "c")
        self.assertEqual("array", str(fmtarg))
        self.assertEqual("array", fmtarg.name)
        self.assertEqual("int *",
                         str(fmtarg.cdecl))
        self.assertEqual("int *arg1",
                         str(fmtarg.cdecl.c_var))
        self.assertEqual("int *other_name",
                         str(fmtarg.cdecl.other))
        self.assertEqual("vector<int> *",
                         str(fmtarg.cxxdecl))
        self.assertEqual("std::vector<int> *other_name",
                         fmtarg.cxxdecl.other)
        self.assertEqual("std::vector<int> *===>xxx<===",
                         fmtarg.cxxdecl.xxx)
    
    def test_arg_dimension(self):
        library = ast.LibraryNode()
        func = library.add_function(
            "void DimensionIn(const int *arg +dimension(10,20))")

        arg = func.ast.declarator.params[0]
        bind = statements.fetch_arg_bind(func, arg, "c")
        statements.set_bind_fmtdict(bind, None)
        fmtdict = bind.fmtdict

        # Empty fmtdict
        fmtarg = fcfmt.FormatGen(func, arg, bind, "c")
        self.assertEqual("",
                         fmtarg.f_allocate_shape)
        self.assertEqual("",
                         fmtarg.c_f_pointer_shape)
        self.assertEqual("",
                         fmtarg.f_cdesc_shape)

        # No f_var_cdesc
        fmtdict.rank = 1
        fmtarg = fcfmt.FormatGen(func, arg, bind, "c")
        self.assertEqual("(===>f_var_cdesc<===%shape(1))",
                         fmtarg.f_allocate_shape)
        self.assertEqual(",\t ===>f_var_cdesc<===%shape(1:1)",
                         fmtarg.c_f_pointer_shape)
        self.assertEqual("\n===>f_var_cdesc<===%shape(1:1) = shape(===>f_var<===)",
                         fmtarg.f_cdesc_shape)

        # scalar
        fmtdict.rank = 0
        fmtarg = fcfmt.FormatGen(func, arg, bind, "c")
        self.assertEqual("",
                         fmtarg.f_allocate_shape)
        self.assertEqual("",
                         fmtarg.c_f_pointer_shape)
        self.assertEqual("",
                         fmtarg.f_cdesc_shape)

        # 2-d array
        fmtdict.rank = 2
        fmtdict.f_var_cdesc = "SHT_arg_cdesc"
        fmtarg = fcfmt.FormatGen(func, arg, bind, "c")
        self.assertEqual("(SHT_arg_cdesc%shape(1),SHT_arg_cdesc%shape(2))",
                         fmtarg.f_allocate_shape)
        self.assertEqual(",\t SHT_arg_cdesc%shape(1:2)",
                         fmtarg.c_f_pointer_shape)
        self.assertEqual("\nSHT_arg_cdesc%shape(1:2) = shape(===>f_var<===)",
                         fmtarg.f_cdesc_shape)

    def test_arg_c_dimension(self):
        library = ast.LibraryNode()
        func = library.add_function(
            "void DimensionIn(const int *arg +dimension(10,20))")

        arg = func.ast.declarator.params[0]
        bind = statements.fetch_arg_bind(func, arg, "c")
        statements.set_bind_fmtdict(bind, None)
        meta = bind.meta
        fmtdict = bind.fmtdict

        # Scalar
        fmtarg = fcfmt.FormatGen(func, arg, bind, "c")
        self.assertEqual("1",
                         fmtarg.c_dimension_size)
        self.assertEqual("",
                         fmtarg.c_array_shape)
        self.assertEqual("1",
                         fmtarg.c_array_size)
        self.assertEqual("",
                         fmtarg.c_extents_decl)
        self.assertEqual("NULL",
                         fmtarg.c_extents_use)
        self.assertEqual("NULL",
                         fmtarg.c_lower_use)
        
        # No c_var_cdesc
        meta["dim_shape"] = ["10"]
        fmtarg = fcfmt.FormatGen(func, arg, bind, "c")
        self.assertEqual("(10)",
                         fmtarg.c_dimension_size)
        self.assertEqual("\n===>c_var_cdesc<===->shape[0] = 10;",
                         fmtarg.c_array_shape)
        self.assertEqual("===>c_var_cdesc<===->shape[0]",
                         fmtarg.c_array_size)
        self.assertEqual("CFI_index_t ===>c_local_extents<===[] = {10};\n",
                         fmtarg.c_extents_decl)
        self.assertEqual("===>c_local_extents<===",
                         fmtarg.c_extents_use)
        self.assertEqual("===>c_helper_lower_bounds_CFI<===",
                         fmtarg.c_lower_use)

        # 2-d array
        fmtdict.c_var_cdesc = "SHT"
        fmtdict.c_local_extents = "SHT_extents"
        fmtdict.c_helper_lower_bounds_CFI = "SHT_lower"
        meta["dim_shape"] = ["10", "20"]
        fmtarg = fcfmt.FormatGen(func, arg, bind, "c")
        self.assertEqual("(10)*(20)",
                         fmtarg.c_dimension_size)
        self.assertEqual("\nSHT->shape[0] = 10;\nSHT->shape[1] = 20;",
                         fmtarg.c_array_shape)
        self.assertEqual("SHT->shape[0]*\tSHT->shape[1]",
                         fmtarg.c_array_size)
        self.assertEqual("CFI_index_t SHT_extents[] = {10,\t 20};\n",
                         fmtarg.c_extents_decl)
        self.assertEqual("SHT_extents",
                         fmtarg.c_extents_use)
        self.assertEqual("SHT_lower",
                         fmtarg.c_lower_use)

if __name__ == "__main__":
    unittest.main()
