# Copyright (c) 2017-2024, Lawrence Livermore National Security, LLC and
# other Shroud Project Developers.
# See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)

"""
Create and manage typemaps used to convert between languages.

An initial typemap is created for struct, class, typedef in the
process of parsing.

It is then later expanded when the node in ast.py is created --
EnumNode, ClassNode.

Typemaps can also be created directly from the YAML file
in the typemaps section..
These represent types that are defined in another YAML file but 
need to be used in a different YAML file.


"""
from __future__ import print_function

from . import error
from . import util

# translation table to convert type name to flat name
# unsigned int -> unsigned_int
# vector<int>  -> vector_int
try:
    # Python 2
    from string import maketrans
    def flatten_name(name, flat_trantab = maketrans("< ", "__")):
        return name.replace("::","_").translate(flat_trantab, ">")
except:
    def flatten_name(name, flat_trantab="".maketrans("< ", "__", ">")):
        return name.replace("::","_").translate(flat_trantab)

void_typemap = None


class Typemap(object):
    """Collect fields for an argument.
    This used to be a dict but a class has better access semantics:
       i.attr vs d['attr']
    It also initializes default values to avoid  d.get('attr', default)

    c_header and cxx_header are used for interface and are written into 
    a header file for use by C or C++.
    For example, size_t uses <stddef.h> and <cstddef>.

    wrap_header is used for generated wrappers for shadow classes.
    Contains struct definitions for capsules from Fortran.

    impl_header is used for implementation, i.e. the wrap.cpp file.
    For example, std::string uses <string>. <string> should not be in
    the interface since the wrapper is a C API.

    A new typemap is created for each class, struct, and typedef.

    A new typemap is created for each templated class/struct
    instantiation:
        - decl: template<typename T> class A
          cxx_template:
          - instantiation: <int>
    """

    # Array of known keys with default values
    _order = (
        ("flat_name", None),  # Name when used by wrapper identifiers
        ("template_suffix", None),  # Name when used by wrapper identifiers
                                    # when added to class/struct format.
                    # Set from format.template_suffix in YAML for class.
        ("ntemplate_args", 0), # Number of template arguments
        ("base", "unknown"),  # Base type: 'string', 'integer', 'real', 'complex'
        ("typedef", None),  # Initialize from existing type (name of type)
                            # A Typemap instance
        ("cpp_if", None),  # C preprocessor test for c_header
        ("idtor", "0"),  # index of capsule_data destructor
        ("ast", None),  # Abstract syntax tree (typedef)
        ("cxx_type", None),  # Name of type in C++, including namespace
        ("cxx_to_c", None),  # Expression to convert from C++ to C
        # None implies {cxx_var} i.e. no conversion
        ("cxx_to_ci", None), # convert from C++ to Fortran interface (ex. enums)
        (
            "cxx_header",
            [],
        ),  # Name of C++ header file required for implementation
        ("cxx_instantiation", None), # Dict of instantiated template types.
            # None = non-templated type.
            # index example ["<int>"] = Typemap for instantiated class.
        # For example, if cxx_to_c was a function
        ("c_type", None),  # Name of type in C, used with C wrapper
        ("c_header", []),  # Name of C header file required for type
        ("c_to_cxx", None),  # Expression to convert from C to C++
        # None implies {c_var}  i.e. no conversion
        ("ci_type", None),   # C interface type
        ("c_return_code", None),
        ("f_class", None),  # Used with type-bound procedures
        ("f_type", None),  # Name of type in Fortran -- integer(C_INT)
        ("f_kind", None),  # Fortran kind            -- C_INT
        ("f_to_c", None),  # Expression to convert from Fortran to C
        ("i_type", None),  # Type for C interface    -- int
        ("i_module", None), # Fortran modules needed for interface  (dictionary)
        ("f_module_name", None), # Name of module which contains f_type
                                 # and f_derived_type and f_capsule_data_type
        ("f_derived_type", None),  # Fortran derived type name
        ("f_capsule_data_type", None),  # Fortran derived type to match C struct
        ("f_module", None),  # Fortran modules needed for type  (dictionary)
        ("f_cast", "{f_var}"),  # Expression to convert to type
                                # e.g. intrinsics such as INT and REAL.
        ("impl_header", []), # implementation header
        ("wrap_header", []), # generated wrapper header
        # Python
        ("PY_format", "O"),  # 'format unit' for PyArg_Parse
        ("PY_PyTypeObject", None),  # variable name of PyTypeObject instance
        ("PY_PyObject", None),  # typedef name of PyObject instance
        ("PY_ctor", None),  # expression to create object.
        # ex. PyFloat_FromDouble({c_deref}{c_var})
        ("PY_get", None),  # expression to create type from PyObject.
        # ex. PyFloat_AsDouble({py_var})
        ("py_ctype", None),        # returned by Py_get ex. "Py_complex"
        ("pytype_to_pyctor", None),  # Used with py_ctype, passed to PY_ctor
        ("pytype_to_cxx", None),  # Used with py_ctype
        ("cxx_to_pytype", None),  # Used with py_ctype
        # Name of converter function with prototype (PyObject *, void *).
        ("PY_to_object", None),  # PyBuild - object=converter(address)
        (
            "PY_from_object",
            None,
        ),  # PyArg_Parse - status=converter(object, address);
        ("PY_to_object_idtor", None),  # object=converter(address, idtor)
        ("PY_build_arg", None),  # argument for Py_BuildValue
        ("PY_build_format", None),  # 'format unit' for Py_BuildValue
        ("PY_struct_as", None),  # For struct - "class" or "list"
        ("PYN_typenum", None),  # NumPy typenum enumeration
        (
            "PYN_descr",
            None,
        ),  # Name of PyArray_Descr variable to describe type (for structs)
        # Lua
        ("LUA_type", "LUA_TNONE"),
        ("LUA_pop", "POP"),
        ("LUA_push", "PUSH"),
        ("LUA_statements", {}),
        ("sgroup", "unknown"),  # statement group. ex. native, string, vector
        ("sh_type", "SH_TYPE_OTHER"),
        ("cfi_type", "CFI_type_other"),
        ("is_enum", False),
        ("export", False),      # If True, export to YAML file.
        ("__line__", None),
    )

    _keyorder, _valueorder = zip(*_order)

    # valid fields
    defaults = dict(_order)

    deprecated = dict(
        # v0.14
        f_c_module="i_module",
        f_c_module_line="i_module_line",
        f_c_type="i_type",
    )

    def __init__(self, name, **kw):
        """
        Args:
            name - name of typemap.
        """
        self.name = name
        #        for key, defvalue in self.defaults.items():
        #            setattr(self, key, defvalue)
        self.__dict__.update(self.defaults)  # set all default values
        self.update(kw)
        if self.cxx_type and not self.flat_name:
            # Do not override an explicitly set value.
            self.compute_flat_name()

    def __deepcopy__(self, memo):
        """Do not deepcopy.
        Instead must explicitly create and register with a different name.
        """
        return self

    def update(self, dct):
        """Add options from dictionary to self.

        Args:
            dct - dictionary-like object.
        """
        for key, value in dct.items():
            if key in ["c_header", "cxx_header", "impl_header", "wrap_header"]:
                # Blank delimited strings to list
                if isinstance(value,list):
                    setattr(self, key, value)
                else:
                    setattr(self, key, value.split())
            elif key in self.defaults:
                setattr(self, key, value)
            elif key in self.deprecated:
                setattr(self, self.deprecated[key], value)
                cursor = error.get_cursor()
                cursor.deprecated("Typemap %s: Replacing deprecated field '%s' with '%s'" %
                                  (self.name, key, self.deprecated[key]))
            else:
                cursor = error.get_cursor()
                cursor.warning("Typemap %s: Unknown key '%s'" % (
                    self.name, key))

    def finalize(self):
        """Compute some fields based on other fields."""
        if self.cxx_type and not self.flat_name:
            # Do not override an explicitly set value.
            self.compute_flat_name()

    def XXXcopy(self):
        ntypemap = Typemap(self.name)
        ntypemap.update(self._to_dict())
        return ntypemap

    def copy_from_typemap(self, node):
        """Copy default fields from node.
        Used to update an existing Typemap.
        """
        for key, defvalue in self.defaults.items():
            value = getattr(node, key)
            setattr(self, key, value)

    def clone_as(self, name):
        """Creates a new Typemap.

        Args:
            name - name of new instance.
        """
        ntypemap = Typemap(name, **self._to_dict())
        return ntypemap

    def compute_flat_name(self):
        """Compute the flat_name.
        Can be called after cxx_type is set
        such as after clone_as.

        cxx_type will not be set for template arguments.
        Only set if None. Complex is set explicitly since
        C and C++ have totally different names  (double complex vs complex<double>)
        """
        if self.flat_name is None:
            self.flat_name = flatten_name(self.cxx_type)

    def _to_dict(self):
        """Convert instance to a dictionary for json.
        """
        # only export non-default values
        dct = {}
        for key, defvalue in self.defaults.items():
            value = getattr(self, key)
            if value is not defvalue:
                dct[key] = value
        return dct

    def __repr__(self):
        # only print non-default values
        args = []
        for key, defvalue in self.defaults.items():
            value = getattr(self, key)
            if value is not defvalue:
                if isinstance(value, str):
                    args.append("{0}='{1}'".format(key, value))
                else:
                    args.append("{0}={1}".format(key, value))
        return "Typemap('%s', " % self.name + ",".join(args) + ")"

    def __as_yaml__(self, indent, output):
        """Write out entire typemap as YAML.

        Args:
            indent -
            output -
        """
        util.as_yaml(self, self._keyorder, indent, output)

    def __export_yaml__(self, output, mode="all"):
        """Write out a subset of a wrapped type.
        Other fields can be derived from these values.

        Args:
            output -
        """
        # Temporary dictionary to allow convert on header fields.
        if mode == "all":
            order = self._keyorder
        else: # class
            # To be used by other libraries which import shadow types. 
            if self.base == "shadow":
                order = [
                    "base",
                    "wrap_header",
                ]
            elif self.base == "struct":
                order = [
                    "base",
                    "wrap_header",
                ]
            else:
                order = [
                    "base",
                    "f_kind",
                ]
            order.extend([
#                "cxx_type",  # same as the dict key
                "cxx_header",
                "c_header",
                "c_type",
                "f_module_name",
                "f_derived_type",
                "f_capsule_data_type",
                "f_to_c",
            ])
                
        util.as_yaml(self, order, output)


def default_typemap():
    def_types = dict(
        void=Typemap(
            "void",
            c_type="void",
            cxx_type="void",
            # fortran='subroutine',
            f_type="type(C_PTR)",
            f_module_name="iso_c_binding",
            i_module=dict(iso_c_binding=["C_PTR"]),
            PY_ctor="PyCapsule_New({ctor_expr}, NULL, NULL)",
            sh_type="SH_TYPE_CPTR",
            cfi_type="CFI_type_intptr_t",
            base="void",
            sgroup="void",
        ),
        short=Typemap(
            "short",
            c_type="short",
            cxx_type="short",
            f_cast="int({f_var}, C_SHORT)",
            f_type="integer(C_SHORT)",
            f_kind="C_SHORT",
            f_module_name="iso_c_binding",
            f_module=dict(iso_c_binding=["C_SHORT"]),
            PY_format="h",
            PY_ctor="PyInt_FromLong({ctor_expr})",
            PY_get="PyInt_AsLong({py_var})",
            PYN_typenum="NPY_SHORT",
            LUA_type="LUA_TNUMBER",
            LUA_pop="lua_tointeger({LUA_state_var}, {LUA_index})",
            LUA_push="lua_pushinteger({LUA_state_var}, {push_arg})",
            sgroup="native",
            base="integer",
            sh_type="SH_TYPE_SHORT",
            cfi_type="CFI_type_short",
        ),
        int=Typemap(
            "int",
            c_type="int",
            cxx_type="int",
            f_cast="int({f_var}, C_INT)",
            f_type="integer(C_INT)",
            f_kind="C_INT",
            f_module_name="iso_c_binding",
            f_module=dict(iso_c_binding=["C_INT"]),
            PY_format="i",
            PY_ctor="PyInt_FromLong({ctor_expr})",
            PY_get="PyInt_AsLong({py_var})",
            PYN_typenum="NPY_INT",
            LUA_type="LUA_TNUMBER",
            LUA_pop="lua_tointeger({LUA_state_var}, {LUA_index})",
            LUA_push="lua_pushinteger({LUA_state_var}, {push_arg})",
            sgroup="native",
            base="integer",
            sh_type="SH_TYPE_INT",
            cfi_type="CFI_type_int",
        ),
        long=Typemap(
            "long",
            c_type="long",
            cxx_type="long",
            f_cast="int({f_var}, C_LONG)",
            f_type="integer(C_LONG)",
            f_kind="C_LONG",
            f_module_name="iso_c_binding",
            f_module=dict(iso_c_binding=["C_LONG"]),
            PY_format="l",
            PY_ctor="PyInt_FromLong({ctor_expr})",
            PY_get="PyInt_AsLong({py_var})",
            PYN_typenum="NPY_LONG",
            LUA_type="LUA_TNUMBER",
            LUA_pop="lua_tointeger({LUA_state_var}, {LUA_index})",
            LUA_push="lua_pushinteger({LUA_state_var}, {push_arg})",
            sgroup="native",
            base="integer",
            sh_type="SH_TYPE_LONG",
            cfi_type="CFI_type_long",
        ),
        long_long=Typemap(
            "long_long",
            c_type="long long",
            cxx_type="long long",
            f_cast="int({f_var}, C_LONG_LONG)",
            f_type="integer(C_LONG_LONG)",
            f_kind="C_LONG_LONG",
            f_module_name="iso_c_binding",
            f_module=dict(iso_c_binding=["C_LONG_LONG"]),
            PY_format="L",
            # #- PY_ctor='PyInt_FromLong({ctor_expr})',
            PYN_typenum="NPY_LONGLONG",
            LUA_type="LUA_TNUMBER",
            LUA_pop="lua_tointeger({LUA_state_var}, {LUA_index})",
            LUA_push="lua_pushinteger({LUA_state_var}, {push_arg})",
            sgroup="native",
            base="integer",
            sh_type="SH_TYPE_LONG_LONG",
            cfi_type="CFI_type_long_long",
        ),
        unsigned_short=Typemap(
            "unsigned_short",
            c_type="unsigned short",
            cxx_type="unsigned short",
            f_cast="int({f_var}, C_SHORT)",
            f_type="integer(C_SHORT)",
            f_kind="C_SHORT",
            f_module_name="iso_c_binding",
            f_module=dict(iso_c_binding=["C_SHORT"]),
            PY_format="H",
            PY_ctor="PyInt_FromLong({ctor_expr})",
            PY_get="PyInt_AsLong({py_var})",
            PYN_typenum="NPY_SHORT",
            LUA_type="LUA_TNUMBER",
            LUA_pop="lua_tointeger({LUA_state_var}, {LUA_index})",
            LUA_push="lua_pushinteger({LUA_state_var}, {push_arg})",
            sgroup="native",
            base="integer",
            sh_type="SH_TYPE_UNSIGNED_SHORT",
            cfi_type="CFI_type_short",
        ),
        unsigned_int=Typemap(
            "unsigned_int",
            c_type="unsigned int",
            cxx_type="unsigned int",
            f_cast="int({f_var}, C_INT)",
            f_type="integer(C_INT)",
            f_kind="C_INT",
            f_module_name="iso_c_binding",
            f_module=dict(iso_c_binding=["C_INT"]),
            PY_format="I",
            PY_ctor="PyInt_FromLong({ctor_expr})",
            PY_get="PyInt_AsLong({py_var})",
            PYN_typenum="NPY_INT",
            LUA_type="LUA_TNUMBER",
            LUA_pop="lua_tointeger({LUA_state_var}, {LUA_index})",
            LUA_push="lua_pushinteger({LUA_state_var}, {push_arg})",
            sgroup="native",
            base="integer",
            sh_type="SH_TYPE_UNSIGNED_INT",
            cfi_type="CFI_type_int",
        ),
        unsigned_long=Typemap(
            "unsigned_long",
            c_type="unsigned long",
            cxx_type="unsigned long",
            f_cast="int({f_var}, C_LONG)",
            f_type="integer(C_LONG)",
            f_kind="C_LONG",
            f_module_name="iso_c_binding",
            f_module=dict(iso_c_binding=["C_LONG"]),
            PY_format="k",
            PY_ctor="PyInt_FromLong({ctor_expr})",
            PY_get="PyInt_AsLong({py_var})",
            PYN_typenum="NPY_LONG",
            LUA_type="LUA_TNUMBER",
            LUA_pop="lua_tointeger({LUA_state_var}, {LUA_index})",
            LUA_push="lua_pushinteger({LUA_state_var}, {push_arg})",
            sgroup="native",
            base="integer",
            sh_type="SH_TYPE_UNSIGNED_LONG",
            cfi_type="CFI_type_long",
        ),
        unsigned_long_long=Typemap(
            "unsigned_long_long",
            c_type="unsigned long long",
            cxx_type="unsigned long long",
            f_cast="int({f_var}, C_LONG_LONG)",
            f_type="integer(C_LONG_LONG)",
            f_kind="C_LONG_LONG",
            f_module_name="iso_c_binding",
            f_module=dict(iso_c_binding=["C_LONG_LONG"]),
            PY_format="K",
            # #- PY_ctor='PyInt_FromLong({ctor_expr})',
            PYN_typenum="NPY_LONGLONG",
            LUA_type="LUA_TNUMBER",
            LUA_pop="lua_tointeger({LUA_state_var}, {LUA_index})",
            LUA_push="lua_pushinteger({LUA_state_var}, {push_arg})",
            sgroup="native",
            base="integer",
            sh_type="SH_TYPE_UNSIGNED_LONG_LONG",
            cfi_type="CFI_type_long_long",
        ),
        size_t=Typemap(
            "size_t",
            c_type="size_t",
            cxx_type="size_t",
            c_header="<stddef.h>",
            cxx_header="<cstddef>",
            f_cast="int({f_var}, C_SIZE_T)",
            f_type="integer(C_SIZE_T)",
            f_kind="C_SIZE_T",
            f_module_name="iso_c_binding",
            f_module=dict(iso_c_binding=["C_SIZE_T"]),
            PY_format="n",
            PY_ctor="PyInt_FromSize_t({ctor_expr})",
            LUA_type="LUA_TNUMBER",
            LUA_pop="lua_tointeger({LUA_state_var}, {LUA_index})",
            LUA_push="lua_pushinteger({LUA_state_var}, {push_arg})",
            sgroup="native",
            base="integer",
            sh_type="SH_TYPE_SIZE_T",
            cfi_type="CFI_type_size_t",
        ),
        # XXX- sized based types for Python
        int8_t=Typemap(
            "int8_t",
            c_type="int8_t",
            cxx_type="int8_t",
            c_header="<stdint.h>",
            cxx_header="<cstdint>",
            f_cast="int({f_var}, C_INT8_T)",
            f_type="integer(C_INT8_T)",
            f_kind="C_INT8_T",
            f_module_name="iso_c_binding",
            f_module=dict(iso_c_binding=["C_INT8_T"]),
            PY_format="i",
            PY_ctor="PyInt_FromLong({ctor_expr})",
            PY_get="PyInt_AsLong({py_var})",
            PYN_typenum="NPY_INT8",
            LUA_type="LUA_TNUMBER",
            LUA_pop="lua_tointeger({LUA_state_var}, {LUA_index})",
            LUA_push="lua_pushinteger({LUA_state_var}, {push_arg})",
            sgroup="native",
            base="integer",
            sh_type="SH_TYPE_INT8_T",
            cfi_type="CFI_type_int8_t",
        ),
        int16_t=Typemap(
            "int16_t",
            c_type="int16_t",
            cxx_type="int16_t",
            c_header="<stdint.h>",
            cxx_header="<cstdint>",
            f_cast="int({f_var}, C_INT16_T)",
            f_type="integer(C_INT16_T)",
            f_kind="C_INT16_T",
            f_module_name="iso_c_binding",
            f_module=dict(iso_c_binding=["C_INT16_T"]),
            PY_format="i",
            PY_ctor="PyInt_FromLong({ctor_expr})",
            PY_get="PyInt_AsLong({py_var})",
            PYN_typenum="NPY_INT16",
            LUA_type="LUA_TNUMBER",
            LUA_pop="lua_tointeger({LUA_state_var}, {LUA_index})",
            LUA_push="lua_pushinteger({LUA_state_var}, {push_arg})",
            sgroup="native",
            base="integer",
            sh_type="SH_TYPE_INT16_T",
            cfi_type="CFI_type_int16_t",
        ),
        int32_t=Typemap(
            "int32_t",
            c_type="int32_t",
            cxx_type="int32_t",
            c_header="<stdint.h>",
            cxx_header="<cstdint>",
            f_cast="int({f_var}, C_INT32_T)",
            f_type="integer(C_INT32_T)",
            f_kind="C_INT32_T",
            f_module_name="iso_c_binding",
            f_module=dict(iso_c_binding=["C_INT32_T"]),
            PY_format="i",
            PY_ctor="PyInt_FromLong({ctor_expr})",
            PY_get="PyInt_AsLong({py_var})",
            PYN_typenum="NPY_INT32",
            LUA_type="LUA_TNUMBER",
            LUA_pop="lua_tointeger({LUA_state_var}, {LUA_index})",
            LUA_push="lua_pushinteger({LUA_state_var}, {push_arg})",
            sgroup="native",
            base="integer",
            sh_type="SH_TYPE_INT32_T",
            cfi_type="CFI_type_int32_t",
        ),
        int64_t=Typemap(
            "int64_t",
            c_type="int64_t",
            cxx_type="int64_t",
            c_header="<stdint.h>",
            cxx_header="<cstdint>",
            f_cast="int({f_var}, C_INT64_T)",
            f_type="integer(C_INT64_T)",
            f_kind="C_INT64_T",
            f_module_name="iso_c_binding",
            f_module=dict(iso_c_binding=["C_INT64_T"]),
            PY_format="L",
            PY_ctor="PyInt_FromLong({ctor_expr})",
            PY_get="PyInt_AsLong({py_var})",
            PYN_typenum="NPY_INT64",
            LUA_type="LUA_TNUMBER",
            LUA_pop="lua_tointeger({LUA_state_var}, {LUA_index})",
            LUA_push="lua_pushinteger({LUA_state_var}, {push_arg})",
            sgroup="native",
            base="integer",
            sh_type="SH_TYPE_INT64_T",
            cfi_type="CFI_type_int64_t",
        ),
        # XXX- sized based types for Python
        uint8_t=Typemap(
            "uint8_t",
            c_type="uint8_t",
            cxx_type="uint8_t",
            c_header="<stdint.h>",
            cxx_header="<cstdint>",
            f_cast="int({f_var}, C_INT8_T)",
            f_type="integer(C_INT8_T)",
            f_kind="C_INT8_T",
            f_module_name="iso_c_binding",
            f_module=dict(iso_c_binding=["C_INT8_T"]),
            PY_format="i",
            PY_ctor="PyInt_FromLong({ctor_expr})",
            PY_get="PyInt_AsLong({py_var})",
            PYN_typenum="NPY_UINT8",
            LUA_type="LUA_TNUMBER",
            LUA_pop="lua_tointeger({LUA_state_var}, {LUA_index})",
            LUA_push="lua_pushinteger({LUA_state_var}, {push_arg})",
            sgroup="native",
            base="integer",
            sh_type="SH_TYPE_UINT8_T",
            cfi_type="CFI_type_int8_t",
        ),
        uint16_t=Typemap(
            "uint16_t",
            c_type="uint16_t",
            cxx_type="uint16_t",
            c_header="<stdint.h>",
            cxx_header="<cstdint>",
            f_cast="int({f_var}, C_INT16_T)",
            f_type="integer(C_INT16_T)",
            f_kind="C_INT16_T",
            f_module_name="iso_c_binding",
            f_module=dict(iso_c_binding=["C_INT16_T"]),
            PY_format="i",
            PY_ctor="PyInt_FromLong({ctor_expr})",
            PY_get="PyInt_AsLong({py_var})",
            PYN_typenum="NPY_UINT16",
            LUA_type="LUA_TNUMBER",
            LUA_pop="lua_tointeger({LUA_state_var}, {LUA_index})",
            LUA_push="lua_pushinteger({LUA_state_var}, {push_arg})",
            sgroup="native",
            base="integer",
            sh_type="SH_TYPE_UINT16_T",
            cfi_type="CFI_type_int16_t",
        ),
        uint32_t=Typemap(
            "uint32_t",
            c_type="uint32_t",
            cxx_type="uint32_t",
            c_header="<stdint.h>",
            cxx_header="<cstdint>",
            f_cast="int({f_var}, C_INT32_T)",
            f_type="integer(C_INT32_T)",
            f_kind="C_INT32_T",
            f_module_name="iso_c_binding",
            f_module=dict(iso_c_binding=["C_INT32_T"]),
            PY_format="i",
            PY_ctor="PyInt_FromLong({ctor_expr})",
            PY_get="PyInt_AsLong({py_var})",
            PYN_typenum="NPY_UINT32",
            LUA_type="LUA_TNUMBER",
            LUA_pop="lua_tointeger({LUA_state_var}, {LUA_index})",
            LUA_push="lua_pushinteger({LUA_state_var}, {push_arg})",
            sgroup="native",
            base="integer",
            sh_type="SH_TYPE_UINT32_T",
            cfi_type="CFI_type_int32_t",
        ),
        uint64_t=Typemap(
            "uint64_t",
            c_type="uint64_t",
            cxx_type="uint64_t",
            c_header="<stdint.h>",
            cxx_header="<cstdint>",
            f_cast="int({f_var}, C_INT64_T)",
            f_type="integer(C_INT64_T)",
            f_kind="C_INT64_T",
            f_module_name="iso_c_binding",
            f_module=dict(iso_c_binding=["C_INT64_T"]),
            PY_format="L",
            PY_ctor="PyInt_FromLong({ctor_expr})",
            PY_get="PyInt_AsLong({py_var})",
            PYN_typenum="NPY_UINT64",
            LUA_type="LUA_TNUMBER",
            LUA_pop="lua_tointeger({LUA_state_var}, {LUA_index})",
            LUA_push="lua_pushinteger({LUA_state_var}, {push_arg})",
            sgroup="native",
            base="integer",
            sh_type="SH_TYPE_UINT64_T",
            cfi_type="CFI_type_int64_t",
        ),
        float=Typemap(
            "float",
            c_type="float",
            cxx_type="float",
            f_cast="real({f_var}, C_FLOAT)",
            f_type="real(C_FLOAT)",
            f_kind="C_FLOAT",
            f_module_name="iso_c_binding",
            f_module=dict(iso_c_binding=["C_FLOAT"]),
            PY_format="f",
            PY_ctor="PyFloat_FromDouble({ctor_expr})",
            PY_get="PyFloat_AsDouble({py_var})",
            PYN_typenum="NPY_FLOAT",
            LUA_type="LUA_TNUMBER",
            LUA_pop="lua_tonumber({LUA_state_var}, {LUA_index})",
            LUA_push="lua_pushnumber({LUA_state_var}, {push_arg})",
            sgroup="native",
            base="real",
            sh_type="SH_TYPE_FLOAT",
            cfi_type="CFI_type_float",
        ),
        double=Typemap(
            "double",
            c_type="double",
            cxx_type="double",
            f_cast="real({f_var}, C_DOUBLE)",
            f_type="real(C_DOUBLE)",
            f_kind="C_DOUBLE",
            f_module_name="iso_c_binding",
            f_module=dict(iso_c_binding=["C_DOUBLE"]),
            PY_format="d",
            PY_ctor="PyFloat_FromDouble({ctor_expr})",
            PY_get="PyFloat_AsDouble({py_var})",
            PYN_typenum="NPY_DOUBLE",
            LUA_type="LUA_TNUMBER",
            LUA_pop="lua_tonumber({LUA_state_var}, {LUA_index})",
            LUA_push="lua_pushnumber({LUA_state_var}, {push_arg})",
            sgroup="native",
            base="real",
            sh_type="SH_TYPE_DOUBLE",
            cfi_type="CFI_type_double",
        ),
        float_complex=Typemap(   # _Complex
            "float_complex",
            c_type="float complex",
            cxx_type="std::complex<float>",
            flat_name="float_complex",
            c_header="<complex.h>",
            cxx_header="<complex>",
            f_cast="cmplx({f_var}, C_FLOAT_COMPLEX)",
            f_type="complex(C_FLOAT_COMPLEX)",
            f_kind="C_FLOAT_COMPLEX",
            f_module_name="iso_c_binding",
            f_module=dict(iso_c_binding=["C_FLOAT_COMPLEX"]),
            PY_format="D",
            py_ctype="Py_complex",
            pytype_to_pyctor="creal({ctor_expr}), cimag({ctor_expr})",
            pytype_to_cxx="{work_var}.real + {work_var}.imag * I",
            cxx_to_pytype="{py_var}.real = creal({cxx_var});\n{py_var}.imag = cimag({cxx_var});",
            PY_ctor="PyComplex_FromDoubles(\t{ctor_expr})",
            PY_get="PyComplex_AsCComplex({py_var})",
            PY_build_arg="&{ctype_var}",
            PYN_typenum="NPY_DOUBLE",
            LUA_type="LUA_TNUMBER",
            LUA_pop="lua_tonumber({LUA_state_var}, {LUA_index})",
            LUA_push="lua_pushnumber({LUA_state_var}, {push_arg})",
            sgroup="native",
            base="complex",
            sh_type="SH_TYPE_FLOAT_COMPLEX",
            cfi_type="CFI_type_float_Complex",
        ),
        double_complex=Typemap(   # _Complex
            "double_complex",
            c_type="double complex",
            cxx_type="std::complex<double>",
            flat_name="double_complex",
            c_header="<complex.h>",
            cxx_header="<complex>",
            f_cast="cmplx({f_var}, C_DOUBLE_COMPLEX)",
            f_type="complex(C_DOUBLE_COMPLEX)",
            f_kind="C_DOUBLE_COMPLEX",
            f_module_name="iso_c_binding",
            f_module=dict(iso_c_binding=["C_DOUBLE_COMPLEX"]),
            PY_format="D",
            PY_get="PyComplex_AsCComplex({py_var})",
            py_ctype="Py_complex",
            pytype_to_pyctor="creal({ctor_expr}), cimag({ctor_expr})",
            pytype_to_cxx="{work_var}.real + {work_var}.imag * I",
            cxx_to_pytype="{ctype_var}.real = creal({cxx_var});\n{ctype_var}.imag = cimag({cxx_var});",
            # fmt.work_ctor = "std::complex(\tcvalue.real, cvalue.imag)"
            # creal(), cimag()
            # std::real(), std::imag()
            # xx.real(), xx.imag()
            PY_ctor="PyComplex_FromDoubles(\t{ctor_expr})", # double real, double imag
            PY_build_arg="&{ctype_var}",
            PYN_typenum="NPY_DOUBLE",
            LUA_type="LUA_TNUMBER",
            LUA_pop="lua_tonumber({LUA_state_var}, {LUA_index})",
            LUA_push="lua_pushnumber({LUA_state_var}, {push_arg})",
            sgroup="native",
            base="complex",
            sh_type="SH_TYPE_DOUBLE_COMPLEX",
            cfi_type="CFI_type_double_Complex",
        ),
        bool=Typemap(
            "bool",
            c_type="bool",
            cxx_type="bool",
            c_header="<stdbool.h>",
            f_type="logical",
            f_kind="C_BOOL",
            i_type="logical(C_BOOL)",
            f_module_name="iso_c_binding",
            f_module=dict(iso_c_binding=["C_BOOL"]),
            # XXX PY_format='p',  # Python 3.3 or greater
            # Use py_statements.x.ctor instead of PY_ctor. This code will always be
            # added.  Older version of Python can not create a bool directly from
            # from Py_BuildValue.
            # #- PY_ctor='PyBool_FromLong({c_var})',
            PY_PyTypeObject="PyBool_Type",
            PY_get="PyObject_IsTrue({py_var})",
            PYN_typenum="NPY_BOOL",
            LUA_type="LUA_TBOOLEAN",
            LUA_pop="lua_toboolean({LUA_state_var}, {LUA_index})",
            LUA_push="lua_pushboolean({LUA_state_var}, {push_arg})",
            base="bool",
            sgroup="bool",
            sh_type="SH_TYPE_BOOL",
            cfi_type="CFI_type_Bool",
        ),
        # implies null terminated string
        char=Typemap(
            "char",
            cxx_type="char",
            c_type="char",  # XXX - char *
            f_type="character(*)",
            f_kind="C_CHAR",
            f_module_name="iso_c_binding",
            i_type="character(kind=C_CHAR)",
            i_module=dict(iso_c_binding=["C_CHAR"]),
            PY_format="s",
            PY_ctor="PyString_FromString({ctor_expr})",
#            PY_get="PyString_AsString({py_var})",
            PYN_typenum="NPY_INTP",  # void *    # XXX - 
            LUA_type="LUA_TSTRING",
            LUA_pop="lua_tostring({LUA_state_var}, {LUA_index})",
            LUA_push="lua_pushstring({LUA_state_var}, {push_arg})",
            base="string",
            sgroup="char",
        ),
        # C++ std::string
        string=Typemap(
            "std::string",
            cxx_type="std::string",
            cxx_to_c="{cxx_var}{cxx_member}c_str()",  # cxx_member is . or ->
            c_type="char",  # XXX - char *
            impl_header=["<string>"],
            f_type="character(*)",
            f_kind="C_CHAR",
            f_module_name="iso_c_binding",
            i_type="character(kind=C_CHAR)",
            i_module=dict(iso_c_binding=["C_CHAR"]),
            PY_format="s",
            PY_ctor="PyString_FromStringAndSize({ctor_expr})",
            PY_build_format="s#",
            # XXX need cast after PY_SSIZE_T_CLEAN
            PY_build_arg="{cxx_var}{cxx_member}data(),\t {cxx_var}{cxx_member}size()",
            LUA_type="LUA_TSTRING",
            LUA_pop="lua_tostring({LUA_state_var}, {LUA_index})",
            LUA_push="lua_pushstring({LUA_state_var}, {push_arg})",
            base="string",
            sgroup="string",
        ),
        # C++ std::vector
        # No c_type or f_type, use attr[template]
        # C++03 "The elements of a vector are stored contiguously" (23.2.4/1).
        vector=Typemap(
            "std::vector",
            ntemplate_args=1,
            cxx_type="std::vector<{cxx_T}>",
            cxx_header="<vector>",
            # #- cxx_to_c='{cxx_var}.data()',  # C++11
            # #- cxx_to_c='{cxx_var}{cxx_member}empty() ? NULL : &{cxx_var}[0]', # C++03)
            # custom code for templates
            #            PY_format='s',
            #            PY_ctor='PyString_FromString({c_var})',
            #            LUA_type='LUA_TSTRING',
            #            LUA_pop='lua_tostring({LUA_state_var}, {LUA_index})',
            #            LUA_push='lua_pushstring({LUA_state_var}, {push_arg})',
            impl_header=["<vector>"],
            base="vector",
            sgroup="vector",
        ),
        MPI_Comm=Typemap(
            "MPI_Comm",
            cxx_type="MPI_Comm",
            c_header="mpi.h",
            cxx_header="mpi.h",
            c_type="MPI_Fint",
            # usually, MPI_Fint will be equivalent to int
            f_type="integer",
            f_kind="C_INT",
            f_module_name="iso_c_binding",
            i_type="integer(C_INT)",
            i_module=dict(iso_c_binding=["C_INT"]),
            cxx_to_c="MPI_Comm_c2f({cxx_var})",
            c_to_cxx="MPI_Comm_f2c({c_var})",
        ),
    )

    # Rename to actual types.
    # It is not possible to do dict(std::string=...)
    def_types["std::string"] = def_types["string"]
    del def_types["string"]
    def_types["std::vector"] = def_types["vector"]
    del def_types["vector"]

    # One typemap for all template parameters.
    type_name = "--template-parameter--"
    ntypemap = Typemap(
        type_name,
        base="template",
        sgroup="template",
        c_type="c_T",
        cxx_type="cxx_T",
        f_type="f_T",
    )
    def_types[type_name] = ntypemap

    global void_typemap
    void_typemap = def_types["void"] # GGG

    return def_types

def check_for_missing_typemap_fields(cxx_name, fields, names):
    missing = []
    for field_name in names:
        if field_name not in fields:
            missing.append(field_name)
    if missing:
        raise RuntimeError(
            "typemap {} requires fields {}".format(cxx_name, ", ".join(missing))
        )

def create_native_typemap_from_fields(cxx_name, fields, library):
    """Create a typemap from fields.
    Used when creating typemap from YAML. (from regression/forward.yaml)
    Used when the base type is 'integer' or 'real'.

        typemap:
        - type: indextype
          fields:
            base: integer
            f_kind: INDEXTYPE
            f_module_name: typemap_mod

    Parameters:
    -----------
    cxx_name : str
    fields : dictionary object.
    library : ast.LibraryNode.
    """
    check_for_missing_typemap_fields(cxx_name, fields, [
        "f_kind",
        "f_module_name",
    ])
    
    fmt = library.fmtdict
    ntypemap = Typemap(
        cxx_name,
        sgroup="native",
        cxx_type=cxx_name,
        c_type=cxx_name,
        f_cast=None,  # Override Typemap default
    )
    ntypemap.update(fields)
    fill_native_typemap_defaults(ntypemap, fmt)
    ntypemap.finalize()
    library.symtab.add_typedef(cxx_name, ntypemap)
    return ntypemap


# Map typemap.base to fortran intrinsic function.
base_cast = dict(
    integer='int',
    real='real',
)

def fill_native_typemap_defaults(ntypemap, fmt):
    """Add some defaults to integer or real typemap.
    When dumping typemaps to a file, only a subset is written
    since the rest are boilerplate.  This function restores
    the boilerplate.

    Some fields can be derived from f_kind.

    Parameters
    ----------
    ntypemap : typemap.Typemap.
    fmt : util.Scope
    """
    if ntypemap.f_type is None:
        ntypemap.f_type = "{}({})".format(ntypemap.base, ntypemap.f_kind)
    if ntypemap.f_cast is None:
        ntypemap.f_cast = "{}({{f_var}}, {})".format(base_cast[ntypemap.base], ntypemap.f_kind)
    if ntypemap.f_module is None:
        ntypemap.f_module = {ntypemap.f_module_name: [ntypemap.f_kind]}


def fill_enum_typemap(node, ftypemap):
    """Fill an enum typemap with wrapping fields.
    The typemap is created in declast.Enum.

# XXX    Create a typemap similar to an int.
# XXX    C++ enums are converted to a C int.

    ftypemap is how the enum is represented in the Fortran wrapper.
    Typically an 'int'.

    Args:
        node - EnumNode instance.
    """
    # Using abstract_decl in the converters, maps to any typedef declaration.
    fmt_enum = node.fmtdict

    ntypemap = node.typemap
    if ntypemap is None:
        raise RuntimeError("Missing typemap on EnumNode")
    else:
        ntypemap.copy_from_typemap(ftypemap)
        ntypemap.is_enum = True
        ntypemap.ci_type = ftypemap.c_type
        ntypemap.sgroup = "enum"

        # Include the generated C header file for the
        # C declaration of the C++ enum.
        ntypemap.c_header = [fmt_enum.C_header_filename]
        ntypemap.cxx_header = [fmt_enum.C_header_filename]
        
        language = node.get_language()

        if language == "c":
            ntypemap.c_type = "enum %s" % ntypemap.name
            
            # XXX - These are used with Python wrapper and ParseTupleAndKeyword.
            ntypemap.cxx_type = util.wformat(
                "enum {namespace_scope}{enum_name}", fmt_enum
            )
            ntypemap.c_to_cxx = util.wformat(
                "(enum {namespace_scope}{enum_name}) {{c_var}}", fmt_enum
            )
            ntypemap.c_to_cxx = "({cxx_abstract_decl}) {c_var}"
            ntypemap.cxx_to_c = "(%s) {cxx_var}" % ntypemap.c_type

            ntypemap.cxx_to_ci = "(%s) {cxx_var}" % ntypemap.ci_type

        else:
            ntypemap.c_type = "enum %s" % fmt_enum.C_enum_type

            ntypemap.cxx_type = util.wformat(
                "{namespace_scope}{enum_name}", fmt_enum
            )
            ntypemap.c_to_cxx = util.wformat(
                "static_cast<{namespace_scope}{enum_name}>({{c_var}})", fmt_enum
            )
            ntypemap.cxx_to_c = "static_cast<{c_abstract_decl}>({cxx_var})"
            
            ntypemap.cxx_to_ci = "static_cast<%s>({cxx_var})" % ntypemap.ci_type
        ntypemap.compute_flat_name()
    return ntypemap

def compute_class_typemap_derived_fields(ntypemap, fields):
    """Compute typemap fields which are derived from other fields.

    If fields has a user provided value, do not compute default.
    """
    # compute names derived from other values
    if "f_class" not in fields:
        ntypemap.f_class = "class(%s)" % ntypemap.f_derived_type
        # XXX f_kind
    if "f_type" not in fields:
        ntypemap.f_type = "type(%s)" % ntypemap.f_derived_type
    if "i_type" not in fields:
        ntypemap.i_type = "type(%s)" % ntypemap.f_capsule_data_type

def create_class_typemap_from_fields(cxx_name, fields, library):
    """Create a typemap from fields.
    Used when creating typemap from YAML. (from regression/forward.yaml)
    Used when the base type is 'shadow''.

        typemap:
        - type: tutorial::Class1
          fields:
            base: shadow

    Parameters:
    -----------
    cxx_name : str
    fields : dictionary object.
    library : ast.LibraryNode.
    """
    check_for_missing_typemap_fields(cxx_name, fields, [
        "f_derived_type",
        "f_module_name",
    ])
    
    fmt_class = library.fmtdict
    ntypemap = Typemap(
        cxx_name,
        base="shadow", sgroup="shadow",
        cxx_type=cxx_name,
        f_capsule_data_type="missing-f_capsule_data_type",
        f_derived_type=cxx_name,

        cxx_to_c="static_cast<{c_const}void *>(\t{cxx_addr}{cxx_var})",
        c_to_cxx="static_cast<{c_const}%s *>\t({c_var}->addr)" % cxx_name,
        LUA_type="LUA_TUSERDATA",
        LUA_pop=(
            "\t({LUA_userdata_type} *)\t luaL_checkudata"
            '(\t{LUA_state_var}, 1, "{LUA_metadata}")'
        ),
    )
    ntypemap.update(fields)
    compute_class_typemap_derived_fields(ntypemap, fields)

    if "f_module" not in fields:
        ntypemap.f_module = {ntypemap.f_module_name: [ntypemap.f_derived_type]}
    if "i_module" not in fields:
        ntypemap.i_module = {ntypemap.f_module_name: [ntypemap.f_capsule_data_type]}
    
    ntypemap.finalize()
    library.symtab.add_typedef(cxx_name, ntypemap)

def create_class_typemap(node, fields={}):
    """Create a typemap from a ClassNode.
    The class ClassNode can result from template instantiation in generate
    and not while parsing.
    Use fields to override defaults.

    The c_type and f_capsule_data_type are a struct which contains
    a pointer to the C++ memory and information on how to delete the memory.

    Args:
        node - ast.ClassNode.
        fields - dictionary-like object.
    """
    fmt_class = node.fmtdict
    cxx_name = util.wformat("{namespace_scope}{cxx_class}", fmt_class)

    ntypemap = Typemap(cxx_name)
    node.typemap = ntypemap
    fill_class_typemap(node, fields)
    node.symtab.register_typemap(cxx_name, ntypemap)
    return ntypemap

def fill_class_typemap(node, fields={}):
    """Fill a class typemap with wrapping fields.

    The typemap already exists in the node.
    """
    ntypemap = node.typemap
    fmt_class = node.fmtdict
    cxx_type = util.wformat("{namespace_scope}{cxx_type}", fmt_class)

    # unname = util.un_camel(name)
    f_name = fmt_class.cxx_class.lower()
    c_name = fmt_class.C_prefix + fmt_class.C_name_scope[:-1]
    ntypemap.update(dict(
        base="shadow",       # GGG already set but may be wrapped differently
        sgroup="shadow",
        cxx_type=cxx_type,
        impl_header=node.find_header(),
        wrap_header=fmt_class.C_header_utility,
        c_type=c_name,
        f_module_name=fmt_class.F_module_name,
        f_derived_type=fmt_class.F_derived_name,
        f_capsule_data_type=fmt_class.F_capsule_data_type,
        # #- f_to_c='{f_var}%%%s()' % fmt_class.F_name_instance_get, # XXX - develop test
        sh_type="SH_TYPE_OTHER",
        cfi_type="CFI_type_other",

        cxx_to_c="static_cast<{c_const}void *>(\t{cxx_addr}{cxx_var})",
        c_to_cxx="static_cast<{c_const}%s *>\t({c_var}->addr)" % cxx_type,
        LUA_type = "LUA_TUSERDATA",
        LUA_pop = (
            "\t({LUA_userdata_type} *)\t luaL_checkudata"
            '(\t{LUA_state_var}, 1, "{LUA_metadata}")'
        )
    ))
    # import classes which are wrapped by this module
    # XXX - deal with namespaces vs modules
    
    ntypemap.update(fields)
    compute_class_typemap_derived_fields(ntypemap, fields)

    if "f_to_c" not in fields:
        ntypemap.f_to_c = "{f_var}%%%s" % fmt_class.F_derived_member

    if "f_module" not in fields:
        ntypemap.f_module = {ntypemap.f_module_name: [ntypemap.f_derived_type]}
    if "i_module" not in fields:
        ntypemap.i_module = {ntypemap.f_module_name: [ntypemap.f_capsule_data_type]}

    ntypemap.finalize()

    fmt_class.C_type_name = ntypemap.c_type


def compute_struct_typemap_derived_fields(ntypemap, fields):
    """Compute typemap fields which are derived from other fields.

    If fields has a user provided value, do not compute default.
    """
    if "f_kind" not in fields:
        ntypemap.f_kind = ntypemap.f_derived_type
    if "f_type" not in fields:
        ntypemap.f_type = "type(%s)" % ntypemap.f_derived_type

def create_struct_typemap_from_fields(cxx_name, fields, library):
    """Create a struct typemap from fields.
    Used when creating typemap from YAML. (from regression/forward.yaml)
    Used when the base type is 'struct'.

        typemap:
        - type: tutorial::Struct1
          fields:
            base: struct

    Parameters:
    -----------
    cxx_name : str
    fields : dictionary object.
    library : ast.LibraryNode.
    """
    check_for_missing_typemap_fields(cxx_name, fields, [
        "f_derived_type",
        "f_module_name",
    ])

    fmt_class = library.fmtdict
    ntypemap = Typemap(
        cxx_name,
        base="struct", sgroup="struct",
        c_type=cxx_name,
        cxx_type=cxx_name,
        f_type = "type(%s)" % cxx_name,
    )
    ntypemap.update(fields)

    # Add defaults for missing names
    if ntypemap.f_derived_type is None:
        ntypemap.f_derived_type  = ntypemap.name

    compute_struct_typemap_derived_fields(ntypemap, fields)

    if "f_module" not in fields:
        ntypemap.f_module = {ntypemap.f_module_name: [ntypemap.f_derived_type]}
#    if "i_module" not in fields:
#        ntypemap.i_module = {ntypemap.f_module_name: [ntypemap.f_capsule_data_type]}

    library.symtab.add_typedef(cxx_name, ntypemap)

def create_struct_typemap(node, fields={}):
    """Create a typemap for a struct from a ClassNode.
    Use fields to override defaults.

    Parameters:
    -----------
    node : ast.ClassNode
    fields : dictionary-like object.
    """
    fmt_class = node.fmtdict
    cxx_name = util.wformat("{namespace_scope}{cxx_class}", fmt_class)

    ntypemap = Typemap(cxx_name)
    node.typemap = ntypemap
    fill_struct_typemap(node, fields)
    node.symtab.register_typemap(cxx_name, ntypemap)
    return ntypemap

def fill_struct_typemap(node, fields={}):
    """Fill a struct typemap with wrapping fields.

    The typemap already exists in the node.
    Use fields to override defaults.

    Parameters:
    -----------
    node : ast.ClassNode
    fields : dictionary-like object.
    """
    ntypemap = node.typemap
    fmt_class = node.fmtdict
    cxx_type = util.wformat("{namespace_scope}{cxx_type}", fmt_class)

    # unname = util.un_camel(name)
    f_name = fmt_class.cxx_class.lower()
    c_name = fmt_class.C_prefix + f_name
    ntypemap.update(dict(
        base="struct",        # GGG already set but may be wrapped differently
        sgroup="struct",
        cxx_type=cxx_type,
        c_type=c_name,
        f_module_name=fmt_class.F_module_name,
        f_derived_type=fmt_class.F_derived_name,
        PYN_descr=fmt_class.PY_struct_array_descr_variable,
        sh_type="SH_TYPE_STRUCT",
        cfi_type="CFI_type_struct",

        LUA_type = "LUA_TUSERDATA",
        LUA_pop = (
            "\t({LUA_userdata_type} *)\t luaL_checkudata"
            '(\t{LUA_state_var}, 1, "{LUA_metadata}")'
        ),
    ))

    libnode = node.get_LibraryNode()
    language = libnode.language
    if language == "c":
        # The struct from the user's library is used.
        # XXX - if struct in class, uses class.cxx_header?
        ntypemap.c_header = libnode.cxx_header
        ntypemap.c_type = ntypemap.cxx_type
    ntypemap.PY_struct_as = node.options.PY_struct_arg
    
    ntypemap.update(fields)
    compute_struct_typemap_derived_fields(ntypemap, fields)

    if "f_module" not in fields:
        ntypemap.f_module = {ntypemap.f_module_name: [ntypemap.f_derived_type]}
    if "i_module" not in fields:
        ntypemap.i_module = {ntypemap.f_module_name: [ntypemap.f_derived_type]}
        
    if ntypemap.cxx_type and not ntypemap.flat_name:
            ntypemap.compute_flat_name()

    fmt_class.C_type_name = ntypemap.c_type

def create_fcnptr_typemap(symtab, name):
    # GGG - Similar to how typemaps are created in class Struct
    """Create a typemap for a function pointer
    The typemap contains a the declaration.

    Args:
        node - ast.TypedefNode
        fields - dictionary
    """
    raise NotImplemented
    type_name = symtab.scopename + name
    ntypemap = Typemap(
        type_name,
        base="procedure",
        sgroup="procedure",
    )
    # Check if all fields are C compatible
#            ntypemap.compute_flat_name()
    
    symtab.register_typemap(type_name, ntypemap)
    return ntypemap

def fill_fcnptr_typemap(node, fields={}):
    """Create a typemap for a function pointer
    The typemap contains a the declaration.

    Args:
        node - ast.TypedefNode
        fields - dictionary
    """
#    print("XXXXXXXX", dir(node))
#    print("  name", node.name)   # return type of function pointer
#    print("  typemap", node.typemap)   # return type of function pointer
#    raise NotImplementedError(
#        "Function pointers not supported in typedef"
#    )
    raise NotImplemented
    fmt = node.fmtdict
    cxx_name = node.ast.name
    fmt.typedef_name = cxx_name
    cxx_name = util.wformat("{namespace_scope}{typedef_name}", fmt)
    cxx_type = cxx_name
#    cxx_type = util.wformat("{namespace_scope}{cxx_type}", fmt)
    c_type = fmt.C_prefix + cxx_name
    ntypemap = Typemap(
        cxx_name,
        base="procedure",
        sgroup="procedure",
        c_type="c_type",
        cxx_type="cxx_type",
        f_type="XXXf_type",
    )
    # Check if all fields are C compatible
#            ntypemap.compute_flat_name()
    
    ntypemap.update(fields)
    register_typemap(cxx_name, ntypemap)
    return ntypemap

def fill_typedef_typemap(node, fields={}):
    """Fill a typedef typemap with wrapping fields.

    The typemap already exists in the node.

    base stays the same.
    f_kind will be a generated parameter
       integer, parameter :: IndexType = C_INT

    impl_header is not needed.  It will be defined
    when the base type is defined.
    """
    ntypemap = node.typemap
    if ntypemap is None:
        raise RuntimeError("Missing typemap on TypedefNode")
    fmtdict = node.fmtdict
#    cxx_name = util.wformat("{namespace_scope}{cxx_class}", fmtdict)
    cxx_type = util.wformat("{namespace_scope}{class_scope}{cxx_type}", fmtdict)

#    f_name = fmtdict.F_name_scope[:-1]
#    c_name = fmtdict.C_prefix + fmtdict.C_name_scope[:-1]
    f_name = fmtdict.F_name_typedef
    c_name = fmtdict.C_name_typedef
#    print("XXX   fill_typedef_typemap  f={}  c={}".format(f_name, c_name))

    # Define equivalent parameter for Fortran
#    node.f_define_parameter = "integer, parameter :: {} = {}".format(
#        f_name, ntypemap.f_kind)

##############################    print("XXXXX DD", ntypemap.name, fmtdict.C_header_filename)
    ntypemap.update(dict(
        cxx_type=cxx_type,
        wrap_header=fmtdict.C_header_filename,
        c_type=c_name,

        f_type = "{}({})".format(ntypemap.base, f_name),
        f_kind=f_name,
        #XXX f_cast  using f_name
        f_module_name=fmtdict.F_module_name,
#        f_derived_type=fmtdict.F_derived_name,
#        f_capsule_data_type=fmtdict.F_capsule_data_type,
#        f_module={fmtdict.F_module_name: [fmtdict.F_derived_name]},
        # #- f_to_c='{f_var}%%%s()' % fmtdict.F_name_instance_get, # XXX - develop test
#        f_to_c="{f_var}%%%s" % fmtdict.F_derived_member,
#        sh_type="SH_TYPE_OTHER",
#        cfi_type="CFI_type_other",
    ))

    if ntypemap.base in ["shadow", "struct"]:
        ntypemap.f_type = "type({})".format(f_name)
    elif ntypemap.base == "integer":
        ntypemap.f_cast = "int({f_var}, %s)" % f_name
    elif ntypemap.base == "real":
        ntypemap.f_cast = "real({f_var}, %s)" % f_name
    
    # USE names which are wrapped by this module
    # XXX - deal with namespaces vs modules
    ntypemap.f_module = {fmtdict.F_module_name: [f_name]}
    ntypemap.i_module = {fmtdict.F_module_name: [f_name]}
    ntypemap.update(fields)
#    fill_typedef_typemap_defaults(ntypemap, fmtdict)
    ntypemap.finalize()

def return_user_types(typemaps):  # typemaps -> dict
    """Return a dictionary of user defined types."""
    dct = {}
    for key, ntypemap in typemaps.items():
        if ntypemap.name == "--template-parameter--":
            continue
        elif ntypemap.sgroup in ["shadow", "struct", "template", "enum"]:
            dct[key] = ntypemap
        elif ntypemap.is_enum:
            dct[key] = ntypemap
        elif hasattr(ntypemap, "is_typedef"):
            dct[key] = ntypemap
    return dct
