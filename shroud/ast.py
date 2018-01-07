
from . import util

class AstNode(object):
    def update_options_from_dict(self, node):
        """Update options from node.
        """
        if 'options' in node and \
                node['options'] is not None:
            if not isinstance(node['options'], dict):
                raise TypeError("options must be a dictionary")
            self.options.update(node['options'], replace=True)

    def option_to_fmt(self):
        """Set fmt based on options dictionary.
        """
        for name in ['C_prefix', 'F_C_prefix', 
                     'C_this', 'C_result', 'CPP_this',
                     'F_this', 'F_result', 'F_derived_member',
                     'C_string_result_as_arg', 'F_string_result_as_arg',
                     'C_header_filename_suffix',
                     'C_impl_filename_suffix',
                     'F_filename_suffix',
                     'PY_header_filename_suffix',
                     'PY_impl_filename_suffix',
                     'PY_result',
                     'LUA_header_filename_suffix',
                     'LUA_impl_filename_suffix',
                     'LUA_result']:
            if name in self.options:
                setattr(self._fmt, name, self.options[name])

    def eval_template(self, node, name, tname='', fmt=None):
        """fmt[name] = node[name] or option[name + tname + '_template']
        """
        if fmt is None:
            fmt = self._fmt
        if name in node:
            setattr(fmt, name, node[name])
        else:
            tname = name + tname + '_template'
            setattr(fmt, name, util.wformat(self.options[tname], fmt))

    def _to_dict(self):
        """Convert to dictionary.
        Used by util.ExpandedEncoder.
        """
        return dict()
        


class LibraryNode(AstNode):
    def __init__(self, node=None):
        """Populate LibraryNode from a dictionary.

        fields = value
        options:
        classes:
        functions:

        """

        self.language = 'c++'     # input language: c or c++
        self.options = self.default_options()
#        wrapp.add_templates(def_options)
#        wrapl.add_templates(def_options)

        if node is None:
            node = dict()

        self.node = node
        if 'language' in node:
            language = node['language'].lower()
            if language not in ['c', 'c++']:
                raise RuntimeError("language must be 'c' or 'c++'")
            self.language = node['language']

        self.update_options_from_dict(node)

        self.default_format(node)
        self.option_to_fmt()

#        self.fmt_stack.append(fmt_library)

        # default some options based on other options
        self.eval_template(node, 'C_header_filename', '_library')
        self.eval_template(node, 'C_impl_filename', '_library')
        # All class/methods and functions may go into this file or
        # just functions.
        self.eval_template(node, 'F_module_name', '_library')
        self.eval_template(node, 'F_impl_filename', '_library')

    def default_options(self):
        """default options."""
        def_options = util.Options(
            parent=None,
            debug=False,   # print additional debug info

            F_module_per_class=True,
            F_string_len_trim=True,
            F_force_wrapper=False,

            wrap_c=True,
            wrap_fortran=True,
            wrap_python=False,
            wrap_lua=False,

            doxygen=True,       # create doxygen comments
            show_splicer_comments=True,

            # blank for functions, set in classes.
            class_prefix_template='{class_lower}_',

            YAML_type_filename_template='{library_lower}_types.yaml',

            C_header_filename_library_template='wrap{library}.{C_header_filename_suffix}',
            C_impl_filename_library_template='wrap{library}.{C_impl_filename_suffix}',

            C_header_filename_class_template='wrap{cpp_class}.{C_header_filename_suffix}',
            C_impl_filename_class_template='wrap{cpp_class}.{C_impl_filename_suffix}',

            C_name_template=(
                '{C_prefix}{class_prefix}{underscore_name}{function_suffix}'),

            C_bufferify_suffix='_bufferify',
            C_var_len_template = 'N{c_var}',         # argument for result of len(arg)
            C_var_trim_template = 'L{c_var}',        # argument for result of len_trim(arg)
            C_var_size_template = 'S{c_var}',        # argument for result of size(arg)

            # Fortran's names for C functions
            F_C_prefix='c_',
            F_C_name_template=(
                '{F_C_prefix}{class_prefix}{underscore_name}{function_suffix}'),

            F_name_impl_template=(
                '{class_prefix}{underscore_name}{function_suffix}'),

            F_name_function_template='{underscore_name}{function_suffix}',
            F_name_generic_template='{underscore_name}',

            F_module_name_library_template='{library_lower}_mod',
            F_impl_filename_library_template='wrapf{library_lower}.{F_filename_suffix}',

            F_module_name_class_template='{class_lower}_mod',
            F_impl_filename_class_template='wrapf{cpp_class}.{F_filename_suffix}',

            F_name_instance_get='get_instance',
            F_name_instance_set='set_instance',
            F_name_associated='associated',

            LUA_module_name_template='{library_lower}',
            LUA_module_filename_template=(
                'lua{library}module.{LUA_impl_filename_suffix}'),
            LUA_header_filename_template=(
                'lua{library}module.{LUA_header_filename_suffix}'),
            LUA_userdata_type_template='{LUA_prefix}{cpp_class}_Type',
            LUA_userdata_member_template='self',
            LUA_module_reg_template='{LUA_prefix}{library}_Reg',
            LUA_class_reg_template='{LUA_prefix}{cpp_class}_Reg',
            LUA_metadata_template='{cpp_class}.metatable',
            LUA_ctor_name_template='{cpp_class}',
            LUA_name_template='{function_name}',
            LUA_name_impl_template='{LUA_prefix}{class_prefix}{underscore_name}',

            PY_module_filename_template=(
                'py{library}module.{PY_impl_filename_suffix}'),
            PY_header_filename_template=(
                'py{library}module.{PY_header_filename_suffix}'),
            PY_helper_filename_template=(
                'py{library}helper.{PY_impl_filename_suffix}'),
            PY_PyTypeObject_template='{PY_prefix}{cpp_class}_Type',
            PY_PyObject_template='{PY_prefix}{cpp_class}',
            PY_type_filename_template=(
                'py{cpp_class}type.{PY_impl_filename_suffix}'),
            PY_name_impl_template=(
                '{PY_prefix}{class_prefix}{underscore_name}{function_suffix}'),
            )
        return def_options

    def default_format(self, node):
        """Set format dictionary.
        """

        self._fmt = util.Options(None)
        fmt_library = self._fmt

        if 'library' in node:
            fmt_library.library = node['library']
        else:
            fmt_library.library = 'default_library'
        fmt_library.library_lower = fmt_library.library.lower()
        fmt_library.library_upper = fmt_library.library.upper()
        fmt_library.function_suffix = ''   # assume no suffix
        fmt_library.C_prefix = self.options.get(
            'C_prefix', fmt_library.library_upper[:3] + '_')
        fmt_library.F_C_prefix = self.options['F_C_prefix']
        if 'namespace' in node and node['namespace']:
            fmt_library.namespace_scope = (
                '::'.join(node['namespace'].split()) + '::')
        else:
            fmt_library.namespace_scope = ''

        # set default values for fields which may be unset.
        fmt_library.class_prefix = ''
#        fmt_library.c_ptr = ''
#        fmt_library.c_const = ''
        fmt_library.CPP_this_call = ''
        fmt_library.CPP_template = ''
        fmt_library.C_pre_call = ''
        fmt_library.C_post_call = ''

        fmt_library.C_this = 'self'
        fmt_library.C_result = 'SHT_rv'
        fmt_library.c_temp = 'SHT_'

        fmt_library.CPP_this = 'SH_this'

        fmt_library.F_this = 'obj'
        fmt_library.F_result = 'SHT_rv'
        fmt_library.F_derived_member = 'voidptr'

        fmt_library.C_string_result_as_arg = 'SHF_rv'
        fmt_library.F_string_result_as_arg = ''

        fmt_library.F_filename_suffix = 'f'

        # don't have to worry about argument names in Python wrappers
        # so skip the SH_ prefix by default.
        fmt_library.PY_result = 'rv'
        fmt_library.LUA_result = 'rv'

        if self.language == 'c':
            fmt_library.C_header_filename_suffix = 'h'
            fmt_library.C_impl_filename_suffix = 'c'

            fmt_library.PY_header_filename_suffix = 'h'
            fmt_library.PY_impl_filename_suffix = 'c'

            fmt_library.LUA_header_filename_suffix = 'h'
            fmt_library.LUA_impl_filename_suffix = 'c'

            fmt_library.stdlib  = ''
        else:
            fmt_library.C_header_filename_suffix = 'h'
            fmt_library.C_impl_filename_suffix = 'cpp'

            fmt_library.PY_header_filename_suffix = 'hpp'
            fmt_library.PY_impl_filename_suffix = 'cpp'

            fmt_library.LUA_header_filename_suffix = 'hpp'
            fmt_library.LUA_impl_filename_suffix = 'cpp'

            fmt_library.stdlib  = 'std::'

    def XX_to_dict(self):
        """Convert to dictionary.
        Used by util.ExpandedEncoder.
        """
        return dict(
            _fmt=self._fmt
        )

class ClassNode(AstNode):
    def __init__(self):
        pass

class FunctionNode(AstNode):
    def __init__(self):
        pass

