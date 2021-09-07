################################################################################
# 
#  MIT License
# 
#  Copyright (c) 2020-2021 Advanced Micro Devices, Inc.
# 
#  Permission is hereby granted, free of charge, to any person obtaining a copy
#  of this software and associated documentation files (the "Software"), to deal
#  in the Software without restriction, including without limitation the rights
#  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#  copies of the Software, and to permit persons to whom the Software is
#  furnished to do so, subject to the following conditions:
# 
#  The above copyright notice and this permission notice shall be included in all
#  copies or substantial portions of the Software.
# 
#  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#  SOFTWARE.
# 
################################################################################
import sys
import os
import inspect
from copy import deepcopy
import subprocess

# NOTE: if following set to True, better parse '-V 0' to conv_driver
# since result can never be correct
MC_DEBUG_IGNORE_LDS_IO = False
MC_DEBUG_IGNORE_GLOBAL_IO = False

class mc_get_version_t(object):
    def __init__(self):
        self.called = 0
        self.version = ''
    def __call__(self):
        if self.called == 0:
            self.called = 1
            '''
            get current version of generator, by git command
            '''
            cmd = ['git', 'rev-parse', 'HEAD']
            na_version = 'UNKNOWN_VERSION'
            try:
                p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr = subprocess.STDOUT)
                try:
                    (out, _) = p.communicate()
                    if p.returncode != 0:
                        self.version = na_version
                    else:
                        self.version = out.decode('utf-8').strip()
                except Exception as e:
                    self.version = na_version
            except Exception as e:
                self.version = na_version
        return self.version

mc_get_version = mc_get_version_t()

class _mc_indent_context_manager_t(object):
    def __init__(self, indent, enter_func=None, exit_func=None):
        self.indent = indent
        self.enter_func = enter_func
        self.exit_func = exit_func
    def __enter__(self):
        if self.enter_func:
            self.enter_func()
        if self.indent:
            self.indent.inc()
    def __exit__(self, type, value, traceback):
        if self.indent:
            self.indent.dec()
        if self.exit_func:
            self.exit_func()

class _mc_indent_t(object):
    def __init__(self, indent_size_per_level, indent_char = ' '):
        self.indent_size_per_level = indent_size_per_level
        self.level = 0
        self.indent_char = indent_char
        self.indent = ''
    def __call__(self):
        return self.indent
    def _update_indent(self):
        self.indent = ''.join(self.indent_char
            for i in range(self.indent_size_per_level * self.level))
    def inc(self):
        self.level += 1
        self._update_indent()
    def dec(self):
        if self.level > 0:
            self.level -= 1
        self._update_indent()
    def set(self, level):
        self.level = level
        self._update_indent()
    def get(self):
        return self.level

class mc_emit_to_string_t(object):
    def __init__(self, indent = _mc_indent_t(4)):
        self.indent = indent
        self.string_buffer = ''
    def emit(self, s):
        self.string_buffer += self.indent() + s + '\n'
    def open(self):
        pass
    def close(self):
        pass
    def indent_context(self, enter_func=None, exit_func=None):
        return _mc_indent_context_manager_t(self.indent, enter_func, exit_func)
    def inc_indent(self):
        self.indent.inc()
    def dec_indent(self):
        self.indent.dec()
    def get_buffer(self):
        return self.string_buffer
    def set_indent(self, level):
        self.indent.set(level)
    def get_indent(self):
        return self.indent.get()

class mc_emit_to_iostream_t(object):
    def __init__(self, indent = _mc_indent_t(4)):
        self.indent = indent
    def emit(self, s):
        print(self.indent() + s)
    def open(self):
        pass
    def close(self):
        pass
    def indent_context(self, enter_func=None, exit_func=None):
        return _mc_indent_context_manager_t(self.indent, enter_func, exit_func)
    def inc_indent(self):
        self.indent.inc()
    def dec_indent(self):
        self.indent.dec()
    def set_indent(self, level):
        self.indent.set(level)
    def get_indent(self):
        return self.indent.get()

class mc_emit_to_file_t(object):
    # TODO: exception check
    def __init__(self, file_name, indent = _mc_indent_t(4)):
        self.file_name = file_name
        self.f = None
        self.indent = indent
    def __del__(self):
        self.close()
    def emit(self, s):
        if self.f:
            if MC_DEBUG_IGNORE_LDS_IO or MC_DEBUG_IGNORE_GLOBAL_IO:
                s2 = s.split('\n')
                ignore_list = list()
                if MC_DEBUG_IGNORE_LDS_IO:
                    ignore_list.extend(['ds_read', 'ds_write',  's_barrier'])
                    # ignore_list.extend(['ds_write'])
                if MC_DEBUG_IGNORE_GLOBAL_IO:
                    ignore_list.extend(['buffer_load', 's_waitcnt vmcnt'])
                for iss, ss in enumerate(s2):
                    need_emit = True
                    for i in ignore_list:
                        if ss.strip().startswith(i):
                            need_emit = False
                            break
                    if need_emit:
                        self.f.write((self.indent() if iss == 0 else '') + ss + '\n')
            else:
                self.f.write(self.indent() + s + '\n')

    def emit_license(self):
        '''
        emit license should per file
        '''
        self.emit('/*******************************************************************************')
        self.emit(' *')
        self.emit(' * MIT License')
        self.emit(' *')
        self.emit(' * Copyright (c) 2020-2021 Advanced Micro Devices, Inc.')
        self.emit(' *')
        self.emit(' * Permission is hereby granted, free of charge, to any person obtaining a copy')
        self.emit(' * of this software and associated documentation files (the "Software"), to deal')
        self.emit(' * in the Software without restriction, including without limitation the rights')
        self.emit(' * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell')
        self.emit(' * copies of the Software, and to permit persons to whom the Software is')
        self.emit(' * furnished to do so, subject to the following conditions:')
        self.emit(' *')
        self.emit(' * The above copyright notice and this permission notice shall be included in all')
        self.emit(' * copies or substantial portions of the Software.')
        self.emit(' *')
        self.emit(' * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR')
        self.emit(' * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,')
        self.emit(' * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE')
        self.emit(' * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER')
        self.emit(' * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,')
        self.emit(' * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE')
        self.emit(' * SOFTWARE.')
        self.emit(' *')
        self.emit(' *******************************************************************************/')

    def open(self):
        if self.f == None:
            try:
                self.f = open(self.file_name, "w")
            except IOError as e:
                print("can't open file:{}({})".format(self.file_name, e))
                sys.exit()

            self.emit_license()
            self.emit('; generated by igemm_codegen.py ({})'.format(mc_get_version()))
            self.emit(';')
            os.fsync(self.f)

    def close(self):
        if self.f != None:
            os.fsync(self.f)
            self.f.close()
            self.f = None
    def indent_context(self,enter_func=None, exit_func=None):
        return _mc_indent_context_manager_t(self.indent,enter_func, exit_func)
    def inc_indent(self):
        self.indent.inc()
    def dec_indent(self):
        self.indent.dec()
    def set_indent(self, level):
        self.indent.set(level)
    def get_indent(self):
        return self.indent.get()

class mc_deferred_emit_t(object):
    '''
    print to string buffer and manage indent
    '''
    def __init__(self, upper_emitter):
        self.indent = upper_emitter.indent  # manage the indent here
        self.buffer = ''
        self.is_first_line = True
    def emit(self, s):
        if self.is_first_line:
            self.buffer += s
            self.is_first_line = False
        else:
            self.buffer += '\n' + self.indent() + s

    def open(self):
        pass
    def close(self):
        pass
    def indent_context(self, enter_func=None, exit_func=None):
        return _mc_indent_context_manager_t(self.indent, enter_func, exit_func)
    def inc_indent(self):
        self.indent.inc()
    def dec_indent(self):
        self.indent.dec()
    def set_indent(self, level):
        self.indent.set(level)
    def get_indent(self):
        return self.indent.get()
    def get_buffer(self):
        return self.buffer

class mc_asm_printer_t(object):
    '''
    this is the MC
    '''
    def __init__(self, emitter, arch_config):
        self.emitter = emitter
        self.emitter.open()
        self.deferred_buffer = ''
        self.global_bucket = set()          # for uniqueness
        self.unique_emitter_dict = dict()
        self.arch_config = arch_config

    def __del__(self):
        self.emitter.close()
    
    def close(self):
        self.emitter.close()

    def insert_unique(self, k, v):
        # TODO: better check valid emitter
        assert type(k) is str
        assert hasattr(v, 'emit'), 'insert a object must have attribute "emit()"!'
        if k not in self.unique_emitter_dict:
            self.unique_emitter_dict[k] = v

    def emit_all_unique(self):
        # Note! sort by name here!
        for k, v in sorted(self.unique_emitter_dict.items()):
            v.emit()

    def emit(self, s):
        self.emitter.emit(s)

    def emit_empty_line(self):
        indent_level = self.emitter.get_indent()
        self.emitter.set_indent(0)
        self.emitter.emit('')
        self.emitter.set_indent(indent_level)

    def emit_macro_indented(self, macro_define_str):
        def macro_enter():
            self.emit(macro_define_str)
        def macro_exit():
            self.emit('.endm')
            self.emit_empty_line()
        return self.indent_context(macro_enter, macro_exit)

    def emit_macro_desc(self, *misc):
        self.emit('; ' + ' '.join( '{}'.format(e) for e in misc))

    def emit_front(self, s):
        indent_level = self.emitter.get_indent()
        self.emitter.set_indent(0)
        self.emitter.emit(s)
        self.emitter.set_indent(indent_level)

    def inc_indent(self):
        self.emitter.inc_indent()
    def dec_indent(self):
        self.emitter.dec_indent()
    def indent_context(self,enter_func=None, exit_func=None):
        return self.emitter.indent_context(enter_func,exit_func)

    def deferred_context(self):
        class deferred_context_t(object):
            def __init__(self, outter):
                self.outter = outter
                self.original_emitter = outter.emitter
                self.deferred_emitter = mc_deferred_emit_t(self.original_emitter)
            def __enter__(self):
                self.outter.emitter = self.deferred_emitter
            def __exit__(self, type, value, traceback):
                self.outter.emitter = self.original_emitter
                self.outter.deferred_buffer = self.deferred_emitter.get_buffer()
        return deferred_context_t(self)

    def get_deferred(self):
        return self.deferred_buffer

    def inject(self, other):
        '''
        useful to inject some control func here
        '''
        #def _emit_unique_wrapper():
        #    self.emit_unique(other)
        def _macro_desc_wrapper(*misc):
            self.emit_macro_desc(inspect.cleandoc(other.__doc__), *misc)
        other._emit = self.emit
        other._emit_empty_line = self.emit_empty_line
        other._emit_macro_indented = self.emit_macro_indented
        other._emit_macro_desc = _macro_desc_wrapper
        other._emit_front = self.emit_front
        #other._emit_unique = _emit_unique_wrapper
        other._inc_indent = self.inc_indent
        other._dec_indent = self.dec_indent
        other._indent_context = self.indent_context
        other._deferred_context = self.deferred_context
        other._get_deferred = self.get_deferred
        other._insert_unique = self.insert_unique

class mc_base_t(object):
    '''
    helper class for mc inject, any class need do emit should inherit from this
    '''
    def __init__(self, mc):
        self.mc = mc
        mc.inject(self)

'''
current using global mc
'''
_current_mc = None
def mc_set_current(mc):
    '''
    TODO: find a better way to set the global variable
    '''
    global _current_mc
    _current_mc = mc

def mc_get_current():
    return _current_mc
