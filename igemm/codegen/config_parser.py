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

class config_section_t(object):
    def __init__(self, name):
        self.dict = {'name': name}
    def get_name(self):
        return self.dict['name']
    def __getitem__(self, key):
        return self.dict[key]
    def __setitem__(self, key, value):
        self.dict[key] = value
    def __iter__(self):
        return self.dict.__iter__()
    def __contains__(self, key):
        return key in self.dict
    def to_dict(self):
        return self.dict

class config_content_t(object):
    def __init__(self):
        self.sections = []
    def add_section(self, section):
        assert type(section) is config_section_t
        self.sections.append(section)
    def __getitem__(self, index):
        return self.sections[index]
    def __len__(self):
        return len(self.sections)

    def dump(self):
        print('total sections:{}'.format(len(self)))
        for section in self:
            print('[{}]'.format(section.get_name()))
            for key in section:
                print('  {} = {} (type:{})'.format(key, section[key], type(section[key])))
    
    def get_section(self, section_name):
        section_list = []
        for section in self:
            if section.get_name() == section_name:
                section_list.append(section)
        if len(section_list) == 0:
            print('no section with name {}'.format(section_name))
        return section_list


class config_parser_t(object):
    def __init__(self, config_file):
        self.config_file = config_file

    def parse(self):
        # return a list of section, each section is key-value pair
        def is_empty(line):
            if len(line) == 0:
                return True
            return False
        def is_section(line):
            if line[0] == '[' and line[-1] == ']':
                return True
            return False
        def is_comment(line):
            if line[0] == '#' or line[0] == ';':
                return True
            return False
        def remove_trailing_comment(line):
            if '#' in line:
                return line.split('#')[0]
            if ';' in line:
                return line.split(';')[0]
            return line
        def is_value_int(value):
            try:
                int(value)
                return True
            except ValueError:
                return False
        def is_value_int(value):
            try:
                int(value)
                return True
            except ValueError:
                return False

        def is_value_float(value):
            if is_value_int(value):
                return False
            try:
                float(value)
                return True
            except ValueError:
                return False

        def is_value_string(value):
            if (value[0] == '\'' and value[-1] == '\'') or \
                (value[0] == '\"' and value[-1] == '\"'):
                return True
            return False

        def is_value_list(value):
            # [x,x,x,x]
            if value[0] == '[' and value[-1] == ']':
                tok = value[1:-1].split(',')
                for i in range(len(tok)):
                    tok[i] = tok[i].strip()
                for i in range(len(tok)):
                    if tok[i] == '':
                        return False
                return True
            return False

        def is_value_range(value):
            # ( [start], end, [step] )
            if value[0] == '(' and value[-1] == ')':
                tok = value[1:-1].split(',')
                for i in range(len(tok)):
                    tok[i] = tok[i].strip()
                for i in range(len(tok)):
                    if tok[i] == '':
                        return False
                if len(tok) == 1 or len(tok) == 2 or len(tok) == 3:
                    return True
                return False
            return False

        def is_value_dict(value):
            if value[0] == '{' and value[-1] == '}':
                tok = value[1:-1].split(',')
                for i in range(len(tok)):
                    tok[i] = tok[i].strip()
                for i in range(len(tok)):
                    if tok[i] == '':
                        return False
                    key_value_pair = tok[i].split('=')
                    if len(key_value_pair) != 2:
                        return False
                    k, v = key_value_pair[0].strip(), key_value_pair[1].strip()
                    if not is_value_string(k):
                        return False
                    if not (is_value_int(v) or \
                            is_value_float(v) or \
                            is_value_string(v) or \
                            is_value_list(v) or \
                            is_value_range(v)):
                        # TODO: recursive dict not supported
                        return False
                return True
            return False

        def parse_value(value):
            if is_value_int(value):
                return int(value)
            if is_value_float(value):
                return float(value)
            if is_value_string(value):
                return str(value[1:-1].strip())
            if is_value_list(value):
                tok = value[1:-1].split(',')
                for i in range(len(tok)):
                    tok[i] = tok[i].strip()
                some_list = []
                for t in tok:
                    if is_value_int(t):
                        some_list.append(int(t))
                    elif is_value_float(t):
                        some_list.append(float(t))
                    elif is_value_string(t):
                        some_list.append(str(t[1:-1].strip()))
                    else:
                        print("value \"{}\" not suitable for list".format(t))
                        sys.exit(-1)
                return some_list
            if is_value_range(value):
                tok = value[1:-1].split(',')
                for i in range(len(tok)):
                    tok[i] = tok[i].strip()
                    if not is_value_int(tok[i]):
                        print("value \"{}\" not suitable for range".format(tok[i]))
                        sys.exit(-1)
                if len(tok) == 1:
                    return range(int(tok[0]))
                if len(tok) == 2:
                    return range(int(tok[0]), int(tok[1]))
                if len(tok) == 3:
                    return range(int(tok[0]), int(tok[1]), int(tok[2]))
                assert False, "should not happen"
            if is_value_dict(value):
                tok = value[1:-1].split(',')
                some_dict = dict()
                for i in range(len(tok)):
                    tok[i] = tok[i].strip()
                    key_value_pair = tok[i].split('=')
                    k, v = key_value_pair[0].strip(), key_value_pair[1].strip()
                    vv = parse_value(v)     # safe to recursively call here, to better create a value type
                    some_dict[k[1:-1]] = vv
                return some_dict

            # finaly return string
            return value

        config_content = config_content_t()
        with open(self.config_file) as f:
            current_section = None
            lines = f.readlines()
            for x in lines:
                line = remove_trailing_comment(x).strip()
                if is_empty(line):
                    continue
                if is_comment(line):
                    continue
                if is_section(line):
                    if current_section is not None:
                        # append to list
                        config_content.add_section(current_section)
                    section_name = line[1:-1].strip()
                    current_section = config_section_t(section_name)
                else:
                    assert current_section is not None
                    tok = line.split('=', 1)
                    if len(tok) != 2:
                        print("fail to parse current line :\"{}\", tok:{}".format(line, tok))
                        sys.exit(-1)
                    key = tok[0].strip()
                    value = tok[1].strip()
                    if key is '' or value is '':
                        print("fail to parse key/value of line :\"{}\"".format(line))
                        sys.exit(-1)
                    if key not in current_section:
                        current_section[key] = parse_value(value)
                    else:
                        print("duplicate key :\"{}\" in current section".format(key))
                        sys.exit(-1)
            if current_section is not None:
                config_content.add_section(current_section)
        return config_content
    def __call__(self):
        return self.parse()

# if __name__ == '__main__':
#     config_parser = config_parser_t("v4r1.conf")
#     config_content = config_parser()
#     config_content.dump()
