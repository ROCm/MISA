import sys

class config_section_t(object):
    def __init__(self, name):
        self.name = name
        self.dict = {}
    def get_name(self):
        return self.name
    def __getitem__(self, key):
        return self.dict[key]
    def __setitem__(self, key, value):
        self.dict[key] = value
    def __contains__(self, key):
        return key in self.dict

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
            print('section:{}'.format(section.get_name()))
            


class config_parser_t(object):
    def __init__(self, config_file):
        self.config_file = config_file

    def parse(self):
        # return a list of section, each section is key-value pair
        def is_section(line):
            if line[0] == '[' and line[-1] == ']':
                return True
            return False
        def is_comment(line):
            if line[0] == '#' or line[0] == ';':
                return True
            return False
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

        def parse_value(value):
            if is_value_int(value):
                return int(value)
            if is_value_float(value):
                return float(value)
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
                    return range(int(tok[0]), int(tok[1]), int(tok[1]))
                assert False, "should not happen"

        config_content = config_content_t()
        with open(self.config_file) as f:
            current_section = None
            lines = f.readlines()
            for x in lines:
                line = x.strip()
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
                    tok = line.split('=')
                    if len(tok != 2):
                        print("fail to parse current line :\"{}\"".format(line))
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
        return config_content
    def __call__(self):
        return self.parse()
