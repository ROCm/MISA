import sys
import fileinput
import copy
import re

class replace_inst_t:
    def __init__(self, origin_name, function_name):
        self.origin_name = origin_name
        self.function_name = function_name

replace_list = [
    replace_inst_t('v_add_u32', 'v_add_nc_u32'),
    replace_inst_t('v_sub_u32', 'v_sub_nc_u32'),
    replace_inst_t('v_subrev_u32', 'v_subrev_nc_u32')
]

def do_remove_asm_comment(original_line):
    new_line = copy.copy(original_line)
    new_line = re.sub('\s*;[\s\w\*\+/-]*\)', ')', new_line)
    return new_line

def do_replace(original_line, replace_item):
    new_line = copy.copy(original_line)
    new_line = re.sub(f'self._emit\(f[\"\']{replace_item.origin_name} ', f'self._emit({replace_item.function_name}(', new_line)
    new_line = re.sub('[\"\']\)', '))', new_line)
    new_line = re.sub('[vs]?\[\{(.*?)\}\]', r'\1', new_line)
    new_line = re.sub('\{(.*?)\}', r'\1', new_line)

    new_line = do_remove_asm_comment(new_line)

    return new_line


def do_replace_v_cmp(original_line):
    new_line = copy.copy(original_line)
    new_line = re.sub('self._emit\(f[\"\'](v_cmp\w*) ', r'self._emit(\1(', new_line)
    new_line = re.sub('[\"\']\)', '))', new_line)
    new_line = re.sub('[vs]?\[\{(.*?)\}\]', r'\1', new_line)
    new_line = re.sub('\{(.*?)\}', r'\1', new_line)

    new_line = re.sub('[,]*\s*vcc\s*[,]*\s*', '', new_line)

    new_line = do_remove_asm_comment(new_line)

    return new_line

def do_replace_v_cndmask(original_line):
    new_line = copy.copy(original_line)
    new_line = re.sub('self._emit\(f[\"\'](v_cndmask\w*) ', r'self._emit(\1(', new_line)
    new_line = re.sub('[\"\']\)', '))', new_line)
    new_line = re.sub('[vs]?\[\{(.*?)\}\]', r'\1', new_line)
    new_line = re.sub('\{(.*?)\}', r'\1', new_line)

    new_line = re.sub('[,]*\s*vcc\s*[,]*\s*', '', new_line)

    new_line = do_remove_asm_comment(new_line)

    return new_line


if __name__ == "__main__":
    file = sys.argv[1]
    with fileinput.FileInput(file, inplace=True, backup='.bak') as input:
        for line in input:
            found = False

            # 1st try simple function replacement
            for replace_item in replace_list:
                if re.search(f'self._emit\(f[\"\']{replace_item.origin_name} ', line) != None:
                    new_line = do_replace(line, replace_item)
                    print(new_line, end='')
                    found = True
                    break
                    # sys.stderr.write(f'{new_line}\n')

            # 2nd, modify the v_cmp* inst
            if not found:
                if re.search('self._emit\(f[\"\']v_cmp\w* ', line) != None:
                    new_line = do_replace_v_cmp(line)
                    print(new_line, end='')
                    found = True

            # 3rd, modify the v_cndmask* inst
            if not found:
                if re.search('self._emit\(f[\"\']v_cndmask\w* ', line) != None:
                    new_line = do_replace_v_cndmask(line)
                    print(new_line, end='')
                    found = True

            if not found:
                print(line, end='')
