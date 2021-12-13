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
# pylint: disable=maybe-no-member

import math
import copy

class tensor_descriptor(object):
    def __init__(self, trans, lower_ids, upper_ids):
        # list of trans_tuples, this is a iterative record of every transform history
        self.trans = copy.deepcopy(trans)       # -> attension! copy every trans history, not reference
        self.lower_ids = lower_ids.copy()
        self.upper_ids = upper_ids.copy()

    def calculate_offset(self, coord, verbose = False):
        '''
        iterate to the original low dim
        '''
        current_upper_coord = coord.copy()
        for itrans in range(len(self.trans) - 1, -1, -1):
            trans = self.trans[itrans]
            tmp_lower_coord = list()
            for t, u in zip(trans, self.upper_ids[itrans]):
                co = [current_upper_coord[i] for i in u]
                tmp_lower_coord.append(t.calculate_lower_index(co))

            unsorted_lower_coord = tensor_util_flattern(tmp_lower_coord)
            unsorted_lower_dims = tensor_util_flattern(self.lower_ids[itrans])

            _, sorted_lower_coord = tensor_util_sort_pairs(unsorted_lower_dims, unsorted_lower_coord)
            if verbose:
                ss = ''
                for t in trans:
                    ss += str(type(t)) + '->'
                print(f'    up coord:{current_upper_coord}, low coord:{sorted_lower_coord}, {ss}')
            current_upper_coord = sorted_lower_coord

        return current_upper_coord[0]

    def get_lengths(self):
        '''
        upper lengths
        '''
        last_trans = self.trans[-1]
        # last_upper_ids = self.upper_ids[-1]

        lengths = list()
        for trans in last_trans:
            lengths.extend(trans.get_upper_lengths())

        unsorted_lengths = tensor_util_flattern(lengths)
        unsorted_dims = tensor_util_flattern(self.upper_ids[-1])

        _, sorted_lengths = tensor_util_sort_pairs(unsorted_dims, unsorted_lengths)

        return sorted_lengths

    def get_length(self, idx):
        lengths = self.get_lengths()
        assert idx < len(lengths)
        return lengths[idx]

    def get_dims(self):
        # upper dims, or visible dims
        last_upper_ids = self.upper_ids[-1]
        # print(*self.upper_ids[-1])
        return len(tensor_util_flattern(last_upper_ids))

class array_like_iterator(object):
    def __init__(self, array_like):
        self.__array_like = array_like
        self.__index = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.__index >= len(self.__array_like):
            raise StopIteration
        item = self.__array_like[self.__index]
        self.__index += 1
        return item

class trans_tuple(object):
    def __init__(self, *args):
        self.content = list()
        for x in args:
            if type(x) == list:
                self.content.append(x.copy())
            elif type(x) == tuple:
                self.content.append(list(x))
            elif type(x) == int:
                self.content.append([x])
            else:
                # any trans like type
                self.content.append(x)

    def __len__(self):
        return len(self.content)

    def __iter__(self):
        return array_like_iterator(self)

    def __getitem__(self, idx):
        return self.content[idx]

    def flattern(self):
        # return a list of flatterned item
        flat_content = list()
        for x in self.content:
            if type(x) == list:
                flat_content.extend(x)
            else:
                flat_content.extend([x])
        return flat_content

def make_tuple(*args):
    return trans_tuple(*args)

def make_naive_tensor_descriptor(lengths, strides):
    assert type(lengths) == list
    assert type(strides) == list
    assert len(lengths) == len(strides)
    trans     = [make_tuple(trans_embed(lengths, strides))]
    lower_ids = [make_tuple(0)]
    upper_ids = [make_tuple([x for x in range(len(lengths))])]

    return tensor_descriptor(trans, lower_ids, upper_ids)

def make_naive_tensor_descriptor_packed(lengths):
    assert type(lengths) == list
    trans     = [make_tuple(trans_unmerge(lengths))]
    lower_ids = [make_tuple(0)]
    upper_ids = [make_tuple([x for x in range(len(lengths))])]

    return tensor_descriptor(trans, lower_ids, upper_ids)

def make_transform_tensor_descriptor(old_desc, new_trans, new_lower_ids, new_upper_ids):
    '''
    we need to concat transformation in old_desc
    '''
    assert (type(new_trans), type(new_lower_ids), type(new_upper_ids)) == (trans_tuple, trans_tuple, trans_tuple)
    assert len(new_trans) == len(new_lower_ids) and len(new_trans) == len(new_upper_ids),   \
            f'trans len:{len(new_trans)}, lower_ids len:{len(new_lower_ids)}, upper_ids len:{len(new_upper_ids)}'
    assert sorted(new_lower_ids.flattern()) == [x for x in range(len(new_lower_ids.flattern()))], \
            f'new_lower_ids must be unique :{new_lower_ids.flattern()}'
    assert sorted(new_upper_ids.flattern()) == [x for x in range(len(new_upper_ids.flattern()))], \
            f'new_upper_ids must be unique :{new_upper_ids.flattern()}'
    assert len(old_desc.get_lengths()) == len(new_lower_ids.flattern()), \
            f'old_desc lengths size:{old_desc.get_lengths()} not match new_lower_ids size:{new_lower_ids.flattern()}'

    all_trans     = [*old_desc.trans, new_trans]
    all_lower_ids = [*old_desc.lower_ids, new_lower_ids]
    all_upper_ids = [*old_desc.upper_ids, new_upper_ids]

    return tensor_descriptor(all_trans, all_lower_ids, all_upper_ids)

def move_tensor_coordinate(desc, coord, coord_step):
    '''
    for simplicity, currently coord, coord_step are just python list
    and we will return a new coord, leave the input coord unchanged
    
    NOTE: move coord will not consider over flow and carry to next dim
    e.g. lengths:[1, 4, 2], coord:[0, 0, 0], step: [1, 2, 1], step also is length 1*2*1
        0# : [0, 0, 0], [0, 1, 0]
        1# : [0, 0, 1], [0, 1, 1]
        2# : [0, 2, 0], [0, 3, 0]
        3# : [0, 2, 1], [0, 3, 1]
    '''
    assert len(coord) == len(coord_step)
    assert len(coord) == len(desc.get_lengths())

    def step_1d(co, le, st):
        new_co = co + st
        overflow = False
        if st > 0:
            if new_co >= le:
                new_co = 0
                overflow = True
        elif st < 0:
            if new_co < 0:
                new_co = le + st
                overflow = True
        else:
            assert False, '0 step has no meaning'
        return overflow, new_co

    lengths = desc.get_lengths()
    acc_idx = len(lengths) - 1       # from right to left. TODO: order

    new_coord = coord.copy()
    while True:
        if acc_idx < 0:
            break
        overflow, new_coord[acc_idx] = step_1d(new_coord[acc_idx],
                                                lengths[acc_idx],
                                                coord_step[acc_idx])
        if overflow:
            acc_idx -= 1
        else:
            break           

    return new_coord

def move_grouped_slice_start_coord(desc, coord_step, index = 0):
    '''
    move trans_grouped_slice.start_coord with coord_step
    index specify which trans_grouped_slice to update, incase there are multiple such transpose

    after this function, desc will be modified
    '''
    def step_start_coord(t, step):
        cur_desc = make_naive_tensor_descriptor_packed(t.low_lengths)
        new_coord = move_tensor_coordinate(cur_desc, t.start_coord, coord_step)
        # print(f'__ {t.start_coord} -> {new_coord}, step:{step}')
        t.start_coord = new_coord
    search_idx = 0
    for itrans in range(len(desc.trans)):
        trans = desc.trans[itrans]
        for it in range(len(trans)):
            t = trans[it]
            if type(t) == trans_grouped_slice:
                if search_idx == index:
                    step_start_coord(t, coord_step)
                else:
                    search_idx += 1

################################
# tensor utility

def tensor_util_split_lengths(groups, lengths, order, mask = list()):
    assert len(order) == len(lengths)
    if len(mask) == 0:
        mask = tensor_util_uniform_sequence_gen(len(order), 1)
    assert len(mask) == len(order)
    g = groups
    split_lengths = lengths.copy()
    for i in order:
        if not mask[i]:
            continue
        s = math.gcd(lengths[i], g)
        g = g // s
        split_lengths[i] = lengths[i] // s
        if g == 1:
            break
    assert g == 1, f'can not evenly split groups:{groups}, among lengths:{lengths}'
    return split_lengths

def tensor_util_arithmetic_sequence_gen(start, end, step):
    assert (end - start) % step == 0
    num = (end - start) // step
    seq = [ start + i * step for i in range(num)]
    return seq

def tensor_util_uniform_sequence_gen(size, value):
    return [value] * size

def tensor_util_flattern(nd_lengths):
    lengths = list()
    for x in nd_lengths:
        if type(x) == list:
            lengths.extend(x)
        elif type(x) == int:
            lengths.extend([x])
    return lengths

def tensor_util_reduce(lengths, func, init_value):
    from functools import reduce
    return reduce(func, lengths, init_value)

def tensor_util_sort_pairs(keys, values):
    assert len(keys) == len(values)
    pairs = [[k, v] for k, v in sorted(zip(keys, values), key = lambda pair : pair[0])]
    zipped_pairs = list(zip(*pairs))
    return list(zipped_pairs[0]), list(zipped_pairs[1])

def tensor_util_reverse_exclusive_scan(lengths, func, init_value):
    lengths_scan = [0] * len(lengths)
    r = init_value
    for i in range(len(lengths) - 1, 0, -1):
        lengths_scan[i] = r
        r = func(r, lengths[i])
    lengths_scan[0] = r
    return lengths_scan

################################
# tensor transformation:
#  |  trans #1 : lower -(trans)-> upper
#  |  trans #2 :                  lower -(trans)-> upper
#  |  trans #3 :                                   lower -(trans)-> upper
#  v
# chained
# 
# to init a transform, feed in constructor with lower length,
# and call get_upper_lengths() to get the length after transform

def _assert_trans_is_valid_upper_idx(trans, upper_idx):
    assert len(upper_idx) == trans.get_upper_dims(), f'upper_idx len:{len(upper_idx)}, upper dims len:{trans.get_upper_dims()}'
    assert all(idx < dim for idx, dim in zip(upper_idx, trans.up_lengths)), f'upper idx:{upper_idx} out-of-bound of up_lengths:{trans.get_upper_lengths()}'

class trans_unmerge(object):
    def __init__(self, up_lengths):
        self.up_lengths = up_lengths.copy()
        self.up_lengths_scan = tensor_util_reverse_exclusive_scan(up_lengths, lambda a, b: a*b, 1)

    def get_upper_lengths(self):
        return self.up_lengths

    def get_lower_dims(self):
        return 1

    def get_upper_dims(self):
        return len(self.up_lengths)

    def calculate_lower_index(self, upper_idx):
        _assert_trans_is_valid_upper_idx(self, upper_idx)
        return [sum(x * y for x, y in zip(upper_idx, self.up_lengths_scan))]

class trans_merge(object):
    def __init__(self, low_lengths):
        self.low_lengths = low_lengths.copy()
        self.low_lengths_scan = tensor_util_reverse_exclusive_scan(low_lengths, lambda a, b: a*b, 1)
        self.up_lengths = [tensor_util_reduce(low_lengths, lambda a, b: a*b, 1)]

    def get_upper_lengths(self):
        return self.up_lengths

    def get_lower_dims(self):
        return len(self.low_lengths)

    def get_upper_dims(self):
        return 1

    def calculate_lower_index(self, upper_idx):
        _assert_trans_is_valid_upper_idx(self, upper_idx)
        lower_idx = [0] * self.get_lower_dims()
        idx = upper_idx[0]
        # print(f'    @@ upper_idx:{upper_idx}, lower_lengths:{self.low_lengths}, low_lengths_scan:{self.low_lengths_scan}')
        for i in range(self.get_lower_dims() - 1):
            lower_idx[i] = idx // self.low_lengths_scan[i]
            idx -= lower_idx[i] * self.low_lengths_scan[i]
        lower_idx[-1] = idx
        return lower_idx

class trans_embed(object):
    def __init__(self, up_lengths, coefficients):
        assert len(up_lengths) == len(coefficients)
        self.up_lengths = up_lengths.copy()
        self.coefficients = coefficients.copy()

    def get_upper_lengths(self):
        return self.up_lengths

    def get_lower_dims(self):
        return 1

    def get_upper_dims(self):
        return len(self.up_lengths)

    def calculate_lower_index(self, upper_idx):
        _assert_trans_is_valid_upper_idx(self, upper_idx)
        return [sum(x * y for x, y in zip(upper_idx, self.coefficients))]

class trans_grouped_slice(object):
    def __init__(self, low_lengths, start_coord, slice_lengths):
        '''
        e.g.
        low_lengths     : [4, 2, 1]
        start_coord     : [0, 0, 0]
        slice_lengths   : [1, 2, 1]
        
        hence will slice in index
            slice 1#: [0, 0, 0], [0, 1, 0]
            slice 1#: [1, 0, 0], [1, 1, 0]
            slice 1#: [2, 0, 0], [2, 1, 0]
            slice 1#: [3, 0, 0], [3, 1, 0]
        '''
        assert len(low_lengths) == len(start_coord)
        assert len(low_lengths) == len(slice_lengths)
        for s, e, l in zip(start_coord, slice_lengths, low_lengths):
            assert s <= l and s <= e and e <= l
        self.low_lengths = low_lengths.copy()
        self.start_coord = start_coord.copy()
        self.up_lengths = slice_lengths.copy()

    def get_upper_lengths(self):
        return self.up_lengths

    def get_lower_dims(self):
        return len(self.up_lengths)

    def get_upper_dims(self):
        return len(self.up_lengths)

    def calculate_lower_index(self, upper_idx):
        _assert_trans_is_valid_upper_idx(self, upper_idx)
        return [x + y for x, y in zip(self.start_coord, upper_idx)]

class trans_vectorize(object):
    def __init__(self, low_length, vector_size):
        self.vector_size = vector_size
        self.up_lengths = [low_length // vector_size]

    def get_upper_lengths(self):
        return self.up_lengths

    def get_lower_dims(self):
        return 1

    def get_upper_dims(self):
        return 1

    def calculate_lower_index(self, upper_idx):
        _assert_trans_is_valid_upper_idx(self, upper_idx)
        return [upper_idx[0] * self.vector_size]

class trans_passthrough(object):
    def __init__(self, low_length):
        self.up_lengths = [low_length]
    
    def get_upper_lengths(self):
        return self.up_lengths

    def get_lower_dims(self):
        return 1

    def get_upper_dims(self):
        return 1

    def calculate_lower_index(self, upper_idx):
        _assert_trans_is_valid_upper_idx(self, upper_idx)
        return upper_idx
