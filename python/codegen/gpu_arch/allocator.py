from abc import abstractclassmethod
from typing import Dict, OrderedDict
from python.codegen.gpu_data_types import reg_block
from python.codegen.gpu_instruct import instruction_type, inst_base
from sortedcontainers import SortedDict

class alloc_unit():
    def __init__(self, l_offset, pos, r_offset) -> None:
        self.l_offset = l_offset
        self.pos = pos
        self.r_offset = r_offset

class base_allocator():
    def __init__(self, mem_size, mem_granularity=1) -> None:
        self.mem_size = mem_size
        self.mem_granularity = mem_granularity
        self.r_pos = 0
    @abstractclassmethod
    def malloc(self, size, alignment):
        pass

    def malloc_opt(self, size, **options):
        pass

    def malloc_at_pos(self, size, position, **options):
        pass

    @abstractclassmethod
    def realloc_dec(self, ptr, size):
        pass

    def mfree(self, ptr):
        pass

    def get_required_size(self):
        'registers required for curent allocation'
        return self.r_pos

    def get_free_dwords_cnt(self):
        return self.r_pos


class stack_allocator(base_allocator):
    def __init__(self, mem_size, mem_granularity=1) -> None:
        super().__init__(mem_size,  mem_granularity=mem_granularity)
        #self.stack = [alloc_unit(0,0,0)]
        self.stack = SortedDict()
        self.stack[0] = alloc_unit(0,0,0)
        self._cur_last_pos = 0

    def malloc(self, size, alignment=0) -> int: 
        _cur_last_pos = self._cur_last_pos
        position = _cur_last_pos
        if alignment:
            aligned_offset = ((position + alignment - 1) // alignment) * alignment
            _cur_last_pos = aligned_offset
            position = aligned_offset
        _cur_last_pos += size
        
        new_unit = alloc_unit(self._cur_last_pos, position, position+size)

        if (self.r_pos < _cur_last_pos):
            self.r_pos = _cur_last_pos
        self._cur_last_pos = _cur_last_pos

        self.stack[position] = new_unit

        return position

    def malloc_at_pos(self, size, position):
        _cur_last_pos = self._cur_last_pos

        if(position < _cur_last_pos):
            assert False
        _cur_last_pos = position + size
        
        if (self.r_pos < _cur_last_pos):
            self.r_pos = _cur_last_pos
        self._cur_last_pos = _cur_last_pos

        return position

    def unit_split(self, general_unit_block:reg_block, sub_units):
        general_block_position = general_unit_block.position
        general_size = general_unit_block.dwords
        
        general_unit = self.stack.pop(general_unit_block.position)[1]
        last_pos = general_unit.l_offset

        for i in range(sub_units):
            cur_sub_unit = sub_units[i]
            pos = general_block_position + cur_sub_unit[0]
            sz = cur_sub_unit[1]
            new_unit = alloc_unit(last_pos, pos, pos+sz)
            last_pos = pos+sz
            self.stack[pos] = new_unit

    def mfree(self, pos):
        self.stack.pop(pos)
