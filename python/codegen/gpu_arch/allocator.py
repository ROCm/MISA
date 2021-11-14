from abc import ABC, abstractclassmethod
from typing import Dict, List, OrderedDict
from python.codegen.gpu_data_types import reg_block
from python.codegen.gpu_instruct import instruction_type, inst_base
from sortedcontainers import SortedDict



class base_allocator(ABC):
    def __init__(self, mem_size, mem_granularity=1) -> None:
        self.mem_size = mem_size
        self.mem_granularity = mem_granularity
        self.r_pos = 0
    @abstractclassmethod
    def malloc(self, size, alignment) -> int:
        pass

    def malloc_opt(self, size, **options) -> int:
        pass

    @abstractclassmethod
    def malloc_at_fixed_pos(self, size, position, **options):
        pass

    #@abstractclassmethod
    #def realloc_dec(self, ptr, size):
    #    pass

    @abstractclassmethod
    def mfree(self, ptr):
        pass

    def get_required_size(self):
        'registers required for curent allocation'
        return self.r_pos

    def get_free_dwords_cnt(self):
        return self.r_pos


class stack_allocator(base_allocator):

    class alloc_unit():
        def __init__(self, l_offset, pos, r_offset) -> None:
            self.l_offset = l_offset
            self.pos = pos
            self.r_offset = r_offset

    def __init__(self, mem_size, mem_granularity=1) -> None:
        super().__init__(mem_size,  mem_granularity=mem_granularity)
        #self.stack = [alloc_unit(0,0,0)]
        self.stack = SortedDict()
        self.stack[0] = stack_allocator.alloc_unit(0,0,0)
        self._cur_last_pos = 0

    def malloc(self, size, alignment=0) -> int: 
        _cur_last_pos = self._cur_last_pos
        position = _cur_last_pos
        if alignment:
            aligned_offset = ((position + alignment - 1) // alignment) * alignment
            _cur_last_pos = aligned_offset
            position = aligned_offset
        _cur_last_pos += size
        
        new_unit = stack_allocator.alloc_unit(self._cur_last_pos, position, position+size)

        if (self.r_pos < _cur_last_pos):
            self.r_pos = _cur_last_pos
        self._cur_last_pos = _cur_last_pos

        self.stack[position] = new_unit

        return position

    def malloc_at_fixed_pos(self, size, position):
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
        
        general_unit = self.stack.pop(general_unit_block.position)
        last_pos = general_unit.l_offset

        for i in range(sub_units):
            cur_sub_unit = sub_units[i]
            pos = general_block_position + cur_sub_unit[0]
            sz = cur_sub_unit[1]
            new_unit = stack_allocator.alloc_unit(last_pos, pos, pos+sz)
            last_pos = pos+sz
            self.stack[pos] = new_unit

    def mfree(self, pos):
        self.stack.pop(pos)
        try:
            last:stack_allocator.alloc_unit =  self.stack.peekitem()[1]
            self._cur_last_pos = last.r_offset
        except IndexError:
            self._cur_last_pos = 0

    def get_required_size(self):
        'registers required for curent allocation'
        return self.r_pos


class onDemand_allocator(base_allocator):

    class alloc_unit():
        def __init__(self, pos, size) -> None:
            self.pos = pos
            self.pos_end = pos + size
            self.size = size

    def __init__(self, mem_size, mem_granularity=1) -> None:
        super().__init__(mem_size,  mem_granularity=mem_granularity)
        #self.stack = [alloc_unit(0,0,0)]
        self.allocated_units = SortedDict()
        self.allocated_units[0] = self.alloc_unit(0,0)

        self.unallocated_units = SortedDict()
        self.unallocated_units[0] = self.alloc_unit(0,mem_size)

        self._cur_max_size = 0
        self._limit = mem_size
    
    def _update_alloc_size(self):
        try:
            total_allocated = self.unallocated_units.peekitem()[1].pos
        except IndexError:            
            total_allocated = self._cur_max_size

        if(self._cur_max_size < total_allocated):
            self._cur_max_size = total_allocated


    def malloc(self, size, alignment=1) -> int: 
        cur_max_size = self._cur_max_size
        
        selected_block:onDemand_allocator.alloc_unit = self.unallocated_units.peekitem(0)[1]

        for i in self.unallocated_units:
            cur_block = self.unallocated_units[i]
            i_size = cur_block.size
            if(i_size >= size):
                aligned_offset = ((cur_block.pos % alignment) - alignment) if alignment else cur_block.pos
                if(i_size >= size + aligned_offset):
                    if(cur_block.size < selected_block.size):
                        selected_block = cur_block

        if(selected_block.size<size):
            assert(False)

        s_pos = selected_block.pos
        self.unallocated_units.pop(s_pos)
        position = s_pos
        if alignment > 1:
            position = ((s_pos + alignment - 1) // alignment) * alignment
            if(position > s_pos):
                self.unallocated_units[s_pos] = onDemand_allocator.alloc_unit(s_pos, position-s_pos)
        
        r_unallocated_space = (selected_block.size + selected_block.pos) - (position + size)
        if(r_unallocated_space > 0):
            self.unallocated_units[position + size] = onDemand_allocator.alloc_unit(position + size, r_unallocated_space)
        
        new_allocated_unit = onDemand_allocator.alloc_unit(position, size)
        
        self.allocated_units[position] = new_allocated_unit

        self._update_alloc_size()

        return position

    def malloc_at_fixed_pos(self, size, position):
        
        selected_block:onDemand_allocator.alloc_unit = None

        for i in self.unallocated_units:
            block = self.unallocated_units[i]
            b_size = block.size
            b_pos = block.pos

            if(b_pos <= position):
                if(b_pos + b_size >= position + size):
                    selected_block = block

        if(selected_block == None):
            assert(False)

        s_pos = selected_block.pos
        if (s_pos < position):
            self.unallocated_units[s_pos] = onDemand_allocator.alloc_unit(s_pos, position-s_pos)
        else:
            self.unallocated_units.pop(s_pos)
        
        r_unallocated_space = (selected_block.size + selected_block.pos) - (position + size)
        if(r_unallocated_space > 0):
            self.unallocated_units[position + size] = onDemand_allocator.alloc_unit(position + size, r_unallocated_space)
        
        new_allocated_unit = onDemand_allocator.alloc_unit(position, size)
        self.allocated_units[position] = new_allocated_unit
        
        self._update_alloc_size()

        return position

    def unit_split(self, general_unit_block:reg_block, sub_units):
        
        general_block_position = general_unit_block.position
        general_size = general_unit_block.dwords
        
        #general_unit = self.allocated_units.pop(general_block_position)
        self.mfree(general_block_position)

        last_pos = general_block_position
        
        for i in range(sub_units):
            cur_unit_offset, cur_unit_sz = sub_units[i]
            pos = general_block_position + cur_unit_offset
            self.malloc_at_fixed_pos(cur_unit_sz, pos)

    def mfree(self, pos):
        
        selected_unit = self.allocated_units.pop(pos)

        l_neighbour:onDemand_allocator.alloc_unit = None
        r_neighbour:onDemand_allocator.alloc_unit = None

        for i in self.unallocated_units:
            block = self.unallocated_units[i]
            if(block.pos < pos):
                l_neighbour = block
            if(block.pos > pos):
                r_neighbour = block

        new_unallocated_unit = onDemand_allocator.alloc_unit(selected_unit.pos, selected_unit.size)
        if(l_neighbour):
            if(l_neighbour.pos_end == pos):
                new_unallocated_unit.pos = l_neighbour.pos
                new_unallocated_unit.size += l_neighbour.size
                self.unallocated_units.pop(l_neighbour.pos)
        
        if(r_neighbour):
            if(r_neighbour.pos == new_unallocated_unit.pos_end):
                new_unallocated_unit.pos_end = r_neighbour.pos_end
                new_unallocated_unit.size += r_neighbour.size
                self.unallocated_units.pop(r_neighbour.pos)

        self.unallocated_units[new_unallocated_unit.pos] = new_unallocated_unit

    def get_required_size(self):
        'registers required for curent allocation'
        return self._cur_max_size
