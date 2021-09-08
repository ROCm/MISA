from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum, auto
from typing import List, Optional, Union
from python.codegen.gpu_reg_block import reg_block

class inst_mode(Enum):
    allocation: auto
    emmition: auto

class instruction_type(Enum):
    SMEM = 'SMEM'
    VMEM = 'VMEM'
    VOP1 = 'VOP1'

class inst_base(ABC):
    __slots__ = ['label', 'inst_type']
    def __init__(self, inst_type:instruction_type, label:str) -> None:
        self.inst_type:instruction_type = inst_type
        self.label = label

    @abstractmethod
    def __str__(self) -> str:
        return f'{self.label}'

    def emit(self) -> str:
        return self.__str__()

class inst_caller_base(ABC):
    __slots__ = ['ic']
    def __init__(self, insturction_container:List[inst_base]) -> None:
        self.ic = insturction_container

    def ic_pb(self, inst):
        self.ic.append(inst)
        return inst

class SMEM_base(inst_base):
    __slots__ = ['sdata','sbase','soffset', 'glc_dlc']
    def __init__(self,
            label: str, sdata:reg_block, sbase:reg_block, soffset=0,
            glc:bool = False, dlc:bool = False
            ) -> None:
        super().__init__(instruction_type.SMEM, label)
        self.sdata = sdata
        self.sbase = sbase
        self.soffset = soffset
        self.glc_dlc = [' glc' if glc else '', ' dlc' if dlc else '']
    
    def __str__(self):
        return f'{self.label} {self.sdata}, {self.sbase}, 0+{self.soffset}{self.glc_dlc[0]}{self.glc_dlc[1]}'

class SMEM_instr_caller(inst_caller_base):
    
    def s_load_dword(self, dwords_cnt:int, sdata:reg_block, sbase:reg_block, soffset=0, glc=False, dlc=False) -> SMEM_base:
        label = f's_load_dwordx{dwords_cnt}' if dwords_cnt > 1 else 's_load_dword'
        return SMEM_base(label, sdata, sbase, soffset)

    def s_buffer_load_dword(self, dwords_cnt:int, sdst:reg_block, sbase:reg_block, soffset=0, glc=False, dlc=False) -> SMEM_base:
        label = f's_buffer_load_dwordx{dwords_cnt}' if dwords_cnt > 1 else 's_buffer_load_dword'
        return SMEM_base(label, sdst, sbase, soffset, glc, dlc)

    def s_buffer_store_dword(self, dwords_cnt:int, sdata:reg_block, sbase:reg_block, soffset=0, glc=False) -> SMEM_base:
        label = f's_buffer_store_dwordx{dwords_cnt}' if dwords_cnt > 1 else 's_buffer_store_dword'
        return SMEM_base(label, sdata, sbase, soffset)

class VMEM_instr_caller(inst_caller_base):
    
    def BUFFER_LOAD(self, size:str, *args):
        label = f'BUFFER_LOAD_{size}'

        return 

class VOP1_base(inst_base):
    __slots__ = ['vdst','src']
    def __init__(self, label: str, vdst:Union[reg_block,None], src:Union[reg_block,None]):
        super().__init__(instruction_type.VOP1, label)
        self.vdst = vdst
        self.src = src
    

    def __str__(self) -> str:
        vdst = self.vdst
        if(vdst !=None):
            return f'{self.label} {vdst}, {self.src}'
        else:
            return f'{self.label}'
    
    @classmethod
    def part_init(cls, label:str):
        inst = VOP1_base(label, None, None)
        return inst

    def set_args(self, vdst:Union[reg_block,None]=None, src:Union[reg_block,None]=None):
        self.vdst = vdst
        self.src = src
        return self
    
    def set_args2(self):
        self.vdst = None
        self.src = None
        return self

    def emit(self):
        return self.__str__()

class VOP1_instr_caller(inst_caller_base):

    def __init__(self, insturction_container) -> None:
        super().__init__(insturction_container)

    def v_clrexcp(self):
        return self.ic_pb(VOP1_base('v_clrexcp', None,None))
    def v_nop(self):
        return self.ic_pb(VOP1_base('v_nop', None,None))
    def v_pipeflush(self):
        return self.ic_pb(VOP1_base('v_pipeflush', None,None))

    def v_bfrev_b32(self, vdst:reg_block,src:reg_block):
        return self.ic_pb(VOP1_base('v_bfrev_b32', vdst,src))
    def v_ceil_f16 (self, vdst:reg_block,src:reg_block):
        return self.ic_pb(VOP1_base('v_ceil_f16', vdst,src))
    def v_ceil_f32 (self, vdst:reg_block,src:reg_block):
        return self.ic_pb(VOP1_base('v_ceil_f32', vdst,src))
    def v_ceil_f64 (self, vdst:reg_block,src:reg_block):
        return self.ic_pb(VOP1_base('v_ceil_f64', vdst,src))
    
    #def __getattribute__(self, name: str):
    #    if name in ['v_clrexcp', 'v_nop', 'v_pipeflush']:
    #        #return functools.partial(VOP1_base,name, None,None)
    #        return self.ic_pb(VOP1_base.part_init(name).set_args2
    #    else:
    #        return self.ic_pb(VOP1_base.part_init(name).set_args
    #        #return functools.partial(VOP1_base,name)
        
    def v_cos_f16         (self, vdst, src):
        return self.ic_pb(VOP1_base('v_cos_f16', vdst,src))
    def v_cos_f32         (self, vdst, src):
        return self.ic_pb(VOP1_base('v_cos_f32', vdst,src))
    def v_cvt_f16_f32     (self, vdst, src):
        return self.ic_pb(VOP1_base('v_cvt_f16_f32', vdst,src))
    def v_cvt_f16_i16     (self, vdst, src):
        return self.ic_pb(VOP1_base('v_cvt_f16_i16', vdst,src))
    def v_cvt_f16_u16     (self, vdst, src):
        return self.ic_pb(VOP1_base('v_cvt_f16_u16', vdst,src))
    def v_cvt_f32_f16     (self, vdst, src):
        return self.ic_pb(VOP1_base('v_cvt_f32_f16', vdst,src))
    def v_cvt_f32_f64     (self, vdst, src):
        return self.ic_pb(VOP1_base('v_cvt_f32_f64', vdst,src))
    def v_cvt_f32_i32     (self, vdst, src):
        return self.ic_pb(VOP1_base('v_cvt_f32_i32', vdst,src))
    def v_cvt_f32_u32     (self, vdst, src):
        return self.ic_pb(VOP1_base('v_cvt_f32_u32', vdst,src))
    def v_cvt_f32_ubyte0  (self, vdst, src):
        return self.ic_pb(VOP1_base('v_cvt_f32_ubyte0', vdst,src))
    def v_cvt_f32_ubyte1  (self, vdst, src):
        return self.ic_pb(VOP1_base('v_cvt_f32_ubyte1', vdst,src))
    def v_cvt_f32_ubyte2  (self, vdst, src):
        return self.ic_pb(VOP1_base('v_cvt_f32_ubyte2', vdst,src))
    def v_cvt_f32_ubyte3  (self, vdst, src):
        return self.ic_pb(VOP1_base('v_cvt_f32_ubyte3', vdst,src))
    def v_cvt_f64_f32     (self, vdst, src):
        return self.ic_pb(VOP1_base('v_cvt_f64_f32', vdst,src))
    def v_cvt_f64_i32     (self, vdst, src):
        return self.ic_pb(VOP1_base('v_cvt_f64_i32', vdst,src))
    def v_cvt_f64_u32     (self, vdst, src):
        return self.ic_pb(VOP1_base('v_cvt_f64_u32', vdst,src))
    def v_cvt_flr_i32_f32 (self, vdst, src):
        return self.ic_pb(VOP1_base('v_cvt_flr_i32_f32', vdst,src))
    def v_cvt_i16_f16     (self, vdst, src):
        return self.ic_pb(VOP1_base('v_cvt_i16_f16', vdst,src))
    
    def v_cvt_i32_f32    (self, vdst, src):
        return self.ic_pb(VOP1_base('v_cvt_i32_f32', vdst,src))
    def v_cvt_i32_f64      (self, vdst, src):
        return self.ic_pb(VOP1_base('v_cvt_i32_f64',vdst,src))
    def v_cvt_norm_i16_f16 (self, vdst, src):
        return self.ic_pb(VOP1_base('v_cvt_norm_i16_f16',vdst,src))
    def v_cvt_norm_u16_f16 (self, vdst, src):
        return self.ic_pb(VOP1_base('v_cvt_norm_u16_f16',vdst,src))
    def v_cvt_off_f32_i4   (self, vdst, src):
        return self.ic_pb(VOP1_base('v_cvt_off_f32_i4',vdst,src))
    def v_cvt_rpi_i32_f32  (self, vdst, src):
        return self.ic_pb(VOP1_base('v_cvt_rpi_i32_f32',vdst,src))
    def v_cvt_u16_f16      (self, vdst, src):
        return self.ic_pb(VOP1_base('v_cvt_u16_f16',vdst,src))
    def v_cvt_u32_f32      (self, vdst, src):
        return self.ic_pb(VOP1_base('v_cvt_u32_f32',vdst,src))
    def v_cvt_u32_f64      (self, vdst, src):
        return self.ic_pb(VOP1_base('v_cvt_u32_f64',vdst,src))
    def v_exp_f16          (self, vdst, src):
        return self.ic_pb(VOP1_base('v_exp_f16',vdst,src))
    def v_exp_f32          (self, vdst, src):
        return self.ic_pb(VOP1_base('v_exp_f32',vdst,src))
    def v_ffbh_i32         (self, vdst, src):
        return self.ic_pb(VOP1_base('v_ffbh_i32',vdst,src))
    def v_ffbh_u32         (self, vdst, src):
        return self.ic_pb(VOP1_base('v_ffbh_u32',vdst,src))
    def v_ffbl_b32         (self, vdst, src):
        return self.ic_pb(VOP1_base('v_ffbl_b32',vdst,src))
    def v_floor_f16        (self, vdst, src):
        return self.ic_pb(VOP1_base('v_floor_f16',vdst,src))
    def v_floor_f32        (self, vdst, src):
        return self.ic_pb(VOP1_base('v_floor_f32',vdst,src))
    def v_floor_f64        (self, vdst, src):
        return self.ic_pb(VOP1_base('v_floor_f64',vdst,src))
    def v_fract_f16        (self, vdst, src):
        return self.ic_pb(VOP1_base('v_fract_f16',vdst,src))
    def v_fract_f32        (self, vdst, src):
        return self.ic_pb(VOP1_base('v_fract_f32',vdst,src))
    def v_fract_f64        (self, vdst, src):
        return self.ic_pb(VOP1_base('v_fract_f64',vdst,src))
    def v_frexp_exp_i16_f16(self, vdst, src):
        return self.ic_pb(VOP1_base('v_frexp_exp_i16_f16(self,',vdst,src))
    def v_frexp_exp_i32_f32(self, vdst, src):
        return self.ic_pb(VOP1_base('v_frexp_exp_i32_f32(self,',vdst,src))
    def v_frexp_exp_i32_f64(self, vdst, src):
        return self.ic_pb(VOP1_base('v_frexp_exp_i32_f64(self,',vdst,src))
    def v_frexp_mant_f16   (self, vdst, src):
        return self.ic_pb(VOP1_base('v_frexp_mant_f16',vdst,src))
    def v_frexp_mant_f32   (self, vdst, src):
        return self.ic_pb(VOP1_base('v_frexp_mant_f32',vdst,src))
    def v_frexp_mant_f64   (self, vdst, src):
        return self.ic_pb(VOP1_base('v_frexp_mant_f64',vdst,src))
    def v_log_f16          (self, vdst, src):
        return self.ic_pb(VOP1_base('v_log_f16',vdst,src))
    def v_log_f32          (self, vdst, src):
        return self.ic_pb(VOP1_base('v_log_f32',vdst,src))
    def v_mov_b32          (self, vdst, src):
        return self.ic_pb(VOP1_base('v_mov_b32',vdst,src))
    def v_movreld_b32      (self, vdst, src):
        return self.ic_pb(VOP1_base('v_movreld_b32',vdst,src))
    def v_movrels_b32      (self, vdst, vsrc):
        return self.ic_pb(VOP1_base('v_movrels_b32',vdst,vsrc))
    def v_movrelsd_2_b32   (self, vdst, vsrc):
        return self.ic_pb(VOP1_base('v_movrelsd_2_b32',vdst,vsrc))
    def v_movrelsd_b32     (self, vdst, vsrc):
        return self.ic_pb(VOP1_base('v_movrelsd_b32',vdst,vsrc))
    def v_not_b32          (self, vdst, src):
        return self.ic_pb(VOP1_base('v_not_b32',vdst,src))
    def v_rcp_f16          (self, vdst, src):
        return self.ic_pb(VOP1_base('v_rcp_f16',vdst,src))
    def v_rcp_f32          (self, vdst, src):
        return self.ic_pb(VOP1_base('v_rcp_f32',vdst,src))
    def v_rcp_f64          (self, vdst, src):
        return self.ic_pb(VOP1_base('v_rcp_f64',vdst,src))
    def v_rcp_iflag_f32    (self, vdst, src):
        return self.ic_pb(VOP1_base('v_rcp_iflag_f32',vdst,src))
    def v_readfirstlane_b32(self, sdst, src):
        return self.ic_pb(VOP1_base('v_readfirstlane_b32', sdst,src))
    def v_rndne_f16        (self, vdst, src):
        return self.ic_pb(VOP1_base('v_rndne_f16',vdst,src))
    def v_rndne_f32        (self, vdst, src):
        return self.ic_pb(VOP1_base('v_rndne_f32',vdst,src))
    def v_rndne_f64        (self, vdst, src):
        return self.ic_pb(VOP1_base('v_rndne_f64',vdst,src))
    def v_rsq_f16          (self, vdst, src):
        return self.ic_pb(VOP1_base('v_rsq_f16',vdst,src))
    def v_rsq_f32          (self, vdst, src):
        return self.ic_pb(VOP1_base('v_rsq_f32',vdst,src))
    def v_rsq_f64          (self, vdst, src):
        return self.ic_pb(VOP1_base('v_rsq_f64',vdst,src))
    def v_sat_pk_u8_i16    (self, vdst_pack,src):
        return self.ic_pb(VOP1_base('v_sat_pk_u8_i16',vdst_pack,src))
    def v_sin_f16          (self, vdst, src):
        return self.ic_pb(VOP1_base('v_sin_f16',vdst,src))
    def v_sin_f32          (self, vdst, src):
        return self.ic_pb(VOP1_base('v_sin_f32',vdst,src))
    def v_sqrt_f16         (self, vdst, src):
        return self.ic_pb(VOP1_base('v_sqrt_f16',vdst,src))
    def v_sqrt_f32         (self, vdst, src):
        return self.ic_pb(VOP1_base('v_sqrt_f32',vdst,src))
    def v_sqrt_f64         (self, vdst, src):
        return self.ic_pb(VOP1_base('v_sqrt_f64',vdst,src))
    def v_swap_b32         (self, vdst, vsrc):
        return self.ic_pb(VOP1_base('v_swap_b32',vdst,vsrc))
    def v_swaprel_b32      (self, vdst, vsrc):
        return self.ic_pb(VOP1_base('v_swaprel_b32',vdst,vsrc))
    def v_trunc_f16        (self, vdst, src):
        return self.ic_pb(VOP1_base('v_trunc_f16',vdst,src))
    def v_trunc_f32        (self, vdst, src):
        return self.ic_pb(VOP1_base('v_trunc_f32',vdst,src))
    def v_trunc_f64        (self, vdst, src):
        return self.ic_pb(VOP1_base('v_trunc_f64',vdst,src))

class SOP1_base(inst_base):
    __slots__ = ['arg1','arg2']
    def __init__(self, label: str, dst:Union[reg_block,None], src:Union[reg_block,None,str]=None):
        super().__init__(instruction_type.VOP1, label)
        self.arg1 = dst
        self.arg2 = src
    
    def __str__(self) -> str:
        arg2 = self.arg2
        arg1 = self.arg1
        if(arg2 !=None):
            return f'{self.label} {arg1}, {arg2}'
        else:
            return f'{self.label} {arg1}'

    def set_args(self, vdst:Union[reg_block,None]=None, src:Union[reg_block,None]=None):
        self.vdst = vdst
        self.src = src
        return self
    
    def set_args2(self, arg1:Union[reg_block,None]=None,):
        self.arg1 = arg1
        self.arg2 = None
        return self

    def emit(self):
        return self.__str__()

class SOP1_instr_caller(inst_caller_base):
    def __init__(self, insturction_container) -> None:
        super().__init__(insturction_container)

    def s_abs_i32              (self, sdst, ssrc):
        return self.ic_pb(SOP1_base('s_abs_i32', sdst, ssrc))
    def s_and_saveexec_b32     (self, sdst, ssrc): 
        return self.ic_pb(SOP1_base('s_and_saveexec_b32', sdst, ssrc))
    def s_and_saveexec_b64     (self, sdst, ssrc): 
        return self.ic_pb(SOP1_base('s_and_saveexec_b64', sdst, ssrc))
    def s_andn1_saveexec_b32   (self, sdst, ssrc): 
        return self.ic_pb(SOP1_base('s_andn1_saveexec_b32', sdst, ssrc))
    def s_andn1_saveexec_b64   (self, sdst, ssrc): 
        return self.ic_pb(SOP1_base('s_andn1_saveexec_b64', sdst, ssrc))
    def s_andn1_wrexec_b32     (self, sdst, ssrc): 
        return self.ic_pb(SOP1_base('s_andn1_wrexec_b32', sdst, ssrc))
    def s_andn1_wrexec_b64     (self, sdst, ssrc): 
        return self.ic_pb(SOP1_base('s_andn1_wrexec_b64', sdst, ssrc))
    def s_andn2_saveexec_b32   (self, sdst, ssrc): 
        return self.ic_pb(SOP1_base('s_andn2_saveexec_b32', sdst, ssrc))
    def s_andn2_saveexec_b64   (self, sdst, ssrc): 
        return self.ic_pb(SOP1_base('s_andn2_saveexec_b64', sdst, ssrc))
    def s_andn2_wrexec_b32     (self, sdst, ssrc): 
        return self.ic_pb(SOP1_base('s_andn2_wrexec_b32', sdst, ssrc))
    def s_andn2_wrexec_b64     (self, sdst, ssrc): 
        return self.ic_pb(SOP1_base('s_andn2_wrexec_b64', sdst, ssrc))
    def s_bcnt0_i32_b32        (self, sdst, ssrc): 
        return self.ic_pb(SOP1_base('s_bcnt0_i32_b32', sdst, ssrc))
    def s_bcnt0_i32_b64        (self, sdst, ssrc): 
        return self.ic_pb(SOP1_base('s_bcnt0_i32_b64', sdst, ssrc))
    def s_bcnt1_i32_b32        (self, sdst, ssrc): 
        return self.ic_pb(SOP1_base('s_bcnt1_i32_b32', sdst, ssrc))
    def s_bcnt1_i32_b64        (self, sdst, ssrc): 
        return self.ic_pb(SOP1_base('s_bcnt1_i32_b64', sdst, ssrc))
    def s_bitreplicate_b64_b32 (self, sdst, ssrc): 
        return self.ic_pb(SOP1_base('s_bitreplicate_b64_b32', sdst, ssrc))
    def s_bitset0_b32          (self, sdst, ssrc): 
        return self.ic_pb(SOP1_base('s_bitset0_b32', sdst, ssrc))
    def s_bitset0_b64          (self, sdst, ssrc): 
        return self.ic_pb(SOP1_base('s_bitset0_b64', sdst, ssrc))
    def s_bitset1_b32          (self, sdst, ssrc): 
        return self.ic_pb(SOP1_base('s_bitset1_b32', sdst, ssrc))
    def s_bitset1_b64          (self, sdst, ssrc): 
        return self.ic_pb(SOP1_base('s_bitset1_b64', sdst, ssrc))
    def s_brev_b32             (self, sdst, ssrc): 
        return self.ic_pb(SOP1_base('s_brev_b32', sdst, ssrc))
    def s_brev_b64             (self, sdst, ssrc): 
        return self.ic_pb(SOP1_base('s_brev_b64', sdst, ssrc))
    def s_cmov_b32             (self, sdst, ssrc): 
        return self.ic_pb(SOP1_base('s_cmov_b32', sdst, ssrc))
    def s_cmov_b64             (self, sdst, ssrc): 
        return self.ic_pb(SOP1_base('s_cmov_b64', sdst, ssrc))
    def s_ff0_i32_b32          (self, sdst, ssrc): 
        return self.ic_pb(SOP1_base('s_ff0_i32_b32', sdst, ssrc))
    def s_ff0_i32_b64          (self, sdst, ssrc): 
        return self.ic_pb(SOP1_base('s_ff0_i32_b64', sdst, ssrc))
    def s_ff1_i32_b32          (self, sdst, ssrc): 
        return self.ic_pb(SOP1_base('s_ff1_i32_b32', sdst, ssrc))
    def s_ff1_i32_b64          (self, sdst, ssrc): 
        return self.ic_pb(SOP1_base('s_ff1_i32_b64', sdst, ssrc))
    def s_flbit_i32            (self, sdst, ssrc): 
        return self.ic_pb(SOP1_base('s_flbit_i32', sdst, ssrc))
    def s_flbit_i32_b32        (self, sdst, ssrc): 
        return self.ic_pb(SOP1_base('s_flbit_i32_b32', sdst, ssrc))
    def s_flbit_i32_b64        (self, sdst, ssrc): 
        return self.ic_pb(SOP1_base('s_flbit_i32_b64', sdst, ssrc))
    def s_flbit_i32_i64        (self, sdst, ssrc): 
        return self.ic_pb(SOP1_base('s_flbit_i32_i64', sdst, ssrc))
    def s_getpc_b64            (self, sdst      ): 
        return self.ic_pb(SOP1_base('s_getpc_b64', sdst ))
    def s_mov_b32              (self, sdst, ssrc): 
        return self.ic_pb(SOP1_base('s_mov_b32', sdst, ssrc))
    def s_mov_b64              (self, sdst, ssrc): 
        return self.ic_pb(SOP1_base('s_mov_b64', sdst, ssrc))
    def s_movreld_b32          (self, sdst, ssrc): 
        return self.ic_pb(SOP1_base('s_movreld_b32', sdst, ssrc))
    def s_movreld_b64          (self, sdst, ssrc): 
        return self.ic_pb(SOP1_base('s_movreld_b64', sdst, ssrc))
    def s_movrels_b32          (self, sdst, ssrc): 
        return self.ic_pb(SOP1_base('s_movrels_b32', sdst, ssrc))
    def s_movrels_b64          (self, sdst, ssrc): 
        return self.ic_pb(SOP1_base('s_movrels_b64', sdst, ssrc))
    def s_movrelsd_2_b32       (self, sdst, ssrc): 
        return self.ic_pb(SOP1_base('s_movrelsd_2_b32', sdst, ssrc))
    def s_nand_saveexec_b32    (self, sdst, ssrc): 
        return self.ic_pb(SOP1_base('s_nand_saveexec_b32', sdst, ssrc))
    def s_nand_saveexec_b64    (self, sdst, ssrc): 
        return self.ic_pb(SOP1_base('s_nand_saveexec_b64', sdst, ssrc))
    def s_nor_saveexec_b32     (self, sdst, ssrc): 
        return self.ic_pb(SOP1_base('s_nor_saveexec_b32', sdst, ssrc))
    def s_nor_saveexec_b64     (self, sdst, ssrc): 
        return self.ic_pb(SOP1_base('s_nor_saveexec_b64', sdst, ssrc))
    def s_not_b32              (self, sdst, ssrc): 
        return self.ic_pb(SOP1_base('s_not_b32', sdst, ssrc))
    def s_not_b64              (self, sdst, ssrc): 
        return self.ic_pb(SOP1_base('s_not_b64', sdst, ssrc))
    def s_or_saveexec_b32      (self, sdst, ssrc): 
        return self.ic_pb(SOP1_base('s_or_saveexec_b32', sdst, ssrc))
    def s_or_saveexec_b64      (self, sdst, ssrc): 
        return self.ic_pb(SOP1_base('s_or_saveexec_b64', sdst, ssrc))
    def s_orn1_saveexec_b32    (self, sdst, ssrc): 
        return self.ic_pb(SOP1_base('s_orn1_saveexec_b32', sdst, ssrc))
    def s_orn1_saveexec_b64    (self, sdst, ssrc): 
        return self.ic_pb(SOP1_base('s_orn1_saveexec_b64', sdst, ssrc))
    def s_orn2_saveexec_b32    (self, sdst, ssrc): 
        return self.ic_pb(SOP1_base('s_orn2_saveexec_b32', sdst, ssrc))
    def s_orn2_saveexec_b64    (self, sdst, ssrc): 
        return self.ic_pb(SOP1_base('s_orn2_saveexec_b64', sdst, ssrc))
    def s_quadmask_b32         (self, sdst, ssrc): 
        return self.ic_pb(SOP1_base('s_quadmask_b32', sdst, ssrc))
    def s_quadmask_b64         (self, sdst, ssrc): 
        return self.ic_pb(SOP1_base('s_quadmask_b64', sdst, ssrc))
    def s_rfe_b64              (self,       ssrc): 
        return self.ic_pb(SOP1_base('s_rfe_b64', ssrc ))
    def s_setpc_b64            (self,       ssrc): 
        return self.ic_pb(SOP1_base('s_setpc_b64', ssrc ))
    def s_sext_i32_i16         (self, sdst, ssrc): 
        return self.ic_pb(SOP1_base('s_sext_i32_i16', sdst, ssrc))
    def s_sext_i32_i8          (self, sdst, ssrc): 
        return self.ic_pb(SOP1_base('s_sext_i32_i8', sdst, ssrc))
    def s_swappc_b64           (self, sdst, ssrc): 
        return self.ic_pb(SOP1_base('s_swappc_b64', sdst, ssrc))
    def s_wqm_b32              (self, sdst, ssrc): 
        return self.ic_pb(SOP1_base('s_wqm_b32', sdst, ssrc))
    def s_wqm_b64              (self, sdst, ssrc): 
        return self.ic_pb(SOP1_base('s_wqm_b64', sdst, ssrc))
    def s_xnor_saveexec_b32    (self, sdst, ssrc): 
        return self.ic_pb(SOP1_base('s_xnor_saveexec_b32', sdst, ssrc))
    def s_xnor_saveexec_b64    (self, sdst, ssrc): 
        return self.ic_pb(SOP1_base('s_xnor_saveexec_b64', sdst, ssrc))
    def s_xor_saveexec_b32     (self, sdst, ssrc): 
        return self.ic_pb(SOP1_base('s_xor_saveexec_b32', sdst, ssrc))
    def s_xor_saveexec_b64     (self, sdst, ssrc): 
        return self.ic_pb(SOP1_base('s_xor_saveexec_b64', sdst, ssrc))

class gpu_instructions_caller(VOP1_instr_caller, VMEM_instr_caller, SMEM_instr_caller, SOP1_instr_caller):
    def __init__(self, insturction_container) -> None:
        super().__init__(insturction_container)

class instruction_ctrl():
    __slots__ = ['instructions_caller', 'instructions_list']

    def __init__(self) -> None:
        instructions_list:List[inst_base] = []
        self.instructions_caller = gpu_instructions_caller(instructions_list)
        self.instructions_list = instructions_list
    
    def _emmit_all(self, emmiter):
        e = emmiter
        for i in self.instructions_list:
            e(f'{i}')
    
    def _emmit_range(self, emmiter, strt:int, end:int):
        e = emmiter
        i_list = self.instructions_list
        for i in i_list:
            e(f'{i}')