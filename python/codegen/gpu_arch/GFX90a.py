from .gfx90a.GFX90a_instruction_set import *


class gfx9_instructions_caller(ds_instr_caller,
    flat_instr_caller, mimg_instr_caller, mtbuf_instr_caller,
    mubuf_instr_caller, smem_instr_caller, sop1_instr_caller,
    sop2_instr_caller, sopc_instr_caller, sopk_instr_caller, sopp_instr_caller,
    vop1_instr_caller, vop2_instr_caller, vop3_instr_caller,
    vop3p_instr_caller, vopc_instr_caller):
    def __init__(self, insturction_container) -> None:
        super().__init__(insturction_container)