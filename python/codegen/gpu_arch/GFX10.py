from .gfx10XX.GFX10 import *
from .gfx10XX.GFX1011 import *


class gfx1011_1012(dpp16_instr_caller_gfx10Ex,dpp8_instr_caller_gfx10Ex,
 vop2_instr_caller_gfx10Ex, vop3p_instr_caller_gfx10Ex):
 def __init__(self, insturction_container) -> None:
        super().__init__(insturction_container)

class gfx10_instructions_caller(dpp16_instr_caller, dpp8_instr_caller, ds_instr_caller,
    exp_instr_caller, flat_instr_caller, mimg_instr_caller, mtbuf_instr_caller,
    mubuf_instr_caller, sdwa_instr_caller, smem_instr_caller, sop1_instr_caller,
    sop2_instr_caller, sopc_instr_caller, sopk_instr_caller, sopp_instr_caller,
    vintrp_instr_caller, vop1_instr_caller, vop2_instr_caller, vop3_instr_caller,
    vop3p_instr_caller, vopc_instr_caller, gfx1011_1012):
    def __init__(self, insturction_container) -> None:
        super().__init__(insturction_container)