from .gfx9xx_base.GFX908_instruction_set import *
from .GFX9 import *


class gfx908_ex( flat_instr_caller_gfx908Ex, mubuf_instr_caller_gfx908Ex, vop2_instr_caller_gfx908Ex, vop3_instr_caller_gfx908Ex,
    vop3p_instr_caller_gfx908Ex):
    def __init__(self, insturction_container) -> None:
        super().__init__(insturction_container)

class gfx908_instructions_caller(gfx9_instructions_caller, gfx908_ex):
    def __init__(self, insturction_container) -> None:
        super().__init__(insturction_container)