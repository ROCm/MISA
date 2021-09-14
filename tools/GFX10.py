from python.codegen.gpu_instruct import * 

class dpp16_base(inst_base): 
	def __init__(self, INSTRUCTION:str, DST0:Union[reg_block,None], DST1:Union[reg_block,None], SRC0:Union[reg_block,None], SRC1:Union[reg_block,None], SRC2:Union[reg_block,None], MODIFIERS:str): 
		super().__init__(instruction_type.SMEM, INSTRUCTION)
		self.DST0 = DST0 
		self.DST1 = DST1 
		self.SRC0 = SRC0 
		self.SRC1 = SRC1 
		self.SRC2 = SRC2 
		self.MODIFIERS = MODIFIERS 
	def __str__(self): 
		args_l = filter(None.__ne__, [self.DST0,self.DST1,self.SRC0,self.SRC1,self.SRC2]) 
		return f"{self.label} {','.join(map(str, args_l))} {self.MODIFIERS}" 
class dpp16_instr_caller(inst_caller_base): 
	def __init__(self, insturction_container) -> None:
     		super().__init__(insturction_container)
	def v_add_co_ci_u32_dpp(self, gfx10_vcc:reg_block, gfx10_vsrc:reg_block, gfx10_vsrc:reg_block, gfx10_vcc:reg_block, MODIFIERS:str=''):
		return self.ic_pb(dpp16_base('v_add_co_ci_u32_dpp', None, gfx10_vcc, gfx10_vsrc, gfx10_vsrc, gfx10_vcc, MODIFIERS))
	def v_add_f16_dpp(self, gfx10_vsrc:reg_block, gfx10_vsrc:reg_block, MODIFIERS:str=''):
		return self.ic_pb(dpp16_base('v_add_f16_dpp', None, None, gfx10_vsrc, gfx10_vsrc, None, MODIFIERS))
	def v_add_f32_dpp(self, gfx10_vsrc:reg_block, gfx10_vsrc:reg_block, MODIFIERS:str=''):
		return self.ic_pb(dpp16_base('v_add_f32_dpp', None, None, gfx10_vsrc, gfx10_vsrc, None, MODIFIERS))
	def v_add_nc_u32_dpp(self, gfx10_vsrc:reg_block, gfx10_vsrc:reg_block, MODIFIERS:str=''):
		return self.ic_pb(dpp16_base('v_add_nc_u32_dpp', None, None, gfx10_vsrc, gfx10_vsrc, None, MODIFIERS))
	def v_and_b32_dpp(self, gfx10_vsrc:reg_block, gfx10_vsrc:reg_block, MODIFIERS:str=''):
		return self.ic_pb(dpp16_base('v_and_b32_dpp', None, None, gfx10_vsrc, gfx10_vsrc, None, MODIFIERS))
	def v_ashrrev_i32_dpp(self, gfx10_vsrc:reg_block, gfx10_vsrc:reg_block, MODIFIERS:str=''):
		return self.ic_pb(dpp16_base('v_ashrrev_i32_dpp', None, None, gfx10_vsrc, gfx10_vsrc, None, MODIFIERS))
	def v_bfrev_b32_dpp(self, gfx10_vsrc:reg_block, MODIFIERS:str=''):
		return self.ic_pb(dpp16_base('v_bfrev_b32_dpp', None, None, gfx10_vsrc, None, None, MODIFIERS))
	def v_ceil_f16_dpp(self, gfx10_vsrc:reg_block, MODIFIERS:str=''):
		return self.ic_pb(dpp16_base('v_ceil_f16_dpp', None, None, gfx10_vsrc, None, None, MODIFIERS))
	def v_ceil_f32_dpp(self, gfx10_vsrc:reg_block, MODIFIERS:str=''):
		return self.ic_pb(dpp16_base('v_ceil_f32_dpp', None, None, gfx10_vsrc, None, None, MODIFIERS))
	def v_cndmask_b32_dpp(self, gfx10_vsrc:reg_block, gfx10_vsrc:reg_block, gfx10_vcc:reg_block, MODIFIERS:str=''):
		return self.ic_pb(dpp16_base('v_cndmask_b32_dpp', None, None, gfx10_vsrc, gfx10_vsrc, gfx10_vcc, MODIFIERS))
	def v_cos_f16_dpp(self, gfx10_vsrc:reg_block, MODIFIERS:str=''):
		return self.ic_pb(dpp16_base('v_cos_f16_dpp', None, None, gfx10_vsrc, None, None, MODIFIERS))
	def v_cos_f32_dpp(self, gfx10_vsrc:reg_block, MODIFIERS:str=''):
		return self.ic_pb(dpp16_base('v_cos_f32_dpp', None, None, gfx10_vsrc, None, None, MODIFIERS))
	def v_cvt_f16_f32_dpp(self, gfx10_vsrc:reg_block, MODIFIERS:str=''):
		return self.ic_pb(dpp16_base('v_cvt_f16_f32_dpp', None, None, gfx10_vsrc, None, None, MODIFIERS))
	def v_cvt_f16_i16_dpp(self, gfx10_vsrc:reg_block, MODIFIERS:str=''):
		return self.ic_pb(dpp16_base('v_cvt_f16_i16_dpp', None, None, gfx10_vsrc, None, None, MODIFIERS))
	def v_cvt_f16_u16_dpp(self, gfx10_vsrc:reg_block, MODIFIERS:str=''):
		return self.ic_pb(dpp16_base('v_cvt_f16_u16_dpp', None, None, gfx10_vsrc, None, None, MODIFIERS))
	def v_cvt_f32_f16_dpp(self, gfx10_vsrc:reg_block, MODIFIERS:str=''):
		return self.ic_pb(dpp16_base('v_cvt_f32_f16_dpp', None, None, gfx10_vsrc, None, None, MODIFIERS))
	def v_cvt_f32_i32_dpp(self, gfx10_vsrc:reg_block, MODIFIERS:str=''):
		return self.ic_pb(dpp16_base('v_cvt_f32_i32_dpp', None, None, gfx10_vsrc, None, None, MODIFIERS))
	def v_cvt_f32_u32_dpp(self, gfx10_vsrc:reg_block, MODIFIERS:str=''):
		return self.ic_pb(dpp16_base('v_cvt_f32_u32_dpp', None, None, gfx10_vsrc, None, None, MODIFIERS))
	def v_cvt_f32_ubyte0_dpp(self, gfx10_vsrc:reg_block, MODIFIERS:str=''):
		return self.ic_pb(dpp16_base('v_cvt_f32_ubyte0_dpp', None, None, gfx10_vsrc, None, None, MODIFIERS))
	def v_cvt_f32_ubyte1_dpp(self, gfx10_vsrc:reg_block, MODIFIERS:str=''):
		return self.ic_pb(dpp16_base('v_cvt_f32_ubyte1_dpp', None, None, gfx10_vsrc, None, None, MODIFIERS))
	def v_cvt_f32_ubyte2_dpp(self, gfx10_vsrc:reg_block, MODIFIERS:str=''):
		return self.ic_pb(dpp16_base('v_cvt_f32_ubyte2_dpp', None, None, gfx10_vsrc, None, None, MODIFIERS))
	def v_cvt_f32_ubyte3_dpp(self, gfx10_vsrc:reg_block, MODIFIERS:str=''):
		return self.ic_pb(dpp16_base('v_cvt_f32_ubyte3_dpp', None, None, gfx10_vsrc, None, None, MODIFIERS))
	def v_cvt_flr_i32_f32_dpp(self, gfx10_vsrc:reg_block, MODIFIERS:str=''):
		return self.ic_pb(dpp16_base('v_cvt_flr_i32_f32_dpp', None, None, gfx10_vsrc, None, None, MODIFIERS))
	def v_cvt_i16_f16_dpp(self, gfx10_vsrc:reg_block, MODIFIERS:str=''):
		return self.ic_pb(dpp16_base('v_cvt_i16_f16_dpp', None, None, gfx10_vsrc, None, None, MODIFIERS))
	def v_cvt_i32_f32_dpp(self, gfx10_vsrc:reg_block, MODIFIERS:str=''):
		return self.ic_pb(dpp16_base('v_cvt_i32_f32_dpp', None, None, gfx10_vsrc, None, None, MODIFIERS))
	def v_cvt_norm_i16_f16_dpp(self, gfx10_vsrc:reg_block, MODIFIERS:str=''):
		return self.ic_pb(dpp16_base('v_cvt_norm_i16_f16_dpp', None, None, gfx10_vsrc, None, None, MODIFIERS))
	def v_cvt_norm_u16_f16_dpp(self, gfx10_vsrc:reg_block, MODIFIERS:str=''):
		return self.ic_pb(dpp16_base('v_cvt_norm_u16_f16_dpp', None, None, gfx10_vsrc, None, None, MODIFIERS))
	def v_cvt_off_f32_i4_dpp(self, gfx10_vsrc:reg_block, MODIFIERS:str=''):
		return self.ic_pb(dpp16_base('v_cvt_off_f32_i4_dpp', None, None, gfx10_vsrc, None, None, MODIFIERS))
	def v_cvt_rpi_i32_f32_dpp(self, gfx10_vsrc:reg_block, MODIFIERS:str=''):
		return self.ic_pb(dpp16_base('v_cvt_rpi_i32_f32_dpp', None, None, gfx10_vsrc, None, None, MODIFIERS))
	def v_cvt_u16_f16_dpp(self, gfx10_vsrc:reg_block, MODIFIERS:str=''):
		return self.ic_pb(dpp16_base('v_cvt_u16_f16_dpp', None, None, gfx10_vsrc, None, None, MODIFIERS))
	def v_cvt_u32_f32_dpp(self, gfx10_vsrc:reg_block, MODIFIERS:str=''):
		return self.ic_pb(dpp16_base('v_cvt_u32_f32_dpp', None, None, gfx10_vsrc, None, None, MODIFIERS))
	def v_exp_f16_dpp(self, gfx10_vsrc:reg_block, MODIFIERS:str=''):
		return self.ic_pb(dpp16_base('v_exp_f16_dpp', None, None, gfx10_vsrc, None, None, MODIFIERS))
	def v_exp_f32_dpp(self, gfx10_vsrc:reg_block, MODIFIERS:str=''):
		return self.ic_pb(dpp16_base('v_exp_f32_dpp', None, None, gfx10_vsrc, None, None, MODIFIERS))
	def v_ffbh_i32_dpp(self, gfx10_vsrc:reg_block, MODIFIERS:str=''):
		return self.ic_pb(dpp16_base('v_ffbh_i32_dpp', None, None, gfx10_vsrc, None, None, MODIFIERS))
	def v_ffbh_u32_dpp(self, gfx10_vsrc:reg_block, MODIFIERS:str=''):
		return self.ic_pb(dpp16_base('v_ffbh_u32_dpp', None, None, gfx10_vsrc, None, None, MODIFIERS))
	def v_ffbl_b32_dpp(self, gfx10_vsrc:reg_block, MODIFIERS:str=''):
		return self.ic_pb(dpp16_base('v_ffbl_b32_dpp', None, None, gfx10_vsrc, None, None, MODIFIERS))
	def v_floor_f16_dpp(self, gfx10_vsrc:reg_block, MODIFIERS:str=''):
		return self.ic_pb(dpp16_base('v_floor_f16_dpp', None, None, gfx10_vsrc, None, None, MODIFIERS))
	def v_floor_f32_dpp(self, gfx10_vsrc:reg_block, MODIFIERS:str=''):
		return self.ic_pb(dpp16_base('v_floor_f32_dpp', None, None, gfx10_vsrc, None, None, MODIFIERS))
	def v_fmac_f16_dpp(self, gfx10_vsrc:reg_block, gfx10_vsrc:reg_block, MODIFIERS:str=''):
		return self.ic_pb(dpp16_base('v_fmac_f16_dpp', None, None, gfx10_vsrc, gfx10_vsrc, None, MODIFIERS))
	def v_fmac_f32_dpp(self, gfx10_vsrc:reg_block, gfx10_vsrc:reg_block, MODIFIERS:str=''):
		return self.ic_pb(dpp16_base('v_fmac_f32_dpp', None, None, gfx10_vsrc, gfx10_vsrc, None, MODIFIERS))
	def v_fract_f16_dpp(self, gfx10_vsrc:reg_block, MODIFIERS:str=''):
		return self.ic_pb(dpp16_base('v_fract_f16_dpp', None, None, gfx10_vsrc, None, None, MODIFIERS))
	def v_fract_f32_dpp(self, gfx10_vsrc:reg_block, MODIFIERS:str=''):
		return self.ic_pb(dpp16_base('v_fract_f32_dpp', None, None, gfx10_vsrc, None, None, MODIFIERS))
	def v_frexp_exp_i16_f16_dpp(self, gfx10_vsrc:reg_block, MODIFIERS:str=''):
		return self.ic_pb(dpp16_base('v_frexp_exp_i16_f16_dpp', None, None, gfx10_vsrc, None, None, MODIFIERS))
	def v_frexp_exp_i32_f32_dpp(self, gfx10_vsrc:reg_block, MODIFIERS:str=''):
		return self.ic_pb(dpp16_base('v_frexp_exp_i32_f32_dpp', None, None, gfx10_vsrc, None, None, MODIFIERS))
	def v_frexp_mant_f16_dpp(self, gfx10_vsrc:reg_block, MODIFIERS:str=''):
		return self.ic_pb(dpp16_base('v_frexp_mant_f16_dpp', None, None, gfx10_vsrc, None, None, MODIFIERS))
	def v_frexp_mant_f32_dpp(self, gfx10_vsrc:reg_block, MODIFIERS:str=''):
		return self.ic_pb(dpp16_base('v_frexp_mant_f32_dpp', None, None, gfx10_vsrc, None, None, MODIFIERS))
	def v_ldexp_f16_dpp(self, gfx10_vsrc:reg_block, gfx10_vsrc:reg_block, MODIFIERS:str=''):
		return self.ic_pb(dpp16_base('v_ldexp_f16_dpp', None, None, gfx10_vsrc, gfx10_vsrc, None, MODIFIERS))
	def v_log_f16_dpp(self, gfx10_vsrc:reg_block, MODIFIERS:str=''):
		return self.ic_pb(dpp16_base('v_log_f16_dpp', None, None, gfx10_vsrc, None, None, MODIFIERS))
	def v_log_f32_dpp(self, gfx10_vsrc:reg_block, MODIFIERS:str=''):
		return self.ic_pb(dpp16_base('v_log_f32_dpp', None, None, gfx10_vsrc, None, None, MODIFIERS))
	def v_lshlrev_b32_dpp(self, gfx10_vsrc:reg_block, gfx10_vsrc:reg_block, MODIFIERS:str=''):
		return self.ic_pb(dpp16_base('v_lshlrev_b32_dpp', None, None, gfx10_vsrc, gfx10_vsrc, None, MODIFIERS))
	def v_lshrrev_b32_dpp(self, gfx10_vsrc:reg_block, gfx10_vsrc:reg_block, MODIFIERS:str=''):
		return self.ic_pb(dpp16_base('v_lshrrev_b32_dpp', None, None, gfx10_vsrc, gfx10_vsrc, None, MODIFIERS))
	def v_mac_f32_dpp(self, gfx10_vsrc:reg_block, gfx10_vsrc:reg_block, MODIFIERS:str=''):
		return self.ic_pb(dpp16_base('v_mac_f32_dpp', None, None, gfx10_vsrc, gfx10_vsrc, None, MODIFIERS))
	def v_max_f16_dpp(self, gfx10_vsrc:reg_block, gfx10_vsrc:reg_block, MODIFIERS:str=''):
		return self.ic_pb(dpp16_base('v_max_f16_dpp', None, None, gfx10_vsrc, gfx10_vsrc, None, MODIFIERS))
	def v_max_f32_dpp(self, gfx10_vsrc:reg_block, gfx10_vsrc:reg_block, MODIFIERS:str=''):
		return self.ic_pb(dpp16_base('v_max_f32_dpp', None, None, gfx10_vsrc, gfx10_vsrc, None, MODIFIERS))
	def v_max_i32_dpp(self, gfx10_vsrc:reg_block, gfx10_vsrc:reg_block, MODIFIERS:str=''):
		return self.ic_pb(dpp16_base('v_max_i32_dpp', None, None, gfx10_vsrc, gfx10_vsrc, None, MODIFIERS))
	def v_max_u32_dpp(self, gfx10_vsrc:reg_block, gfx10_vsrc:reg_block, MODIFIERS:str=''):
		return self.ic_pb(dpp16_base('v_max_u32_dpp', None, None, gfx10_vsrc, gfx10_vsrc, None, MODIFIERS))
	def v_min_f16_dpp(self, gfx10_vsrc:reg_block, gfx10_vsrc:reg_block, MODIFIERS:str=''):
		return self.ic_pb(dpp16_base('v_min_f16_dpp', None, None, gfx10_vsrc, gfx10_vsrc, None, MODIFIERS))
	def v_min_f32_dpp(self, gfx10_vsrc:reg_block, gfx10_vsrc:reg_block, MODIFIERS:str=''):
		return self.ic_pb(dpp16_base('v_min_f32_dpp', None, None, gfx10_vsrc, gfx10_vsrc, None, MODIFIERS))
	def v_min_i32_dpp(self, gfx10_vsrc:reg_block, gfx10_vsrc:reg_block, MODIFIERS:str=''):
		return self.ic_pb(dpp16_base('v_min_i32_dpp', None, None, gfx10_vsrc, gfx10_vsrc, None, MODIFIERS))
	def v_min_u32_dpp(self, gfx10_vsrc:reg_block, gfx10_vsrc:reg_block, MODIFIERS:str=''):
		return self.ic_pb(dpp16_base('v_min_u32_dpp', None, None, gfx10_vsrc, gfx10_vsrc, None, MODIFIERS))
	def v_mov_b32_dpp(self, gfx10_vsrc:reg_block, MODIFIERS:str=''):
		return self.ic_pb(dpp16_base('v_mov_b32_dpp', None, None, gfx10_vsrc, None, None, MODIFIERS))
	def v_movreld_b32_dpp(self, gfx10_vsrc:reg_block, MODIFIERS:str=''):
		return self.ic_pb(dpp16_base('v_movreld_b32_dpp', None, None, gfx10_vsrc, None, None, MODIFIERS))
	def v_movrels_b32_dpp(self, gfx10_vsrc:reg_block, MODIFIERS:str=''):
		return self.ic_pb(dpp16_base('v_movrels_b32_dpp', None, None, gfx10_vsrc, None, None, MODIFIERS))
	def v_movrelsd_2_b32_dpp(self, gfx10_vsrc:reg_block, MODIFIERS:str=''):
		return self.ic_pb(dpp16_base('v_movrelsd_2_b32_dpp', None, None, gfx10_vsrc, None, None, MODIFIERS))
	def v_movrelsd_b32_dpp(self, gfx10_vsrc:reg_block, MODIFIERS:str=''):
		return self.ic_pb(dpp16_base('v_movrelsd_b32_dpp', None, None, gfx10_vsrc, None, None, MODIFIERS))
	def v_mul_f16_dpp(self, gfx10_vsrc:reg_block, gfx10_vsrc:reg_block, MODIFIERS:str=''):
		return self.ic_pb(dpp16_base('v_mul_f16_dpp', None, None, gfx10_vsrc, gfx10_vsrc, None, MODIFIERS))
	def v_mul_f32_dpp(self, gfx10_vsrc:reg_block, gfx10_vsrc:reg_block, MODIFIERS:str=''):
		return self.ic_pb(dpp16_base('v_mul_f32_dpp', None, None, gfx10_vsrc, gfx10_vsrc, None, MODIFIERS))
	def v_mul_hi_i32_i24_dpp(self, gfx10_vsrc:reg_block, gfx10_vsrc:reg_block, MODIFIERS:str=''):
		return self.ic_pb(dpp16_base('v_mul_hi_i32_i24_dpp', None, None, gfx10_vsrc, gfx10_vsrc, None, MODIFIERS))
	def v_mul_hi_u32_u24_dpp(self, gfx10_vsrc:reg_block, gfx10_vsrc:reg_block, MODIFIERS:str=''):
		return self.ic_pb(dpp16_base('v_mul_hi_u32_u24_dpp', None, None, gfx10_vsrc, gfx10_vsrc, None, MODIFIERS))
	def v_mul_i32_i24_dpp(self, gfx10_vsrc:reg_block, gfx10_vsrc:reg_block, MODIFIERS:str=''):
		return self.ic_pb(dpp16_base('v_mul_i32_i24_dpp', None, None, gfx10_vsrc, gfx10_vsrc, None, MODIFIERS))
	def v_mul_legacy_f32_dpp(self, gfx10_vsrc:reg_block, gfx10_vsrc:reg_block, MODIFIERS:str=''):
		return self.ic_pb(dpp16_base('v_mul_legacy_f32_dpp', None, None, gfx10_vsrc, gfx10_vsrc, None, MODIFIERS))
	def v_mul_u32_u24_dpp(self, gfx10_vsrc:reg_block, gfx10_vsrc:reg_block, MODIFIERS:str=''):
		return self.ic_pb(dpp16_base('v_mul_u32_u24_dpp', None, None, gfx10_vsrc, gfx10_vsrc, None, MODIFIERS))
	def v_not_b32_dpp(self, gfx10_vsrc:reg_block, MODIFIERS:str=''):
		return self.ic_pb(dpp16_base('v_not_b32_dpp', None, None, gfx10_vsrc, None, None, MODIFIERS))
	def v_or_b32_dpp(self, gfx10_vsrc:reg_block, gfx10_vsrc:reg_block, MODIFIERS:str=''):
		return self.ic_pb(dpp16_base('v_or_b32_dpp', None, None, gfx10_vsrc, gfx10_vsrc, None, MODIFIERS))
	def v_rcp_f16_dpp(self, gfx10_vsrc:reg_block, MODIFIERS:str=''):
		return self.ic_pb(dpp16_base('v_rcp_f16_dpp', None, None, gfx10_vsrc, None, None, MODIFIERS))
	def v_rcp_f32_dpp(self, gfx10_vsrc:reg_block, MODIFIERS:str=''):
		return self.ic_pb(dpp16_base('v_rcp_f32_dpp', None, None, gfx10_vsrc, None, None, MODIFIERS))
	def v_rcp_iflag_f32_dpp(self, gfx10_vsrc:reg_block, MODIFIERS:str=''):
		return self.ic_pb(dpp16_base('v_rcp_iflag_f32_dpp', None, None, gfx10_vsrc, None, None, MODIFIERS))
	def v_rndne_f16_dpp(self, gfx10_vsrc:reg_block, MODIFIERS:str=''):
		return self.ic_pb(dpp16_base('v_rndne_f16_dpp', None, None, gfx10_vsrc, None, None, MODIFIERS))
	def v_rndne_f32_dpp(self, gfx10_vsrc:reg_block, MODIFIERS:str=''):
		return self.ic_pb(dpp16_base('v_rndne_f32_dpp', None, None, gfx10_vsrc, None, None, MODIFIERS))
	def v_rsq_f16_dpp(self, gfx10_vsrc:reg_block, MODIFIERS:str=''):
		return self.ic_pb(dpp16_base('v_rsq_f16_dpp', None, None, gfx10_vsrc, None, None, MODIFIERS))
	def v_rsq_f32_dpp(self, gfx10_vsrc:reg_block, MODIFIERS:str=''):
		return self.ic_pb(dpp16_base('v_rsq_f32_dpp', None, None, gfx10_vsrc, None, None, MODIFIERS))
	def v_sat_pk_u8_i16_dpp(self, gfx10_vsrc:reg_block, MODIFIERS:str=''):
		return self.ic_pb(dpp16_base('v_sat_pk_u8_i16_dpp', None, None, gfx10_vsrc, None, None, MODIFIERS))
	def v_sin_f16_dpp(self, gfx10_vsrc:reg_block, MODIFIERS:str=''):
		return self.ic_pb(dpp16_base('v_sin_f16_dpp', None, None, gfx10_vsrc, None, None, MODIFIERS))
	def v_sin_f32_dpp(self, gfx10_vsrc:reg_block, MODIFIERS:str=''):
		return self.ic_pb(dpp16_base('v_sin_f32_dpp', None, None, gfx10_vsrc, None, None, MODIFIERS))
	def v_sqrt_f16_dpp(self, gfx10_vsrc:reg_block, MODIFIERS:str=''):
		return self.ic_pb(dpp16_base('v_sqrt_f16_dpp', None, None, gfx10_vsrc, None, None, MODIFIERS))
	def v_sqrt_f32_dpp(self, gfx10_vsrc:reg_block, MODIFIERS:str=''):
		return self.ic_pb(dpp16_base('v_sqrt_f32_dpp', None, None, gfx10_vsrc, None, None, MODIFIERS))
	def v_sub_co_ci_u32_dpp(self, gfx10_vcc:reg_block, gfx10_vsrc:reg_block, gfx10_vsrc:reg_block, gfx10_vcc:reg_block, MODIFIERS:str=''):
		return self.ic_pb(dpp16_base('v_sub_co_ci_u32_dpp', None, gfx10_vcc, gfx10_vsrc, gfx10_vsrc, gfx10_vcc, MODIFIERS))
	def v_sub_f16_dpp(self, gfx10_vsrc:reg_block, gfx10_vsrc:reg_block, MODIFIERS:str=''):
		return self.ic_pb(dpp16_base('v_sub_f16_dpp', None, None, gfx10_vsrc, gfx10_vsrc, None, MODIFIERS))
	def v_sub_f32_dpp(self, gfx10_vsrc:reg_block, gfx10_vsrc:reg_block, MODIFIERS:str=''):
		return self.ic_pb(dpp16_base('v_sub_f32_dpp', None, None, gfx10_vsrc, gfx10_vsrc, None, MODIFIERS))
	def v_sub_nc_u32_dpp(self, gfx10_vsrc:reg_block, gfx10_vsrc:reg_block, MODIFIERS:str=''):
		return self.ic_pb(dpp16_base('v_sub_nc_u32_dpp', None, None, gfx10_vsrc, gfx10_vsrc, None, MODIFIERS))
	def v_subrev_co_ci_u32_dpp(self, gfx10_vcc:reg_block, gfx10_vsrc:reg_block, gfx10_vsrc:reg_block, gfx10_vcc:reg_block, MODIFIERS:str=''):
		return self.ic_pb(dpp16_base('v_subrev_co_ci_u32_dpp', None, gfx10_vcc, gfx10_vsrc, gfx10_vsrc, gfx10_vcc, MODIFIERS))
	def v_subrev_f16_dpp(self, gfx10_vsrc:reg_block, gfx10_vsrc:reg_block, MODIFIERS:str=''):
		return self.ic_pb(dpp16_base('v_subrev_f16_dpp', None, None, gfx10_vsrc, gfx10_vsrc, None, MODIFIERS))
	def v_subrev_f32_dpp(self, gfx10_vsrc:reg_block, gfx10_vsrc:reg_block, MODIFIERS:str=''):
		return self.ic_pb(dpp16_base('v_subrev_f32_dpp', None, None, gfx10_vsrc, gfx10_vsrc, None, MODIFIERS))
	def v_subrev_nc_u32_dpp(self, gfx10_vsrc:reg_block, gfx10_vsrc:reg_block, MODIFIERS:str=''):
		return self.ic_pb(dpp16_base('v_subrev_nc_u32_dpp', None, None, gfx10_vsrc, gfx10_vsrc, None, MODIFIERS))
	def v_trunc_f16_dpp(self, gfx10_vsrc:reg_block, MODIFIERS:str=''):
		return self.ic_pb(dpp16_base('v_trunc_f16_dpp', None, None, gfx10_vsrc, None, None, MODIFIERS))
	def v_trunc_f32_dpp(self, gfx10_vsrc:reg_block, MODIFIERS:str=''):
		return self.ic_pb(dpp16_base('v_trunc_f32_dpp', None, None, gfx10_vsrc, None, None, MODIFIERS))
	def v_xnor_b32_dpp(self, gfx10_vsrc:reg_block, gfx10_vsrc:reg_block, MODIFIERS:str=''):
		return self.ic_pb(dpp16_base('v_xnor_b32_dpp', None, None, gfx10_vsrc, gfx10_vsrc, None, MODIFIERS))
	def v_xor_b32_dpp(self, gfx10_vsrc:reg_block, gfx10_vsrc:reg_block, MODIFIERS:str=''):
		return self.ic_pb(dpp16_base('v_xor_b32_dpp', None, None, gfx10_vsrc, gfx10_vsrc, None, MODIFIERS))
class dpp8_base(inst_base): 
	def __init__(self, INSTRUCTION:str, DST0:Union[reg_block,None], DST1:Union[reg_block,None], SRC0:Union[reg_block,None], SRC1:Union[reg_block,None], SRC2:Union[reg_block,None], MODIFIERS:str): 
		super().__init__(instruction_type.SMEM, INSTRUCTION)
		self.DST0 = DST0 
		self.DST1 = DST1 
		self.SRC0 = SRC0 
		self.SRC1 = SRC1 
		self.SRC2 = SRC2 
		self.MODIFIERS = MODIFIERS 
	def __str__(self): 
		args_l = filter(None.__ne__, [self.DST0,self.DST1,self.SRC0,self.SRC1,self.SRC2]) 
		return f"{self.label} {','.join(map(str, args_l))} {self.MODIFIERS}" 
class dpp8_instr_caller(inst_caller_base): 
	def __init__(self, insturction_container) -> None:
     		super().__init__(insturction_container)
	def v_add_co_ci_u32_dpp(self, gfx10_vcc:reg_block, gfx10_vsrc:reg_block, gfx10_vsrc:reg_block, gfx10_vcc:reg_block, MODIFIERS:str=''):
		return self.ic_pb(dpp8_base('v_add_co_ci_u32_dpp', None, gfx10_vcc, gfx10_vsrc, gfx10_vsrc, gfx10_vcc, MODIFIERS))
	def v_add_f16_dpp(self, gfx10_vsrc:reg_block, gfx10_vsrc:reg_block, MODIFIERS:str=''):
		return self.ic_pb(dpp8_base('v_add_f16_dpp', None, None, gfx10_vsrc, gfx10_vsrc, None, MODIFIERS))
	def v_add_f32_dpp(self, gfx10_vsrc:reg_block, gfx10_vsrc:reg_block, MODIFIERS:str=''):
		return self.ic_pb(dpp8_base('v_add_f32_dpp', None, None, gfx10_vsrc, gfx10_vsrc, None, MODIFIERS))
	def v_add_nc_u32_dpp(self, gfx10_vsrc:reg_block, gfx10_vsrc:reg_block, MODIFIERS:str=''):
		return self.ic_pb(dpp8_base('v_add_nc_u32_dpp', None, None, gfx10_vsrc, gfx10_vsrc, None, MODIFIERS))
	def v_and_b32_dpp(self, gfx10_vsrc:reg_block, gfx10_vsrc:reg_block, MODIFIERS:str=''):
		return self.ic_pb(dpp8_base('v_and_b32_dpp', None, None, gfx10_vsrc, gfx10_vsrc, None, MODIFIERS))
	def v_ashrrev_i32_dpp(self, gfx10_vsrc:reg_block, gfx10_vsrc:reg_block, MODIFIERS:str=''):
		return self.ic_pb(dpp8_base('v_ashrrev_i32_dpp', None, None, gfx10_vsrc, gfx10_vsrc, None, MODIFIERS))
	def v_bfrev_b32_dpp(self, gfx10_vsrc:reg_block, MODIFIERS:str=''):
		return self.ic_pb(dpp8_base('v_bfrev_b32_dpp', None, None, gfx10_vsrc, None, None, MODIFIERS))
	def v_ceil_f16_dpp(self, gfx10_vsrc:reg_block, MODIFIERS:str=''):
		return self.ic_pb(dpp8_base('v_ceil_f16_dpp', None, None, gfx10_vsrc, None, None, MODIFIERS))
	def v_ceil_f32_dpp(self, gfx10_vsrc:reg_block, MODIFIERS:str=''):
		return self.ic_pb(dpp8_base('v_ceil_f32_dpp', None, None, gfx10_vsrc, None, None, MODIFIERS))
	def v_cndmask_b32_dpp(self, gfx10_vsrc:reg_block, gfx10_vsrc:reg_block, gfx10_vcc:reg_block, MODIFIERS:str=''):
		return self.ic_pb(dpp8_base('v_cndmask_b32_dpp', None, None, gfx10_vsrc, gfx10_vsrc, gfx10_vcc, MODIFIERS))
	def v_cos_f16_dpp(self, gfx10_vsrc:reg_block, MODIFIERS:str=''):
		return self.ic_pb(dpp8_base('v_cos_f16_dpp', None, None, gfx10_vsrc, None, None, MODIFIERS))
	def v_cos_f32_dpp(self, gfx10_vsrc:reg_block, MODIFIERS:str=''):
		return self.ic_pb(dpp8_base('v_cos_f32_dpp', None, None, gfx10_vsrc, None, None, MODIFIERS))
	def v_cvt_f16_f32_dpp(self, gfx10_vsrc:reg_block, MODIFIERS:str=''):
		return self.ic_pb(dpp8_base('v_cvt_f16_f32_dpp', None, None, gfx10_vsrc, None, None, MODIFIERS))
	def v_cvt_f16_i16_dpp(self, gfx10_vsrc:reg_block, MODIFIERS:str=''):
		return self.ic_pb(dpp8_base('v_cvt_f16_i16_dpp', None, None, gfx10_vsrc, None, None, MODIFIERS))
	def v_cvt_f16_u16_dpp(self, gfx10_vsrc:reg_block, MODIFIERS:str=''):
		return self.ic_pb(dpp8_base('v_cvt_f16_u16_dpp', None, None, gfx10_vsrc, None, None, MODIFIERS))
	def v_cvt_f32_f16_dpp(self, gfx10_vsrc:reg_block, MODIFIERS:str=''):
		return self.ic_pb(dpp8_base('v_cvt_f32_f16_dpp', None, None, gfx10_vsrc, None, None, MODIFIERS))
	def v_cvt_f32_i32_dpp(self, gfx10_vsrc:reg_block, MODIFIERS:str=''):
		return self.ic_pb(dpp8_base('v_cvt_f32_i32_dpp', None, None, gfx10_vsrc, None, None, MODIFIERS))
	def v_cvt_f32_u32_dpp(self, gfx10_vsrc:reg_block, MODIFIERS:str=''):
		return self.ic_pb(dpp8_base('v_cvt_f32_u32_dpp', None, None, gfx10_vsrc, None, None, MODIFIERS))
	def v_cvt_f32_ubyte0_dpp(self, gfx10_vsrc:reg_block, MODIFIERS:str=''):
		return self.ic_pb(dpp8_base('v_cvt_f32_ubyte0_dpp', None, None, gfx10_vsrc, None, None, MODIFIERS))
	def v_cvt_f32_ubyte1_dpp(self, gfx10_vsrc:reg_block, MODIFIERS:str=''):
		return self.ic_pb(dpp8_base('v_cvt_f32_ubyte1_dpp', None, None, gfx10_vsrc, None, None, MODIFIERS))
	def v_cvt_f32_ubyte2_dpp(self, gfx10_vsrc:reg_block, MODIFIERS:str=''):
		return self.ic_pb(dpp8_base('v_cvt_f32_ubyte2_dpp', None, None, gfx10_vsrc, None, None, MODIFIERS))
	def v_cvt_f32_ubyte3_dpp(self, gfx10_vsrc:reg_block, MODIFIERS:str=''):
		return self.ic_pb(dpp8_base('v_cvt_f32_ubyte3_dpp', None, None, gfx10_vsrc, None, None, MODIFIERS))
	def v_cvt_flr_i32_f32_dpp(self, gfx10_vsrc:reg_block, MODIFIERS:str=''):
		return self.ic_pb(dpp8_base('v_cvt_flr_i32_f32_dpp', None, None, gfx10_vsrc, None, None, MODIFIERS))
	def v_cvt_i16_f16_dpp(self, gfx10_vsrc:reg_block, MODIFIERS:str=''):
		return self.ic_pb(dpp8_base('v_cvt_i16_f16_dpp', None, None, gfx10_vsrc, None, None, MODIFIERS))
	def v_cvt_i32_f32_dpp(self, gfx10_vsrc:reg_block, MODIFIERS:str=''):
		return self.ic_pb(dpp8_base('v_cvt_i32_f32_dpp', None, None, gfx10_vsrc, None, None, MODIFIERS))
	def v_cvt_norm_i16_f16_dpp(self, gfx10_vsrc:reg_block, MODIFIERS:str=''):
		return self.ic_pb(dpp8_base('v_cvt_norm_i16_f16_dpp', None, None, gfx10_vsrc, None, None, MODIFIERS))
	def v_cvt_norm_u16_f16_dpp(self, gfx10_vsrc:reg_block, MODIFIERS:str=''):
		return self.ic_pb(dpp8_base('v_cvt_norm_u16_f16_dpp', None, None, gfx10_vsrc, None, None, MODIFIERS))
	def v_cvt_off_f32_i4_dpp(self, gfx10_vsrc:reg_block, MODIFIERS:str=''):
		return self.ic_pb(dpp8_base('v_cvt_off_f32_i4_dpp', None, None, gfx10_vsrc, None, None, MODIFIERS))
	def v_cvt_rpi_i32_f32_dpp(self, gfx10_vsrc:reg_block, MODIFIERS:str=''):
		return self.ic_pb(dpp8_base('v_cvt_rpi_i32_f32_dpp', None, None, gfx10_vsrc, None, None, MODIFIERS))
	def v_cvt_u16_f16_dpp(self, gfx10_vsrc:reg_block, MODIFIERS:str=''):
		return self.ic_pb(dpp8_base('v_cvt_u16_f16_dpp', None, None, gfx10_vsrc, None, None, MODIFIERS))
	def v_cvt_u32_f32_dpp(self, gfx10_vsrc:reg_block, MODIFIERS:str=''):
		return self.ic_pb(dpp8_base('v_cvt_u32_f32_dpp', None, None, gfx10_vsrc, None, None, MODIFIERS))
	def v_exp_f16_dpp(self, gfx10_vsrc:reg_block, MODIFIERS:str=''):
		return self.ic_pb(dpp8_base('v_exp_f16_dpp', None, None, gfx10_vsrc, None, None, MODIFIERS))
	def v_exp_f32_dpp(self, gfx10_vsrc:reg_block, MODIFIERS:str=''):
		return self.ic_pb(dpp8_base('v_exp_f32_dpp', None, None, gfx10_vsrc, None, None, MODIFIERS))
	def v_ffbh_i32_dpp(self, gfx10_vsrc:reg_block, MODIFIERS:str=''):
		return self.ic_pb(dpp8_base('v_ffbh_i32_dpp', None, None, gfx10_vsrc, None, None, MODIFIERS))
	def v_ffbh_u32_dpp(self, gfx10_vsrc:reg_block, MODIFIERS:str=''):
		return self.ic_pb(dpp8_base('v_ffbh_u32_dpp', None, None, gfx10_vsrc, None, None, MODIFIERS))
	def v_ffbl_b32_dpp(self, gfx10_vsrc:reg_block, MODIFIERS:str=''):
		return self.ic_pb(dpp8_base('v_ffbl_b32_dpp', None, None, gfx10_vsrc, None, None, MODIFIERS))
	def v_floor_f16_dpp(self, gfx10_vsrc:reg_block, MODIFIERS:str=''):
		return self.ic_pb(dpp8_base('v_floor_f16_dpp', None, None, gfx10_vsrc, None, None, MODIFIERS))
	def v_floor_f32_dpp(self, gfx10_vsrc:reg_block, MODIFIERS:str=''):
		return self.ic_pb(dpp8_base('v_floor_f32_dpp', None, None, gfx10_vsrc, None, None, MODIFIERS))
	def v_fmac_f16_dpp(self, gfx10_vsrc:reg_block, gfx10_vsrc:reg_block, MODIFIERS:str=''):
		return self.ic_pb(dpp8_base('v_fmac_f16_dpp', None, None, gfx10_vsrc, gfx10_vsrc, None, MODIFIERS))
	def v_fmac_f32_dpp(self, gfx10_vsrc:reg_block, gfx10_vsrc:reg_block, MODIFIERS:str=''):
		return self.ic_pb(dpp8_base('v_fmac_f32_dpp', None, None, gfx10_vsrc, gfx10_vsrc, None, MODIFIERS))
	def v_fract_f16_dpp(self, gfx10_vsrc:reg_block, MODIFIERS:str=''):
		return self.ic_pb(dpp8_base('v_fract_f16_dpp', None, None, gfx10_vsrc, None, None, MODIFIERS))
	def v_fract_f32_dpp(self, gfx10_vsrc:reg_block, MODIFIERS:str=''):
		return self.ic_pb(dpp8_base('v_fract_f32_dpp', None, None, gfx10_vsrc, None, None, MODIFIERS))
	def v_frexp_exp_i16_f16_dpp(self, gfx10_vsrc:reg_block, MODIFIERS:str=''):
		return self.ic_pb(dpp8_base('v_frexp_exp_i16_f16_dpp', None, None, gfx10_vsrc, None, None, MODIFIERS))
	def v_frexp_exp_i32_f32_dpp(self, gfx10_vsrc:reg_block, MODIFIERS:str=''):
		return self.ic_pb(dpp8_base('v_frexp_exp_i32_f32_dpp', None, None, gfx10_vsrc, None, None, MODIFIERS))
	def v_frexp_mant_f16_dpp(self, gfx10_vsrc:reg_block, MODIFIERS:str=''):
		return self.ic_pb(dpp8_base('v_frexp_mant_f16_dpp', None, None, gfx10_vsrc, None, None, MODIFIERS))
	def v_frexp_mant_f32_dpp(self, gfx10_vsrc:reg_block, MODIFIERS:str=''):
		return self.ic_pb(dpp8_base('v_frexp_mant_f32_dpp', None, None, gfx10_vsrc, None, None, MODIFIERS))
	def v_ldexp_f16_dpp(self, gfx10_vsrc:reg_block, gfx10_vsrc:reg_block, MODIFIERS:str=''):
		return self.ic_pb(dpp8_base('v_ldexp_f16_dpp', None, None, gfx10_vsrc, gfx10_vsrc, None, MODIFIERS))
	def v_log_f16_dpp(self, gfx10_vsrc:reg_block, MODIFIERS:str=''):
		return self.ic_pb(dpp8_base('v_log_f16_dpp', None, None, gfx10_vsrc, None, None, MODIFIERS))
	def v_log_f32_dpp(self, gfx10_vsrc:reg_block, MODIFIERS:str=''):
		return self.ic_pb(dpp8_base('v_log_f32_dpp', None, None, gfx10_vsrc, None, None, MODIFIERS))
	def v_lshlrev_b32_dpp(self, gfx10_vsrc:reg_block, gfx10_vsrc:reg_block, MODIFIERS:str=''):
		return self.ic_pb(dpp8_base('v_lshlrev_b32_dpp', None, None, gfx10_vsrc, gfx10_vsrc, None, MODIFIERS))
	def v_lshrrev_b32_dpp(self, gfx10_vsrc:reg_block, gfx10_vsrc:reg_block, MODIFIERS:str=''):
		return self.ic_pb(dpp8_base('v_lshrrev_b32_dpp', None, None, gfx10_vsrc, gfx10_vsrc, None, MODIFIERS))
	def v_mac_f32_dpp(self, gfx10_vsrc:reg_block, gfx10_vsrc:reg_block, MODIFIERS:str=''):
		return self.ic_pb(dpp8_base('v_mac_f32_dpp', None, None, gfx10_vsrc, gfx10_vsrc, None, MODIFIERS))
	def v_max_f16_dpp(self, gfx10_vsrc:reg_block, gfx10_vsrc:reg_block, MODIFIERS:str=''):
		return self.ic_pb(dpp8_base('v_max_f16_dpp', None, None, gfx10_vsrc, gfx10_vsrc, None, MODIFIERS))
	def v_max_f32_dpp(self, gfx10_vsrc:reg_block, gfx10_vsrc:reg_block, MODIFIERS:str=''):
		return self.ic_pb(dpp8_base('v_max_f32_dpp', None, None, gfx10_vsrc, gfx10_vsrc, None, MODIFIERS))
	def v_max_i32_dpp(self, gfx10_vsrc:reg_block, gfx10_vsrc:reg_block, MODIFIERS:str=''):
		return self.ic_pb(dpp8_base('v_max_i32_dpp', None, None, gfx10_vsrc, gfx10_vsrc, None, MODIFIERS))
	def v_max_u32_dpp(self, gfx10_vsrc:reg_block, gfx10_vsrc:reg_block, MODIFIERS:str=''):
		return self.ic_pb(dpp8_base('v_max_u32_dpp', None, None, gfx10_vsrc, gfx10_vsrc, None, MODIFIERS))
	def v_min_f16_dpp(self, gfx10_vsrc:reg_block, gfx10_vsrc:reg_block, MODIFIERS:str=''):
		return self.ic_pb(dpp8_base('v_min_f16_dpp', None, None, gfx10_vsrc, gfx10_vsrc, None, MODIFIERS))
	def v_min_f32_dpp(self, gfx10_vsrc:reg_block, gfx10_vsrc:reg_block, MODIFIERS:str=''):
		return self.ic_pb(dpp8_base('v_min_f32_dpp', None, None, gfx10_vsrc, gfx10_vsrc, None, MODIFIERS))
	def v_min_i32_dpp(self, gfx10_vsrc:reg_block, gfx10_vsrc:reg_block, MODIFIERS:str=''):
		return self.ic_pb(dpp8_base('v_min_i32_dpp', None, None, gfx10_vsrc, gfx10_vsrc, None, MODIFIERS))
	def v_min_u32_dpp(self, gfx10_vsrc:reg_block, gfx10_vsrc:reg_block, MODIFIERS:str=''):
		return self.ic_pb(dpp8_base('v_min_u32_dpp', None, None, gfx10_vsrc, gfx10_vsrc, None, MODIFIERS))
	def v_mov_b32_dpp(self, gfx10_vsrc:reg_block, MODIFIERS:str=''):
		return self.ic_pb(dpp8_base('v_mov_b32_dpp', None, None, gfx10_vsrc, None, None, MODIFIERS))
	def v_movreld_b32_dpp(self, gfx10_vsrc:reg_block, MODIFIERS:str=''):
		return self.ic_pb(dpp8_base('v_movreld_b32_dpp', None, None, gfx10_vsrc, None, None, MODIFIERS))
	def v_movrels_b32_dpp(self, gfx10_vsrc:reg_block, MODIFIERS:str=''):
		return self.ic_pb(dpp8_base('v_movrels_b32_dpp', None, None, gfx10_vsrc, None, None, MODIFIERS))
	def v_movrelsd_2_b32_dpp(self, gfx10_vsrc:reg_block, MODIFIERS:str=''):
		return self.ic_pb(dpp8_base('v_movrelsd_2_b32_dpp', None, None, gfx10_vsrc, None, None, MODIFIERS))
	def v_movrelsd_b32_dpp(self, gfx10_vsrc:reg_block, MODIFIERS:str=''):
		return self.ic_pb(dpp8_base('v_movrelsd_b32_dpp', None, None, gfx10_vsrc, None, None, MODIFIERS))
	def v_mul_f16_dpp(self, gfx10_vsrc:reg_block, gfx10_vsrc:reg_block, MODIFIERS:str=''):
		return self.ic_pb(dpp8_base('v_mul_f16_dpp', None, None, gfx10_vsrc, gfx10_vsrc, None, MODIFIERS))
	def v_mul_f32_dpp(self, gfx10_vsrc:reg_block, gfx10_vsrc:reg_block, MODIFIERS:str=''):
		return self.ic_pb(dpp8_base('v_mul_f32_dpp', None, None, gfx10_vsrc, gfx10_vsrc, None, MODIFIERS))
	def v_mul_hi_i32_i24_dpp(self, gfx10_vsrc:reg_block, gfx10_vsrc:reg_block, MODIFIERS:str=''):
		return self.ic_pb(dpp8_base('v_mul_hi_i32_i24_dpp', None, None, gfx10_vsrc, gfx10_vsrc, None, MODIFIERS))
	def v_mul_hi_u32_u24_dpp(self, gfx10_vsrc:reg_block, gfx10_vsrc:reg_block, MODIFIERS:str=''):
		return self.ic_pb(dpp8_base('v_mul_hi_u32_u24_dpp', None, None, gfx10_vsrc, gfx10_vsrc, None, MODIFIERS))
	def v_mul_i32_i24_dpp(self, gfx10_vsrc:reg_block, gfx10_vsrc:reg_block, MODIFIERS:str=''):
		return self.ic_pb(dpp8_base('v_mul_i32_i24_dpp', None, None, gfx10_vsrc, gfx10_vsrc, None, MODIFIERS))
	def v_mul_legacy_f32_dpp(self, gfx10_vsrc:reg_block, gfx10_vsrc:reg_block, MODIFIERS:str=''):
		return self.ic_pb(dpp8_base('v_mul_legacy_f32_dpp', None, None, gfx10_vsrc, gfx10_vsrc, None, MODIFIERS))
	def v_mul_u32_u24_dpp(self, gfx10_vsrc:reg_block, gfx10_vsrc:reg_block, MODIFIERS:str=''):
		return self.ic_pb(dpp8_base('v_mul_u32_u24_dpp', None, None, gfx10_vsrc, gfx10_vsrc, None, MODIFIERS))
	def v_not_b32_dpp(self, gfx10_vsrc:reg_block, MODIFIERS:str=''):
		return self.ic_pb(dpp8_base('v_not_b32_dpp', None, None, gfx10_vsrc, None, None, MODIFIERS))
	def v_or_b32_dpp(self, gfx10_vsrc:reg_block, gfx10_vsrc:reg_block, MODIFIERS:str=''):
		return self.ic_pb(dpp8_base('v_or_b32_dpp', None, None, gfx10_vsrc, gfx10_vsrc, None, MODIFIERS))
	def v_rcp_f16_dpp(self, gfx10_vsrc:reg_block, MODIFIERS:str=''):
		return self.ic_pb(dpp8_base('v_rcp_f16_dpp', None, None, gfx10_vsrc, None, None, MODIFIERS))
	def v_rcp_f32_dpp(self, gfx10_vsrc:reg_block, MODIFIERS:str=''):
		return self.ic_pb(dpp8_base('v_rcp_f32_dpp', None, None, gfx10_vsrc, None, None, MODIFIERS))
	def v_rcp_iflag_f32_dpp(self, gfx10_vsrc:reg_block, MODIFIERS:str=''):
		return self.ic_pb(dpp8_base('v_rcp_iflag_f32_dpp', None, None, gfx10_vsrc, None, None, MODIFIERS))
	def v_rndne_f16_dpp(self, gfx10_vsrc:reg_block, MODIFIERS:str=''):
		return self.ic_pb(dpp8_base('v_rndne_f16_dpp', None, None, gfx10_vsrc, None, None, MODIFIERS))
	def v_rndne_f32_dpp(self, gfx10_vsrc:reg_block, MODIFIERS:str=''):
		return self.ic_pb(dpp8_base('v_rndne_f32_dpp', None, None, gfx10_vsrc, None, None, MODIFIERS))
	def v_rsq_f16_dpp(self, gfx10_vsrc:reg_block, MODIFIERS:str=''):
		return self.ic_pb(dpp8_base('v_rsq_f16_dpp', None, None, gfx10_vsrc, None, None, MODIFIERS))
	def v_rsq_f32_dpp(self, gfx10_vsrc:reg_block, MODIFIERS:str=''):
		return self.ic_pb(dpp8_base('v_rsq_f32_dpp', None, None, gfx10_vsrc, None, None, MODIFIERS))
	def v_sat_pk_u8_i16_dpp(self, gfx10_vsrc:reg_block, MODIFIERS:str=''):
		return self.ic_pb(dpp8_base('v_sat_pk_u8_i16_dpp', None, None, gfx10_vsrc, None, None, MODIFIERS))
	def v_sin_f16_dpp(self, gfx10_vsrc:reg_block, MODIFIERS:str=''):
		return self.ic_pb(dpp8_base('v_sin_f16_dpp', None, None, gfx10_vsrc, None, None, MODIFIERS))
	def v_sin_f32_dpp(self, gfx10_vsrc:reg_block, MODIFIERS:str=''):
		return self.ic_pb(dpp8_base('v_sin_f32_dpp', None, None, gfx10_vsrc, None, None, MODIFIERS))
	def v_sqrt_f16_dpp(self, gfx10_vsrc:reg_block, MODIFIERS:str=''):
		return self.ic_pb(dpp8_base('v_sqrt_f16_dpp', None, None, gfx10_vsrc, None, None, MODIFIERS))
	def v_sqrt_f32_dpp(self, gfx10_vsrc:reg_block, MODIFIERS:str=''):
		return self.ic_pb(dpp8_base('v_sqrt_f32_dpp', None, None, gfx10_vsrc, None, None, MODIFIERS))
	def v_sub_co_ci_u32_dpp(self, gfx10_vcc:reg_block, gfx10_vsrc:reg_block, gfx10_vsrc:reg_block, gfx10_vcc:reg_block, MODIFIERS:str=''):
		return self.ic_pb(dpp8_base('v_sub_co_ci_u32_dpp', None, gfx10_vcc, gfx10_vsrc, gfx10_vsrc, gfx10_vcc, MODIFIERS))
	def v_sub_f16_dpp(self, gfx10_vsrc:reg_block, gfx10_vsrc:reg_block, MODIFIERS:str=''):
		return self.ic_pb(dpp8_base('v_sub_f16_dpp', None, None, gfx10_vsrc, gfx10_vsrc, None, MODIFIERS))
	def v_sub_f32_dpp(self, gfx10_vsrc:reg_block, gfx10_vsrc:reg_block, MODIFIERS:str=''):
		return self.ic_pb(dpp8_base('v_sub_f32_dpp', None, None, gfx10_vsrc, gfx10_vsrc, None, MODIFIERS))
	def v_sub_nc_u32_dpp(self, gfx10_vsrc:reg_block, gfx10_vsrc:reg_block, MODIFIERS:str=''):
		return self.ic_pb(dpp8_base('v_sub_nc_u32_dpp', None, None, gfx10_vsrc, gfx10_vsrc, None, MODIFIERS))
	def v_subrev_co_ci_u32_dpp(self, gfx10_vcc:reg_block, gfx10_vsrc:reg_block, gfx10_vsrc:reg_block, gfx10_vcc:reg_block, MODIFIERS:str=''):
		return self.ic_pb(dpp8_base('v_subrev_co_ci_u32_dpp', None, gfx10_vcc, gfx10_vsrc, gfx10_vsrc, gfx10_vcc, MODIFIERS))
	def v_subrev_f16_dpp(self, gfx10_vsrc:reg_block, gfx10_vsrc:reg_block, MODIFIERS:str=''):
		return self.ic_pb(dpp8_base('v_subrev_f16_dpp', None, None, gfx10_vsrc, gfx10_vsrc, None, MODIFIERS))
	def v_subrev_f32_dpp(self, gfx10_vsrc:reg_block, gfx10_vsrc:reg_block, MODIFIERS:str=''):
		return self.ic_pb(dpp8_base('v_subrev_f32_dpp', None, None, gfx10_vsrc, gfx10_vsrc, None, MODIFIERS))
	def v_subrev_nc_u32_dpp(self, gfx10_vsrc:reg_block, gfx10_vsrc:reg_block, MODIFIERS:str=''):
		return self.ic_pb(dpp8_base('v_subrev_nc_u32_dpp', None, None, gfx10_vsrc, gfx10_vsrc, None, MODIFIERS))
	def v_trunc_f16_dpp(self, gfx10_vsrc:reg_block, MODIFIERS:str=''):
		return self.ic_pb(dpp8_base('v_trunc_f16_dpp', None, None, gfx10_vsrc, None, None, MODIFIERS))
	def v_trunc_f32_dpp(self, gfx10_vsrc:reg_block, MODIFIERS:str=''):
		return self.ic_pb(dpp8_base('v_trunc_f32_dpp', None, None, gfx10_vsrc, None, None, MODIFIERS))
	def v_xnor_b32_dpp(self, gfx10_vsrc:reg_block, gfx10_vsrc:reg_block, MODIFIERS:str=''):
		return self.ic_pb(dpp8_base('v_xnor_b32_dpp', None, None, gfx10_vsrc, gfx10_vsrc, None, MODIFIERS))
	def v_xor_b32_dpp(self, gfx10_vsrc:reg_block, gfx10_vsrc:reg_block, MODIFIERS:str=''):
		return self.ic_pb(dpp8_base('v_xor_b32_dpp', None, None, gfx10_vsrc, gfx10_vsrc, None, MODIFIERS))
class ds_base(inst_base): 
	def __init__(self, INSTRUCTION:str, DST:Union[reg_block,None], SRC0:Union[reg_block,None], SRC1:Union[reg_block,None], SRC2:Union[reg_block,None], MODIFIERS:str): 
		super().__init__(instruction_type.SMEM, INSTRUCTION)
		self.DST = DST 
		self.SRC0 = SRC0 
		self.SRC1 = SRC1 
		self.SRC2 = SRC2 
		self.MODIFIERS = MODIFIERS 
	def __str__(self): 
		args_l = filter(None.__ne__, [self.DST,self.SRC0,self.SRC1,self.SRC2]) 
		return f"{self.label} {','.join(map(str, args_l))} {self.MODIFIERS}" 
class ds_instr_caller(inst_caller_base): 
	def __init__(self, insturction_container) -> None:
     		super().__init__(insturction_container)
	def ds_add_f32(self, gfx10_vdata:reg_block, MODIFIERS:str=''):
		return self.ic_pb(ds_base('ds_add_f32', None, None, gfx10_vdata, None, MODIFIERS))
	def ds_add_rtn_f32(self, gfx10_vaddr:reg_block, gfx10_vdata:reg_block, MODIFIERS:str=''):
		return self.ic_pb(ds_base('ds_add_rtn_f32', None, gfx10_vaddr, gfx10_vdata, None, MODIFIERS))
	def ds_add_rtn_u32(self, gfx10_vaddr:reg_block, gfx10_vdata:reg_block, MODIFIERS:str=''):
		return self.ic_pb(ds_base('ds_add_rtn_u32', None, gfx10_vaddr, gfx10_vdata, None, MODIFIERS))
	def ds_add_rtn_u64(self, gfx10_vaddr:reg_block, gfx10_vdata_1:reg_block, MODIFIERS:str=''):
		return self.ic_pb(ds_base('ds_add_rtn_u64', None, gfx10_vaddr, gfx10_vdata_1, None, MODIFIERS))
	def ds_add_src2_f32(self, MODIFIERS:str=''):
		return self.ic_pb(ds_base('ds_add_src2_f32', None, None, None, None, MODIFIERS))
	def ds_add_src2_u32(self, MODIFIERS:str=''):
		return self.ic_pb(ds_base('ds_add_src2_u32', None, None, None, None, MODIFIERS))
	def ds_add_src2_u64(self, MODIFIERS:str=''):
		return self.ic_pb(ds_base('ds_add_src2_u64', None, None, None, None, MODIFIERS))
	def ds_add_u32(self, gfx10_vdata:reg_block, MODIFIERS:str=''):
		return self.ic_pb(ds_base('ds_add_u32', None, None, gfx10_vdata, None, MODIFIERS))
	def ds_add_u64(self, gfx10_vdata_1:reg_block, MODIFIERS:str=''):
		return self.ic_pb(ds_base('ds_add_u64', None, None, gfx10_vdata_1, None, MODIFIERS))
	def ds_and_b32(self, gfx10_vdata:reg_block, MODIFIERS:str=''):
		return self.ic_pb(ds_base('ds_and_b32', None, None, gfx10_vdata, None, MODIFIERS))
	def ds_and_b64(self, gfx10_vdata_1:reg_block, MODIFIERS:str=''):
		return self.ic_pb(ds_base('ds_and_b64', None, None, gfx10_vdata_1, None, MODIFIERS))
	def ds_and_rtn_b32(self, gfx10_vaddr:reg_block, gfx10_vdata:reg_block, MODIFIERS:str=''):
		return self.ic_pb(ds_base('ds_and_rtn_b32', None, gfx10_vaddr, gfx10_vdata, None, MODIFIERS))
	def ds_and_rtn_b64(self, gfx10_vaddr:reg_block, gfx10_vdata_1:reg_block, MODIFIERS:str=''):
		return self.ic_pb(ds_base('ds_and_rtn_b64', None, gfx10_vaddr, gfx10_vdata_1, None, MODIFIERS))
	def ds_and_src2_b32(self, MODIFIERS:str=''):
		return self.ic_pb(ds_base('ds_and_src2_b32', None, None, None, None, MODIFIERS))
	def ds_and_src2_b64(self, MODIFIERS:str=''):
		return self.ic_pb(ds_base('ds_and_src2_b64', None, None, None, None, MODIFIERS))
	def ds_append(self, MODIFIERS:str=''):
		return self.ic_pb(ds_base('ds_append', None, None, None, None, MODIFIERS))
	def ds_bpermute_b32(self, gfx10_vaddr:reg_block, gfx10_vdata:reg_block, MODIFIERS:str=''):
		return self.ic_pb(ds_base('ds_bpermute_b32', None, gfx10_vaddr, gfx10_vdata, None, MODIFIERS))
	def ds_cmpst_b32(self, gfx10_vdata0:reg_block, gfx10_vdata1:reg_block, MODIFIERS:str=''):
		return self.ic_pb(ds_base('ds_cmpst_b32', None, None, gfx10_vdata0, gfx10_vdata1, MODIFIERS))
	def ds_cmpst_b64(self, gfx10_vdata0_1:reg_block, gfx10_vdata1_1:reg_block, MODIFIERS:str=''):
		return self.ic_pb(ds_base('ds_cmpst_b64', None, None, gfx10_vdata0_1, gfx10_vdata1_1, MODIFIERS))
	def ds_cmpst_f32(self, gfx10_vdata0:reg_block, gfx10_vdata1:reg_block, MODIFIERS:str=''):
		return self.ic_pb(ds_base('ds_cmpst_f32', None, None, gfx10_vdata0, gfx10_vdata1, MODIFIERS))
	def ds_cmpst_f64(self, gfx10_vdata0_1:reg_block, gfx10_vdata1_1:reg_block, MODIFIERS:str=''):
		return self.ic_pb(ds_base('ds_cmpst_f64', None, None, gfx10_vdata0_1, gfx10_vdata1_1, MODIFIERS))
	def ds_cmpst_rtn_b32(self, gfx10_vaddr:reg_block, gfx10_vdata0:reg_block, gfx10_vdata1:reg_block, MODIFIERS:str=''):
		return self.ic_pb(ds_base('ds_cmpst_rtn_b32', None, gfx10_vaddr, gfx10_vdata0, gfx10_vdata1, MODIFIERS))
	def ds_cmpst_rtn_b64(self, gfx10_vaddr:reg_block, gfx10_vdata0_1:reg_block, gfx10_vdata1_1:reg_block, MODIFIERS:str=''):
		return self.ic_pb(ds_base('ds_cmpst_rtn_b64', None, gfx10_vaddr, gfx10_vdata0_1, gfx10_vdata1_1, MODIFIERS))
	def ds_cmpst_rtn_f32(self, gfx10_vaddr:reg_block, gfx10_vdata0:reg_block, gfx10_vdata1:reg_block, MODIFIERS:str=''):
		return self.ic_pb(ds_base('ds_cmpst_rtn_f32', None, gfx10_vaddr, gfx10_vdata0, gfx10_vdata1, MODIFIERS))
	def ds_cmpst_rtn_f64(self, gfx10_vaddr:reg_block, gfx10_vdata0_1:reg_block, gfx10_vdata1_1:reg_block, MODIFIERS:str=''):
		return self.ic_pb(ds_base('ds_cmpst_rtn_f64', None, gfx10_vaddr, gfx10_vdata0_1, gfx10_vdata1_1, MODIFIERS))
	def ds_condxchg32_rtn_b64(self, gfx10_vaddr:reg_block, gfx10_vdata_1:reg_block, MODIFIERS:str=''):
		return self.ic_pb(ds_base('ds_condxchg32_rtn_b64', None, gfx10_vaddr, gfx10_vdata_1, None, MODIFIERS))
	def ds_consume(self, MODIFIERS:str=''):
		return self.ic_pb(ds_base('ds_consume', None, None, None, None, MODIFIERS))
	def ds_dec_rtn_u32(self, gfx10_vaddr:reg_block, gfx10_vdata:reg_block, MODIFIERS:str=''):
		return self.ic_pb(ds_base('ds_dec_rtn_u32', None, gfx10_vaddr, gfx10_vdata, None, MODIFIERS))
	def ds_dec_rtn_u64(self, gfx10_vaddr:reg_block, gfx10_vdata_1:reg_block, MODIFIERS:str=''):
		return self.ic_pb(ds_base('ds_dec_rtn_u64', None, gfx10_vaddr, gfx10_vdata_1, None, MODIFIERS))
	def ds_dec_src2_u32(self, MODIFIERS:str=''):
		return self.ic_pb(ds_base('ds_dec_src2_u32', None, None, None, None, MODIFIERS))
	def ds_dec_src2_u64(self, MODIFIERS:str=''):
		return self.ic_pb(ds_base('ds_dec_src2_u64', None, None, None, None, MODIFIERS))
	def ds_dec_u32(self, gfx10_vdata:reg_block, MODIFIERS:str=''):
		return self.ic_pb(ds_base('ds_dec_u32', None, None, gfx10_vdata, None, MODIFIERS))
	def ds_dec_u64(self, gfx10_vdata_1:reg_block, MODIFIERS:str=''):
		return self.ic_pb(ds_base('ds_dec_u64', None, None, gfx10_vdata_1, None, MODIFIERS))
	def ds_gws_barrier(self, MODIFIERS:str=''):
		return self.ic_pb(ds_base('ds_gws_barrier', None, None, None, None, MODIFIERS))
	def ds_gws_init(self, MODIFIERS:str=''):
		return self.ic_pb(ds_base('ds_gws_init', None, None, None, None, MODIFIERS))
	def ds_gws_sema_br(self, MODIFIERS:str=''):
		return self.ic_pb(ds_base('ds_gws_sema_br', None, None, None, None, MODIFIERS))
	def ds_gws_sema_p(self, MODIFIERS:str=''):
		return self.ic_pb(ds_base('ds_gws_sema_p', None, None, None, None, MODIFIERS))
	def ds_gws_sema_release_all(self, MODIFIERS:str=''):
		return self.ic_pb(ds_base('ds_gws_sema_release_all', None, None, None, None, MODIFIERS))
	def ds_gws_sema_v(self, MODIFIERS:str=''):
		return self.ic_pb(ds_base('ds_gws_sema_v', None, None, None, None, MODIFIERS))
	def ds_inc_rtn_u32(self, gfx10_vaddr:reg_block, gfx10_vdata:reg_block, MODIFIERS:str=''):
		return self.ic_pb(ds_base('ds_inc_rtn_u32', None, gfx10_vaddr, gfx10_vdata, None, MODIFIERS))
	def ds_inc_rtn_u64(self, gfx10_vaddr:reg_block, gfx10_vdata_1:reg_block, MODIFIERS:str=''):
		return self.ic_pb(ds_base('ds_inc_rtn_u64', None, gfx10_vaddr, gfx10_vdata_1, None, MODIFIERS))
	def ds_inc_src2_u32(self, MODIFIERS:str=''):
		return self.ic_pb(ds_base('ds_inc_src2_u32', None, None, None, None, MODIFIERS))
	def ds_inc_src2_u64(self, MODIFIERS:str=''):
		return self.ic_pb(ds_base('ds_inc_src2_u64', None, None, None, None, MODIFIERS))
	def ds_inc_u32(self, gfx10_vdata:reg_block, MODIFIERS:str=''):
		return self.ic_pb(ds_base('ds_inc_u32', None, None, gfx10_vdata, None, MODIFIERS))
	def ds_inc_u64(self, gfx10_vdata_1:reg_block, MODIFIERS:str=''):
		return self.ic_pb(ds_base('ds_inc_u64', None, None, gfx10_vdata_1, None, MODIFIERS))
	def ds_max_f32(self, gfx10_vdata:reg_block, MODIFIERS:str=''):
		return self.ic_pb(ds_base('ds_max_f32', None, None, gfx10_vdata, None, MODIFIERS))
	def ds_max_f64(self, gfx10_vdata_1:reg_block, MODIFIERS:str=''):
		return self.ic_pb(ds_base('ds_max_f64', None, None, gfx10_vdata_1, None, MODIFIERS))
	def ds_max_i32(self, gfx10_vdata:reg_block, MODIFIERS:str=''):
		return self.ic_pb(ds_base('ds_max_i32', None, None, gfx10_vdata, None, MODIFIERS))
	def ds_max_i64(self, gfx10_vdata_1:reg_block, MODIFIERS:str=''):
		return self.ic_pb(ds_base('ds_max_i64', None, None, gfx10_vdata_1, None, MODIFIERS))
	def ds_max_rtn_f32(self, gfx10_vaddr:reg_block, gfx10_vdata:reg_block, MODIFIERS:str=''):
		return self.ic_pb(ds_base('ds_max_rtn_f32', None, gfx10_vaddr, gfx10_vdata, None, MODIFIERS))
	def ds_max_rtn_f64(self, gfx10_vaddr:reg_block, gfx10_vdata_1:reg_block, MODIFIERS:str=''):
		return self.ic_pb(ds_base('ds_max_rtn_f64', None, gfx10_vaddr, gfx10_vdata_1, None, MODIFIERS))
	def ds_max_rtn_i32(self, gfx10_vaddr:reg_block, gfx10_vdata:reg_block, MODIFIERS:str=''):
		return self.ic_pb(ds_base('ds_max_rtn_i32', None, gfx10_vaddr, gfx10_vdata, None, MODIFIERS))
	def ds_max_rtn_i64(self, gfx10_vaddr:reg_block, gfx10_vdata_1:reg_block, MODIFIERS:str=''):
		return self.ic_pb(ds_base('ds_max_rtn_i64', None, gfx10_vaddr, gfx10_vdata_1, None, MODIFIERS))
	def ds_max_rtn_u32(self, gfx10_vaddr:reg_block, gfx10_vdata:reg_block, MODIFIERS:str=''):
		return self.ic_pb(ds_base('ds_max_rtn_u32', None, gfx10_vaddr, gfx10_vdata, None, MODIFIERS))
	def ds_max_rtn_u64(self, gfx10_vaddr:reg_block, gfx10_vdata_1:reg_block, MODIFIERS:str=''):
		return self.ic_pb(ds_base('ds_max_rtn_u64', None, gfx10_vaddr, gfx10_vdata_1, None, MODIFIERS))
	def ds_max_src2_f32(self, MODIFIERS:str=''):
		return self.ic_pb(ds_base('ds_max_src2_f32', None, None, None, None, MODIFIERS))
	def ds_max_src2_f64(self, MODIFIERS:str=''):
		return self.ic_pb(ds_base('ds_max_src2_f64', None, None, None, None, MODIFIERS))
	def ds_max_src2_i32(self, MODIFIERS:str=''):
		return self.ic_pb(ds_base('ds_max_src2_i32', None, None, None, None, MODIFIERS))
	def ds_max_src2_i64(self, MODIFIERS:str=''):
		return self.ic_pb(ds_base('ds_max_src2_i64', None, None, None, None, MODIFIERS))
	def ds_max_src2_u32(self, MODIFIERS:str=''):
		return self.ic_pb(ds_base('ds_max_src2_u32', None, None, None, None, MODIFIERS))
	def ds_max_src2_u64(self, MODIFIERS:str=''):
		return self.ic_pb(ds_base('ds_max_src2_u64', None, None, None, None, MODIFIERS))
	def ds_max_u32(self, gfx10_vdata:reg_block, MODIFIERS:str=''):
		return self.ic_pb(ds_base('ds_max_u32', None, None, gfx10_vdata, None, MODIFIERS))
	def ds_max_u64(self, gfx10_vdata_1:reg_block, MODIFIERS:str=''):
		return self.ic_pb(ds_base('ds_max_u64', None, None, gfx10_vdata_1, None, MODIFIERS))
	def ds_min_f32(self, gfx10_vdata:reg_block, MODIFIERS:str=''):
		return self.ic_pb(ds_base('ds_min_f32', None, None, gfx10_vdata, None, MODIFIERS))
	def ds_min_f64(self, gfx10_vdata_1:reg_block, MODIFIERS:str=''):
		return self.ic_pb(ds_base('ds_min_f64', None, None, gfx10_vdata_1, None, MODIFIERS))
	def ds_min_i32(self, gfx10_vdata:reg_block, MODIFIERS:str=''):
		return self.ic_pb(ds_base('ds_min_i32', None, None, gfx10_vdata, None, MODIFIERS))
	def ds_min_i64(self, gfx10_vdata_1:reg_block, MODIFIERS:str=''):
		return self.ic_pb(ds_base('ds_min_i64', None, None, gfx10_vdata_1, None, MODIFIERS))
	def ds_min_rtn_f32(self, gfx10_vaddr:reg_block, gfx10_vdata:reg_block, MODIFIERS:str=''):
		return self.ic_pb(ds_base('ds_min_rtn_f32', None, gfx10_vaddr, gfx10_vdata, None, MODIFIERS))
	def ds_min_rtn_f64(self, gfx10_vaddr:reg_block, gfx10_vdata_1:reg_block, MODIFIERS:str=''):
		return self.ic_pb(ds_base('ds_min_rtn_f64', None, gfx10_vaddr, gfx10_vdata_1, None, MODIFIERS))
	def ds_min_rtn_i32(self, gfx10_vaddr:reg_block, gfx10_vdata:reg_block, MODIFIERS:str=''):
		return self.ic_pb(ds_base('ds_min_rtn_i32', None, gfx10_vaddr, gfx10_vdata, None, MODIFIERS))
	def ds_min_rtn_i64(self, gfx10_vaddr:reg_block, gfx10_vdata_1:reg_block, MODIFIERS:str=''):
		return self.ic_pb(ds_base('ds_min_rtn_i64', None, gfx10_vaddr, gfx10_vdata_1, None, MODIFIERS))
	def ds_min_rtn_u32(self, gfx10_vaddr:reg_block, gfx10_vdata:reg_block, MODIFIERS:str=''):
		return self.ic_pb(ds_base('ds_min_rtn_u32', None, gfx10_vaddr, gfx10_vdata, None, MODIFIERS))
	def ds_min_rtn_u64(self, gfx10_vaddr:reg_block, gfx10_vdata_1:reg_block, MODIFIERS:str=''):
		return self.ic_pb(ds_base('ds_min_rtn_u64', None, gfx10_vaddr, gfx10_vdata_1, None, MODIFIERS))
	def ds_min_src2_f32(self, MODIFIERS:str=''):
		return self.ic_pb(ds_base('ds_min_src2_f32', None, None, None, None, MODIFIERS))
	def ds_min_src2_f64(self, MODIFIERS:str=''):
		return self.ic_pb(ds_base('ds_min_src2_f64', None, None, None, None, MODIFIERS))
	def ds_min_src2_i32(self, MODIFIERS:str=''):
		return self.ic_pb(ds_base('ds_min_src2_i32', None, None, None, None, MODIFIERS))
	def ds_min_src2_i64(self, MODIFIERS:str=''):
		return self.ic_pb(ds_base('ds_min_src2_i64', None, None, None, None, MODIFIERS))
	def ds_min_src2_u32(self, MODIFIERS:str=''):
		return self.ic_pb(ds_base('ds_min_src2_u32', None, None, None, None, MODIFIERS))
	def ds_min_src2_u64(self, MODIFIERS:str=''):
		return self.ic_pb(ds_base('ds_min_src2_u64', None, None, None, None, MODIFIERS))
	def ds_min_u32(self, gfx10_vdata:reg_block, MODIFIERS:str=''):
		return self.ic_pb(ds_base('ds_min_u32', None, None, gfx10_vdata, None, MODIFIERS))
	def ds_min_u64(self, gfx10_vdata_1:reg_block, MODIFIERS:str=''):
		return self.ic_pb(ds_base('ds_min_u64', None, None, gfx10_vdata_1, None, MODIFIERS))
	def ds_mskor_b32(self, gfx10_vdata0:reg_block, gfx10_vdata1:reg_block, MODIFIERS:str=''):
		return self.ic_pb(ds_base('ds_mskor_b32', None, None, gfx10_vdata0, gfx10_vdata1, MODIFIERS))
	def ds_mskor_b64(self, gfx10_vdata0_1:reg_block, gfx10_vdata1_1:reg_block, MODIFIERS:str=''):
		return self.ic_pb(ds_base('ds_mskor_b64', None, None, gfx10_vdata0_1, gfx10_vdata1_1, MODIFIERS))
	def ds_mskor_rtn_b32(self, gfx10_vaddr:reg_block, gfx10_vdata0:reg_block, gfx10_vdata1:reg_block, MODIFIERS:str=''):
		return self.ic_pb(ds_base('ds_mskor_rtn_b32', None, gfx10_vaddr, gfx10_vdata0, gfx10_vdata1, MODIFIERS))
	def ds_mskor_rtn_b64(self, gfx10_vaddr:reg_block, gfx10_vdata0_1:reg_block, gfx10_vdata1_1:reg_block, MODIFIERS:str=''):
		return self.ic_pb(ds_base('ds_mskor_rtn_b64', None, gfx10_vaddr, gfx10_vdata0_1, gfx10_vdata1_1, MODIFIERS))
	def ds_nop(self):
		return self.ic_pb(ds_base('ds_nop', None, None, None, None))
	def ds_or_b32(self, gfx10_vdata:reg_block, MODIFIERS:str=''):
		return self.ic_pb(ds_base('ds_or_b32', None, None, gfx10_vdata, None, MODIFIERS))
	def ds_or_b64(self, gfx10_vdata_1:reg_block, MODIFIERS:str=''):
		return self.ic_pb(ds_base('ds_or_b64', None, None, gfx10_vdata_1, None, MODIFIERS))
	def ds_or_rtn_b32(self, gfx10_vaddr:reg_block, gfx10_vdata:reg_block, MODIFIERS:str=''):
		return self.ic_pb(ds_base('ds_or_rtn_b32', None, gfx10_vaddr, gfx10_vdata, None, MODIFIERS))
	def ds_or_rtn_b64(self, gfx10_vaddr:reg_block, gfx10_vdata_1:reg_block, MODIFIERS:str=''):
		return self.ic_pb(ds_base('ds_or_rtn_b64', None, gfx10_vaddr, gfx10_vdata_1, None, MODIFIERS))
	def ds_or_src2_b32(self, MODIFIERS:str=''):
		return self.ic_pb(ds_base('ds_or_src2_b32', None, None, None, None, MODIFIERS))
	def ds_or_src2_b64(self, MODIFIERS:str=''):
		return self.ic_pb(ds_base('ds_or_src2_b64', None, None, None, None, MODIFIERS))
	def ds_ordered_count(self, gfx10_vaddr:reg_block, MODIFIERS:str=''):
		return self.ic_pb(ds_base('ds_ordered_count', None, gfx10_vaddr, None, None, MODIFIERS))
	def ds_permute_b32(self, gfx10_vaddr:reg_block, gfx10_vdata:reg_block, MODIFIERS:str=''):
		return self.ic_pb(ds_base('ds_permute_b32', None, gfx10_vaddr, gfx10_vdata, None, MODIFIERS))
	def ds_read2_b32(self, gfx10_vaddr:reg_block, MODIFIERS:str=''):
		return self.ic_pb(ds_base('ds_read2_b32', None, gfx10_vaddr, None, None, MODIFIERS))
	def ds_read2_b64(self, gfx10_vaddr:reg_block, MODIFIERS:str=''):
		return self.ic_pb(ds_base('ds_read2_b64', None, gfx10_vaddr, None, None, MODIFIERS))
	def ds_read2st64_b32(self, gfx10_vaddr:reg_block, MODIFIERS:str=''):
		return self.ic_pb(ds_base('ds_read2st64_b32', None, gfx10_vaddr, None, None, MODIFIERS))
	def ds_read2st64_b64(self, gfx10_vaddr:reg_block, MODIFIERS:str=''):
		return self.ic_pb(ds_base('ds_read2st64_b64', None, gfx10_vaddr, None, None, MODIFIERS))
	def ds_read_addtid_b32(self, MODIFIERS:str=''):
		return self.ic_pb(ds_base('ds_read_addtid_b32', None, None, None, None, MODIFIERS))
	def ds_read_b128(self, gfx10_vaddr:reg_block, MODIFIERS:str=''):
		return self.ic_pb(ds_base('ds_read_b128', None, gfx10_vaddr, None, None, MODIFIERS))
	def ds_read_b32(self, gfx10_vaddr:reg_block, MODIFIERS:str=''):
		return self.ic_pb(ds_base('ds_read_b32', None, gfx10_vaddr, None, None, MODIFIERS))
	def ds_read_b64(self, gfx10_vaddr:reg_block, MODIFIERS:str=''):
		return self.ic_pb(ds_base('ds_read_b64', None, gfx10_vaddr, None, None, MODIFIERS))
	def ds_read_b96(self, gfx10_vaddr:reg_block, MODIFIERS:str=''):
		return self.ic_pb(ds_base('ds_read_b96', None, gfx10_vaddr, None, None, MODIFIERS))
	def ds_read_i16(self, gfx10_vaddr:reg_block, MODIFIERS:str=''):
		return self.ic_pb(ds_base('ds_read_i16', None, gfx10_vaddr, None, None, MODIFIERS))
	def ds_read_i8(self, gfx10_vaddr:reg_block, MODIFIERS:str=''):
		return self.ic_pb(ds_base('ds_read_i8', None, gfx10_vaddr, None, None, MODIFIERS))
	def ds_read_i8_d16(self, gfx10_vaddr:reg_block, MODIFIERS:str=''):
		return self.ic_pb(ds_base('ds_read_i8_d16', None, gfx10_vaddr, None, None, MODIFIERS))
	def ds_read_i8_d16_hi(self, gfx10_vaddr:reg_block, MODIFIERS:str=''):
		return self.ic_pb(ds_base('ds_read_i8_d16_hi', None, gfx10_vaddr, None, None, MODIFIERS))
	def ds_read_u16(self, gfx10_vaddr:reg_block, MODIFIERS:str=''):
		return self.ic_pb(ds_base('ds_read_u16', None, gfx10_vaddr, None, None, MODIFIERS))
	def ds_read_u16_d16(self, gfx10_vaddr:reg_block, MODIFIERS:str=''):
		return self.ic_pb(ds_base('ds_read_u16_d16', None, gfx10_vaddr, None, None, MODIFIERS))
	def ds_read_u16_d16_hi(self, gfx10_vaddr:reg_block, MODIFIERS:str=''):
		return self.ic_pb(ds_base('ds_read_u16_d16_hi', None, gfx10_vaddr, None, None, MODIFIERS))
	def ds_read_u8(self, gfx10_vaddr:reg_block, MODIFIERS:str=''):
		return self.ic_pb(ds_base('ds_read_u8', None, gfx10_vaddr, None, None, MODIFIERS))
	def ds_read_u8_d16(self, gfx10_vaddr:reg_block, MODIFIERS:str=''):
		return self.ic_pb(ds_base('ds_read_u8_d16', None, gfx10_vaddr, None, None, MODIFIERS))
	def ds_read_u8_d16_hi(self, gfx10_vaddr:reg_block, MODIFIERS:str=''):
		return self.ic_pb(ds_base('ds_read_u8_d16_hi', None, gfx10_vaddr, None, None, MODIFIERS))
	def ds_rsub_rtn_u32(self, gfx10_vaddr:reg_block, gfx10_vdata:reg_block, MODIFIERS:str=''):
		return self.ic_pb(ds_base('ds_rsub_rtn_u32', None, gfx10_vaddr, gfx10_vdata, None, MODIFIERS))
	def ds_rsub_rtn_u64(self, gfx10_vaddr:reg_block, gfx10_vdata_1:reg_block, MODIFIERS:str=''):
		return self.ic_pb(ds_base('ds_rsub_rtn_u64', None, gfx10_vaddr, gfx10_vdata_1, None, MODIFIERS))
	def ds_rsub_src2_u32(self, MODIFIERS:str=''):
		return self.ic_pb(ds_base('ds_rsub_src2_u32', None, None, None, None, MODIFIERS))
	def ds_rsub_src2_u64(self, MODIFIERS:str=''):
		return self.ic_pb(ds_base('ds_rsub_src2_u64', None, None, None, None, MODIFIERS))
	def ds_rsub_u32(self, gfx10_vdata:reg_block, MODIFIERS:str=''):
		return self.ic_pb(ds_base('ds_rsub_u32', None, None, gfx10_vdata, None, MODIFIERS))
	def ds_rsub_u64(self, gfx10_vdata_1:reg_block, MODIFIERS:str=''):
		return self.ic_pb(ds_base('ds_rsub_u64', None, None, gfx10_vdata_1, None, MODIFIERS))
	def ds_sub_rtn_u32(self, gfx10_vaddr:reg_block, gfx10_vdata:reg_block, MODIFIERS:str=''):
		return self.ic_pb(ds_base('ds_sub_rtn_u32', None, gfx10_vaddr, gfx10_vdata, None, MODIFIERS))
	def ds_sub_rtn_u64(self, gfx10_vaddr:reg_block, gfx10_vdata_1:reg_block, MODIFIERS:str=''):
		return self.ic_pb(ds_base('ds_sub_rtn_u64', None, gfx10_vaddr, gfx10_vdata_1, None, MODIFIERS))
	def ds_sub_src2_u32(self, MODIFIERS:str=''):
		return self.ic_pb(ds_base('ds_sub_src2_u32', None, None, None, None, MODIFIERS))
	def ds_sub_src2_u64(self, MODIFIERS:str=''):
		return self.ic_pb(ds_base('ds_sub_src2_u64', None, None, None, None, MODIFIERS))
	def ds_sub_u32(self, gfx10_vdata:reg_block, MODIFIERS:str=''):
		return self.ic_pb(ds_base('ds_sub_u32', None, None, gfx10_vdata, None, MODIFIERS))
	def ds_sub_u64(self, gfx10_vdata_1:reg_block, MODIFIERS:str=''):
		return self.ic_pb(ds_base('ds_sub_u64', None, None, gfx10_vdata_1, None, MODIFIERS))
	def ds_swizzle_b32(self, gfx10_vaddr:reg_block, MODIFIERS:str=''):
		return self.ic_pb(ds_base('ds_swizzle_b32', None, gfx10_vaddr, None, None, MODIFIERS))
	def ds_wrap_rtn_b32(self, gfx10_vaddr:reg_block, gfx10_vdata0:reg_block, gfx10_vdata1:reg_block, MODIFIERS:str=''):
		return self.ic_pb(ds_base('ds_wrap_rtn_b32', None, gfx10_vaddr, gfx10_vdata0, gfx10_vdata1, MODIFIERS))
	def ds_write2_b32(self, gfx10_vdata0:reg_block, gfx10_vdata1:reg_block, MODIFIERS:str=''):
		return self.ic_pb(ds_base('ds_write2_b32', None, None, gfx10_vdata0, gfx10_vdata1, MODIFIERS))
	def ds_write2_b64(self, gfx10_vdata0_1:reg_block, gfx10_vdata1_1:reg_block, MODIFIERS:str=''):
		return self.ic_pb(ds_base('ds_write2_b64', None, None, gfx10_vdata0_1, gfx10_vdata1_1, MODIFIERS))
	def ds_write2st64_b32(self, gfx10_vdata0:reg_block, gfx10_vdata1:reg_block, MODIFIERS:str=''):
		return self.ic_pb(ds_base('ds_write2st64_b32', None, None, gfx10_vdata0, gfx10_vdata1, MODIFIERS))
	def ds_write2st64_b64(self, gfx10_vdata0_1:reg_block, gfx10_vdata1_1:reg_block, MODIFIERS:str=''):
		return self.ic_pb(ds_base('ds_write2st64_b64', None, None, gfx10_vdata0_1, gfx10_vdata1_1, MODIFIERS))
	def ds_write_addtid_b32(self, MODIFIERS:str=''):
		return self.ic_pb(ds_base('ds_write_addtid_b32', None, None, None, None, MODIFIERS))
	def ds_write_b128(self, gfx10_vdata_2:reg_block, MODIFIERS:str=''):
		return self.ic_pb(ds_base('ds_write_b128', None, None, gfx10_vdata_2, None, MODIFIERS))
	def ds_write_b16(self, gfx10_vdata:reg_block, MODIFIERS:str=''):
		return self.ic_pb(ds_base('ds_write_b16', None, None, gfx10_vdata, None, MODIFIERS))
	def ds_write_b16_d16_hi(self, gfx10_vdata:reg_block, MODIFIERS:str=''):
		return self.ic_pb(ds_base('ds_write_b16_d16_hi', None, None, gfx10_vdata, None, MODIFIERS))
	def ds_write_b32(self, gfx10_vdata:reg_block, MODIFIERS:str=''):
		return self.ic_pb(ds_base('ds_write_b32', None, None, gfx10_vdata, None, MODIFIERS))
	def ds_write_b64(self, gfx10_vdata_1:reg_block, MODIFIERS:str=''):
		return self.ic_pb(ds_base('ds_write_b64', None, None, gfx10_vdata_1, None, MODIFIERS))
	def ds_write_b8(self, gfx10_vdata:reg_block, MODIFIERS:str=''):
		return self.ic_pb(ds_base('ds_write_b8', None, None, gfx10_vdata, None, MODIFIERS))
	def ds_write_b8_d16_hi(self, gfx10_vdata:reg_block, MODIFIERS:str=''):
		return self.ic_pb(ds_base('ds_write_b8_d16_hi', None, None, gfx10_vdata, None, MODIFIERS))
	def ds_write_b96(self, gfx10_vdata_3:reg_block, MODIFIERS:str=''):
		return self.ic_pb(ds_base('ds_write_b96', None, None, gfx10_vdata_3, None, MODIFIERS))
	def ds_write_src2_b32(self, MODIFIERS:str=''):
		return self.ic_pb(ds_base('ds_write_src2_b32', None, None, None, None, MODIFIERS))
	def ds_write_src2_b64(self, MODIFIERS:str=''):
		return self.ic_pb(ds_base('ds_write_src2_b64', None, None, None, None, MODIFIERS))
	def ds_wrxchg2_rtn_b32(self, gfx10_vaddr:reg_block, gfx10_vdata0:reg_block, gfx10_vdata1:reg_block, MODIFIERS:str=''):
		return self.ic_pb(ds_base('ds_wrxchg2_rtn_b32', None, gfx10_vaddr, gfx10_vdata0, gfx10_vdata1, MODIFIERS))
	def ds_wrxchg2_rtn_b64(self, gfx10_vaddr:reg_block, gfx10_vdata0_1:reg_block, gfx10_vdata1_1:reg_block, MODIFIERS:str=''):
		return self.ic_pb(ds_base('ds_wrxchg2_rtn_b64', None, gfx10_vaddr, gfx10_vdata0_1, gfx10_vdata1_1, MODIFIERS))
	def ds_wrxchg2st64_rtn_b32(self, gfx10_vaddr:reg_block, gfx10_vdata0:reg_block, gfx10_vdata1:reg_block, MODIFIERS:str=''):
		return self.ic_pb(ds_base('ds_wrxchg2st64_rtn_b32', None, gfx10_vaddr, gfx10_vdata0, gfx10_vdata1, MODIFIERS))
	def ds_wrxchg2st64_rtn_b64(self, gfx10_vaddr:reg_block, gfx10_vdata0_1:reg_block, gfx10_vdata1_1:reg_block, MODIFIERS:str=''):
		return self.ic_pb(ds_base('ds_wrxchg2st64_rtn_b64', None, gfx10_vaddr, gfx10_vdata0_1, gfx10_vdata1_1, MODIFIERS))
	def ds_wrxchg_rtn_b32(self, gfx10_vaddr:reg_block, gfx10_vdata:reg_block, MODIFIERS:str=''):
		return self.ic_pb(ds_base('ds_wrxchg_rtn_b32', None, gfx10_vaddr, gfx10_vdata, None, MODIFIERS))
	def ds_wrxchg_rtn_b64(self, gfx10_vaddr:reg_block, gfx10_vdata_1:reg_block, MODIFIERS:str=''):
		return self.ic_pb(ds_base('ds_wrxchg_rtn_b64', None, gfx10_vaddr, gfx10_vdata_1, None, MODIFIERS))
	def ds_xor_b32(self, gfx10_vdata:reg_block, MODIFIERS:str=''):
		return self.ic_pb(ds_base('ds_xor_b32', None, None, gfx10_vdata, None, MODIFIERS))
	def ds_xor_b64(self, gfx10_vdata_1:reg_block, MODIFIERS:str=''):
		return self.ic_pb(ds_base('ds_xor_b64', None, None, gfx10_vdata_1, None, MODIFIERS))
	def ds_xor_rtn_b32(self, gfx10_vaddr:reg_block, gfx10_vdata:reg_block, MODIFIERS:str=''):
		return self.ic_pb(ds_base('ds_xor_rtn_b32', None, gfx10_vaddr, gfx10_vdata, None, MODIFIERS))
	def ds_xor_rtn_b64(self, gfx10_vaddr:reg_block, gfx10_vdata_1:reg_block, MODIFIERS:str=''):
		return self.ic_pb(ds_base('ds_xor_rtn_b64', None, gfx10_vaddr, gfx10_vdata_1, None, MODIFIERS))
	def ds_xor_src2_b32(self, MODIFIERS:str=''):
		return self.ic_pb(ds_base('ds_xor_src2_b32', None, None, None, None, MODIFIERS))
	def ds_xor_src2_b64(self, MODIFIERS:str=''):
		return self.ic_pb(ds_base('ds_xor_src2_b64', None, None, None, None, MODIFIERS))
class exp_base(inst_base): 
	def __init__(self, INSTRUCTION:str, DST:Union[reg_block,None], SRC0:Union[reg_block,None], SRC1:Union[reg_block,None], SRC2:Union[reg_block,None], SRC3:Union[reg_block,None], MODIFIERS:str): 
		super().__init__(instruction_type.SMEM, INSTRUCTION)
		self.DST = DST 
		self.SRC0 = SRC0 
		self.SRC1 = SRC1 
		self.SRC2 = SRC2 
		self.SRC3 = SRC3 
		self.MODIFIERS = MODIFIERS 
	def __str__(self): 
		args_l = filter(None.__ne__, [self.DST,self.SRC0,self.SRC1,self.SRC2,self.SRC3]) 
		return f"{self.label} {','.join(map(str, args_l))} {self.MODIFIERS}" 
class exp_instr_caller(inst_caller_base): 
	def __init__(self, insturction_container) -> None:
     		super().__init__(insturction_container)
	def exp(self, gfx10_vsrc_1:reg_block, gfx10_vsrc_1:reg_block, gfx10_vsrc_1:reg_block, gfx10_vsrc_1:reg_block, MODIFIERS:str=''):
		return self.ic_pb(exp_base('exp', None, gfx10_vsrc_1, gfx10_vsrc_1, gfx10_vsrc_1, gfx10_vsrc_1, MODIFIERS))
class flat_base(inst_base): 
	def __init__(self, INSTRUCTION:str, DST:Union[reg_block,None], SRC0:Union[reg_block,None], SRC1:Union[reg_block,None], SRC2:Union[reg_block,None], MODIFIERS:str): 
		super().__init__(instruction_type.SMEM, INSTRUCTION)
		self.DST = DST 
		self.SRC0 = SRC0 
		self.SRC1 = SRC1 
		self.SRC2 = SRC2 
		self.MODIFIERS = MODIFIERS 
	def __str__(self): 
		args_l = filter(None.__ne__, [self.DST,self.SRC0,self.SRC1,self.SRC2]) 
		return f"{self.label} {','.join(map(str, args_l))} {self.MODIFIERS}" 
class flat_instr_caller(inst_caller_base): 
	def __init__(self, insturction_container) -> None:
     		super().__init__(insturction_container)
	def flat_atomic_add(self, gfx10_vaddr_1:reg_block, gfx10_vdata:reg_block, MODIFIERS:str=''):
		return self.ic_pb(flat_base('flat_atomic_add', None, gfx10_vaddr_1, gfx10_vdata, None, MODIFIERS))
	def flat_atomic_add_x2(self, gfx10_vaddr_1:reg_block, gfx10_vdata_1:reg_block, MODIFIERS:str=''):
		return self.ic_pb(flat_base('flat_atomic_add_x2', None, gfx10_vaddr_1, gfx10_vdata_1, None, MODIFIERS))
	def flat_atomic_and(self, gfx10_vaddr_1:reg_block, gfx10_vdata:reg_block, MODIFIERS:str=''):
		return self.ic_pb(flat_base('flat_atomic_and', None, gfx10_vaddr_1, gfx10_vdata, None, MODIFIERS))
	def flat_atomic_and_x2(self, gfx10_vaddr_1:reg_block, gfx10_vdata_1:reg_block, MODIFIERS:str=''):
		return self.ic_pb(flat_base('flat_atomic_and_x2', None, gfx10_vaddr_1, gfx10_vdata_1, None, MODIFIERS))
	def flat_atomic_cmpswap(self, gfx10_vaddr_1:reg_block, gfx10_vdata_1:reg_block, MODIFIERS:str=''):
		return self.ic_pb(flat_base('flat_atomic_cmpswap', None, gfx10_vaddr_1, gfx10_vdata_1, None, MODIFIERS))
	def flat_atomic_cmpswap_x2(self, gfx10_vaddr_1:reg_block, gfx10_vdata_2:reg_block, MODIFIERS:str=''):
		return self.ic_pb(flat_base('flat_atomic_cmpswap_x2', None, gfx10_vaddr_1, gfx10_vdata_2, None, MODIFIERS))
	def flat_atomic_dec(self, gfx10_vaddr_1:reg_block, gfx10_vdata:reg_block, MODIFIERS:str=''):
		return self.ic_pb(flat_base('flat_atomic_dec', None, gfx10_vaddr_1, gfx10_vdata, None, MODIFIERS))
	def flat_atomic_dec_x2(self, gfx10_vaddr_1:reg_block, gfx10_vdata_1:reg_block, MODIFIERS:str=''):
		return self.ic_pb(flat_base('flat_atomic_dec_x2', None, gfx10_vaddr_1, gfx10_vdata_1, None, MODIFIERS))
	def flat_atomic_fcmpswap(self, gfx10_vaddr_1:reg_block, gfx10_vdata_1:reg_block, MODIFIERS:str=''):
		return self.ic_pb(flat_base('flat_atomic_fcmpswap', None, gfx10_vaddr_1, gfx10_vdata_1, None, MODIFIERS))
	def flat_atomic_fcmpswap_x2(self, gfx10_vaddr_1:reg_block, gfx10_vdata_2:reg_block, MODIFIERS:str=''):
		return self.ic_pb(flat_base('flat_atomic_fcmpswap_x2', None, gfx10_vaddr_1, gfx10_vdata_2, None, MODIFIERS))
	def flat_atomic_fmax(self, gfx10_vaddr_1:reg_block, gfx10_vdata:reg_block, MODIFIERS:str=''):
		return self.ic_pb(flat_base('flat_atomic_fmax', None, gfx10_vaddr_1, gfx10_vdata, None, MODIFIERS))
	def flat_atomic_fmax_x2(self, gfx10_vaddr_1:reg_block, gfx10_vdata_1:reg_block, MODIFIERS:str=''):
		return self.ic_pb(flat_base('flat_atomic_fmax_x2', None, gfx10_vaddr_1, gfx10_vdata_1, None, MODIFIERS))
	def flat_atomic_fmin(self, gfx10_vaddr_1:reg_block, gfx10_vdata:reg_block, MODIFIERS:str=''):
		return self.ic_pb(flat_base('flat_atomic_fmin', None, gfx10_vaddr_1, gfx10_vdata, None, MODIFIERS))
	def flat_atomic_fmin_x2(self, gfx10_vaddr_1:reg_block, gfx10_vdata_1:reg_block, MODIFIERS:str=''):
		return self.ic_pb(flat_base('flat_atomic_fmin_x2', None, gfx10_vaddr_1, gfx10_vdata_1, None, MODIFIERS))
	def flat_atomic_inc(self, gfx10_vaddr_1:reg_block, gfx10_vdata:reg_block, MODIFIERS:str=''):
		return self.ic_pb(flat_base('flat_atomic_inc', None, gfx10_vaddr_1, gfx10_vdata, None, MODIFIERS))
	def flat_atomic_inc_x2(self, gfx10_vaddr_1:reg_block, gfx10_vdata_1:reg_block, MODIFIERS:str=''):
		return self.ic_pb(flat_base('flat_atomic_inc_x2', None, gfx10_vaddr_1, gfx10_vdata_1, None, MODIFIERS))
	def flat_atomic_or(self, gfx10_vaddr_1:reg_block, gfx10_vdata:reg_block, MODIFIERS:str=''):
		return self.ic_pb(flat_base('flat_atomic_or', None, gfx10_vaddr_1, gfx10_vdata, None, MODIFIERS))
	def flat_atomic_or_x2(self, gfx10_vaddr_1:reg_block, gfx10_vdata_1:reg_block, MODIFIERS:str=''):
		return self.ic_pb(flat_base('flat_atomic_or_x2', None, gfx10_vaddr_1, gfx10_vdata_1, None, MODIFIERS))
	def flat_atomic_smax(self, gfx10_vaddr_1:reg_block, gfx10_vdata:reg_block, MODIFIERS:str=''):
		return self.ic_pb(flat_base('flat_atomic_smax', None, gfx10_vaddr_1, gfx10_vdata, None, MODIFIERS))
	def flat_atomic_smax_x2(self, gfx10_vaddr_1:reg_block, gfx10_vdata_1:reg_block, MODIFIERS:str=''):
		return self.ic_pb(flat_base('flat_atomic_smax_x2', None, gfx10_vaddr_1, gfx10_vdata_1, None, MODIFIERS))
	def flat_atomic_smin(self, gfx10_vaddr_1:reg_block, gfx10_vdata:reg_block, MODIFIERS:str=''):
		return self.ic_pb(flat_base('flat_atomic_smin', None, gfx10_vaddr_1, gfx10_vdata, None, MODIFIERS))
	def flat_atomic_smin_x2(self, gfx10_vaddr_1:reg_block, gfx10_vdata_1:reg_block, MODIFIERS:str=''):
		return self.ic_pb(flat_base('flat_atomic_smin_x2', None, gfx10_vaddr_1, gfx10_vdata_1, None, MODIFIERS))
	def flat_atomic_sub(self, gfx10_vaddr_1:reg_block, gfx10_vdata:reg_block, MODIFIERS:str=''):
		return self.ic_pb(flat_base('flat_atomic_sub', None, gfx10_vaddr_1, gfx10_vdata, None, MODIFIERS))
	def flat_atomic_sub_x2(self, gfx10_vaddr_1:reg_block, gfx10_vdata_1:reg_block, MODIFIERS:str=''):
		return self.ic_pb(flat_base('flat_atomic_sub_x2', None, gfx10_vaddr_1, gfx10_vdata_1, None, MODIFIERS))
	def flat_atomic_swap(self, gfx10_vaddr_1:reg_block, gfx10_vdata:reg_block, MODIFIERS:str=''):
		return self.ic_pb(flat_base('flat_atomic_swap', None, gfx10_vaddr_1, gfx10_vdata, None, MODIFIERS))
	def flat_atomic_swap_x2(self, gfx10_vaddr_1:reg_block, gfx10_vdata_1:reg_block, MODIFIERS:str=''):
		return self.ic_pb(flat_base('flat_atomic_swap_x2', None, gfx10_vaddr_1, gfx10_vdata_1, None, MODIFIERS))
	def flat_atomic_umax(self, gfx10_vaddr_1:reg_block, gfx10_vdata:reg_block, MODIFIERS:str=''):
		return self.ic_pb(flat_base('flat_atomic_umax', None, gfx10_vaddr_1, gfx10_vdata, None, MODIFIERS))
	def flat_atomic_umax_x2(self, gfx10_vaddr_1:reg_block, gfx10_vdata_1:reg_block, MODIFIERS:str=''):
		return self.ic_pb(flat_base('flat_atomic_umax_x2', None, gfx10_vaddr_1, gfx10_vdata_1, None, MODIFIERS))
	def flat_atomic_umin(self, gfx10_vaddr_1:reg_block, gfx10_vdata:reg_block, MODIFIERS:str=''):
		return self.ic_pb(flat_base('flat_atomic_umin', None, gfx10_vaddr_1, gfx10_vdata, None, MODIFIERS))
	def flat_atomic_umin_x2(self, gfx10_vaddr_1:reg_block, gfx10_vdata_1:reg_block, MODIFIERS:str=''):
		return self.ic_pb(flat_base('flat_atomic_umin_x2', None, gfx10_vaddr_1, gfx10_vdata_1, None, MODIFIERS))
	def flat_atomic_xor(self, gfx10_vaddr_1:reg_block, gfx10_vdata:reg_block, MODIFIERS:str=''):
		return self.ic_pb(flat_base('flat_atomic_xor', None, gfx10_vaddr_1, gfx10_vdata, None, MODIFIERS))
	def flat_atomic_xor_x2(self, gfx10_vaddr_1:reg_block, gfx10_vdata_1:reg_block, MODIFIERS:str=''):
		return self.ic_pb(flat_base('flat_atomic_xor_x2', None, gfx10_vaddr_1, gfx10_vdata_1, None, MODIFIERS))
	def flat_load_dword(self, gfx10_vaddr_1:reg_block, MODIFIERS:str=''):
		return self.ic_pb(flat_base('flat_load_dword', None, gfx10_vaddr_1, None, None, MODIFIERS))
	def flat_load_dwordx2(self, gfx10_vaddr_1:reg_block, MODIFIERS:str=''):
		return self.ic_pb(flat_base('flat_load_dwordx2', None, gfx10_vaddr_1, None, None, MODIFIERS))
	def flat_load_dwordx3(self, gfx10_vaddr_1:reg_block, MODIFIERS:str=''):
		return self.ic_pb(flat_base('flat_load_dwordx3', None, gfx10_vaddr_1, None, None, MODIFIERS))
	def flat_load_dwordx4(self, gfx10_vaddr_1:reg_block, MODIFIERS:str=''):
		return self.ic_pb(flat_base('flat_load_dwordx4', None, gfx10_vaddr_1, None, None, MODIFIERS))
	def flat_load_sbyte(self, gfx10_vaddr_1:reg_block, MODIFIERS:str=''):
		return self.ic_pb(flat_base('flat_load_sbyte', None, gfx10_vaddr_1, None, None, MODIFIERS))
	def flat_load_sbyte_d16(self, gfx10_vaddr_1:reg_block, MODIFIERS:str=''):
		return self.ic_pb(flat_base('flat_load_sbyte_d16', None, gfx10_vaddr_1, None, None, MODIFIERS))
	def flat_load_sbyte_d16_hi(self, gfx10_vaddr_1:reg_block, MODIFIERS:str=''):
		return self.ic_pb(flat_base('flat_load_sbyte_d16_hi', None, gfx10_vaddr_1, None, None, MODIFIERS))
	def flat_load_short_d16(self, gfx10_vaddr_1:reg_block, MODIFIERS:str=''):
		return self.ic_pb(flat_base('flat_load_short_d16', None, gfx10_vaddr_1, None, None, MODIFIERS))
	def flat_load_short_d16_hi(self, gfx10_vaddr_1:reg_block, MODIFIERS:str=''):
		return self.ic_pb(flat_base('flat_load_short_d16_hi', None, gfx10_vaddr_1, None, None, MODIFIERS))
	def flat_load_sshort(self, gfx10_vaddr_1:reg_block, MODIFIERS:str=''):
		return self.ic_pb(flat_base('flat_load_sshort', None, gfx10_vaddr_1, None, None, MODIFIERS))
	def flat_load_ubyte(self, gfx10_vaddr_1:reg_block, MODIFIERS:str=''):
		return self.ic_pb(flat_base('flat_load_ubyte', None, gfx10_vaddr_1, None, None, MODIFIERS))
	def flat_load_ubyte_d16(self, gfx10_vaddr_1:reg_block, MODIFIERS:str=''):
		return self.ic_pb(flat_base('flat_load_ubyte_d16', None, gfx10_vaddr_1, None, None, MODIFIERS))
	def flat_load_ubyte_d16_hi(self, gfx10_vaddr_1:reg_block, MODIFIERS:str=''):
		return self.ic_pb(flat_base('flat_load_ubyte_d16_hi', None, gfx10_vaddr_1, None, None, MODIFIERS))
	def flat_load_ushort(self, gfx10_vaddr_1:reg_block, MODIFIERS:str=''):
		return self.ic_pb(flat_base('flat_load_ushort', None, gfx10_vaddr_1, None, None, MODIFIERS))
	def flat_store_byte(self, gfx10_vdata:reg_block, MODIFIERS:str=''):
		return self.ic_pb(flat_base('flat_store_byte', None, None, gfx10_vdata, None, MODIFIERS))
	def flat_store_byte_d16_hi(self, gfx10_vdata:reg_block, MODIFIERS:str=''):
		return self.ic_pb(flat_base('flat_store_byte_d16_hi', None, None, gfx10_vdata, None, MODIFIERS))
	def flat_store_dword(self, gfx10_vdata:reg_block, MODIFIERS:str=''):
		return self.ic_pb(flat_base('flat_store_dword', None, None, gfx10_vdata, None, MODIFIERS))
	def flat_store_dwordx2(self, gfx10_vdata_1:reg_block, MODIFIERS:str=''):
		return self.ic_pb(flat_base('flat_store_dwordx2', None, None, gfx10_vdata_1, None, MODIFIERS))
	def flat_store_dwordx3(self, gfx10_vdata_3:reg_block, MODIFIERS:str=''):
		return self.ic_pb(flat_base('flat_store_dwordx3', None, None, gfx10_vdata_3, None, MODIFIERS))
	def flat_store_dwordx4(self, gfx10_vdata_2:reg_block, MODIFIERS:str=''):
		return self.ic_pb(flat_base('flat_store_dwordx4', None, None, gfx10_vdata_2, None, MODIFIERS))
	def flat_store_short(self, gfx10_vdata:reg_block, MODIFIERS:str=''):
		return self.ic_pb(flat_base('flat_store_short', None, None, gfx10_vdata, None, MODIFIERS))
	def flat_store_short_d16_hi(self, gfx10_vdata:reg_block, MODIFIERS:str=''):
		return self.ic_pb(flat_base('flat_store_short_d16_hi', None, None, gfx10_vdata, None, MODIFIERS))
	def global_atomic_add(self, gfx10_vaddr_2:reg_block, gfx10_vdata:reg_block, gfx10_saddr:reg_block, MODIFIERS:str=''):
		return self.ic_pb(flat_base('global_atomic_add', None, gfx10_vaddr_2, gfx10_vdata, gfx10_saddr, MODIFIERS))
	def global_atomic_add_x2(self, gfx10_vaddr_2:reg_block, gfx10_vdata_1:reg_block, gfx10_saddr:reg_block, MODIFIERS:str=''):
		return self.ic_pb(flat_base('global_atomic_add_x2', None, gfx10_vaddr_2, gfx10_vdata_1, gfx10_saddr, MODIFIERS))
	def global_atomic_and(self, gfx10_vaddr_2:reg_block, gfx10_vdata:reg_block, gfx10_saddr:reg_block, MODIFIERS:str=''):
		return self.ic_pb(flat_base('global_atomic_and', None, gfx10_vaddr_2, gfx10_vdata, gfx10_saddr, MODIFIERS))
	def global_atomic_and_x2(self, gfx10_vaddr_2:reg_block, gfx10_vdata_1:reg_block, gfx10_saddr:reg_block, MODIFIERS:str=''):
		return self.ic_pb(flat_base('global_atomic_and_x2', None, gfx10_vaddr_2, gfx10_vdata_1, gfx10_saddr, MODIFIERS))
	def global_atomic_cmpswap(self, gfx10_vaddr_2:reg_block, gfx10_vdata_1:reg_block, gfx10_saddr:reg_block, MODIFIERS:str=''):
		return self.ic_pb(flat_base('global_atomic_cmpswap', None, gfx10_vaddr_2, gfx10_vdata_1, gfx10_saddr, MODIFIERS))
	def global_atomic_cmpswap_x2(self, gfx10_vaddr_2:reg_block, gfx10_vdata_2:reg_block, gfx10_saddr:reg_block, MODIFIERS:str=''):
		return self.ic_pb(flat_base('global_atomic_cmpswap_x2', None, gfx10_vaddr_2, gfx10_vdata_2, gfx10_saddr, MODIFIERS))
	def global_atomic_dec(self, gfx10_vaddr_2:reg_block, gfx10_vdata:reg_block, gfx10_saddr:reg_block, MODIFIERS:str=''):
		return self.ic_pb(flat_base('global_atomic_dec', None, gfx10_vaddr_2, gfx10_vdata, gfx10_saddr, MODIFIERS))
	def global_atomic_dec_x2(self, gfx10_vaddr_2:reg_block, gfx10_vdata_1:reg_block, gfx10_saddr:reg_block, MODIFIERS:str=''):
		return self.ic_pb(flat_base('global_atomic_dec_x2', None, gfx10_vaddr_2, gfx10_vdata_1, gfx10_saddr, MODIFIERS))
	def global_atomic_fmax(self, gfx10_vaddr_2:reg_block, gfx10_vdata:reg_block, gfx10_saddr:reg_block, MODIFIERS:str=''):
		return self.ic_pb(flat_base('global_atomic_fmax', None, gfx10_vaddr_2, gfx10_vdata, gfx10_saddr, MODIFIERS))
	def global_atomic_fmax_x2(self, gfx10_vaddr_2:reg_block, gfx10_vdata_1:reg_block, gfx10_saddr:reg_block, MODIFIERS:str=''):
		return self.ic_pb(flat_base('global_atomic_fmax_x2', None, gfx10_vaddr_2, gfx10_vdata_1, gfx10_saddr, MODIFIERS))
	def global_atomic_fmin(self, gfx10_vaddr_2:reg_block, gfx10_vdata:reg_block, gfx10_saddr:reg_block, MODIFIERS:str=''):
		return self.ic_pb(flat_base('global_atomic_fmin', None, gfx10_vaddr_2, gfx10_vdata, gfx10_saddr, MODIFIERS))
	def global_atomic_fmin_x2(self, gfx10_vaddr_2:reg_block, gfx10_vdata_1:reg_block, gfx10_saddr:reg_block, MODIFIERS:str=''):
		return self.ic_pb(flat_base('global_atomic_fmin_x2', None, gfx10_vaddr_2, gfx10_vdata_1, gfx10_saddr, MODIFIERS))
	def global_atomic_inc(self, gfx10_vaddr_2:reg_block, gfx10_vdata:reg_block, gfx10_saddr:reg_block, MODIFIERS:str=''):
		return self.ic_pb(flat_base('global_atomic_inc', None, gfx10_vaddr_2, gfx10_vdata, gfx10_saddr, MODIFIERS))
	def global_atomic_inc_x2(self, gfx10_vaddr_2:reg_block, gfx10_vdata_1:reg_block, gfx10_saddr:reg_block, MODIFIERS:str=''):
		return self.ic_pb(flat_base('global_atomic_inc_x2', None, gfx10_vaddr_2, gfx10_vdata_1, gfx10_saddr, MODIFIERS))
	def global_atomic_or(self, gfx10_vaddr_2:reg_block, gfx10_vdata:reg_block, gfx10_saddr:reg_block, MODIFIERS:str=''):
		return self.ic_pb(flat_base('global_atomic_or', None, gfx10_vaddr_2, gfx10_vdata, gfx10_saddr, MODIFIERS))
	def global_atomic_or_x2(self, gfx10_vaddr_2:reg_block, gfx10_vdata_1:reg_block, gfx10_saddr:reg_block, MODIFIERS:str=''):
		return self.ic_pb(flat_base('global_atomic_or_x2', None, gfx10_vaddr_2, gfx10_vdata_1, gfx10_saddr, MODIFIERS))
	def global_atomic_smax(self, gfx10_vaddr_2:reg_block, gfx10_vdata:reg_block, gfx10_saddr:reg_block, MODIFIERS:str=''):
		return self.ic_pb(flat_base('global_atomic_smax', None, gfx10_vaddr_2, gfx10_vdata, gfx10_saddr, MODIFIERS))
	def global_atomic_smax_x2(self, gfx10_vaddr_2:reg_block, gfx10_vdata_1:reg_block, gfx10_saddr:reg_block, MODIFIERS:str=''):
		return self.ic_pb(flat_base('global_atomic_smax_x2', None, gfx10_vaddr_2, gfx10_vdata_1, gfx10_saddr, MODIFIERS))
	def global_atomic_smin(self, gfx10_vaddr_2:reg_block, gfx10_vdata:reg_block, gfx10_saddr:reg_block, MODIFIERS:str=''):
		return self.ic_pb(flat_base('global_atomic_smin', None, gfx10_vaddr_2, gfx10_vdata, gfx10_saddr, MODIFIERS))
	def global_atomic_smin_x2(self, gfx10_vaddr_2:reg_block, gfx10_vdata_1:reg_block, gfx10_saddr:reg_block, MODIFIERS:str=''):
		return self.ic_pb(flat_base('global_atomic_smin_x2', None, gfx10_vaddr_2, gfx10_vdata_1, gfx10_saddr, MODIFIERS))
	def global_atomic_sub(self, gfx10_vaddr_2:reg_block, gfx10_vdata:reg_block, gfx10_saddr:reg_block, MODIFIERS:str=''):
		return self.ic_pb(flat_base('global_atomic_sub', None, gfx10_vaddr_2, gfx10_vdata, gfx10_saddr, MODIFIERS))
	def global_atomic_sub_x2(self, gfx10_vaddr_2:reg_block, gfx10_vdata_1:reg_block, gfx10_saddr:reg_block, MODIFIERS:str=''):
		return self.ic_pb(flat_base('global_atomic_sub_x2', None, gfx10_vaddr_2, gfx10_vdata_1, gfx10_saddr, MODIFIERS))
	def global_atomic_swap(self, gfx10_vaddr_2:reg_block, gfx10_vdata:reg_block, gfx10_saddr:reg_block, MODIFIERS:str=''):
		return self.ic_pb(flat_base('global_atomic_swap', None, gfx10_vaddr_2, gfx10_vdata, gfx10_saddr, MODIFIERS))
	def global_atomic_swap_x2(self, gfx10_vaddr_2:reg_block, gfx10_vdata_1:reg_block, gfx10_saddr:reg_block, MODIFIERS:str=''):
		return self.ic_pb(flat_base('global_atomic_swap_x2', None, gfx10_vaddr_2, gfx10_vdata_1, gfx10_saddr, MODIFIERS))
	def global_atomic_umax(self, gfx10_vaddr_2:reg_block, gfx10_vdata:reg_block, gfx10_saddr:reg_block, MODIFIERS:str=''):
		return self.ic_pb(flat_base('global_atomic_umax', None, gfx10_vaddr_2, gfx10_vdata, gfx10_saddr, MODIFIERS))
	def global_atomic_umax_x2(self, gfx10_vaddr_2:reg_block, gfx10_vdata_1:reg_block, gfx10_saddr:reg_block, MODIFIERS:str=''):
		return self.ic_pb(flat_base('global_atomic_umax_x2', None, gfx10_vaddr_2, gfx10_vdata_1, gfx10_saddr, MODIFIERS))
	def global_atomic_umin(self, gfx10_vaddr_2:reg_block, gfx10_vdata:reg_block, gfx10_saddr:reg_block, MODIFIERS:str=''):
		return self.ic_pb(flat_base('global_atomic_umin', None, gfx10_vaddr_2, gfx10_vdata, gfx10_saddr, MODIFIERS))
	def global_atomic_umin_x2(self, gfx10_vaddr_2:reg_block, gfx10_vdata_1:reg_block, gfx10_saddr:reg_block, MODIFIERS:str=''):
		return self.ic_pb(flat_base('global_atomic_umin_x2', None, gfx10_vaddr_2, gfx10_vdata_1, gfx10_saddr, MODIFIERS))
	def global_atomic_xor(self, gfx10_vaddr_2:reg_block, gfx10_vdata:reg_block, gfx10_saddr:reg_block, MODIFIERS:str=''):
		return self.ic_pb(flat_base('global_atomic_xor', None, gfx10_vaddr_2, gfx10_vdata, gfx10_saddr, MODIFIERS))
	def global_atomic_xor_x2(self, gfx10_vaddr_2:reg_block, gfx10_vdata_1:reg_block, gfx10_saddr:reg_block, MODIFIERS:str=''):
		return self.ic_pb(flat_base('global_atomic_xor_x2', None, gfx10_vaddr_2, gfx10_vdata_1, gfx10_saddr, MODIFIERS))
	def global_load_dword(self, gfx10_vaddr_2:reg_block, gfx10_saddr:reg_block, MODIFIERS:str=''):
		return self.ic_pb(flat_base('global_load_dword', None, gfx10_vaddr_2, gfx10_saddr, None, MODIFIERS))
	def global_load_dwordx2(self, gfx10_vaddr_2:reg_block, gfx10_saddr:reg_block, MODIFIERS:str=''):
		return self.ic_pb(flat_base('global_load_dwordx2', None, gfx10_vaddr_2, gfx10_saddr, None, MODIFIERS))
	def global_load_dwordx3(self, gfx10_vaddr_2:reg_block, gfx10_saddr:reg_block, MODIFIERS:str=''):
		return self.ic_pb(flat_base('global_load_dwordx3', None, gfx10_vaddr_2, gfx10_saddr, None, MODIFIERS))
	def global_load_dwordx4(self, gfx10_vaddr_2:reg_block, gfx10_saddr:reg_block, MODIFIERS:str=''):
		return self.ic_pb(flat_base('global_load_dwordx4', None, gfx10_vaddr_2, gfx10_saddr, None, MODIFIERS))
	def global_load_sbyte(self, gfx10_vaddr_2:reg_block, gfx10_saddr:reg_block, MODIFIERS:str=''):
		return self.ic_pb(flat_base('global_load_sbyte', None, gfx10_vaddr_2, gfx10_saddr, None, MODIFIERS))
	def global_load_sbyte_d16(self, gfx10_vaddr_2:reg_block, gfx10_saddr:reg_block, MODIFIERS:str=''):
		return self.ic_pb(flat_base('global_load_sbyte_d16', None, gfx10_vaddr_2, gfx10_saddr, None, MODIFIERS))
	def global_load_sbyte_d16_hi(self, gfx10_vaddr_2:reg_block, gfx10_saddr:reg_block, MODIFIERS:str=''):
		return self.ic_pb(flat_base('global_load_sbyte_d16_hi', None, gfx10_vaddr_2, gfx10_saddr, None, MODIFIERS))
	def global_load_short_d16(self, gfx10_vaddr_2:reg_block, gfx10_saddr:reg_block, MODIFIERS:str=''):
		return self.ic_pb(flat_base('global_load_short_d16', None, gfx10_vaddr_2, gfx10_saddr, None, MODIFIERS))
	def global_load_short_d16_hi(self, gfx10_vaddr_2:reg_block, gfx10_saddr:reg_block, MODIFIERS:str=''):
		return self.ic_pb(flat_base('global_load_short_d16_hi', None, gfx10_vaddr_2, gfx10_saddr, None, MODIFIERS))
	def global_load_sshort(self, gfx10_vaddr_2:reg_block, gfx10_saddr:reg_block, MODIFIERS:str=''):
		return self.ic_pb(flat_base('global_load_sshort', None, gfx10_vaddr_2, gfx10_saddr, None, MODIFIERS))
	def global_load_ubyte(self, gfx10_vaddr_2:reg_block, gfx10_saddr:reg_block, MODIFIERS:str=''):
		return self.ic_pb(flat_base('global_load_ubyte', None, gfx10_vaddr_2, gfx10_saddr, None, MODIFIERS))
	def global_load_ubyte_d16(self, gfx10_vaddr_2:reg_block, gfx10_saddr:reg_block, MODIFIERS:str=''):
		return self.ic_pb(flat_base('global_load_ubyte_d16', None, gfx10_vaddr_2, gfx10_saddr, None, MODIFIERS))
	def global_load_ubyte_d16_hi(self, gfx10_vaddr_2:reg_block, gfx10_saddr:reg_block, MODIFIERS:str=''):
		return self.ic_pb(flat_base('global_load_ubyte_d16_hi', None, gfx10_vaddr_2, gfx10_saddr, None, MODIFIERS))
	def global_load_ushort(self, gfx10_vaddr_2:reg_block, gfx10_saddr:reg_block, MODIFIERS:str=''):
		return self.ic_pb(flat_base('global_load_ushort', None, gfx10_vaddr_2, gfx10_saddr, None, MODIFIERS))
	def global_store_byte(self, gfx10_vdata:reg_block, gfx10_saddr:reg_block, MODIFIERS:str=''):
		return self.ic_pb(flat_base('global_store_byte', None, None, gfx10_vdata, gfx10_saddr, MODIFIERS))
	def global_store_byte_d16_hi(self, gfx10_vdata:reg_block, gfx10_saddr:reg_block, MODIFIERS:str=''):
		return self.ic_pb(flat_base('global_store_byte_d16_hi', None, None, gfx10_vdata, gfx10_saddr, MODIFIERS))
	def global_store_dword(self, gfx10_vdata:reg_block, gfx10_saddr:reg_block, MODIFIERS:str=''):
		return self.ic_pb(flat_base('global_store_dword', None, None, gfx10_vdata, gfx10_saddr, MODIFIERS))
	def global_store_dwordx2(self, gfx10_vdata_1:reg_block, gfx10_saddr:reg_block, MODIFIERS:str=''):
		return self.ic_pb(flat_base('global_store_dwordx2', None, None, gfx10_vdata_1, gfx10_saddr, MODIFIERS))
	def global_store_dwordx3(self, gfx10_vdata_3:reg_block, gfx10_saddr:reg_block, MODIFIERS:str=''):
		return self.ic_pb(flat_base('global_store_dwordx3', None, None, gfx10_vdata_3, gfx10_saddr, MODIFIERS))
	def global_store_dwordx4(self, gfx10_vdata_2:reg_block, gfx10_saddr:reg_block, MODIFIERS:str=''):
		return self.ic_pb(flat_base('global_store_dwordx4', None, None, gfx10_vdata_2, gfx10_saddr, MODIFIERS))
	def global_store_short(self, gfx10_vdata:reg_block, gfx10_saddr:reg_block, MODIFIERS:str=''):
		return self.ic_pb(flat_base('global_store_short', None, None, gfx10_vdata, gfx10_saddr, MODIFIERS))
	def global_store_short_d16_hi(self, gfx10_vdata:reg_block, gfx10_saddr:reg_block, MODIFIERS:str=''):
		return self.ic_pb(flat_base('global_store_short_d16_hi', None, None, gfx10_vdata, gfx10_saddr, MODIFIERS))
	def scratch_load_dword(self, gfx10_vaddr_3:reg_block, gfx10_saddr_1:reg_block, MODIFIERS:str=''):
		return self.ic_pb(flat_base('scratch_load_dword', None, gfx10_vaddr_3, gfx10_saddr_1, None, MODIFIERS))
	def scratch_load_dwordx2(self, gfx10_vaddr_3:reg_block, gfx10_saddr_1:reg_block, MODIFIERS:str=''):
		return self.ic_pb(flat_base('scratch_load_dwordx2', None, gfx10_vaddr_3, gfx10_saddr_1, None, MODIFIERS))
	def scratch_load_dwordx3(self, gfx10_vaddr_3:reg_block, gfx10_saddr_1:reg_block, MODIFIERS:str=''):
		return self.ic_pb(flat_base('scratch_load_dwordx3', None, gfx10_vaddr_3, gfx10_saddr_1, None, MODIFIERS))
	def scratch_load_dwordx4(self, gfx10_vaddr_3:reg_block, gfx10_saddr_1:reg_block, MODIFIERS:str=''):
		return self.ic_pb(flat_base('scratch_load_dwordx4', None, gfx10_vaddr_3, gfx10_saddr_1, None, MODIFIERS))
	def scratch_load_sbyte(self, gfx10_vaddr_3:reg_block, gfx10_saddr_1:reg_block, MODIFIERS:str=''):
		return self.ic_pb(flat_base('scratch_load_sbyte', None, gfx10_vaddr_3, gfx10_saddr_1, None, MODIFIERS))
	def scratch_load_sbyte_d16(self, gfx10_vaddr_3:reg_block, gfx10_saddr_1:reg_block, MODIFIERS:str=''):
		return self.ic_pb(flat_base('scratch_load_sbyte_d16', None, gfx10_vaddr_3, gfx10_saddr_1, None, MODIFIERS))
	def scratch_load_sbyte_d16_hi(self, gfx10_vaddr_3:reg_block, gfx10_saddr_1:reg_block, MODIFIERS:str=''):
		return self.ic_pb(flat_base('scratch_load_sbyte_d16_hi', None, gfx10_vaddr_3, gfx10_saddr_1, None, MODIFIERS))
	def scratch_load_short_d16(self, gfx10_vaddr_3:reg_block, gfx10_saddr_1:reg_block, MODIFIERS:str=''):
		return self.ic_pb(flat_base('scratch_load_short_d16', None, gfx10_vaddr_3, gfx10_saddr_1, None, MODIFIERS))
	def scratch_load_short_d16_hi(self, gfx10_vaddr_3:reg_block, gfx10_saddr_1:reg_block, MODIFIERS:str=''):
		return self.ic_pb(flat_base('scratch_load_short_d16_hi', None, gfx10_vaddr_3, gfx10_saddr_1, None, MODIFIERS))
	def scratch_load_sshort(self, gfx10_vaddr_3:reg_block, gfx10_saddr_1:reg_block, MODIFIERS:str=''):
		return self.ic_pb(flat_base('scratch_load_sshort', None, gfx10_vaddr_3, gfx10_saddr_1, None, MODIFIERS))
	def scratch_load_ubyte(self, gfx10_vaddr_3:reg_block, gfx10_saddr_1:reg_block, MODIFIERS:str=''):
		return self.ic_pb(flat_base('scratch_load_ubyte', None, gfx10_vaddr_3, gfx10_saddr_1, None, MODIFIERS))
	def scratch_load_ubyte_d16(self, gfx10_vaddr_3:reg_block, gfx10_saddr_1:reg_block, MODIFIERS:str=''):
		return self.ic_pb(flat_base('scratch_load_ubyte_d16', None, gfx10_vaddr_3, gfx10_saddr_1, None, MODIFIERS))
	def scratch_load_ubyte_d16_hi(self, gfx10_vaddr_3:reg_block, gfx10_saddr_1:reg_block, MODIFIERS:str=''):
		return self.ic_pb(flat_base('scratch_load_ubyte_d16_hi', None, gfx10_vaddr_3, gfx10_saddr_1, None, MODIFIERS))
	def scratch_load_ushort(self, gfx10_vaddr_3:reg_block, gfx10_saddr_1:reg_block, MODIFIERS:str=''):
		return self.ic_pb(flat_base('scratch_load_ushort', None, gfx10_vaddr_3, gfx10_saddr_1, None, MODIFIERS))
	def scratch_store_byte(self, gfx10_vdata:reg_block, gfx10_saddr_1:reg_block, MODIFIERS:str=''):
		return self.ic_pb(flat_base('scratch_store_byte', None, None, gfx10_vdata, gfx10_saddr_1, MODIFIERS))
	def scratch_store_byte_d16_hi(self, gfx10_vdata:reg_block, gfx10_saddr_1:reg_block, MODIFIERS:str=''):
		return self.ic_pb(flat_base('scratch_store_byte_d16_hi', None, None, gfx10_vdata, gfx10_saddr_1, MODIFIERS))
	def scratch_store_dword(self, gfx10_vdata:reg_block, gfx10_saddr_1:reg_block, MODIFIERS:str=''):
		return self.ic_pb(flat_base('scratch_store_dword', None, None, gfx10_vdata, gfx10_saddr_1, MODIFIERS))
	def scratch_store_dwordx2(self, gfx10_vdata_1:reg_block, gfx10_saddr_1:reg_block, MODIFIERS:str=''):
		return self.ic_pb(flat_base('scratch_store_dwordx2', None, None, gfx10_vdata_1, gfx10_saddr_1, MODIFIERS))
	def scratch_store_dwordx3(self, gfx10_vdata_3:reg_block, gfx10_saddr_1:reg_block, MODIFIERS:str=''):
		return self.ic_pb(flat_base('scratch_store_dwordx3', None, None, gfx10_vdata_3, gfx10_saddr_1, MODIFIERS))
	def scratch_store_dwordx4(self, gfx10_vdata_2:reg_block, gfx10_saddr_1:reg_block, MODIFIERS:str=''):
		return self.ic_pb(flat_base('scratch_store_dwordx4', None, None, gfx10_vdata_2, gfx10_saddr_1, MODIFIERS))
	def scratch_store_short(self, gfx10_vdata:reg_block, gfx10_saddr_1:reg_block, MODIFIERS:str=''):
		return self.ic_pb(flat_base('scratch_store_short', None, None, gfx10_vdata, gfx10_saddr_1, MODIFIERS))
	def scratch_store_short_d16_hi(self, gfx10_vdata:reg_block, gfx10_saddr_1:reg_block, MODIFIERS:str=''):
		return self.ic_pb(flat_base('scratch_store_short_d16_hi', None, None, gfx10_vdata, gfx10_saddr_1, MODIFIERS))
class mimg_base(inst_base): 
	def __init__(self, INSTRUCTION:str, DST:Union[reg_block,None], SRC0:Union[reg_block,None], SRC1:Union[reg_block,None], SRC2:Union[reg_block,None], MODIFIERS:str): 
		super().__init__(instruction_type.SMEM, INSTRUCTION)
		self.DST = DST 
		self.SRC0 = SRC0 
		self.SRC1 = SRC1 
		self.SRC2 = SRC2 
		self.MODIFIERS = MODIFIERS 
	def __str__(self): 
		args_l = filter(None.__ne__, [self.DST,self.SRC0,self.SRC1,self.SRC2]) 
		return f"{self.label} {','.join(map(str, args_l))} {self.MODIFIERS}" 
class mimg_instr_caller(inst_caller_base): 
	def __init__(self, insturction_container) -> None:
     		super().__init__(insturction_container)
	def image_atomic_add(self, gfx10_vaddr_4:reg_block, gfx10_srsrc:reg_block, MODIFIERS:str=''):
		return self.ic_pb(mimg_base('image_atomic_add', None, None, gfx10_vaddr_4, gfx10_srsrc, MODIFIERS))
	def image_atomic_and(self, gfx10_vaddr_4:reg_block, gfx10_srsrc:reg_block, MODIFIERS:str=''):
		return self.ic_pb(mimg_base('image_atomic_and', None, None, gfx10_vaddr_4, gfx10_srsrc, MODIFIERS))
	def image_atomic_cmpswap(self, gfx10_vaddr_4:reg_block, gfx10_srsrc:reg_block, MODIFIERS:str=''):
		return self.ic_pb(mimg_base('image_atomic_cmpswap', None, None, gfx10_vaddr_4, gfx10_srsrc, MODIFIERS))
	def image_atomic_dec(self, gfx10_vaddr_4:reg_block, gfx10_srsrc:reg_block, MODIFIERS:str=''):
		return self.ic_pb(mimg_base('image_atomic_dec', None, None, gfx10_vaddr_4, gfx10_srsrc, MODIFIERS))
	def image_atomic_fcmpswap(self, gfx10_vaddr_4:reg_block, gfx10_srsrc:reg_block, MODIFIERS:str=''):
		return self.ic_pb(mimg_base('image_atomic_fcmpswap', None, None, gfx10_vaddr_4, gfx10_srsrc, MODIFIERS))
	def image_atomic_fmax(self, gfx10_vaddr_4:reg_block, gfx10_srsrc:reg_block, MODIFIERS:str=''):
		return self.ic_pb(mimg_base('image_atomic_fmax', None, None, gfx10_vaddr_4, gfx10_srsrc, MODIFIERS))
	def image_atomic_fmin(self, gfx10_vaddr_4:reg_block, gfx10_srsrc:reg_block, MODIFIERS:str=''):
		return self.ic_pb(mimg_base('image_atomic_fmin', None, None, gfx10_vaddr_4, gfx10_srsrc, MODIFIERS))
	def image_atomic_inc(self, gfx10_vaddr_4:reg_block, gfx10_srsrc:reg_block, MODIFIERS:str=''):
		return self.ic_pb(mimg_base('image_atomic_inc', None, None, gfx10_vaddr_4, gfx10_srsrc, MODIFIERS))
	def image_atomic_or(self, gfx10_vaddr_4:reg_block, gfx10_srsrc:reg_block, MODIFIERS:str=''):
		return self.ic_pb(mimg_base('image_atomic_or', None, None, gfx10_vaddr_4, gfx10_srsrc, MODIFIERS))
	def image_atomic_smax(self, gfx10_vaddr_4:reg_block, gfx10_srsrc:reg_block, MODIFIERS:str=''):
		return self.ic_pb(mimg_base('image_atomic_smax', None, None, gfx10_vaddr_4, gfx10_srsrc, MODIFIERS))
	def image_atomic_smin(self, gfx10_vaddr_4:reg_block, gfx10_srsrc:reg_block, MODIFIERS:str=''):
		return self.ic_pb(mimg_base('image_atomic_smin', None, None, gfx10_vaddr_4, gfx10_srsrc, MODIFIERS))
	def image_atomic_sub(self, gfx10_vaddr_4:reg_block, gfx10_srsrc:reg_block, MODIFIERS:str=''):
		return self.ic_pb(mimg_base('image_atomic_sub', None, None, gfx10_vaddr_4, gfx10_srsrc, MODIFIERS))
	def image_atomic_swap(self, gfx10_vaddr_4:reg_block, gfx10_srsrc:reg_block, MODIFIERS:str=''):
		return self.ic_pb(mimg_base('image_atomic_swap', None, None, gfx10_vaddr_4, gfx10_srsrc, MODIFIERS))
	def image_atomic_umax(self, gfx10_vaddr_4:reg_block, gfx10_srsrc:reg_block, MODIFIERS:str=''):
		return self.ic_pb(mimg_base('image_atomic_umax', None, None, gfx10_vaddr_4, gfx10_srsrc, MODIFIERS))
	def image_atomic_umin(self, gfx10_vaddr_4:reg_block, gfx10_srsrc:reg_block, MODIFIERS:str=''):
		return self.ic_pb(mimg_base('image_atomic_umin', None, None, gfx10_vaddr_4, gfx10_srsrc, MODIFIERS))
	def image_atomic_xor(self, gfx10_vaddr_4:reg_block, gfx10_srsrc:reg_block, MODIFIERS:str=''):
		return self.ic_pb(mimg_base('image_atomic_xor', None, None, gfx10_vaddr_4, gfx10_srsrc, MODIFIERS))
	def image_gather4(self, gfx10_vaddr_4:reg_block, gfx10_srsrc:reg_block, gfx10_ssamp:reg_block, MODIFIERS:str=''):
		return self.ic_pb(mimg_base('image_gather4', None, gfx10_vaddr_4, gfx10_srsrc, gfx10_ssamp, MODIFIERS))
	def image_gather4_b(self, gfx10_vaddr_4:reg_block, gfx10_srsrc:reg_block, gfx10_ssamp:reg_block, MODIFIERS:str=''):
		return self.ic_pb(mimg_base('image_gather4_b', None, gfx10_vaddr_4, gfx10_srsrc, gfx10_ssamp, MODIFIERS))
	def image_gather4_b_cl(self, gfx10_vaddr_4:reg_block, gfx10_srsrc:reg_block, gfx10_ssamp:reg_block, MODIFIERS:str=''):
		return self.ic_pb(mimg_base('image_gather4_b_cl', None, gfx10_vaddr_4, gfx10_srsrc, gfx10_ssamp, MODIFIERS))
	def image_gather4_b_cl_o(self, gfx10_vaddr_4:reg_block, gfx10_srsrc:reg_block, gfx10_ssamp:reg_block, MODIFIERS:str=''):
		return self.ic_pb(mimg_base('image_gather4_b_cl_o', None, gfx10_vaddr_4, gfx10_srsrc, gfx10_ssamp, MODIFIERS))
	def image_gather4_b_o(self, gfx10_vaddr_4:reg_block, gfx10_srsrc:reg_block, gfx10_ssamp:reg_block, MODIFIERS:str=''):
		return self.ic_pb(mimg_base('image_gather4_b_o', None, gfx10_vaddr_4, gfx10_srsrc, gfx10_ssamp, MODIFIERS))
	def image_gather4_c(self, gfx10_vaddr_4:reg_block, gfx10_srsrc:reg_block, gfx10_ssamp:reg_block, MODIFIERS:str=''):
		return self.ic_pb(mimg_base('image_gather4_c', None, gfx10_vaddr_4, gfx10_srsrc, gfx10_ssamp, MODIFIERS))
	def image_gather4_c_b(self, gfx10_vaddr_4:reg_block, gfx10_srsrc:reg_block, gfx10_ssamp:reg_block, MODIFIERS:str=''):
		return self.ic_pb(mimg_base('image_gather4_c_b', None, gfx10_vaddr_4, gfx10_srsrc, gfx10_ssamp, MODIFIERS))
	def image_gather4_c_b_cl(self, gfx10_vaddr_4:reg_block, gfx10_srsrc:reg_block, gfx10_ssamp:reg_block, MODIFIERS:str=''):
		return self.ic_pb(mimg_base('image_gather4_c_b_cl', None, gfx10_vaddr_4, gfx10_srsrc, gfx10_ssamp, MODIFIERS))
	def image_gather4_c_b_cl_o(self, gfx10_vaddr_4:reg_block, gfx10_srsrc:reg_block, gfx10_ssamp:reg_block, MODIFIERS:str=''):
		return self.ic_pb(mimg_base('image_gather4_c_b_cl_o', None, gfx10_vaddr_4, gfx10_srsrc, gfx10_ssamp, MODIFIERS))
	def image_gather4_c_b_o(self, gfx10_vaddr_4:reg_block, gfx10_srsrc:reg_block, gfx10_ssamp:reg_block, MODIFIERS:str=''):
		return self.ic_pb(mimg_base('image_gather4_c_b_o', None, gfx10_vaddr_4, gfx10_srsrc, gfx10_ssamp, MODIFIERS))
	def image_gather4_c_cl(self, gfx10_vaddr_4:reg_block, gfx10_srsrc:reg_block, gfx10_ssamp:reg_block, MODIFIERS:str=''):
		return self.ic_pb(mimg_base('image_gather4_c_cl', None, gfx10_vaddr_4, gfx10_srsrc, gfx10_ssamp, MODIFIERS))
	def image_gather4_c_cl_o(self, gfx10_vaddr_4:reg_block, gfx10_srsrc:reg_block, gfx10_ssamp:reg_block, MODIFIERS:str=''):
		return self.ic_pb(mimg_base('image_gather4_c_cl_o', None, gfx10_vaddr_4, gfx10_srsrc, gfx10_ssamp, MODIFIERS))
	def image_gather4_c_l(self, gfx10_vaddr_4:reg_block, gfx10_srsrc:reg_block, gfx10_ssamp:reg_block, MODIFIERS:str=''):
		return self.ic_pb(mimg_base('image_gather4_c_l', None, gfx10_vaddr_4, gfx10_srsrc, gfx10_ssamp, MODIFIERS))
	def image_gather4_c_l_o(self, gfx10_vaddr_4:reg_block, gfx10_srsrc:reg_block, gfx10_ssamp:reg_block, MODIFIERS:str=''):
		return self.ic_pb(mimg_base('image_gather4_c_l_o', None, gfx10_vaddr_4, gfx10_srsrc, gfx10_ssamp, MODIFIERS))
	def image_gather4_c_lz(self, gfx10_vaddr_4:reg_block, gfx10_srsrc:reg_block, gfx10_ssamp:reg_block, MODIFIERS:str=''):
		return self.ic_pb(mimg_base('image_gather4_c_lz', None, gfx10_vaddr_4, gfx10_srsrc, gfx10_ssamp, MODIFIERS))
	def image_gather4_c_lz_o(self, gfx10_vaddr_4:reg_block, gfx10_srsrc:reg_block, gfx10_ssamp:reg_block, MODIFIERS:str=''):
		return self.ic_pb(mimg_base('image_gather4_c_lz_o', None, gfx10_vaddr_4, gfx10_srsrc, gfx10_ssamp, MODIFIERS))
	def image_gather4_c_o(self, gfx10_vaddr_4:reg_block, gfx10_srsrc:reg_block, gfx10_ssamp:reg_block, MODIFIERS:str=''):
		return self.ic_pb(mimg_base('image_gather4_c_o', None, gfx10_vaddr_4, gfx10_srsrc, gfx10_ssamp, MODIFIERS))
	def image_gather4_cl(self, gfx10_vaddr_4:reg_block, gfx10_srsrc:reg_block, gfx10_ssamp:reg_block, MODIFIERS:str=''):
		return self.ic_pb(mimg_base('image_gather4_cl', None, gfx10_vaddr_4, gfx10_srsrc, gfx10_ssamp, MODIFIERS))
	def image_gather4_cl_o(self, gfx10_vaddr_4:reg_block, gfx10_srsrc:reg_block, gfx10_ssamp:reg_block, MODIFIERS:str=''):
		return self.ic_pb(mimg_base('image_gather4_cl_o', None, gfx10_vaddr_4, gfx10_srsrc, gfx10_ssamp, MODIFIERS))
	def image_gather4_l(self, gfx10_vaddr_4:reg_block, gfx10_srsrc:reg_block, gfx10_ssamp:reg_block, MODIFIERS:str=''):
		return self.ic_pb(mimg_base('image_gather4_l', None, gfx10_vaddr_4, gfx10_srsrc, gfx10_ssamp, MODIFIERS))
	def image_gather4_l_o(self, gfx10_vaddr_4:reg_block, gfx10_srsrc:reg_block, gfx10_ssamp:reg_block, MODIFIERS:str=''):
		return self.ic_pb(mimg_base('image_gather4_l_o', None, gfx10_vaddr_4, gfx10_srsrc, gfx10_ssamp, MODIFIERS))
	def image_gather4_lz(self, gfx10_vaddr_4:reg_block, gfx10_srsrc:reg_block, gfx10_ssamp:reg_block, MODIFIERS:str=''):
		return self.ic_pb(mimg_base('image_gather4_lz', None, gfx10_vaddr_4, gfx10_srsrc, gfx10_ssamp, MODIFIERS))
	def image_gather4_lz_o(self, gfx10_vaddr_4:reg_block, gfx10_srsrc:reg_block, gfx10_ssamp:reg_block, MODIFIERS:str=''):
		return self.ic_pb(mimg_base('image_gather4_lz_o', None, gfx10_vaddr_4, gfx10_srsrc, gfx10_ssamp, MODIFIERS))
	def image_gather4_o(self, gfx10_vaddr_4:reg_block, gfx10_srsrc:reg_block, gfx10_ssamp:reg_block, MODIFIERS:str=''):
		return self.ic_pb(mimg_base('image_gather4_o', None, gfx10_vaddr_4, gfx10_srsrc, gfx10_ssamp, MODIFIERS))
	def image_get_lod(self, gfx10_vaddr_4:reg_block, gfx10_srsrc:reg_block, gfx10_ssamp:reg_block, MODIFIERS:str=''):
		return self.ic_pb(mimg_base('image_get_lod', None, gfx10_vaddr_4, gfx10_srsrc, gfx10_ssamp, MODIFIERS))
	def image_get_resinfo(self, gfx10_vaddr_4:reg_block, gfx10_srsrc:reg_block, MODIFIERS:str=''):
		return self.ic_pb(mimg_base('image_get_resinfo', None, gfx10_vaddr_4, gfx10_srsrc, None, MODIFIERS))
	def image_load(self, gfx10_vaddr_4:reg_block, gfx10_srsrc:reg_block, MODIFIERS:str=''):
		return self.ic_pb(mimg_base('image_load', None, gfx10_vaddr_4, gfx10_srsrc, None, MODIFIERS))
	def image_load_mip(self, gfx10_vaddr_4:reg_block, gfx10_srsrc:reg_block, MODIFIERS:str=''):
		return self.ic_pb(mimg_base('image_load_mip', None, gfx10_vaddr_4, gfx10_srsrc, None, MODIFIERS))
	def image_load_mip_pck(self, gfx10_vaddr_4:reg_block, gfx10_srsrc:reg_block, MODIFIERS:str=''):
		return self.ic_pb(mimg_base('image_load_mip_pck', None, gfx10_vaddr_4, gfx10_srsrc, None, MODIFIERS))
	def image_load_mip_pck_sgn(self, gfx10_vaddr_4:reg_block, gfx10_srsrc:reg_block, MODIFIERS:str=''):
		return self.ic_pb(mimg_base('image_load_mip_pck_sgn', None, gfx10_vaddr_4, gfx10_srsrc, None, MODIFIERS))
	def image_load_pck(self, gfx10_vaddr_4:reg_block, gfx10_srsrc:reg_block, MODIFIERS:str=''):
		return self.ic_pb(mimg_base('image_load_pck', None, gfx10_vaddr_4, gfx10_srsrc, None, MODIFIERS))
	def image_load_pck_sgn(self, gfx10_vaddr_4:reg_block, gfx10_srsrc:reg_block, MODIFIERS:str=''):
		return self.ic_pb(mimg_base('image_load_pck_sgn', None, gfx10_vaddr_4, gfx10_srsrc, None, MODIFIERS))
	def image_sample(self, gfx10_vaddr_4:reg_block, gfx10_srsrc:reg_block, gfx10_ssamp:reg_block, MODIFIERS:str=''):
		return self.ic_pb(mimg_base('image_sample', None, gfx10_vaddr_4, gfx10_srsrc, gfx10_ssamp, MODIFIERS))
	def image_sample_b(self, gfx10_vaddr_4:reg_block, gfx10_srsrc:reg_block, gfx10_ssamp:reg_block, MODIFIERS:str=''):
		return self.ic_pb(mimg_base('image_sample_b', None, gfx10_vaddr_4, gfx10_srsrc, gfx10_ssamp, MODIFIERS))
	def image_sample_b_cl(self, gfx10_vaddr_4:reg_block, gfx10_srsrc:reg_block, gfx10_ssamp:reg_block, MODIFIERS:str=''):
		return self.ic_pb(mimg_base('image_sample_b_cl', None, gfx10_vaddr_4, gfx10_srsrc, gfx10_ssamp, MODIFIERS))
	def image_sample_b_cl_o(self, gfx10_vaddr_4:reg_block, gfx10_srsrc:reg_block, gfx10_ssamp:reg_block, MODIFIERS:str=''):
		return self.ic_pb(mimg_base('image_sample_b_cl_o', None, gfx10_vaddr_4, gfx10_srsrc, gfx10_ssamp, MODIFIERS))
	def image_sample_b_o(self, gfx10_vaddr_4:reg_block, gfx10_srsrc:reg_block, gfx10_ssamp:reg_block, MODIFIERS:str=''):
		return self.ic_pb(mimg_base('image_sample_b_o', None, gfx10_vaddr_4, gfx10_srsrc, gfx10_ssamp, MODIFIERS))
	def image_sample_c(self, gfx10_vaddr_4:reg_block, gfx10_srsrc:reg_block, gfx10_ssamp:reg_block, MODIFIERS:str=''):
		return self.ic_pb(mimg_base('image_sample_c', None, gfx10_vaddr_4, gfx10_srsrc, gfx10_ssamp, MODIFIERS))
	def image_sample_c_b(self, gfx10_vaddr_4:reg_block, gfx10_srsrc:reg_block, gfx10_ssamp:reg_block, MODIFIERS:str=''):
		return self.ic_pb(mimg_base('image_sample_c_b', None, gfx10_vaddr_4, gfx10_srsrc, gfx10_ssamp, MODIFIERS))
	def image_sample_c_b_cl(self, gfx10_vaddr_4:reg_block, gfx10_srsrc:reg_block, gfx10_ssamp:reg_block, MODIFIERS:str=''):
		return self.ic_pb(mimg_base('image_sample_c_b_cl', None, gfx10_vaddr_4, gfx10_srsrc, gfx10_ssamp, MODIFIERS))
	def image_sample_c_b_cl_o(self, gfx10_vaddr_4:reg_block, gfx10_srsrc:reg_block, gfx10_ssamp:reg_block, MODIFIERS:str=''):
		return self.ic_pb(mimg_base('image_sample_c_b_cl_o', None, gfx10_vaddr_4, gfx10_srsrc, gfx10_ssamp, MODIFIERS))
	def image_sample_c_b_o(self, gfx10_vaddr_4:reg_block, gfx10_srsrc:reg_block, gfx10_ssamp:reg_block, MODIFIERS:str=''):
		return self.ic_pb(mimg_base('image_sample_c_b_o', None, gfx10_vaddr_4, gfx10_srsrc, gfx10_ssamp, MODIFIERS))
	def image_sample_c_cd(self, gfx10_vaddr_4:reg_block, gfx10_srsrc:reg_block, gfx10_ssamp:reg_block, MODIFIERS:str=''):
		return self.ic_pb(mimg_base('image_sample_c_cd', None, gfx10_vaddr_4, gfx10_srsrc, gfx10_ssamp, MODIFIERS))
	def image_sample_c_cd_cl(self, gfx10_vaddr_4:reg_block, gfx10_srsrc:reg_block, gfx10_ssamp:reg_block, MODIFIERS:str=''):
		return self.ic_pb(mimg_base('image_sample_c_cd_cl', None, gfx10_vaddr_4, gfx10_srsrc, gfx10_ssamp, MODIFIERS))
	def image_sample_c_cd_cl_g16(self, gfx10_vaddr_4:reg_block, gfx10_srsrc:reg_block, gfx10_ssamp:reg_block, MODIFIERS:str=''):
		return self.ic_pb(mimg_base('image_sample_c_cd_cl_g16', None, gfx10_vaddr_4, gfx10_srsrc, gfx10_ssamp, MODIFIERS))
	def image_sample_c_cd_cl_o(self, gfx10_vaddr_4:reg_block, gfx10_srsrc:reg_block, gfx10_ssamp:reg_block, MODIFIERS:str=''):
		return self.ic_pb(mimg_base('image_sample_c_cd_cl_o', None, gfx10_vaddr_4, gfx10_srsrc, gfx10_ssamp, MODIFIERS))
	def image_sample_c_cd_cl_o_g16(self, gfx10_vaddr_4:reg_block, gfx10_srsrc:reg_block, gfx10_ssamp:reg_block, MODIFIERS:str=''):
		return self.ic_pb(mimg_base('image_sample_c_cd_cl_o_g16', None, gfx10_vaddr_4, gfx10_srsrc, gfx10_ssamp, MODIFIERS))
	def image_sample_c_cd_g16(self, gfx10_vaddr_4:reg_block, gfx10_srsrc:reg_block, gfx10_ssamp:reg_block, MODIFIERS:str=''):
		return self.ic_pb(mimg_base('image_sample_c_cd_g16', None, gfx10_vaddr_4, gfx10_srsrc, gfx10_ssamp, MODIFIERS))
	def image_sample_c_cd_o(self, gfx10_vaddr_4:reg_block, gfx10_srsrc:reg_block, gfx10_ssamp:reg_block, MODIFIERS:str=''):
		return self.ic_pb(mimg_base('image_sample_c_cd_o', None, gfx10_vaddr_4, gfx10_srsrc, gfx10_ssamp, MODIFIERS))
	def image_sample_c_cd_o_g16(self, gfx10_vaddr_4:reg_block, gfx10_srsrc:reg_block, gfx10_ssamp:reg_block, MODIFIERS:str=''):
		return self.ic_pb(mimg_base('image_sample_c_cd_o_g16', None, gfx10_vaddr_4, gfx10_srsrc, gfx10_ssamp, MODIFIERS))
	def image_sample_c_cl(self, gfx10_vaddr_4:reg_block, gfx10_srsrc:reg_block, gfx10_ssamp:reg_block, MODIFIERS:str=''):
		return self.ic_pb(mimg_base('image_sample_c_cl', None, gfx10_vaddr_4, gfx10_srsrc, gfx10_ssamp, MODIFIERS))
	def image_sample_c_cl_o(self, gfx10_vaddr_4:reg_block, gfx10_srsrc:reg_block, gfx10_ssamp:reg_block, MODIFIERS:str=''):
		return self.ic_pb(mimg_base('image_sample_c_cl_o', None, gfx10_vaddr_4, gfx10_srsrc, gfx10_ssamp, MODIFIERS))
	def image_sample_c_d(self, gfx10_vaddr_4:reg_block, gfx10_srsrc:reg_block, gfx10_ssamp:reg_block, MODIFIERS:str=''):
		return self.ic_pb(mimg_base('image_sample_c_d', None, gfx10_vaddr_4, gfx10_srsrc, gfx10_ssamp, MODIFIERS))
	def image_sample_c_d_cl(self, gfx10_vaddr_4:reg_block, gfx10_srsrc:reg_block, gfx10_ssamp:reg_block, MODIFIERS:str=''):
		return self.ic_pb(mimg_base('image_sample_c_d_cl', None, gfx10_vaddr_4, gfx10_srsrc, gfx10_ssamp, MODIFIERS))
	def image_sample_c_d_cl_g16(self, gfx10_vaddr_4:reg_block, gfx10_srsrc:reg_block, gfx10_ssamp:reg_block, MODIFIERS:str=''):
		return self.ic_pb(mimg_base('image_sample_c_d_cl_g16', None, gfx10_vaddr_4, gfx10_srsrc, gfx10_ssamp, MODIFIERS))
	def image_sample_c_d_cl_o(self, gfx10_vaddr_4:reg_block, gfx10_srsrc:reg_block, gfx10_ssamp:reg_block, MODIFIERS:str=''):
		return self.ic_pb(mimg_base('image_sample_c_d_cl_o', None, gfx10_vaddr_4, gfx10_srsrc, gfx10_ssamp, MODIFIERS))
	def image_sample_c_d_cl_o_g16(self, gfx10_vaddr_4:reg_block, gfx10_srsrc:reg_block, gfx10_ssamp:reg_block, MODIFIERS:str=''):
		return self.ic_pb(mimg_base('image_sample_c_d_cl_o_g16', None, gfx10_vaddr_4, gfx10_srsrc, gfx10_ssamp, MODIFIERS))
	def image_sample_c_d_g16(self, gfx10_vaddr_4:reg_block, gfx10_srsrc:reg_block, gfx10_ssamp:reg_block, MODIFIERS:str=''):
		return self.ic_pb(mimg_base('image_sample_c_d_g16', None, gfx10_vaddr_4, gfx10_srsrc, gfx10_ssamp, MODIFIERS))
	def image_sample_c_d_o(self, gfx10_vaddr_4:reg_block, gfx10_srsrc:reg_block, gfx10_ssamp:reg_block, MODIFIERS:str=''):
		return self.ic_pb(mimg_base('image_sample_c_d_o', None, gfx10_vaddr_4, gfx10_srsrc, gfx10_ssamp, MODIFIERS))
	def image_sample_c_d_o_g16(self, gfx10_vaddr_4:reg_block, gfx10_srsrc:reg_block, gfx10_ssamp:reg_block, MODIFIERS:str=''):
		return self.ic_pb(mimg_base('image_sample_c_d_o_g16', None, gfx10_vaddr_4, gfx10_srsrc, gfx10_ssamp, MODIFIERS))
	def image_sample_c_l(self, gfx10_vaddr_4:reg_block, gfx10_srsrc:reg_block, gfx10_ssamp:reg_block, MODIFIERS:str=''):
		return self.ic_pb(mimg_base('image_sample_c_l', None, gfx10_vaddr_4, gfx10_srsrc, gfx10_ssamp, MODIFIERS))
	def image_sample_c_l_o(self, gfx10_vaddr_4:reg_block, gfx10_srsrc:reg_block, gfx10_ssamp:reg_block, MODIFIERS:str=''):
		return self.ic_pb(mimg_base('image_sample_c_l_o', None, gfx10_vaddr_4, gfx10_srsrc, gfx10_ssamp, MODIFIERS))
	def image_sample_c_lz(self, gfx10_vaddr_4:reg_block, gfx10_srsrc:reg_block, gfx10_ssamp:reg_block, MODIFIERS:str=''):
		return self.ic_pb(mimg_base('image_sample_c_lz', None, gfx10_vaddr_4, gfx10_srsrc, gfx10_ssamp, MODIFIERS))
	def image_sample_c_lz_o(self, gfx10_vaddr_4:reg_block, gfx10_srsrc:reg_block, gfx10_ssamp:reg_block, MODIFIERS:str=''):
		return self.ic_pb(mimg_base('image_sample_c_lz_o', None, gfx10_vaddr_4, gfx10_srsrc, gfx10_ssamp, MODIFIERS))
	def image_sample_c_o(self, gfx10_vaddr_4:reg_block, gfx10_srsrc:reg_block, gfx10_ssamp:reg_block, MODIFIERS:str=''):
		return self.ic_pb(mimg_base('image_sample_c_o', None, gfx10_vaddr_4, gfx10_srsrc, gfx10_ssamp, MODIFIERS))
	def image_sample_cd(self, gfx10_vaddr_4:reg_block, gfx10_srsrc:reg_block, gfx10_ssamp:reg_block, MODIFIERS:str=''):
		return self.ic_pb(mimg_base('image_sample_cd', None, gfx10_vaddr_4, gfx10_srsrc, gfx10_ssamp, MODIFIERS))
	def image_sample_cd_cl(self, gfx10_vaddr_4:reg_block, gfx10_srsrc:reg_block, gfx10_ssamp:reg_block, MODIFIERS:str=''):
		return self.ic_pb(mimg_base('image_sample_cd_cl', None, gfx10_vaddr_4, gfx10_srsrc, gfx10_ssamp, MODIFIERS))
	def image_sample_cd_cl_g16(self, gfx10_vaddr_4:reg_block, gfx10_srsrc:reg_block, gfx10_ssamp:reg_block, MODIFIERS:str=''):
		return self.ic_pb(mimg_base('image_sample_cd_cl_g16', None, gfx10_vaddr_4, gfx10_srsrc, gfx10_ssamp, MODIFIERS))
	def image_sample_cd_cl_o(self, gfx10_vaddr_4:reg_block, gfx10_srsrc:reg_block, gfx10_ssamp:reg_block, MODIFIERS:str=''):
		return self.ic_pb(mimg_base('image_sample_cd_cl_o', None, gfx10_vaddr_4, gfx10_srsrc, gfx10_ssamp, MODIFIERS))
	def image_sample_cd_cl_o_g16(self, gfx10_vaddr_4:reg_block, gfx10_srsrc:reg_block, gfx10_ssamp:reg_block, MODIFIERS:str=''):
		return self.ic_pb(mimg_base('image_sample_cd_cl_o_g16', None, gfx10_vaddr_4, gfx10_srsrc, gfx10_ssamp, MODIFIERS))
	def image_sample_cd_g16(self, gfx10_vaddr_4:reg_block, gfx10_srsrc:reg_block, gfx10_ssamp:reg_block, MODIFIERS:str=''):
		return self.ic_pb(mimg_base('image_sample_cd_g16', None, gfx10_vaddr_4, gfx10_srsrc, gfx10_ssamp, MODIFIERS))
	def image_sample_cd_o(self, gfx10_vaddr_4:reg_block, gfx10_srsrc:reg_block, gfx10_ssamp:reg_block, MODIFIERS:str=''):
		return self.ic_pb(mimg_base('image_sample_cd_o', None, gfx10_vaddr_4, gfx10_srsrc, gfx10_ssamp, MODIFIERS))
	def image_sample_cd_o_g16(self, gfx10_vaddr_4:reg_block, gfx10_srsrc:reg_block, gfx10_ssamp:reg_block, MODIFIERS:str=''):
		return self.ic_pb(mimg_base('image_sample_cd_o_g16', None, gfx10_vaddr_4, gfx10_srsrc, gfx10_ssamp, MODIFIERS))
	def image_sample_cl(self, gfx10_vaddr_4:reg_block, gfx10_srsrc:reg_block, gfx10_ssamp:reg_block, MODIFIERS:str=''):
		return self.ic_pb(mimg_base('image_sample_cl', None, gfx10_vaddr_4, gfx10_srsrc, gfx10_ssamp, MODIFIERS))
	def image_sample_cl_o(self, gfx10_vaddr_4:reg_block, gfx10_srsrc:reg_block, gfx10_ssamp:reg_block, MODIFIERS:str=''):
		return self.ic_pb(mimg_base('image_sample_cl_o', None, gfx10_vaddr_4, gfx10_srsrc, gfx10_ssamp, MODIFIERS))
	def image_sample_d(self, gfx10_vaddr_4:reg_block, gfx10_srsrc:reg_block, gfx10_ssamp:reg_block, MODIFIERS:str=''):
		return self.ic_pb(mimg_base('image_sample_d', None, gfx10_vaddr_4, gfx10_srsrc, gfx10_ssamp, MODIFIERS))
	def image_sample_d_cl(self, gfx10_vaddr_4:reg_block, gfx10_srsrc:reg_block, gfx10_ssamp:reg_block, MODIFIERS:str=''):
		return self.ic_pb(mimg_base('image_sample_d_cl', None, gfx10_vaddr_4, gfx10_srsrc, gfx10_ssamp, MODIFIERS))
	def image_sample_d_cl_g16(self, gfx10_vaddr_4:reg_block, gfx10_srsrc:reg_block, gfx10_ssamp:reg_block, MODIFIERS:str=''):
		return self.ic_pb(mimg_base('image_sample_d_cl_g16', None, gfx10_vaddr_4, gfx10_srsrc, gfx10_ssamp, MODIFIERS))
	def image_sample_d_cl_o(self, gfx10_vaddr_4:reg_block, gfx10_srsrc:reg_block, gfx10_ssamp:reg_block, MODIFIERS:str=''):
		return self.ic_pb(mimg_base('image_sample_d_cl_o', None, gfx10_vaddr_4, gfx10_srsrc, gfx10_ssamp, MODIFIERS))
	def image_sample_d_cl_o_g16(self, gfx10_vaddr_4:reg_block, gfx10_srsrc:reg_block, gfx10_ssamp:reg_block, MODIFIERS:str=''):
		return self.ic_pb(mimg_base('image_sample_d_cl_o_g16', None, gfx10_vaddr_4, gfx10_srsrc, gfx10_ssamp, MODIFIERS))
	def image_sample_d_g16(self, gfx10_vaddr_4:reg_block, gfx10_srsrc:reg_block, gfx10_ssamp:reg_block, MODIFIERS:str=''):
		return self.ic_pb(mimg_base('image_sample_d_g16', None, gfx10_vaddr_4, gfx10_srsrc, gfx10_ssamp, MODIFIERS))
	def image_sample_d_o(self, gfx10_vaddr_4:reg_block, gfx10_srsrc:reg_block, gfx10_ssamp:reg_block, MODIFIERS:str=''):
		return self.ic_pb(mimg_base('image_sample_d_o', None, gfx10_vaddr_4, gfx10_srsrc, gfx10_ssamp, MODIFIERS))
	def image_sample_d_o_g16(self, gfx10_vaddr_4:reg_block, gfx10_srsrc:reg_block, gfx10_ssamp:reg_block, MODIFIERS:str=''):
		return self.ic_pb(mimg_base('image_sample_d_o_g16', None, gfx10_vaddr_4, gfx10_srsrc, gfx10_ssamp, MODIFIERS))
	def image_sample_l(self, gfx10_vaddr_4:reg_block, gfx10_srsrc:reg_block, gfx10_ssamp:reg_block, MODIFIERS:str=''):
		return self.ic_pb(mimg_base('image_sample_l', None, gfx10_vaddr_4, gfx10_srsrc, gfx10_ssamp, MODIFIERS))
	def image_sample_l_o(self, gfx10_vaddr_4:reg_block, gfx10_srsrc:reg_block, gfx10_ssamp:reg_block, MODIFIERS:str=''):
		return self.ic_pb(mimg_base('image_sample_l_o', None, gfx10_vaddr_4, gfx10_srsrc, gfx10_ssamp, MODIFIERS))
	def image_sample_lz(self, gfx10_vaddr_4:reg_block, gfx10_srsrc:reg_block, gfx10_ssamp:reg_block, MODIFIERS:str=''):
		return self.ic_pb(mimg_base('image_sample_lz', None, gfx10_vaddr_4, gfx10_srsrc, gfx10_ssamp, MODIFIERS))
	def image_sample_lz_o(self, gfx10_vaddr_4:reg_block, gfx10_srsrc:reg_block, gfx10_ssamp:reg_block, MODIFIERS:str=''):
		return self.ic_pb(mimg_base('image_sample_lz_o', None, gfx10_vaddr_4, gfx10_srsrc, gfx10_ssamp, MODIFIERS))
	def image_sample_o(self, gfx10_vaddr_4:reg_block, gfx10_srsrc:reg_block, gfx10_ssamp:reg_block, MODIFIERS:str=''):
		return self.ic_pb(mimg_base('image_sample_o', None, gfx10_vaddr_4, gfx10_srsrc, gfx10_ssamp, MODIFIERS))
	def image_store(self, gfx10_vaddr_4:reg_block, gfx10_srsrc:reg_block, MODIFIERS:str=''):
		return self.ic_pb(mimg_base('image_store', None, None, gfx10_vaddr_4, gfx10_srsrc, MODIFIERS))
	def image_store_mip(self, gfx10_vaddr_4:reg_block, gfx10_srsrc:reg_block, MODIFIERS:str=''):
		return self.ic_pb(mimg_base('image_store_mip', None, None, gfx10_vaddr_4, gfx10_srsrc, MODIFIERS))
	def image_store_mip_pck(self, gfx10_vaddr_4:reg_block, gfx10_srsrc:reg_block, MODIFIERS:str=''):
		return self.ic_pb(mimg_base('image_store_mip_pck', None, None, gfx10_vaddr_4, gfx10_srsrc, MODIFIERS))
	def image_store_pck(self, gfx10_vaddr_4:reg_block, gfx10_srsrc:reg_block, MODIFIERS:str=''):
		return self.ic_pb(mimg_base('image_store_pck', None, None, gfx10_vaddr_4, gfx10_srsrc, MODIFIERS))
class mtbuf_base(inst_base): 
	def __init__(self, INSTRUCTION:str, DST:Union[reg_block,None], SRC0:Union[reg_block,None], SRC1:Union[reg_block,None], SRC2:Union[reg_block,None], SRC3:Union[reg_block,None], MODIFIERS:str): 
		super().__init__(instruction_type.SMEM, INSTRUCTION)
		self.DST = DST 
		self.SRC0 = SRC0 
		self.SRC1 = SRC1 
		self.SRC2 = SRC2 
		self.SRC3 = SRC3 
		self.MODIFIERS = MODIFIERS 
	def __str__(self): 
		args_l = filter(None.__ne__, [self.DST,self.SRC0,self.SRC1,self.SRC2,self.SRC3]) 
		return f"{self.label} {','.join(map(str, args_l))} {self.MODIFIERS}" 
class mtbuf_instr_caller(inst_caller_base): 
	def __init__(self, insturction_container) -> None:
     		super().__init__(insturction_container)
	def tbuffer_load_format_d16_x(self, gfx10_vaddr_5:Union[reg_block,None], gfx10_srsrc_1:reg_block, gfx10_soffset:reg_block, MODIFIERS:str=''):
		return self.ic_pb(mtbuf_base('tbuffer_load_format_d16_x', None, gfx10_vaddr_5, gfx10_srsrc_1, gfx10_soffset, None, MODIFIERS))
	def tbuffer_load_format_d16_xy(self, gfx10_vaddr_5:Union[reg_block,None], gfx10_srsrc_1:reg_block, gfx10_soffset:reg_block, MODIFIERS:str=''):
		return self.ic_pb(mtbuf_base('tbuffer_load_format_d16_xy', None, gfx10_vaddr_5, gfx10_srsrc_1, gfx10_soffset, None, MODIFIERS))
	def tbuffer_load_format_d16_xyz(self, gfx10_vaddr_5:Union[reg_block,None], gfx10_srsrc_1:reg_block, gfx10_soffset:reg_block, MODIFIERS:str=''):
		return self.ic_pb(mtbuf_base('tbuffer_load_format_d16_xyz', None, gfx10_vaddr_5, gfx10_srsrc_1, gfx10_soffset, None, MODIFIERS))
	def tbuffer_load_format_d16_xyzw(self, gfx10_vaddr_5:Union[reg_block,None], gfx10_srsrc_1:reg_block, gfx10_soffset:reg_block, MODIFIERS:str=''):
		return self.ic_pb(mtbuf_base('tbuffer_load_format_d16_xyzw', None, gfx10_vaddr_5, gfx10_srsrc_1, gfx10_soffset, None, MODIFIERS))
	def tbuffer_load_format_x(self, gfx10_vaddr_5:Union[reg_block,None], gfx10_srsrc_1:reg_block, gfx10_soffset:reg_block, MODIFIERS:str=''):
		return self.ic_pb(mtbuf_base('tbuffer_load_format_x', None, gfx10_vaddr_5, gfx10_srsrc_1, gfx10_soffset, None, MODIFIERS))
	def tbuffer_load_format_xy(self, gfx10_vaddr_5:Union[reg_block,None], gfx10_srsrc_1:reg_block, gfx10_soffset:reg_block, MODIFIERS:str=''):
		return self.ic_pb(mtbuf_base('tbuffer_load_format_xy', None, gfx10_vaddr_5, gfx10_srsrc_1, gfx10_soffset, None, MODIFIERS))
	def tbuffer_load_format_xyz(self, gfx10_vaddr_5:Union[reg_block,None], gfx10_srsrc_1:reg_block, gfx10_soffset:reg_block, MODIFIERS:str=''):
		return self.ic_pb(mtbuf_base('tbuffer_load_format_xyz', None, gfx10_vaddr_5, gfx10_srsrc_1, gfx10_soffset, None, MODIFIERS))
	def tbuffer_load_format_xyzw(self, gfx10_vaddr_5:Union[reg_block,None], gfx10_srsrc_1:reg_block, gfx10_soffset:reg_block, MODIFIERS:str=''):
		return self.ic_pb(mtbuf_base('tbuffer_load_format_xyzw', None, gfx10_vaddr_5, gfx10_srsrc_1, gfx10_soffset, None, MODIFIERS))
	def tbuffer_store_format_d16_x(self, gfx10_vaddr_5:Union[reg_block,None], gfx10_srsrc_1:reg_block, gfx10_soffset:reg_block, MODIFIERS:str=''):
		return self.ic_pb(mtbuf_base('tbuffer_store_format_d16_x', None, None, gfx10_vaddr_5, gfx10_srsrc_1, gfx10_soffset, MODIFIERS))
	def tbuffer_store_format_d16_xy(self, gfx10_vaddr_5:Union[reg_block,None], gfx10_srsrc_1:reg_block, gfx10_soffset:reg_block, MODIFIERS:str=''):
		return self.ic_pb(mtbuf_base('tbuffer_store_format_d16_xy', None, None, gfx10_vaddr_5, gfx10_srsrc_1, gfx10_soffset, MODIFIERS))
	def tbuffer_store_format_d16_xyz(self, gfx10_vaddr_5:Union[reg_block,None], gfx10_srsrc_1:reg_block, gfx10_soffset:reg_block, MODIFIERS:str=''):
		return self.ic_pb(mtbuf_base('tbuffer_store_format_d16_xyz', None, None, gfx10_vaddr_5, gfx10_srsrc_1, gfx10_soffset, MODIFIERS))
	def tbuffer_store_format_d16_xyzw(self, gfx10_vaddr_5:Union[reg_block,None], gfx10_srsrc_1:reg_block, gfx10_soffset:reg_block, MODIFIERS:str=''):
		return self.ic_pb(mtbuf_base('tbuffer_store_format_d16_xyzw', None, None, gfx10_vaddr_5, gfx10_srsrc_1, gfx10_soffset, MODIFIERS))
	def tbuffer_store_format_x(self, gfx10_vaddr_5:Union[reg_block,None], gfx10_srsrc_1:reg_block, gfx10_soffset:reg_block, MODIFIERS:str=''):
		return self.ic_pb(mtbuf_base('tbuffer_store_format_x', None, None, gfx10_vaddr_5, gfx10_srsrc_1, gfx10_soffset, MODIFIERS))
	def tbuffer_store_format_xy(self, gfx10_vaddr_5:Union[reg_block,None], gfx10_srsrc_1:reg_block, gfx10_soffset:reg_block, MODIFIERS:str=''):
		return self.ic_pb(mtbuf_base('tbuffer_store_format_xy', None, None, gfx10_vaddr_5, gfx10_srsrc_1, gfx10_soffset, MODIFIERS))
	def tbuffer_store_format_xyz(self, gfx10_vaddr_5:Union[reg_block,None], gfx10_srsrc_1:reg_block, gfx10_soffset:reg_block, MODIFIERS:str=''):
		return self.ic_pb(mtbuf_base('tbuffer_store_format_xyz', None, None, gfx10_vaddr_5, gfx10_srsrc_1, gfx10_soffset, MODIFIERS))
	def tbuffer_store_format_xyzw(self, gfx10_vaddr_5:Union[reg_block,None], gfx10_srsrc_1:reg_block, gfx10_soffset:reg_block, MODIFIERS:str=''):
		return self.ic_pb(mtbuf_base('tbuffer_store_format_xyzw', None, None, gfx10_vaddr_5, gfx10_srsrc_1, gfx10_soffset, MODIFIERS))
class mubuf_base(inst_base): 
	def __init__(self, INSTRUCTION:str, DST:Union[reg_block,None], SRC0:Union[reg_block,None], SRC1:Union[reg_block,None], SRC2:Union[reg_block,None], SRC3:Union[reg_block,None], MODIFIERS:str): 
		super().__init__(instruction_type.SMEM, INSTRUCTION)
		self.DST = DST 
		self.SRC0 = SRC0 
		self.SRC1 = SRC1 
		self.SRC2 = SRC2 
		self.SRC3 = SRC3 
		self.MODIFIERS = MODIFIERS 
	def __str__(self): 
		args_l = filter(None.__ne__, [self.DST,self.SRC0,self.SRC1,self.SRC2,self.SRC3]) 
		return f"{self.label} {','.join(map(str, args_l))} {self.MODIFIERS}" 
class mubuf_instr_caller(inst_caller_base): 
	def __init__(self, insturction_container) -> None:
     		super().__init__(insturction_container)
	def buffer_atomic_add(self, gfx10_vaddr_5:Union[reg_block,None], gfx10_srsrc_1:reg_block, gfx10_soffset:reg_block, MODIFIERS:str=''):
		return self.ic_pb(mubuf_base('buffer_atomic_add', None, None, gfx10_vaddr_5, gfx10_srsrc_1, gfx10_soffset, MODIFIERS))
	def buffer_atomic_add_x2(self, gfx10_vaddr_5:Union[reg_block,None], gfx10_srsrc_1:reg_block, gfx10_soffset:reg_block, MODIFIERS:str=''):
		return self.ic_pb(mubuf_base('buffer_atomic_add_x2', None, None, gfx10_vaddr_5, gfx10_srsrc_1, gfx10_soffset, MODIFIERS))
	def buffer_atomic_and(self, gfx10_vaddr_5:Union[reg_block,None], gfx10_srsrc_1:reg_block, gfx10_soffset:reg_block, MODIFIERS:str=''):
		return self.ic_pb(mubuf_base('buffer_atomic_and', None, None, gfx10_vaddr_5, gfx10_srsrc_1, gfx10_soffset, MODIFIERS))
	def buffer_atomic_and_x2(self, gfx10_vaddr_5:Union[reg_block,None], gfx10_srsrc_1:reg_block, gfx10_soffset:reg_block, MODIFIERS:str=''):
		return self.ic_pb(mubuf_base('buffer_atomic_and_x2', None, None, gfx10_vaddr_5, gfx10_srsrc_1, gfx10_soffset, MODIFIERS))
	def buffer_atomic_cmpswap(self, gfx10_vaddr_5:Union[reg_block,None], gfx10_srsrc_1:reg_block, gfx10_soffset:reg_block, MODIFIERS:str=''):
		return self.ic_pb(mubuf_base('buffer_atomic_cmpswap', None, None, gfx10_vaddr_5, gfx10_srsrc_1, gfx10_soffset, MODIFIERS))
	def buffer_atomic_cmpswap_x2(self, gfx10_vaddr_5:Union[reg_block,None], gfx10_srsrc_1:reg_block, gfx10_soffset:reg_block, MODIFIERS:str=''):
		return self.ic_pb(mubuf_base('buffer_atomic_cmpswap_x2', None, None, gfx10_vaddr_5, gfx10_srsrc_1, gfx10_soffset, MODIFIERS))
	def buffer_atomic_dec(self, gfx10_vaddr_5:Union[reg_block,None], gfx10_srsrc_1:reg_block, gfx10_soffset:reg_block, MODIFIERS:str=''):
		return self.ic_pb(mubuf_base('buffer_atomic_dec', None, None, gfx10_vaddr_5, gfx10_srsrc_1, gfx10_soffset, MODIFIERS))
	def buffer_atomic_dec_x2(self, gfx10_vaddr_5:Union[reg_block,None], gfx10_srsrc_1:reg_block, gfx10_soffset:reg_block, MODIFIERS:str=''):
		return self.ic_pb(mubuf_base('buffer_atomic_dec_x2', None, None, gfx10_vaddr_5, gfx10_srsrc_1, gfx10_soffset, MODIFIERS))
	def buffer_atomic_fcmpswap(self, gfx10_vaddr_5:Union[reg_block,None], gfx10_srsrc_1:reg_block, gfx10_soffset:reg_block, MODIFIERS:str=''):
		return self.ic_pb(mubuf_base('buffer_atomic_fcmpswap', None, None, gfx10_vaddr_5, gfx10_srsrc_1, gfx10_soffset, MODIFIERS))
	def buffer_atomic_fcmpswap_x2(self, gfx10_vaddr_5:Union[reg_block,None], gfx10_srsrc_1:reg_block, gfx10_soffset:reg_block, MODIFIERS:str=''):
		return self.ic_pb(mubuf_base('buffer_atomic_fcmpswap_x2', None, None, gfx10_vaddr_5, gfx10_srsrc_1, gfx10_soffset, MODIFIERS))
	def buffer_atomic_fmax(self, gfx10_vaddr_5:Union[reg_block,None], gfx10_srsrc_1:reg_block, gfx10_soffset:reg_block, MODIFIERS:str=''):
		return self.ic_pb(mubuf_base('buffer_atomic_fmax', None, None, gfx10_vaddr_5, gfx10_srsrc_1, gfx10_soffset, MODIFIERS))
	def buffer_atomic_fmax_x2(self, gfx10_vaddr_5:Union[reg_block,None], gfx10_srsrc_1:reg_block, gfx10_soffset:reg_block, MODIFIERS:str=''):
		return self.ic_pb(mubuf_base('buffer_atomic_fmax_x2', None, None, gfx10_vaddr_5, gfx10_srsrc_1, gfx10_soffset, MODIFIERS))
	def buffer_atomic_fmin(self, gfx10_vaddr_5:Union[reg_block,None], gfx10_srsrc_1:reg_block, gfx10_soffset:reg_block, MODIFIERS:str=''):
		return self.ic_pb(mubuf_base('buffer_atomic_fmin', None, None, gfx10_vaddr_5, gfx10_srsrc_1, gfx10_soffset, MODIFIERS))
	def buffer_atomic_fmin_x2(self, gfx10_vaddr_5:Union[reg_block,None], gfx10_srsrc_1:reg_block, gfx10_soffset:reg_block, MODIFIERS:str=''):
		return self.ic_pb(mubuf_base('buffer_atomic_fmin_x2', None, None, gfx10_vaddr_5, gfx10_srsrc_1, gfx10_soffset, MODIFIERS))
	def buffer_atomic_inc(self, gfx10_vaddr_5:Union[reg_block,None], gfx10_srsrc_1:reg_block, gfx10_soffset:reg_block, MODIFIERS:str=''):
		return self.ic_pb(mubuf_base('buffer_atomic_inc', None, None, gfx10_vaddr_5, gfx10_srsrc_1, gfx10_soffset, MODIFIERS))
	def buffer_atomic_inc_x2(self, gfx10_vaddr_5:Union[reg_block,None], gfx10_srsrc_1:reg_block, gfx10_soffset:reg_block, MODIFIERS:str=''):
		return self.ic_pb(mubuf_base('buffer_atomic_inc_x2', None, None, gfx10_vaddr_5, gfx10_srsrc_1, gfx10_soffset, MODIFIERS))
	def buffer_atomic_or(self, gfx10_vaddr_5:Union[reg_block,None], gfx10_srsrc_1:reg_block, gfx10_soffset:reg_block, MODIFIERS:str=''):
		return self.ic_pb(mubuf_base('buffer_atomic_or', None, None, gfx10_vaddr_5, gfx10_srsrc_1, gfx10_soffset, MODIFIERS))
	def buffer_atomic_or_x2(self, gfx10_vaddr_5:Union[reg_block,None], gfx10_srsrc_1:reg_block, gfx10_soffset:reg_block, MODIFIERS:str=''):
		return self.ic_pb(mubuf_base('buffer_atomic_or_x2', None, None, gfx10_vaddr_5, gfx10_srsrc_1, gfx10_soffset, MODIFIERS))
	def buffer_atomic_smax(self, gfx10_vaddr_5:Union[reg_block,None], gfx10_srsrc_1:reg_block, gfx10_soffset:reg_block, MODIFIERS:str=''):
		return self.ic_pb(mubuf_base('buffer_atomic_smax', None, None, gfx10_vaddr_5, gfx10_srsrc_1, gfx10_soffset, MODIFIERS))
	def buffer_atomic_smax_x2(self, gfx10_vaddr_5:Union[reg_block,None], gfx10_srsrc_1:reg_block, gfx10_soffset:reg_block, MODIFIERS:str=''):
		return self.ic_pb(mubuf_base('buffer_atomic_smax_x2', None, None, gfx10_vaddr_5, gfx10_srsrc_1, gfx10_soffset, MODIFIERS))
	def buffer_atomic_smin(self, gfx10_vaddr_5:Union[reg_block,None], gfx10_srsrc_1:reg_block, gfx10_soffset:reg_block, MODIFIERS:str=''):
		return self.ic_pb(mubuf_base('buffer_atomic_smin', None, None, gfx10_vaddr_5, gfx10_srsrc_1, gfx10_soffset, MODIFIERS))
	def buffer_atomic_smin_x2(self, gfx10_vaddr_5:Union[reg_block,None], gfx10_srsrc_1:reg_block, gfx10_soffset:reg_block, MODIFIERS:str=''):
		return self.ic_pb(mubuf_base('buffer_atomic_smin_x2', None, None, gfx10_vaddr_5, gfx10_srsrc_1, gfx10_soffset, MODIFIERS))
	def buffer_atomic_sub(self, gfx10_vaddr_5:Union[reg_block,None], gfx10_srsrc_1:reg_block, gfx10_soffset:reg_block, MODIFIERS:str=''):
		return self.ic_pb(mubuf_base('buffer_atomic_sub', None, None, gfx10_vaddr_5, gfx10_srsrc_1, gfx10_soffset, MODIFIERS))
	def buffer_atomic_sub_x2(self, gfx10_vaddr_5:Union[reg_block,None], gfx10_srsrc_1:reg_block, gfx10_soffset:reg_block, MODIFIERS:str=''):
		return self.ic_pb(mubuf_base('buffer_atomic_sub_x2', None, None, gfx10_vaddr_5, gfx10_srsrc_1, gfx10_soffset, MODIFIERS))
	def buffer_atomic_swap(self, gfx10_vaddr_5:Union[reg_block,None], gfx10_srsrc_1:reg_block, gfx10_soffset:reg_block, MODIFIERS:str=''):
		return self.ic_pb(mubuf_base('buffer_atomic_swap', None, None, gfx10_vaddr_5, gfx10_srsrc_1, gfx10_soffset, MODIFIERS))
	def buffer_atomic_swap_x2(self, gfx10_vaddr_5:Union[reg_block,None], gfx10_srsrc_1:reg_block, gfx10_soffset:reg_block, MODIFIERS:str=''):
		return self.ic_pb(mubuf_base('buffer_atomic_swap_x2', None, None, gfx10_vaddr_5, gfx10_srsrc_1, gfx10_soffset, MODIFIERS))
	def buffer_atomic_umax(self, gfx10_vaddr_5:Union[reg_block,None], gfx10_srsrc_1:reg_block, gfx10_soffset:reg_block, MODIFIERS:str=''):
		return self.ic_pb(mubuf_base('buffer_atomic_umax', None, None, gfx10_vaddr_5, gfx10_srsrc_1, gfx10_soffset, MODIFIERS))
	def buffer_atomic_umax_x2(self, gfx10_vaddr_5:Union[reg_block,None], gfx10_srsrc_1:reg_block, gfx10_soffset:reg_block, MODIFIERS:str=''):
		return self.ic_pb(mubuf_base('buffer_atomic_umax_x2', None, None, gfx10_vaddr_5, gfx10_srsrc_1, gfx10_soffset, MODIFIERS))
	def buffer_atomic_umin(self, gfx10_vaddr_5:Union[reg_block,None], gfx10_srsrc_1:reg_block, gfx10_soffset:reg_block, MODIFIERS:str=''):
		return self.ic_pb(mubuf_base('buffer_atomic_umin', None, None, gfx10_vaddr_5, gfx10_srsrc_1, gfx10_soffset, MODIFIERS))
	def buffer_atomic_umin_x2(self, gfx10_vaddr_5:Union[reg_block,None], gfx10_srsrc_1:reg_block, gfx10_soffset:reg_block, MODIFIERS:str=''):
		return self.ic_pb(mubuf_base('buffer_atomic_umin_x2', None, None, gfx10_vaddr_5, gfx10_srsrc_1, gfx10_soffset, MODIFIERS))
	def buffer_atomic_xor(self, gfx10_vaddr_5:Union[reg_block,None], gfx10_srsrc_1:reg_block, gfx10_soffset:reg_block, MODIFIERS:str=''):
		return self.ic_pb(mubuf_base('buffer_atomic_xor', None, None, gfx10_vaddr_5, gfx10_srsrc_1, gfx10_soffset, MODIFIERS))
	def buffer_atomic_xor_x2(self, gfx10_vaddr_5:Union[reg_block,None], gfx10_srsrc_1:reg_block, gfx10_soffset:reg_block, MODIFIERS:str=''):
		return self.ic_pb(mubuf_base('buffer_atomic_xor_x2', None, None, gfx10_vaddr_5, gfx10_srsrc_1, gfx10_soffset, MODIFIERS))
	def buffer_gl0_inv(self):
		return self.ic_pb(mubuf_base('buffer_gl0_inv', None, None, None, None, None))
	def buffer_gl1_inv(self):
		return self.ic_pb(mubuf_base('buffer_gl1_inv', None, None, None, None, None))
	def buffer_load_dword(self, gfx10_vaddr_5:Union[reg_block,None], gfx10_srsrc_1:reg_block, gfx10_soffset:reg_block, MODIFIERS:str=''):
		return self.ic_pb(mubuf_base('buffer_load_dword', None, gfx10_vaddr_5, gfx10_srsrc_1, gfx10_soffset, None, MODIFIERS))
	def buffer_load_dwordx2(self, gfx10_vaddr_5:Union[reg_block,None], gfx10_srsrc_1:reg_block, gfx10_soffset:reg_block, MODIFIERS:str=''):
		return self.ic_pb(mubuf_base('buffer_load_dwordx2', None, gfx10_vaddr_5, gfx10_srsrc_1, gfx10_soffset, None, MODIFIERS))
	def buffer_load_dwordx3(self, gfx10_vaddr_5:Union[reg_block,None], gfx10_srsrc_1:reg_block, gfx10_soffset:reg_block, MODIFIERS:str=''):
		return self.ic_pb(mubuf_base('buffer_load_dwordx3', None, gfx10_vaddr_5, gfx10_srsrc_1, gfx10_soffset, None, MODIFIERS))
	def buffer_load_dwordx4(self, gfx10_vaddr_5:Union[reg_block,None], gfx10_srsrc_1:reg_block, gfx10_soffset:reg_block, MODIFIERS:str=''):
		return self.ic_pb(mubuf_base('buffer_load_dwordx4', None, gfx10_vaddr_5, gfx10_srsrc_1, gfx10_soffset, None, MODIFIERS))
	def buffer_load_format_d16_x(self, gfx10_vaddr_5:Union[reg_block,None], gfx10_srsrc_1:reg_block, gfx10_soffset:reg_block, MODIFIERS:str=''):
		return self.ic_pb(mubuf_base('buffer_load_format_d16_x', None, gfx10_vaddr_5, gfx10_srsrc_1, gfx10_soffset, None, MODIFIERS))
	def buffer_load_format_d16_xy(self, gfx10_vaddr_5:Union[reg_block,None], gfx10_srsrc_1:reg_block, gfx10_soffset:reg_block, MODIFIERS:str=''):
		return self.ic_pb(mubuf_base('buffer_load_format_d16_xy', None, gfx10_vaddr_5, gfx10_srsrc_1, gfx10_soffset, None, MODIFIERS))
	def buffer_load_format_d16_xyz(self, gfx10_vaddr_5:Union[reg_block,None], gfx10_srsrc_1:reg_block, gfx10_soffset:reg_block, MODIFIERS:str=''):
		return self.ic_pb(mubuf_base('buffer_load_format_d16_xyz', None, gfx10_vaddr_5, gfx10_srsrc_1, gfx10_soffset, None, MODIFIERS))
	def buffer_load_format_d16_xyzw(self, gfx10_vaddr_5:Union[reg_block,None], gfx10_srsrc_1:reg_block, gfx10_soffset:reg_block, MODIFIERS:str=''):
		return self.ic_pb(mubuf_base('buffer_load_format_d16_xyzw', None, gfx10_vaddr_5, gfx10_srsrc_1, gfx10_soffset, None, MODIFIERS))
	def buffer_load_format_x(self, gfx10_vaddr_5:Union[reg_block,None], gfx10_srsrc_1:reg_block, gfx10_soffset:reg_block, MODIFIERS:str=''):
		return self.ic_pb(mubuf_base('buffer_load_format_x', None, gfx10_vaddr_5, gfx10_srsrc_1, gfx10_soffset, None, MODIFIERS))
	def buffer_load_format_xy(self, gfx10_vaddr_5:Union[reg_block,None], gfx10_srsrc_1:reg_block, gfx10_soffset:reg_block, MODIFIERS:str=''):
		return self.ic_pb(mubuf_base('buffer_load_format_xy', None, gfx10_vaddr_5, gfx10_srsrc_1, gfx10_soffset, None, MODIFIERS))
	def buffer_load_format_xyz(self, gfx10_vaddr_5:Union[reg_block,None], gfx10_srsrc_1:reg_block, gfx10_soffset:reg_block, MODIFIERS:str=''):
		return self.ic_pb(mubuf_base('buffer_load_format_xyz', None, gfx10_vaddr_5, gfx10_srsrc_1, gfx10_soffset, None, MODIFIERS))
	def buffer_load_format_xyzw(self, gfx10_vaddr_5:Union[reg_block,None], gfx10_srsrc_1:reg_block, gfx10_soffset:reg_block, MODIFIERS:str=''):
		return self.ic_pb(mubuf_base('buffer_load_format_xyzw', None, gfx10_vaddr_5, gfx10_srsrc_1, gfx10_soffset, None, MODIFIERS))
	def buffer_load_sbyte(self, gfx10_vaddr_5:Union[reg_block,None], gfx10_srsrc_1:reg_block, gfx10_soffset:reg_block, MODIFIERS:str=''):
		return self.ic_pb(mubuf_base('buffer_load_sbyte', None, gfx10_vaddr_5, gfx10_srsrc_1, gfx10_soffset, None, MODIFIERS))
	def buffer_load_sbyte_d16(self, gfx10_vaddr_5:Union[reg_block,None], gfx10_srsrc_1:reg_block, gfx10_soffset:reg_block, MODIFIERS:str=''):
		return self.ic_pb(mubuf_base('buffer_load_sbyte_d16', None, gfx10_vaddr_5, gfx10_srsrc_1, gfx10_soffset, None, MODIFIERS))
	def buffer_load_sbyte_d16_hi(self, gfx10_vaddr_5:Union[reg_block,None], gfx10_srsrc_1:reg_block, gfx10_soffset:reg_block, MODIFIERS:str=''):
		return self.ic_pb(mubuf_base('buffer_load_sbyte_d16_hi', None, gfx10_vaddr_5, gfx10_srsrc_1, gfx10_soffset, None, MODIFIERS))
	def buffer_load_short_d16(self, gfx10_vaddr_5:Union[reg_block,None], gfx10_srsrc_1:reg_block, gfx10_soffset:reg_block, MODIFIERS:str=''):
		return self.ic_pb(mubuf_base('buffer_load_short_d16', None, gfx10_vaddr_5, gfx10_srsrc_1, gfx10_soffset, None, MODIFIERS))
	def buffer_load_short_d16_hi(self, gfx10_vaddr_5:Union[reg_block,None], gfx10_srsrc_1:reg_block, gfx10_soffset:reg_block, MODIFIERS:str=''):
		return self.ic_pb(mubuf_base('buffer_load_short_d16_hi', None, gfx10_vaddr_5, gfx10_srsrc_1, gfx10_soffset, None, MODIFIERS))
	def buffer_load_sshort(self, gfx10_vaddr_5:Union[reg_block,None], gfx10_srsrc_1:reg_block, gfx10_soffset:reg_block, MODIFIERS:str=''):
		return self.ic_pb(mubuf_base('buffer_load_sshort', None, gfx10_vaddr_5, gfx10_srsrc_1, gfx10_soffset, None, MODIFIERS))
	def buffer_load_ubyte(self, gfx10_vaddr_5:Union[reg_block,None], gfx10_srsrc_1:reg_block, gfx10_soffset:reg_block, MODIFIERS:str=''):
		return self.ic_pb(mubuf_base('buffer_load_ubyte', None, gfx10_vaddr_5, gfx10_srsrc_1, gfx10_soffset, None, MODIFIERS))
	def buffer_load_ubyte_d16(self, gfx10_vaddr_5:Union[reg_block,None], gfx10_srsrc_1:reg_block, gfx10_soffset:reg_block, MODIFIERS:str=''):
		return self.ic_pb(mubuf_base('buffer_load_ubyte_d16', None, gfx10_vaddr_5, gfx10_srsrc_1, gfx10_soffset, None, MODIFIERS))
	def buffer_load_ubyte_d16_hi(self, gfx10_vaddr_5:Union[reg_block,None], gfx10_srsrc_1:reg_block, gfx10_soffset:reg_block, MODIFIERS:str=''):
		return self.ic_pb(mubuf_base('buffer_load_ubyte_d16_hi', None, gfx10_vaddr_5, gfx10_srsrc_1, gfx10_soffset, None, MODIFIERS))
	def buffer_load_ushort(self, gfx10_vaddr_5:Union[reg_block,None], gfx10_srsrc_1:reg_block, gfx10_soffset:reg_block, MODIFIERS:str=''):
		return self.ic_pb(mubuf_base('buffer_load_ushort', None, gfx10_vaddr_5, gfx10_srsrc_1, gfx10_soffset, None, MODIFIERS))
	def buffer_store_byte(self, gfx10_vaddr_5:Union[reg_block,None], gfx10_srsrc_1:reg_block, gfx10_soffset:reg_block, MODIFIERS:str=''):
		return self.ic_pb(mubuf_base('buffer_store_byte', None, None, gfx10_vaddr_5, gfx10_srsrc_1, gfx10_soffset, MODIFIERS))
	def buffer_store_byte_d16_hi(self, gfx10_vaddr_5:Union[reg_block,None], gfx10_srsrc_1:reg_block, gfx10_soffset:reg_block, MODIFIERS:str=''):
		return self.ic_pb(mubuf_base('buffer_store_byte_d16_hi', None, None, gfx10_vaddr_5, gfx10_srsrc_1, gfx10_soffset, MODIFIERS))
	def buffer_store_dword(self, gfx10_vaddr_5:Union[reg_block,None], gfx10_srsrc_1:reg_block, gfx10_soffset:reg_block, MODIFIERS:str=''):
		return self.ic_pb(mubuf_base('buffer_store_dword', None, None, gfx10_vaddr_5, gfx10_srsrc_1, gfx10_soffset, MODIFIERS))
	def buffer_store_dwordx2(self, gfx10_vaddr_5:Union[reg_block,None], gfx10_srsrc_1:reg_block, gfx10_soffset:reg_block, MODIFIERS:str=''):
		return self.ic_pb(mubuf_base('buffer_store_dwordx2', None, None, gfx10_vaddr_5, gfx10_srsrc_1, gfx10_soffset, MODIFIERS))
	def buffer_store_dwordx3(self, gfx10_vaddr_5:Union[reg_block,None], gfx10_srsrc_1:reg_block, gfx10_soffset:reg_block, MODIFIERS:str=''):
		return self.ic_pb(mubuf_base('buffer_store_dwordx3', None, None, gfx10_vaddr_5, gfx10_srsrc_1, gfx10_soffset, MODIFIERS))
	def buffer_store_dwordx4(self, gfx10_vaddr_5:Union[reg_block,None], gfx10_srsrc_1:reg_block, gfx10_soffset:reg_block, MODIFIERS:str=''):
		return self.ic_pb(mubuf_base('buffer_store_dwordx4', None, None, gfx10_vaddr_5, gfx10_srsrc_1, gfx10_soffset, MODIFIERS))
	def buffer_store_format_d16_x(self, gfx10_vaddr_5:Union[reg_block,None], gfx10_srsrc_1:reg_block, gfx10_soffset:reg_block, MODIFIERS:str=''):
		return self.ic_pb(mubuf_base('buffer_store_format_d16_x', None, None, gfx10_vaddr_5, gfx10_srsrc_1, gfx10_soffset, MODIFIERS))
	def buffer_store_format_d16_xy(self, gfx10_vaddr_5:Union[reg_block,None], gfx10_srsrc_1:reg_block, gfx10_soffset:reg_block, MODIFIERS:str=''):
		return self.ic_pb(mubuf_base('buffer_store_format_d16_xy', None, None, gfx10_vaddr_5, gfx10_srsrc_1, gfx10_soffset, MODIFIERS))
	def buffer_store_format_d16_xyz(self, gfx10_vaddr_5:Union[reg_block,None], gfx10_srsrc_1:reg_block, gfx10_soffset:reg_block, MODIFIERS:str=''):
		return self.ic_pb(mubuf_base('buffer_store_format_d16_xyz', None, None, gfx10_vaddr_5, gfx10_srsrc_1, gfx10_soffset, MODIFIERS))
	def buffer_store_format_d16_xyzw(self, gfx10_vaddr_5:Union[reg_block,None], gfx10_srsrc_1:reg_block, gfx10_soffset:reg_block, MODIFIERS:str=''):
		return self.ic_pb(mubuf_base('buffer_store_format_d16_xyzw', None, None, gfx10_vaddr_5, gfx10_srsrc_1, gfx10_soffset, MODIFIERS))
	def buffer_store_format_x(self, gfx10_vaddr_5:Union[reg_block,None], gfx10_srsrc_1:reg_block, gfx10_soffset:reg_block, MODIFIERS:str=''):
		return self.ic_pb(mubuf_base('buffer_store_format_x', None, None, gfx10_vaddr_5, gfx10_srsrc_1, gfx10_soffset, MODIFIERS))
	def buffer_store_format_xy(self, gfx10_vaddr_5:Union[reg_block,None], gfx10_srsrc_1:reg_block, gfx10_soffset:reg_block, MODIFIERS:str=''):
		return self.ic_pb(mubuf_base('buffer_store_format_xy', None, None, gfx10_vaddr_5, gfx10_srsrc_1, gfx10_soffset, MODIFIERS))
	def buffer_store_format_xyz(self, gfx10_vaddr_5:Union[reg_block,None], gfx10_srsrc_1:reg_block, gfx10_soffset:reg_block, MODIFIERS:str=''):
		return self.ic_pb(mubuf_base('buffer_store_format_xyz', None, None, gfx10_vaddr_5, gfx10_srsrc_1, gfx10_soffset, MODIFIERS))
	def buffer_store_format_xyzw(self, gfx10_vaddr_5:Union[reg_block,None], gfx10_srsrc_1:reg_block, gfx10_soffset:reg_block, MODIFIERS:str=''):
		return self.ic_pb(mubuf_base('buffer_store_format_xyzw', None, None, gfx10_vaddr_5, gfx10_srsrc_1, gfx10_soffset, MODIFIERS))
	def buffer_store_short(self, gfx10_vaddr_5:Union[reg_block,None], gfx10_srsrc_1:reg_block, gfx10_soffset:reg_block, MODIFIERS:str=''):
		return self.ic_pb(mubuf_base('buffer_store_short', None, None, gfx10_vaddr_5, gfx10_srsrc_1, gfx10_soffset, MODIFIERS))
	def buffer_store_short_d16_hi(self, gfx10_vaddr_5:Union[reg_block,None], gfx10_srsrc_1:reg_block, gfx10_soffset:reg_block, MODIFIERS:str=''):
		return self.ic_pb(mubuf_base('buffer_store_short_d16_hi', None, None, gfx10_vaddr_5, gfx10_srsrc_1, gfx10_soffset, MODIFIERS))
class sdwa_base(inst_base): 
	def __init__(self, INSTRUCTION:str, DST0:Union[reg_block,None], DST1:Union[reg_block,None], SRC0:Union[reg_block,None], SRC1:Union[reg_block,None], SRC2:Union[reg_block,None], MODIFIERS:str): 
		super().__init__(instruction_type.SMEM, INSTRUCTION)
		self.DST0 = DST0 
		self.DST1 = DST1 
		self.SRC0 = SRC0 
		self.SRC1 = SRC1 
		self.SRC2 = SRC2 
		self.MODIFIERS = MODIFIERS 
	def __str__(self): 
		args_l = filter(None.__ne__, [self.DST0,self.DST1,self.SRC0,self.SRC1,self.SRC2]) 
		return f"{self.label} {','.join(map(str, args_l))} {self.MODIFIERS}" 
class sdwa_instr_caller(inst_caller_base): 
	def __init__(self, insturction_container) -> None:
     		super().__init__(insturction_container)
	def v_add_co_ci_u32_sdwa(self, gfx10_vcc:reg_block, gfx10_src:reg_block, gfx10_src:reg_block, gfx10_vcc:reg_block, MODIFIERS:str=''):
		return self.ic_pb(sdwa_base('v_add_co_ci_u32_sdwa', None, gfx10_vcc, gfx10_src, gfx10_src, gfx10_vcc, MODIFIERS))
	def v_add_f16_sdwa(self, gfx10_src:reg_block, gfx10_src:reg_block, MODIFIERS:str=''):
		return self.ic_pb(sdwa_base('v_add_f16_sdwa', None, None, gfx10_src, gfx10_src, None, MODIFIERS))
	def v_add_f32_sdwa(self, gfx10_src:reg_block, gfx10_src:reg_block, MODIFIERS:str=''):
		return self.ic_pb(sdwa_base('v_add_f32_sdwa', None, None, gfx10_src, gfx10_src, None, MODIFIERS))
	def v_add_nc_u32_sdwa(self, gfx10_src:reg_block, gfx10_src:reg_block, MODIFIERS:str=''):
		return self.ic_pb(sdwa_base('v_add_nc_u32_sdwa', None, None, gfx10_src, gfx10_src, None, MODIFIERS))
	def v_and_b32_sdwa(self, gfx10_src:reg_block, gfx10_src:reg_block, MODIFIERS:str=''):
		return self.ic_pb(sdwa_base('v_and_b32_sdwa', None, None, gfx10_src, gfx10_src, None, MODIFIERS))
	def v_ashrrev_i32_sdwa(self, gfx10_src:reg_block, gfx10_src:reg_block, MODIFIERS:str=''):
		return self.ic_pb(sdwa_base('v_ashrrev_i32_sdwa', None, None, gfx10_src, gfx10_src, None, MODIFIERS))
	def v_bfrev_b32_sdwa(self, gfx10_src:reg_block, MODIFIERS:str=''):
		return self.ic_pb(sdwa_base('v_bfrev_b32_sdwa', None, None, gfx10_src, None, None, MODIFIERS))
	def v_ceil_f16_sdwa(self, gfx10_src:reg_block, MODIFIERS:str=''):
		return self.ic_pb(sdwa_base('v_ceil_f16_sdwa', None, None, gfx10_src, None, None, MODIFIERS))
	def v_ceil_f32_sdwa(self, gfx10_src:reg_block, MODIFIERS:str=''):
		return self.ic_pb(sdwa_base('v_ceil_f32_sdwa', None, None, gfx10_src, None, None, MODIFIERS))
	def v_cmp_class_f16_sdwa(self, gfx10_src:reg_block, gfx10_src:reg_block, MODIFIERS:str=''):
		return self.ic_pb(sdwa_base('v_cmp_class_f16_sdwa', None, None, gfx10_src, gfx10_src, None, MODIFIERS))
	def v_cmp_class_f32_sdwa(self, gfx10_src:reg_block, gfx10_src:reg_block, MODIFIERS:str=''):
		return self.ic_pb(sdwa_base('v_cmp_class_f32_sdwa', None, None, gfx10_src, gfx10_src, None, MODIFIERS))
	def v_cmp_eq_f16_sdwa(self, gfx10_src:reg_block, gfx10_src:reg_block, MODIFIERS:str=''):
		return self.ic_pb(sdwa_base('v_cmp_eq_f16_sdwa', None, None, gfx10_src, gfx10_src, None, MODIFIERS))
	def v_cmp_eq_f32_sdwa(self, gfx10_src:reg_block, gfx10_src:reg_block, MODIFIERS:str=''):
		return self.ic_pb(sdwa_base('v_cmp_eq_f32_sdwa', None, None, gfx10_src, gfx10_src, None, MODIFIERS))
	def v_cmp_eq_i16_sdwa(self, gfx10_src_1:reg_block, gfx10_src_1:reg_block, MODIFIERS:str=''):
		return self.ic_pb(sdwa_base('v_cmp_eq_i16_sdwa', None, None, gfx10_src_1, gfx10_src_1, None, MODIFIERS))
	def v_cmp_eq_i32_sdwa(self, gfx10_src:reg_block, gfx10_src:reg_block, MODIFIERS:str=''):
		return self.ic_pb(sdwa_base('v_cmp_eq_i32_sdwa', None, None, gfx10_src, gfx10_src, None, MODIFIERS))
	def v_cmp_eq_u16_sdwa(self, gfx10_src_1:reg_block, gfx10_src_1:reg_block, MODIFIERS:str=''):
		return self.ic_pb(sdwa_base('v_cmp_eq_u16_sdwa', None, None, gfx10_src_1, gfx10_src_1, None, MODIFIERS))
	def v_cmp_eq_u32_sdwa(self, gfx10_src:reg_block, gfx10_src:reg_block, MODIFIERS:str=''):
		return self.ic_pb(sdwa_base('v_cmp_eq_u32_sdwa', None, None, gfx10_src, gfx10_src, None, MODIFIERS))
	def v_cmp_f_f16_sdwa(self, gfx10_src:reg_block, gfx10_src:reg_block, MODIFIERS:str=''):
		return self.ic_pb(sdwa_base('v_cmp_f_f16_sdwa', None, None, gfx10_src, gfx10_src, None, MODIFIERS))
	def v_cmp_f_f32_sdwa(self, gfx10_src:reg_block, gfx10_src:reg_block, MODIFIERS:str=''):
		return self.ic_pb(sdwa_base('v_cmp_f_f32_sdwa', None, None, gfx10_src, gfx10_src, None, MODIFIERS))
	def v_cmp_f_i32_sdwa(self, gfx10_src:reg_block, gfx10_src:reg_block, MODIFIERS:str=''):
		return self.ic_pb(sdwa_base('v_cmp_f_i32_sdwa', None, None, gfx10_src, gfx10_src, None, MODIFIERS))
	def v_cmp_f_u32_sdwa(self, gfx10_src:reg_block, gfx10_src:reg_block, MODIFIERS:str=''):
		return self.ic_pb(sdwa_base('v_cmp_f_u32_sdwa', None, None, gfx10_src, gfx10_src, None, MODIFIERS))
	def v_cmp_ge_f16_sdwa(self, gfx10_src:reg_block, gfx10_src:reg_block, MODIFIERS:str=''):
		return self.ic_pb(sdwa_base('v_cmp_ge_f16_sdwa', None, None, gfx10_src, gfx10_src, None, MODIFIERS))
	def v_cmp_ge_f32_sdwa(self, gfx10_src:reg_block, gfx10_src:reg_block, MODIFIERS:str=''):
		return self.ic_pb(sdwa_base('v_cmp_ge_f32_sdwa', None, None, gfx10_src, gfx10_src, None, MODIFIERS))
	def v_cmp_ge_i16_sdwa(self, gfx10_src_1:reg_block, gfx10_src_1:reg_block, MODIFIERS:str=''):
		return self.ic_pb(sdwa_base('v_cmp_ge_i16_sdwa', None, None, gfx10_src_1, gfx10_src_1, None, MODIFIERS))
	def v_cmp_ge_i32_sdwa(self, gfx10_src:reg_block, gfx10_src:reg_block, MODIFIERS:str=''):
		return self.ic_pb(sdwa_base('v_cmp_ge_i32_sdwa', None, None, gfx10_src, gfx10_src, None, MODIFIERS))
	def v_cmp_ge_u16_sdwa(self, gfx10_src_1:reg_block, gfx10_src_1:reg_block, MODIFIERS:str=''):
		return self.ic_pb(sdwa_base('v_cmp_ge_u16_sdwa', None, None, gfx10_src_1, gfx10_src_1, None, MODIFIERS))
	def v_cmp_ge_u32_sdwa(self, gfx10_src:reg_block, gfx10_src:reg_block, MODIFIERS:str=''):
		return self.ic_pb(sdwa_base('v_cmp_ge_u32_sdwa', None, None, gfx10_src, gfx10_src, None, MODIFIERS))
	def v_cmp_gt_f16_sdwa(self, gfx10_src:reg_block, gfx10_src:reg_block, MODIFIERS:str=''):
		return self.ic_pb(sdwa_base('v_cmp_gt_f16_sdwa', None, None, gfx10_src, gfx10_src, None, MODIFIERS))
	def v_cmp_gt_f32_sdwa(self, gfx10_src:reg_block, gfx10_src:reg_block, MODIFIERS:str=''):
		return self.ic_pb(sdwa_base('v_cmp_gt_f32_sdwa', None, None, gfx10_src, gfx10_src, None, MODIFIERS))
	def v_cmp_gt_i16_sdwa(self, gfx10_src_1:reg_block, gfx10_src_1:reg_block, MODIFIERS:str=''):
		return self.ic_pb(sdwa_base('v_cmp_gt_i16_sdwa', None, None, gfx10_src_1, gfx10_src_1, None, MODIFIERS))
	def v_cmp_gt_i32_sdwa(self, gfx10_src:reg_block, gfx10_src:reg_block, MODIFIERS:str=''):
		return self.ic_pb(sdwa_base('v_cmp_gt_i32_sdwa', None, None, gfx10_src, gfx10_src, None, MODIFIERS))
	def v_cmp_gt_u16_sdwa(self, gfx10_src_1:reg_block, gfx10_src_1:reg_block, MODIFIERS:str=''):
		return self.ic_pb(sdwa_base('v_cmp_gt_u16_sdwa', None, None, gfx10_src_1, gfx10_src_1, None, MODIFIERS))
	def v_cmp_gt_u32_sdwa(self, gfx10_src:reg_block, gfx10_src:reg_block, MODIFIERS:str=''):
		return self.ic_pb(sdwa_base('v_cmp_gt_u32_sdwa', None, None, gfx10_src, gfx10_src, None, MODIFIERS))
	def v_cmp_le_f16_sdwa(self, gfx10_src:reg_block, gfx10_src:reg_block, MODIFIERS:str=''):
		return self.ic_pb(sdwa_base('v_cmp_le_f16_sdwa', None, None, gfx10_src, gfx10_src, None, MODIFIERS))
	def v_cmp_le_f32_sdwa(self, gfx10_src:reg_block, gfx10_src:reg_block, MODIFIERS:str=''):
		return self.ic_pb(sdwa_base('v_cmp_le_f32_sdwa', None, None, gfx10_src, gfx10_src, None, MODIFIERS))
	def v_cmp_le_i16_sdwa(self, gfx10_src_1:reg_block, gfx10_src_1:reg_block, MODIFIERS:str=''):
		return self.ic_pb(sdwa_base('v_cmp_le_i16_sdwa', None, None, gfx10_src_1, gfx10_src_1, None, MODIFIERS))
	def v_cmp_le_i32_sdwa(self, gfx10_src:reg_block, gfx10_src:reg_block, MODIFIERS:str=''):
		return self.ic_pb(sdwa_base('v_cmp_le_i32_sdwa', None, None, gfx10_src, gfx10_src, None, MODIFIERS))
	def v_cmp_le_u16_sdwa(self, gfx10_src_1:reg_block, gfx10_src_1:reg_block, MODIFIERS:str=''):
		return self.ic_pb(sdwa_base('v_cmp_le_u16_sdwa', None, None, gfx10_src_1, gfx10_src_1, None, MODIFIERS))
	def v_cmp_le_u32_sdwa(self, gfx10_src:reg_block, gfx10_src:reg_block, MODIFIERS:str=''):
		return self.ic_pb(sdwa_base('v_cmp_le_u32_sdwa', None, None, gfx10_src, gfx10_src, None, MODIFIERS))
	def v_cmp_lg_f16_sdwa(self, gfx10_src:reg_block, gfx10_src:reg_block, MODIFIERS:str=''):
		return self.ic_pb(sdwa_base('v_cmp_lg_f16_sdwa', None, None, gfx10_src, gfx10_src, None, MODIFIERS))
	def v_cmp_lg_f32_sdwa(self, gfx10_src:reg_block, gfx10_src:reg_block, MODIFIERS:str=''):
		return self.ic_pb(sdwa_base('v_cmp_lg_f32_sdwa', None, None, gfx10_src, gfx10_src, None, MODIFIERS))
	def v_cmp_lt_f16_sdwa(self, gfx10_src:reg_block, gfx10_src:reg_block, MODIFIERS:str=''):
		return self.ic_pb(sdwa_base('v_cmp_lt_f16_sdwa', None, None, gfx10_src, gfx10_src, None, MODIFIERS))
	def v_cmp_lt_f32_sdwa(self, gfx10_src:reg_block, gfx10_src:reg_block, MODIFIERS:str=''):
		return self.ic_pb(sdwa_base('v_cmp_lt_f32_sdwa', None, None, gfx10_src, gfx10_src, None, MODIFIERS))
	def v_cmp_lt_i16_sdwa(self, gfx10_src_1:reg_block, gfx10_src_1:reg_block, MODIFIERS:str=''):
		return self.ic_pb(sdwa_base('v_cmp_lt_i16_sdwa', None, None, gfx10_src_1, gfx10_src_1, None, MODIFIERS))
	def v_cmp_lt_i32_sdwa(self, gfx10_src:reg_block, gfx10_src:reg_block, MODIFIERS:str=''):
		return self.ic_pb(sdwa_base('v_cmp_lt_i32_sdwa', None, None, gfx10_src, gfx10_src, None, MODIFIERS))
	def v_cmp_lt_u16_sdwa(self, gfx10_src_1:reg_block, gfx10_src_1:reg_block, MODIFIERS:str=''):
		return self.ic_pb(sdwa_base('v_cmp_lt_u16_sdwa', None, None, gfx10_src_1, gfx10_src_1, None, MODIFIERS))
	def v_cmp_lt_u32_sdwa(self, gfx10_src:reg_block, gfx10_src:reg_block, MODIFIERS:str=''):
		return self.ic_pb(sdwa_base('v_cmp_lt_u32_sdwa', None, None, gfx10_src, gfx10_src, None, MODIFIERS))
	def v_cmp_ne_i16_sdwa(self, gfx10_src_1:reg_block, gfx10_src_1:reg_block, MODIFIERS:str=''):
		return self.ic_pb(sdwa_base('v_cmp_ne_i16_sdwa', None, None, gfx10_src_1, gfx10_src_1, None, MODIFIERS))
	def v_cmp_ne_i32_sdwa(self, gfx10_src:reg_block, gfx10_src:reg_block, MODIFIERS:str=''):
		return self.ic_pb(sdwa_base('v_cmp_ne_i32_sdwa', None, None, gfx10_src, gfx10_src, None, MODIFIERS))
	def v_cmp_ne_u16_sdwa(self, gfx10_src_1:reg_block, gfx10_src_1:reg_block, MODIFIERS:str=''):
		return self.ic_pb(sdwa_base('v_cmp_ne_u16_sdwa', None, None, gfx10_src_1, gfx10_src_1, None, MODIFIERS))
	def v_cmp_ne_u32_sdwa(self, gfx10_src:reg_block, gfx10_src:reg_block, MODIFIERS:str=''):
		return self.ic_pb(sdwa_base('v_cmp_ne_u32_sdwa', None, None, gfx10_src, gfx10_src, None, MODIFIERS))
	def v_cmp_neq_f16_sdwa(self, gfx10_src:reg_block, gfx10_src:reg_block, MODIFIERS:str=''):
		return self.ic_pb(sdwa_base('v_cmp_neq_f16_sdwa', None, None, gfx10_src, gfx10_src, None, MODIFIERS))
	def v_cmp_neq_f32_sdwa(self, gfx10_src:reg_block, gfx10_src:reg_block, MODIFIERS:str=''):
		return self.ic_pb(sdwa_base('v_cmp_neq_f32_sdwa', None, None, gfx10_src, gfx10_src, None, MODIFIERS))
	def v_cmp_nge_f16_sdwa(self, gfx10_src:reg_block, gfx10_src:reg_block, MODIFIERS:str=''):
		return self.ic_pb(sdwa_base('v_cmp_nge_f16_sdwa', None, None, gfx10_src, gfx10_src, None, MODIFIERS))
	def v_cmp_nge_f32_sdwa(self, gfx10_src:reg_block, gfx10_src:reg_block, MODIFIERS:str=''):
		return self.ic_pb(sdwa_base('v_cmp_nge_f32_sdwa', None, None, gfx10_src, gfx10_src, None, MODIFIERS))
	def v_cmp_ngt_f16_sdwa(self, gfx10_src:reg_block, gfx10_src:reg_block, MODIFIERS:str=''):
		return self.ic_pb(sdwa_base('v_cmp_ngt_f16_sdwa', None, None, gfx10_src, gfx10_src, None, MODIFIERS))
	def v_cmp_ngt_f32_sdwa(self, gfx10_src:reg_block, gfx10_src:reg_block, MODIFIERS:str=''):
		return self.ic_pb(sdwa_base('v_cmp_ngt_f32_sdwa', None, None, gfx10_src, gfx10_src, None, MODIFIERS))
	def v_cmp_nle_f16_sdwa(self, gfx10_src:reg_block, gfx10_src:reg_block, MODIFIERS:str=''):
		return self.ic_pb(sdwa_base('v_cmp_nle_f16_sdwa', None, None, gfx10_src, gfx10_src, None, MODIFIERS))
	def v_cmp_nle_f32_sdwa(self, gfx10_src:reg_block, gfx10_src:reg_block, MODIFIERS:str=''):
		return self.ic_pb(sdwa_base('v_cmp_nle_f32_sdwa', None, None, gfx10_src, gfx10_src, None, MODIFIERS))
	def v_cmp_nlg_f16_sdwa(self, gfx10_src:reg_block, gfx10_src:reg_block, MODIFIERS:str=''):
		return self.ic_pb(sdwa_base('v_cmp_nlg_f16_sdwa', None, None, gfx10_src, gfx10_src, None, MODIFIERS))
	def v_cmp_nlg_f32_sdwa(self, gfx10_src:reg_block, gfx10_src:reg_block, MODIFIERS:str=''):
		return self.ic_pb(sdwa_base('v_cmp_nlg_f32_sdwa', None, None, gfx10_src, gfx10_src, None, MODIFIERS))
	def v_cmp_nlt_f16_sdwa(self, gfx10_src:reg_block, gfx10_src:reg_block, MODIFIERS:str=''):
		return self.ic_pb(sdwa_base('v_cmp_nlt_f16_sdwa', None, None, gfx10_src, gfx10_src, None, MODIFIERS))
	def v_cmp_nlt_f32_sdwa(self, gfx10_src:reg_block, gfx10_src:reg_block, MODIFIERS:str=''):
		return self.ic_pb(sdwa_base('v_cmp_nlt_f32_sdwa', None, None, gfx10_src, gfx10_src, None, MODIFIERS))
	def v_cmp_o_f16_sdwa(self, gfx10_src:reg_block, gfx10_src:reg_block, MODIFIERS:str=''):
		return self.ic_pb(sdwa_base('v_cmp_o_f16_sdwa', None, None, gfx10_src, gfx10_src, None, MODIFIERS))
	def v_cmp_o_f32_sdwa(self, gfx10_src:reg_block, gfx10_src:reg_block, MODIFIERS:str=''):
		return self.ic_pb(sdwa_base('v_cmp_o_f32_sdwa', None, None, gfx10_src, gfx10_src, None, MODIFIERS))
	def v_cmp_t_i32_sdwa(self, gfx10_src:reg_block, gfx10_src:reg_block, MODIFIERS:str=''):
		return self.ic_pb(sdwa_base('v_cmp_t_i32_sdwa', None, None, gfx10_src, gfx10_src, None, MODIFIERS))
	def v_cmp_t_u32_sdwa(self, gfx10_src:reg_block, gfx10_src:reg_block, MODIFIERS:str=''):
		return self.ic_pb(sdwa_base('v_cmp_t_u32_sdwa', None, None, gfx10_src, gfx10_src, None, MODIFIERS))
	def v_cmp_tru_f16_sdwa(self, gfx10_src:reg_block, gfx10_src:reg_block, MODIFIERS:str=''):
		return self.ic_pb(sdwa_base('v_cmp_tru_f16_sdwa', None, None, gfx10_src, gfx10_src, None, MODIFIERS))
	def v_cmp_tru_f32_sdwa(self, gfx10_src:reg_block, gfx10_src:reg_block, MODIFIERS:str=''):
		return self.ic_pb(sdwa_base('v_cmp_tru_f32_sdwa', None, None, gfx10_src, gfx10_src, None, MODIFIERS))
	def v_cmp_u_f16_sdwa(self, gfx10_src:reg_block, gfx10_src:reg_block, MODIFIERS:str=''):
		return self.ic_pb(sdwa_base('v_cmp_u_f16_sdwa', None, None, gfx10_src, gfx10_src, None, MODIFIERS))
	def v_cmp_u_f32_sdwa(self, gfx10_src:reg_block, gfx10_src:reg_block, MODIFIERS:str=''):
		return self.ic_pb(sdwa_base('v_cmp_u_f32_sdwa', None, None, gfx10_src, gfx10_src, None, MODIFIERS))
	def v_cmpx_class_f16_sdwa(self, gfx10_src:reg_block, MODIFIERS:str=''):
		return self.ic_pb(sdwa_base('v_cmpx_class_f16_sdwa', None, None, None, gfx10_src, None, MODIFIERS))
	def v_cmpx_class_f32_sdwa(self, gfx10_src:reg_block, MODIFIERS:str=''):
		return self.ic_pb(sdwa_base('v_cmpx_class_f32_sdwa', None, None, None, gfx10_src, None, MODIFIERS))
	def v_cmpx_eq_f16_sdwa(self, gfx10_src:reg_block, MODIFIERS:str=''):
		return self.ic_pb(sdwa_base('v_cmpx_eq_f16_sdwa', None, None, None, gfx10_src, None, MODIFIERS))
	def v_cmpx_eq_f32_sdwa(self, gfx10_src:reg_block, MODIFIERS:str=''):
		return self.ic_pb(sdwa_base('v_cmpx_eq_f32_sdwa', None, None, None, gfx10_src, None, MODIFIERS))
	def v_cmpx_eq_i16_sdwa(self, gfx10_src_1:reg_block, MODIFIERS:str=''):
		return self.ic_pb(sdwa_base('v_cmpx_eq_i16_sdwa', None, None, None, gfx10_src_1, None, MODIFIERS))
	def v_cmpx_eq_i32_sdwa(self, gfx10_src:reg_block, MODIFIERS:str=''):
		return self.ic_pb(sdwa_base('v_cmpx_eq_i32_sdwa', None, None, None, gfx10_src, None, MODIFIERS))
	def v_cmpx_eq_u16_sdwa(self, gfx10_src_1:reg_block, MODIFIERS:str=''):
		return self.ic_pb(sdwa_base('v_cmpx_eq_u16_sdwa', None, None, None, gfx10_src_1, None, MODIFIERS))
	def v_cmpx_eq_u32_sdwa(self, gfx10_src:reg_block, MODIFIERS:str=''):
		return self.ic_pb(sdwa_base('v_cmpx_eq_u32_sdwa', None, None, None, gfx10_src, None, MODIFIERS))
	def v_cmpx_f_f16_sdwa(self, gfx10_src:reg_block, MODIFIERS:str=''):
		return self.ic_pb(sdwa_base('v_cmpx_f_f16_sdwa', None, None, None, gfx10_src, None, MODIFIERS))
	def v_cmpx_f_f32_sdwa(self, gfx10_src:reg_block, MODIFIERS:str=''):
		return self.ic_pb(sdwa_base('v_cmpx_f_f32_sdwa', None, None, None, gfx10_src, None, MODIFIERS))
	def v_cmpx_f_i32_sdwa(self, gfx10_src:reg_block, MODIFIERS:str=''):
		return self.ic_pb(sdwa_base('v_cmpx_f_i32_sdwa', None, None, None, gfx10_src, None, MODIFIERS))
	def v_cmpx_f_u32_sdwa(self, gfx10_src:reg_block, MODIFIERS:str=''):
		return self.ic_pb(sdwa_base('v_cmpx_f_u32_sdwa', None, None, None, gfx10_src, None, MODIFIERS))
	def v_cmpx_ge_f16_sdwa(self, gfx10_src:reg_block, MODIFIERS:str=''):
		return self.ic_pb(sdwa_base('v_cmpx_ge_f16_sdwa', None, None, None, gfx10_src, None, MODIFIERS))
	def v_cmpx_ge_f32_sdwa(self, gfx10_src:reg_block, MODIFIERS:str=''):
		return self.ic_pb(sdwa_base('v_cmpx_ge_f32_sdwa', None, None, None, gfx10_src, None, MODIFIERS))
	def v_cmpx_ge_i16_sdwa(self, gfx10_src_1:reg_block, MODIFIERS:str=''):
		return self.ic_pb(sdwa_base('v_cmpx_ge_i16_sdwa', None, None, None, gfx10_src_1, None, MODIFIERS))
	def v_cmpx_ge_i32_sdwa(self, gfx10_src:reg_block, MODIFIERS:str=''):
		return self.ic_pb(sdwa_base('v_cmpx_ge_i32_sdwa', None, None, None, gfx10_src, None, MODIFIERS))
	def v_cmpx_ge_u16_sdwa(self, gfx10_src_1:reg_block, MODIFIERS:str=''):
		return self.ic_pb(sdwa_base('v_cmpx_ge_u16_sdwa', None, None, None, gfx10_src_1, None, MODIFIERS))
	def v_cmpx_ge_u32_sdwa(self, gfx10_src:reg_block, MODIFIERS:str=''):
		return self.ic_pb(sdwa_base('v_cmpx_ge_u32_sdwa', None, None, None, gfx10_src, None, MODIFIERS))
	def v_cmpx_gt_f16_sdwa(self, gfx10_src:reg_block, MODIFIERS:str=''):
		return self.ic_pb(sdwa_base('v_cmpx_gt_f16_sdwa', None, None, None, gfx10_src, None, MODIFIERS))
	def v_cmpx_gt_f32_sdwa(self, gfx10_src:reg_block, MODIFIERS:str=''):
		return self.ic_pb(sdwa_base('v_cmpx_gt_f32_sdwa', None, None, None, gfx10_src, None, MODIFIERS))
	def v_cmpx_gt_i16_sdwa(self, gfx10_src_1:reg_block, MODIFIERS:str=''):
		return self.ic_pb(sdwa_base('v_cmpx_gt_i16_sdwa', None, None, None, gfx10_src_1, None, MODIFIERS))
	def v_cmpx_gt_i32_sdwa(self, gfx10_src:reg_block, MODIFIERS:str=''):
		return self.ic_pb(sdwa_base('v_cmpx_gt_i32_sdwa', None, None, None, gfx10_src, None, MODIFIERS))
	def v_cmpx_gt_u16_sdwa(self, gfx10_src_1:reg_block, MODIFIERS:str=''):
		return self.ic_pb(sdwa_base('v_cmpx_gt_u16_sdwa', None, None, None, gfx10_src_1, None, MODIFIERS))
	def v_cmpx_gt_u32_sdwa(self, gfx10_src:reg_block, MODIFIERS:str=''):
		return self.ic_pb(sdwa_base('v_cmpx_gt_u32_sdwa', None, None, None, gfx10_src, None, MODIFIERS))
	def v_cmpx_le_f16_sdwa(self, gfx10_src:reg_block, MODIFIERS:str=''):
		return self.ic_pb(sdwa_base('v_cmpx_le_f16_sdwa', None, None, None, gfx10_src, None, MODIFIERS))
	def v_cmpx_le_f32_sdwa(self, gfx10_src:reg_block, MODIFIERS:str=''):
		return self.ic_pb(sdwa_base('v_cmpx_le_f32_sdwa', None, None, None, gfx10_src, None, MODIFIERS))
	def v_cmpx_le_i16_sdwa(self, gfx10_src_1:reg_block, MODIFIERS:str=''):
		return self.ic_pb(sdwa_base('v_cmpx_le_i16_sdwa', None, None, None, gfx10_src_1, None, MODIFIERS))
	def v_cmpx_le_i32_sdwa(self, gfx10_src:reg_block, MODIFIERS:str=''):
		return self.ic_pb(sdwa_base('v_cmpx_le_i32_sdwa', None, None, None, gfx10_src, None, MODIFIERS))
	def v_cmpx_le_u16_sdwa(self, gfx10_src_1:reg_block, MODIFIERS:str=''):
		return self.ic_pb(sdwa_base('v_cmpx_le_u16_sdwa', None, None, None, gfx10_src_1, None, MODIFIERS))
	def v_cmpx_le_u32_sdwa(self, gfx10_src:reg_block, MODIFIERS:str=''):
		return self.ic_pb(sdwa_base('v_cmpx_le_u32_sdwa', None, None, None, gfx10_src, None, MODIFIERS))
	def v_cmpx_lg_f16_sdwa(self, gfx10_src:reg_block, MODIFIERS:str=''):
		return self.ic_pb(sdwa_base('v_cmpx_lg_f16_sdwa', None, None, None, gfx10_src, None, MODIFIERS))
	def v_cmpx_lg_f32_sdwa(self, gfx10_src:reg_block, MODIFIERS:str=''):
		return self.ic_pb(sdwa_base('v_cmpx_lg_f32_sdwa', None, None, None, gfx10_src, None, MODIFIERS))
	def v_cmpx_lt_f16_sdwa(self, gfx10_src:reg_block, MODIFIERS:str=''):
		return self.ic_pb(sdwa_base('v_cmpx_lt_f16_sdwa', None, None, None, gfx10_src, None, MODIFIERS))
	def v_cmpx_lt_f32_sdwa(self, gfx10_src:reg_block, MODIFIERS:str=''):
		return self.ic_pb(sdwa_base('v_cmpx_lt_f32_sdwa', None, None, None, gfx10_src, None, MODIFIERS))
	def v_cmpx_lt_i16_sdwa(self, gfx10_src_1:reg_block, MODIFIERS:str=''):
		return self.ic_pb(sdwa_base('v_cmpx_lt_i16_sdwa', None, None, None, gfx10_src_1, None, MODIFIERS))
	def v_cmpx_lt_i32_sdwa(self, gfx10_src:reg_block, MODIFIERS:str=''):
		return self.ic_pb(sdwa_base('v_cmpx_lt_i32_sdwa', None, None, None, gfx10_src, None, MODIFIERS))
	def v_cmpx_lt_u16_sdwa(self, gfx10_src_1:reg_block, MODIFIERS:str=''):
		return self.ic_pb(sdwa_base('v_cmpx_lt_u16_sdwa', None, None, None, gfx10_src_1, None, MODIFIERS))
	def v_cmpx_lt_u32_sdwa(self, gfx10_src:reg_block, MODIFIERS:str=''):
		return self.ic_pb(sdwa_base('v_cmpx_lt_u32_sdwa', None, None, None, gfx10_src, None, MODIFIERS))
	def v_cmpx_ne_i16_sdwa(self, gfx10_src_1:reg_block, MODIFIERS:str=''):
		return self.ic_pb(sdwa_base('v_cmpx_ne_i16_sdwa', None, None, None, gfx10_src_1, None, MODIFIERS))
	def v_cmpx_ne_i32_sdwa(self, gfx10_src:reg_block, MODIFIERS:str=''):
		return self.ic_pb(sdwa_base('v_cmpx_ne_i32_sdwa', None, None, None, gfx10_src, None, MODIFIERS))
	def v_cmpx_ne_u16_sdwa(self, gfx10_src_1:reg_block, MODIFIERS:str=''):
		return self.ic_pb(sdwa_base('v_cmpx_ne_u16_sdwa', None, None, None, gfx10_src_1, None, MODIFIERS))
	def v_cmpx_ne_u32_sdwa(self, gfx10_src:reg_block, MODIFIERS:str=''):
		return self.ic_pb(sdwa_base('v_cmpx_ne_u32_sdwa', None, None, None, gfx10_src, None, MODIFIERS))
	def v_cmpx_neq_f16_sdwa(self, gfx10_src:reg_block, MODIFIERS:str=''):
		return self.ic_pb(sdwa_base('v_cmpx_neq_f16_sdwa', None, None, None, gfx10_src, None, MODIFIERS))
	def v_cmpx_neq_f32_sdwa(self, gfx10_src:reg_block, MODIFIERS:str=''):
		return self.ic_pb(sdwa_base('v_cmpx_neq_f32_sdwa', None, None, None, gfx10_src, None, MODIFIERS))
	def v_cmpx_nge_f16_sdwa(self, gfx10_src:reg_block, MODIFIERS:str=''):
		return self.ic_pb(sdwa_base('v_cmpx_nge_f16_sdwa', None, None, None, gfx10_src, None, MODIFIERS))
	def v_cmpx_nge_f32_sdwa(self, gfx10_src:reg_block, MODIFIERS:str=''):
		return self.ic_pb(sdwa_base('v_cmpx_nge_f32_sdwa', None, None, None, gfx10_src, None, MODIFIERS))
	def v_cmpx_ngt_f16_sdwa(self, gfx10_src:reg_block, MODIFIERS:str=''):
		return self.ic_pb(sdwa_base('v_cmpx_ngt_f16_sdwa', None, None, None, gfx10_src, None, MODIFIERS))
	def v_cmpx_ngt_f32_sdwa(self, gfx10_src:reg_block, MODIFIERS:str=''):
		return self.ic_pb(sdwa_base('v_cmpx_ngt_f32_sdwa', None, None, None, gfx10_src, None, MODIFIERS))
	def v_cmpx_nle_f16_sdwa(self, gfx10_src:reg_block, MODIFIERS:str=''):
		return self.ic_pb(sdwa_base('v_cmpx_nle_f16_sdwa', None, None, None, gfx10_src, None, MODIFIERS))
	def v_cmpx_nle_f32_sdwa(self, gfx10_src:reg_block, MODIFIERS:str=''):
		return self.ic_pb(sdwa_base('v_cmpx_nle_f32_sdwa', None, None, None, gfx10_src, None, MODIFIERS))
	def v_cmpx_nlg_f16_sdwa(self, gfx10_src:reg_block, MODIFIERS:str=''):
		return self.ic_pb(sdwa_base('v_cmpx_nlg_f16_sdwa', None, None, None, gfx10_src, None, MODIFIERS))
	def v_cmpx_nlg_f32_sdwa(self, gfx10_src:reg_block, MODIFIERS:str=''):
		return self.ic_pb(sdwa_base('v_cmpx_nlg_f32_sdwa', None, None, None, gfx10_src, None, MODIFIERS))
	def v_cmpx_nlt_f16_sdwa(self, gfx10_src:reg_block, MODIFIERS:str=''):
		return self.ic_pb(sdwa_base('v_cmpx_nlt_f16_sdwa', None, None, None, gfx10_src, None, MODIFIERS))
	def v_cmpx_nlt_f32_sdwa(self, gfx10_src:reg_block, MODIFIERS:str=''):
		return self.ic_pb(sdwa_base('v_cmpx_nlt_f32_sdwa', None, None, None, gfx10_src, None, MODIFIERS))
	def v_cmpx_o_f16_sdwa(self, gfx10_src:reg_block, MODIFIERS:str=''):
		return self.ic_pb(sdwa_base('v_cmpx_o_f16_sdwa', None, None, None, gfx10_src, None, MODIFIERS))
	def v_cmpx_o_f32_sdwa(self, gfx10_src:reg_block, MODIFIERS:str=''):
		return self.ic_pb(sdwa_base('v_cmpx_o_f32_sdwa', None, None, None, gfx10_src, None, MODIFIERS))
	def v_cmpx_t_i32_sdwa(self, gfx10_src:reg_block, MODIFIERS:str=''):
		return self.ic_pb(sdwa_base('v_cmpx_t_i32_sdwa', None, None, None, gfx10_src, None, MODIFIERS))
	def v_cmpx_t_u32_sdwa(self, gfx10_src:reg_block, MODIFIERS:str=''):
		return self.ic_pb(sdwa_base('v_cmpx_t_u32_sdwa', None, None, None, gfx10_src, None, MODIFIERS))
	def v_cmpx_tru_f16_sdwa(self, gfx10_src:reg_block, MODIFIERS:str=''):
		return self.ic_pb(sdwa_base('v_cmpx_tru_f16_sdwa', None, None, None, gfx10_src, None, MODIFIERS))
	def v_cmpx_tru_f32_sdwa(self, gfx10_src:reg_block, MODIFIERS:str=''):
		return self.ic_pb(sdwa_base('v_cmpx_tru_f32_sdwa', None, None, None, gfx10_src, None, MODIFIERS))
	def v_cmpx_u_f16_sdwa(self, gfx10_src:reg_block, MODIFIERS:str=''):
		return self.ic_pb(sdwa_base('v_cmpx_u_f16_sdwa', None, None, None, gfx10_src, None, MODIFIERS))
	def v_cmpx_u_f32_sdwa(self, gfx10_src:reg_block, MODIFIERS:str=''):
		return self.ic_pb(sdwa_base('v_cmpx_u_f32_sdwa', None, None, None, gfx10_src, None, MODIFIERS))
	def v_cndmask_b32_sdwa(self, gfx10_src:reg_block, gfx10_src:reg_block, gfx10_vcc:reg_block, MODIFIERS:str=''):
		return self.ic_pb(sdwa_base('v_cndmask_b32_sdwa', None, None, gfx10_src, gfx10_src, gfx10_vcc, MODIFIERS))
	def v_cos_f16_sdwa(self, gfx10_src:reg_block, MODIFIERS:str=''):
		return self.ic_pb(sdwa_base('v_cos_f16_sdwa', None, None, gfx10_src, None, None, MODIFIERS))
	def v_cos_f32_sdwa(self, gfx10_src:reg_block, MODIFIERS:str=''):
		return self.ic_pb(sdwa_base('v_cos_f32_sdwa', None, None, gfx10_src, None, None, MODIFIERS))
	def v_cvt_f16_f32_sdwa(self, gfx10_src:reg_block, MODIFIERS:str=''):
		return self.ic_pb(sdwa_base('v_cvt_f16_f32_sdwa', None, None, gfx10_src, None, None, MODIFIERS))
	def v_cvt_f16_i16_sdwa(self, gfx10_src_1:reg_block, MODIFIERS:str=''):
		return self.ic_pb(sdwa_base('v_cvt_f16_i16_sdwa', None, None, gfx10_src_1, None, None, MODIFIERS))
	def v_cvt_f16_u16_sdwa(self, gfx10_src_1:reg_block, MODIFIERS:str=''):
		return self.ic_pb(sdwa_base('v_cvt_f16_u16_sdwa', None, None, gfx10_src_1, None, None, MODIFIERS))
	def v_cvt_f32_f16_sdwa(self, gfx10_src:reg_block, MODIFIERS:str=''):
		return self.ic_pb(sdwa_base('v_cvt_f32_f16_sdwa', None, None, gfx10_src, None, None, MODIFIERS))
	def v_cvt_f32_i32_sdwa(self, gfx10_src:reg_block, MODIFIERS:str=''):
		return self.ic_pb(sdwa_base('v_cvt_f32_i32_sdwa', None, None, gfx10_src, None, None, MODIFIERS))
	def v_cvt_f32_u32_sdwa(self, gfx10_src:reg_block, MODIFIERS:str=''):
		return self.ic_pb(sdwa_base('v_cvt_f32_u32_sdwa', None, None, gfx10_src, None, None, MODIFIERS))
	def v_cvt_f32_ubyte0_sdwa(self, gfx10_src:reg_block, MODIFIERS:str=''):
		return self.ic_pb(sdwa_base('v_cvt_f32_ubyte0_sdwa', None, None, gfx10_src, None, None, MODIFIERS))
	def v_cvt_f32_ubyte1_sdwa(self, gfx10_src:reg_block, MODIFIERS:str=''):
		return self.ic_pb(sdwa_base('v_cvt_f32_ubyte1_sdwa', None, None, gfx10_src, None, None, MODIFIERS))
	def v_cvt_f32_ubyte2_sdwa(self, gfx10_src:reg_block, MODIFIERS:str=''):
		return self.ic_pb(sdwa_base('v_cvt_f32_ubyte2_sdwa', None, None, gfx10_src, None, None, MODIFIERS))
	def v_cvt_f32_ubyte3_sdwa(self, gfx10_src:reg_block, MODIFIERS:str=''):
		return self.ic_pb(sdwa_base('v_cvt_f32_ubyte3_sdwa', None, None, gfx10_src, None, None, MODIFIERS))
	def v_cvt_flr_i32_f32_sdwa(self, gfx10_src:reg_block, MODIFIERS:str=''):
		return self.ic_pb(sdwa_base('v_cvt_flr_i32_f32_sdwa', None, None, gfx10_src, None, None, MODIFIERS))
	def v_cvt_i16_f16_sdwa(self, gfx10_src:reg_block, MODIFIERS:str=''):
		return self.ic_pb(sdwa_base('v_cvt_i16_f16_sdwa', None, None, gfx10_src, None, None, MODIFIERS))
	def v_cvt_i32_f32_sdwa(self, gfx10_src:reg_block, MODIFIERS:str=''):
		return self.ic_pb(sdwa_base('v_cvt_i32_f32_sdwa', None, None, gfx10_src, None, None, MODIFIERS))
	def v_cvt_norm_i16_f16_sdwa(self, gfx10_src:reg_block, MODIFIERS:str=''):
		return self.ic_pb(sdwa_base('v_cvt_norm_i16_f16_sdwa', None, None, gfx10_src, None, None, MODIFIERS))
	def v_cvt_norm_u16_f16_sdwa(self, gfx10_src:reg_block, MODIFIERS:str=''):
		return self.ic_pb(sdwa_base('v_cvt_norm_u16_f16_sdwa', None, None, gfx10_src, None, None, MODIFIERS))
	def v_cvt_off_f32_i4_sdwa(self, gfx10_src:reg_block, MODIFIERS:str=''):
		return self.ic_pb(sdwa_base('v_cvt_off_f32_i4_sdwa', None, None, gfx10_src, None, None, MODIFIERS))
	def v_cvt_rpi_i32_f32_sdwa(self, gfx10_src:reg_block, MODIFIERS:str=''):
		return self.ic_pb(sdwa_base('v_cvt_rpi_i32_f32_sdwa', None, None, gfx10_src, None, None, MODIFIERS))
	def v_cvt_u16_f16_sdwa(self, gfx10_src:reg_block, MODIFIERS:str=''):
		return self.ic_pb(sdwa_base('v_cvt_u16_f16_sdwa', None, None, gfx10_src, None, None, MODIFIERS))
	def v_cvt_u32_f32_sdwa(self, gfx10_src:reg_block, MODIFIERS:str=''):
		return self.ic_pb(sdwa_base('v_cvt_u32_f32_sdwa', None, None, gfx10_src, None, None, MODIFIERS))
	def v_exp_f16_sdwa(self, gfx10_src:reg_block, MODIFIERS:str=''):
		return self.ic_pb(sdwa_base('v_exp_f16_sdwa', None, None, gfx10_src, None, None, MODIFIERS))
	def v_exp_f32_sdwa(self, gfx10_src:reg_block, MODIFIERS:str=''):
		return self.ic_pb(sdwa_base('v_exp_f32_sdwa', None, None, gfx10_src, None, None, MODIFIERS))
	def v_ffbh_i32_sdwa(self, gfx10_src:reg_block, MODIFIERS:str=''):
		return self.ic_pb(sdwa_base('v_ffbh_i32_sdwa', None, None, gfx10_src, None, None, MODIFIERS))
	def v_ffbh_u32_sdwa(self, gfx10_src:reg_block, MODIFIERS:str=''):
		return self.ic_pb(sdwa_base('v_ffbh_u32_sdwa', None, None, gfx10_src, None, None, MODIFIERS))
	def v_ffbl_b32_sdwa(self, gfx10_src:reg_block, MODIFIERS:str=''):
		return self.ic_pb(sdwa_base('v_ffbl_b32_sdwa', None, None, gfx10_src, None, None, MODIFIERS))
	def v_floor_f16_sdwa(self, gfx10_src:reg_block, MODIFIERS:str=''):
		return self.ic_pb(sdwa_base('v_floor_f16_sdwa', None, None, gfx10_src, None, None, MODIFIERS))
	def v_floor_f32_sdwa(self, gfx10_src:reg_block, MODIFIERS:str=''):
		return self.ic_pb(sdwa_base('v_floor_f32_sdwa', None, None, gfx10_src, None, None, MODIFIERS))
	def v_fract_f16_sdwa(self, gfx10_src:reg_block, MODIFIERS:str=''):
		return self.ic_pb(sdwa_base('v_fract_f16_sdwa', None, None, gfx10_src, None, None, MODIFIERS))
	def v_fract_f32_sdwa(self, gfx10_src:reg_block, MODIFIERS:str=''):
		return self.ic_pb(sdwa_base('v_fract_f32_sdwa', None, None, gfx10_src, None, None, MODIFIERS))
	def v_frexp_exp_i16_f16_sdwa(self, gfx10_src:reg_block, MODIFIERS:str=''):
		return self.ic_pb(sdwa_base('v_frexp_exp_i16_f16_sdwa', None, None, gfx10_src, None, None, MODIFIERS))
	def v_frexp_exp_i32_f32_sdwa(self, gfx10_src:reg_block, MODIFIERS:str=''):
		return self.ic_pb(sdwa_base('v_frexp_exp_i32_f32_sdwa', None, None, gfx10_src, None, None, MODIFIERS))
	def v_frexp_mant_f16_sdwa(self, gfx10_src:reg_block, MODIFIERS:str=''):
		return self.ic_pb(sdwa_base('v_frexp_mant_f16_sdwa', None, None, gfx10_src, None, None, MODIFIERS))
	def v_frexp_mant_f32_sdwa(self, gfx10_src:reg_block, MODIFIERS:str=''):
		return self.ic_pb(sdwa_base('v_frexp_mant_f32_sdwa', None, None, gfx10_src, None, None, MODIFIERS))
	def v_ldexp_f16_sdwa(self, gfx10_src:reg_block, gfx10_src:reg_block, MODIFIERS:str=''):
		return self.ic_pb(sdwa_base('v_ldexp_f16_sdwa', None, None, gfx10_src, gfx10_src, None, MODIFIERS))
	def v_log_f16_sdwa(self, gfx10_src:reg_block, MODIFIERS:str=''):
		return self.ic_pb(sdwa_base('v_log_f16_sdwa', None, None, gfx10_src, None, None, MODIFIERS))
	def v_log_f32_sdwa(self, gfx10_src:reg_block, MODIFIERS:str=''):
		return self.ic_pb(sdwa_base('v_log_f32_sdwa', None, None, gfx10_src, None, None, MODIFIERS))
	def v_lshlrev_b32_sdwa(self, gfx10_src:reg_block, gfx10_src:reg_block, MODIFIERS:str=''):
		return self.ic_pb(sdwa_base('v_lshlrev_b32_sdwa', None, None, gfx10_src, gfx10_src, None, MODIFIERS))
	def v_lshrrev_b32_sdwa(self, gfx10_src:reg_block, gfx10_src:reg_block, MODIFIERS:str=''):
		return self.ic_pb(sdwa_base('v_lshrrev_b32_sdwa', None, None, gfx10_src, gfx10_src, None, MODIFIERS))
	def v_max_f16_sdwa(self, gfx10_src:reg_block, gfx10_src:reg_block, MODIFIERS:str=''):
		return self.ic_pb(sdwa_base('v_max_f16_sdwa', None, None, gfx10_src, gfx10_src, None, MODIFIERS))
	def v_max_f32_sdwa(self, gfx10_src:reg_block, gfx10_src:reg_block, MODIFIERS:str=''):
		return self.ic_pb(sdwa_base('v_max_f32_sdwa', None, None, gfx10_src, gfx10_src, None, MODIFIERS))
	def v_max_i32_sdwa(self, gfx10_src:reg_block, gfx10_src:reg_block, MODIFIERS:str=''):
		return self.ic_pb(sdwa_base('v_max_i32_sdwa', None, None, gfx10_src, gfx10_src, None, MODIFIERS))
	def v_max_u32_sdwa(self, gfx10_src:reg_block, gfx10_src:reg_block, MODIFIERS:str=''):
		return self.ic_pb(sdwa_base('v_max_u32_sdwa', None, None, gfx10_src, gfx10_src, None, MODIFIERS))
	def v_min_f16_sdwa(self, gfx10_src:reg_block, gfx10_src:reg_block, MODIFIERS:str=''):
		return self.ic_pb(sdwa_base('v_min_f16_sdwa', None, None, gfx10_src, gfx10_src, None, MODIFIERS))
	def v_min_f32_sdwa(self, gfx10_src:reg_block, gfx10_src:reg_block, MODIFIERS:str=''):
		return self.ic_pb(sdwa_base('v_min_f32_sdwa', None, None, gfx10_src, gfx10_src, None, MODIFIERS))
	def v_min_i32_sdwa(self, gfx10_src:reg_block, gfx10_src:reg_block, MODIFIERS:str=''):
		return self.ic_pb(sdwa_base('v_min_i32_sdwa', None, None, gfx10_src, gfx10_src, None, MODIFIERS))
	def v_min_u32_sdwa(self, gfx10_src:reg_block, gfx10_src:reg_block, MODIFIERS:str=''):
		return self.ic_pb(sdwa_base('v_min_u32_sdwa', None, None, gfx10_src, gfx10_src, None, MODIFIERS))
	def v_mov_b32_sdwa(self, gfx10_src:reg_block, MODIFIERS:str=''):
		return self.ic_pb(sdwa_base('v_mov_b32_sdwa', None, None, gfx10_src, None, None, MODIFIERS))
	def v_movreld_b32_sdwa(self, gfx10_src:reg_block, MODIFIERS:str=''):
		return self.ic_pb(sdwa_base('v_movreld_b32_sdwa', None, None, gfx10_src, None, None, MODIFIERS))
	def v_movrels_b32_sdwa(self, gfx10_vsrc:reg_block, MODIFIERS:str=''):
		return self.ic_pb(sdwa_base('v_movrels_b32_sdwa', None, None, gfx10_vsrc, None, None, MODIFIERS))
	def v_movrelsd_2_b32_sdwa(self, gfx10_vsrc:reg_block, MODIFIERS:str=''):
		return self.ic_pb(sdwa_base('v_movrelsd_2_b32_sdwa', None, None, gfx10_vsrc, None, None, MODIFIERS))
	def v_movrelsd_b32_sdwa(self, gfx10_vsrc:reg_block, MODIFIERS:str=''):
		return self.ic_pb(sdwa_base('v_movrelsd_b32_sdwa', None, None, gfx10_vsrc, None, None, MODIFIERS))
	def v_mul_f16_sdwa(self, gfx10_src:reg_block, gfx10_src:reg_block, MODIFIERS:str=''):
		return self.ic_pb(sdwa_base('v_mul_f16_sdwa', None, None, gfx10_src, gfx10_src, None, MODIFIERS))
	def v_mul_f32_sdwa(self, gfx10_src:reg_block, gfx10_src:reg_block, MODIFIERS:str=''):
		return self.ic_pb(sdwa_base('v_mul_f32_sdwa', None, None, gfx10_src, gfx10_src, None, MODIFIERS))
	def v_mul_hi_i32_i24_sdwa(self, gfx10_src:reg_block, gfx10_src:reg_block, MODIFIERS:str=''):
		return self.ic_pb(sdwa_base('v_mul_hi_i32_i24_sdwa', None, None, gfx10_src, gfx10_src, None, MODIFIERS))
	def v_mul_hi_u32_u24_sdwa(self, gfx10_src:reg_block, gfx10_src:reg_block, MODIFIERS:str=''):
		return self.ic_pb(sdwa_base('v_mul_hi_u32_u24_sdwa', None, None, gfx10_src, gfx10_src, None, MODIFIERS))
	def v_mul_i32_i24_sdwa(self, gfx10_src:reg_block, gfx10_src:reg_block, MODIFIERS:str=''):
		return self.ic_pb(sdwa_base('v_mul_i32_i24_sdwa', None, None, gfx10_src, gfx10_src, None, MODIFIERS))
	def v_mul_legacy_f32_sdwa(self, gfx10_src:reg_block, gfx10_src:reg_block, MODIFIERS:str=''):
		return self.ic_pb(sdwa_base('v_mul_legacy_f32_sdwa', None, None, gfx10_src, gfx10_src, None, MODIFIERS))
	def v_mul_u32_u24_sdwa(self, gfx10_src:reg_block, gfx10_src:reg_block, MODIFIERS:str=''):
		return self.ic_pb(sdwa_base('v_mul_u32_u24_sdwa', None, None, gfx10_src, gfx10_src, None, MODIFIERS))
	def v_not_b32_sdwa(self, gfx10_src:reg_block, MODIFIERS:str=''):
		return self.ic_pb(sdwa_base('v_not_b32_sdwa', None, None, gfx10_src, None, None, MODIFIERS))
	def v_or_b32_sdwa(self, gfx10_src:reg_block, gfx10_src:reg_block, MODIFIERS:str=''):
		return self.ic_pb(sdwa_base('v_or_b32_sdwa', None, None, gfx10_src, gfx10_src, None, MODIFIERS))
	def v_rcp_f16_sdwa(self, gfx10_src:reg_block, MODIFIERS:str=''):
		return self.ic_pb(sdwa_base('v_rcp_f16_sdwa', None, None, gfx10_src, None, None, MODIFIERS))
	def v_rcp_f32_sdwa(self, gfx10_src:reg_block, MODIFIERS:str=''):
		return self.ic_pb(sdwa_base('v_rcp_f32_sdwa', None, None, gfx10_src, None, None, MODIFIERS))
	def v_rcp_iflag_f32_sdwa(self, gfx10_src:reg_block, MODIFIERS:str=''):
		return self.ic_pb(sdwa_base('v_rcp_iflag_f32_sdwa', None, None, gfx10_src, None, None, MODIFIERS))
	def v_rndne_f16_sdwa(self, gfx10_src:reg_block, MODIFIERS:str=''):
		return self.ic_pb(sdwa_base('v_rndne_f16_sdwa', None, None, gfx10_src, None, None, MODIFIERS))
	def v_rndne_f32_sdwa(self, gfx10_src:reg_block, MODIFIERS:str=''):
		return self.ic_pb(sdwa_base('v_rndne_f32_sdwa', None, None, gfx10_src, None, None, MODIFIERS))
	def v_rsq_f16_sdwa(self, gfx10_src:reg_block, MODIFIERS:str=''):
		return self.ic_pb(sdwa_base('v_rsq_f16_sdwa', None, None, gfx10_src, None, None, MODIFIERS))
	def v_rsq_f32_sdwa(self, gfx10_src:reg_block, MODIFIERS:str=''):
		return self.ic_pb(sdwa_base('v_rsq_f32_sdwa', None, None, gfx10_src, None, None, MODIFIERS))
	def v_sat_pk_u8_i16_sdwa(self, gfx10_src:reg_block, MODIFIERS:str=''):
		return self.ic_pb(sdwa_base('v_sat_pk_u8_i16_sdwa', None, None, gfx10_src, None, None, MODIFIERS))
	def v_sin_f16_sdwa(self, gfx10_src:reg_block, MODIFIERS:str=''):
		return self.ic_pb(sdwa_base('v_sin_f16_sdwa', None, None, gfx10_src, None, None, MODIFIERS))
	def v_sin_f32_sdwa(self, gfx10_src:reg_block, MODIFIERS:str=''):
		return self.ic_pb(sdwa_base('v_sin_f32_sdwa', None, None, gfx10_src, None, None, MODIFIERS))
	def v_sqrt_f16_sdwa(self, gfx10_src:reg_block, MODIFIERS:str=''):
		return self.ic_pb(sdwa_base('v_sqrt_f16_sdwa', None, None, gfx10_src, None, None, MODIFIERS))
	def v_sqrt_f32_sdwa(self, gfx10_src:reg_block, MODIFIERS:str=''):
		return self.ic_pb(sdwa_base('v_sqrt_f32_sdwa', None, None, gfx10_src, None, None, MODIFIERS))
	def v_sub_co_ci_u32_sdwa(self, gfx10_vcc:reg_block, gfx10_src:reg_block, gfx10_src:reg_block, gfx10_vcc:reg_block, MODIFIERS:str=''):
		return self.ic_pb(sdwa_base('v_sub_co_ci_u32_sdwa', None, gfx10_vcc, gfx10_src, gfx10_src, gfx10_vcc, MODIFIERS))
	def v_sub_f16_sdwa(self, gfx10_src:reg_block, gfx10_src:reg_block, MODIFIERS:str=''):
		return self.ic_pb(sdwa_base('v_sub_f16_sdwa', None, None, gfx10_src, gfx10_src, None, MODIFIERS))
	def v_sub_f32_sdwa(self, gfx10_src:reg_block, gfx10_src:reg_block, MODIFIERS:str=''):
		return self.ic_pb(sdwa_base('v_sub_f32_sdwa', None, None, gfx10_src, gfx10_src, None, MODIFIERS))
	def v_sub_nc_u32_sdwa(self, gfx10_src:reg_block, gfx10_src:reg_block, MODIFIERS:str=''):
		return self.ic_pb(sdwa_base('v_sub_nc_u32_sdwa', None, None, gfx10_src, gfx10_src, None, MODIFIERS))
	def v_subrev_co_ci_u32_sdwa(self, gfx10_vcc:reg_block, gfx10_src:reg_block, gfx10_src:reg_block, gfx10_vcc:reg_block, MODIFIERS:str=''):
		return self.ic_pb(sdwa_base('v_subrev_co_ci_u32_sdwa', None, gfx10_vcc, gfx10_src, gfx10_src, gfx10_vcc, MODIFIERS))
	def v_subrev_f16_sdwa(self, gfx10_src:reg_block, gfx10_src:reg_block, MODIFIERS:str=''):
		return self.ic_pb(sdwa_base('v_subrev_f16_sdwa', None, None, gfx10_src, gfx10_src, None, MODIFIERS))
	def v_subrev_f32_sdwa(self, gfx10_src:reg_block, gfx10_src:reg_block, MODIFIERS:str=''):
		return self.ic_pb(sdwa_base('v_subrev_f32_sdwa', None, None, gfx10_src, gfx10_src, None, MODIFIERS))
	def v_subrev_nc_u32_sdwa(self, gfx10_src:reg_block, gfx10_src:reg_block, MODIFIERS:str=''):
		return self.ic_pb(sdwa_base('v_subrev_nc_u32_sdwa', None, None, gfx10_src, gfx10_src, None, MODIFIERS))
	def v_trunc_f16_sdwa(self, gfx10_src:reg_block, MODIFIERS:str=''):
		return self.ic_pb(sdwa_base('v_trunc_f16_sdwa', None, None, gfx10_src, None, None, MODIFIERS))
	def v_trunc_f32_sdwa(self, gfx10_src:reg_block, MODIFIERS:str=''):
		return self.ic_pb(sdwa_base('v_trunc_f32_sdwa', None, None, gfx10_src, None, None, MODIFIERS))
	def v_xnor_b32_sdwa(self, gfx10_src:reg_block, gfx10_src:reg_block, MODIFIERS:str=''):
		return self.ic_pb(sdwa_base('v_xnor_b32_sdwa', None, None, gfx10_src, gfx10_src, None, MODIFIERS))
	def v_xor_b32_sdwa(self, gfx10_src:reg_block, gfx10_src:reg_block, MODIFIERS:str=''):
		return self.ic_pb(sdwa_base('v_xor_b32_sdwa', None, None, gfx10_src, gfx10_src, None, MODIFIERS))
class smem_base(inst_base): 
	def __init__(self, INSTRUCTION:str, DST:Union[reg_block,None], SRC0:Union[reg_block,None], SRC1:Union[reg_block,None], SRC2:Union[reg_block,None], MODIFIERS:str): 
		super().__init__(instruction_type.SMEM, INSTRUCTION)
		self.DST = DST 
		self.SRC0 = SRC0 
		self.SRC1 = SRC1 
		self.SRC2 = SRC2 
		self.MODIFIERS = MODIFIERS 
	def __str__(self): 
		args_l = filter(None.__ne__, [self.DST,self.SRC0,self.SRC1,self.SRC2]) 
		return f"{self.label} {','.join(map(str, args_l))} {self.MODIFIERS}" 
class smem_instr_caller(inst_caller_base): 
	def __init__(self, insturction_container) -> None:
     		super().__init__(insturction_container)
	def s_atc_probe(self, gfx10_sbase:reg_block, gfx10_soffset_1:reg_block):
		return self.ic_pb(smem_base('s_atc_probe', None, None, gfx10_sbase, gfx10_soffset_1))
	def s_atc_probe_buffer(self, gfx10_sbase_1:reg_block, gfx10_soffset_2:reg_block):
		return self.ic_pb(smem_base('s_atc_probe_buffer', None, None, gfx10_sbase_1, gfx10_soffset_2))
	def s_atomic_add(self, gfx10_sbase:reg_block, gfx10_soffset_1:reg_block, MODIFIERS:str=''):
		return self.ic_pb(smem_base('s_atomic_add', None, None, gfx10_sbase, gfx10_soffset_1, MODIFIERS))
	def s_atomic_add_x2(self, gfx10_sbase:reg_block, gfx10_soffset_1:reg_block, MODIFIERS:str=''):
		return self.ic_pb(smem_base('s_atomic_add_x2', None, None, gfx10_sbase, gfx10_soffset_1, MODIFIERS))
	def s_atomic_and(self, gfx10_sbase:reg_block, gfx10_soffset_1:reg_block, MODIFIERS:str=''):
		return self.ic_pb(smem_base('s_atomic_and', None, None, gfx10_sbase, gfx10_soffset_1, MODIFIERS))
	def s_atomic_and_x2(self, gfx10_sbase:reg_block, gfx10_soffset_1:reg_block, MODIFIERS:str=''):
		return self.ic_pb(smem_base('s_atomic_and_x2', None, None, gfx10_sbase, gfx10_soffset_1, MODIFIERS))
	def s_atomic_cmpswap(self, gfx10_sbase:reg_block, gfx10_soffset_1:reg_block, MODIFIERS:str=''):
		return self.ic_pb(smem_base('s_atomic_cmpswap', None, None, gfx10_sbase, gfx10_soffset_1, MODIFIERS))
	def s_atomic_cmpswap_x2(self, gfx10_sbase:reg_block, gfx10_soffset_1:reg_block, MODIFIERS:str=''):
		return self.ic_pb(smem_base('s_atomic_cmpswap_x2', None, None, gfx10_sbase, gfx10_soffset_1, MODIFIERS))
	def s_atomic_dec(self, gfx10_sbase:reg_block, gfx10_soffset_1:reg_block, MODIFIERS:str=''):
		return self.ic_pb(smem_base('s_atomic_dec', None, None, gfx10_sbase, gfx10_soffset_1, MODIFIERS))
	def s_atomic_dec_x2(self, gfx10_sbase:reg_block, gfx10_soffset_1:reg_block, MODIFIERS:str=''):
		return self.ic_pb(smem_base('s_atomic_dec_x2', None, None, gfx10_sbase, gfx10_soffset_1, MODIFIERS))
	def s_atomic_inc(self, gfx10_sbase:reg_block, gfx10_soffset_1:reg_block, MODIFIERS:str=''):
		return self.ic_pb(smem_base('s_atomic_inc', None, None, gfx10_sbase, gfx10_soffset_1, MODIFIERS))
	def s_atomic_inc_x2(self, gfx10_sbase:reg_block, gfx10_soffset_1:reg_block, MODIFIERS:str=''):
		return self.ic_pb(smem_base('s_atomic_inc_x2', None, None, gfx10_sbase, gfx10_soffset_1, MODIFIERS))
	def s_atomic_or(self, gfx10_sbase:reg_block, gfx10_soffset_1:reg_block, MODIFIERS:str=''):
		return self.ic_pb(smem_base('s_atomic_or', None, None, gfx10_sbase, gfx10_soffset_1, MODIFIERS))
	def s_atomic_or_x2(self, gfx10_sbase:reg_block, gfx10_soffset_1:reg_block, MODIFIERS:str=''):
		return self.ic_pb(smem_base('s_atomic_or_x2', None, None, gfx10_sbase, gfx10_soffset_1, MODIFIERS))
	def s_atomic_smax(self, gfx10_sbase:reg_block, gfx10_soffset_1:reg_block, MODIFIERS:str=''):
		return self.ic_pb(smem_base('s_atomic_smax', None, None, gfx10_sbase, gfx10_soffset_1, MODIFIERS))
	def s_atomic_smax_x2(self, gfx10_sbase:reg_block, gfx10_soffset_1:reg_block, MODIFIERS:str=''):
		return self.ic_pb(smem_base('s_atomic_smax_x2', None, None, gfx10_sbase, gfx10_soffset_1, MODIFIERS))
	def s_atomic_smin(self, gfx10_sbase:reg_block, gfx10_soffset_1:reg_block, MODIFIERS:str=''):
		return self.ic_pb(smem_base('s_atomic_smin', None, None, gfx10_sbase, gfx10_soffset_1, MODIFIERS))
	def s_atomic_smin_x2(self, gfx10_sbase:reg_block, gfx10_soffset_1:reg_block, MODIFIERS:str=''):
		return self.ic_pb(smem_base('s_atomic_smin_x2', None, None, gfx10_sbase, gfx10_soffset_1, MODIFIERS))
	def s_atomic_sub(self, gfx10_sbase:reg_block, gfx10_soffset_1:reg_block, MODIFIERS:str=''):
		return self.ic_pb(smem_base('s_atomic_sub', None, None, gfx10_sbase, gfx10_soffset_1, MODIFIERS))
	def s_atomic_sub_x2(self, gfx10_sbase:reg_block, gfx10_soffset_1:reg_block, MODIFIERS:str=''):
		return self.ic_pb(smem_base('s_atomic_sub_x2', None, None, gfx10_sbase, gfx10_soffset_1, MODIFIERS))
	def s_atomic_swap(self, gfx10_sbase:reg_block, gfx10_soffset_1:reg_block, MODIFIERS:str=''):
		return self.ic_pb(smem_base('s_atomic_swap', None, None, gfx10_sbase, gfx10_soffset_1, MODIFIERS))
	def s_atomic_swap_x2(self, gfx10_sbase:reg_block, gfx10_soffset_1:reg_block, MODIFIERS:str=''):
		return self.ic_pb(smem_base('s_atomic_swap_x2', None, None, gfx10_sbase, gfx10_soffset_1, MODIFIERS))
	def s_atomic_umax(self, gfx10_sbase:reg_block, gfx10_soffset_1:reg_block, MODIFIERS:str=''):
		return self.ic_pb(smem_base('s_atomic_umax', None, None, gfx10_sbase, gfx10_soffset_1, MODIFIERS))
	def s_atomic_umax_x2(self, gfx10_sbase:reg_block, gfx10_soffset_1:reg_block, MODIFIERS:str=''):
		return self.ic_pb(smem_base('s_atomic_umax_x2', None, None, gfx10_sbase, gfx10_soffset_1, MODIFIERS))
	def s_atomic_umin(self, gfx10_sbase:reg_block, gfx10_soffset_1:reg_block, MODIFIERS:str=''):
		return self.ic_pb(smem_base('s_atomic_umin', None, None, gfx10_sbase, gfx10_soffset_1, MODIFIERS))
	def s_atomic_umin_x2(self, gfx10_sbase:reg_block, gfx10_soffset_1:reg_block, MODIFIERS:str=''):
		return self.ic_pb(smem_base('s_atomic_umin_x2', None, None, gfx10_sbase, gfx10_soffset_1, MODIFIERS))
	def s_atomic_xor(self, gfx10_sbase:reg_block, gfx10_soffset_1:reg_block, MODIFIERS:str=''):
		return self.ic_pb(smem_base('s_atomic_xor', None, None, gfx10_sbase, gfx10_soffset_1, MODIFIERS))
	def s_atomic_xor_x2(self, gfx10_sbase:reg_block, gfx10_soffset_1:reg_block, MODIFIERS:str=''):
		return self.ic_pb(smem_base('s_atomic_xor_x2', None, None, gfx10_sbase, gfx10_soffset_1, MODIFIERS))
	def s_buffer_atomic_add(self, gfx10_sbase_1:reg_block, gfx10_soffset_2:reg_block, MODIFIERS:str=''):
		return self.ic_pb(smem_base('s_buffer_atomic_add', None, None, gfx10_sbase_1, gfx10_soffset_2, MODIFIERS))
	def s_buffer_atomic_add_x2(self, gfx10_sbase_1:reg_block, gfx10_soffset_2:reg_block, MODIFIERS:str=''):
		return self.ic_pb(smem_base('s_buffer_atomic_add_x2', None, None, gfx10_sbase_1, gfx10_soffset_2, MODIFIERS))
	def s_buffer_atomic_and(self, gfx10_sbase_1:reg_block, gfx10_soffset_2:reg_block, MODIFIERS:str=''):
		return self.ic_pb(smem_base('s_buffer_atomic_and', None, None, gfx10_sbase_1, gfx10_soffset_2, MODIFIERS))
	def s_buffer_atomic_and_x2(self, gfx10_sbase_1:reg_block, gfx10_soffset_2:reg_block, MODIFIERS:str=''):
		return self.ic_pb(smem_base('s_buffer_atomic_and_x2', None, None, gfx10_sbase_1, gfx10_soffset_2, MODIFIERS))
	def s_buffer_atomic_cmpswap(self, gfx10_sbase_1:reg_block, gfx10_soffset_2:reg_block, MODIFIERS:str=''):
		return self.ic_pb(smem_base('s_buffer_atomic_cmpswap', None, None, gfx10_sbase_1, gfx10_soffset_2, MODIFIERS))
	def s_buffer_atomic_cmpswap_x2(self, gfx10_sbase_1:reg_block, gfx10_soffset_2:reg_block, MODIFIERS:str=''):
		return self.ic_pb(smem_base('s_buffer_atomic_cmpswap_x2', None, None, gfx10_sbase_1, gfx10_soffset_2, MODIFIERS))
	def s_buffer_atomic_dec(self, gfx10_sbase_1:reg_block, gfx10_soffset_2:reg_block, MODIFIERS:str=''):
		return self.ic_pb(smem_base('s_buffer_atomic_dec', None, None, gfx10_sbase_1, gfx10_soffset_2, MODIFIERS))
	def s_buffer_atomic_dec_x2(self, gfx10_sbase_1:reg_block, gfx10_soffset_2:reg_block, MODIFIERS:str=''):
		return self.ic_pb(smem_base('s_buffer_atomic_dec_x2', None, None, gfx10_sbase_1, gfx10_soffset_2, MODIFIERS))
	def s_buffer_atomic_inc(self, gfx10_sbase_1:reg_block, gfx10_soffset_2:reg_block, MODIFIERS:str=''):
		return self.ic_pb(smem_base('s_buffer_atomic_inc', None, None, gfx10_sbase_1, gfx10_soffset_2, MODIFIERS))
	def s_buffer_atomic_inc_x2(self, gfx10_sbase_1:reg_block, gfx10_soffset_2:reg_block, MODIFIERS:str=''):
		return self.ic_pb(smem_base('s_buffer_atomic_inc_x2', None, None, gfx10_sbase_1, gfx10_soffset_2, MODIFIERS))
	def s_buffer_atomic_or(self, gfx10_sbase_1:reg_block, gfx10_soffset_2:reg_block, MODIFIERS:str=''):
		return self.ic_pb(smem_base('s_buffer_atomic_or', None, None, gfx10_sbase_1, gfx10_soffset_2, MODIFIERS))
	def s_buffer_atomic_or_x2(self, gfx10_sbase_1:reg_block, gfx10_soffset_2:reg_block, MODIFIERS:str=''):
		return self.ic_pb(smem_base('s_buffer_atomic_or_x2', None, None, gfx10_sbase_1, gfx10_soffset_2, MODIFIERS))
	def s_buffer_atomic_smax(self, gfx10_sbase_1:reg_block, gfx10_soffset_2:reg_block, MODIFIERS:str=''):
		return self.ic_pb(smem_base('s_buffer_atomic_smax', None, None, gfx10_sbase_1, gfx10_soffset_2, MODIFIERS))
	def s_buffer_atomic_smax_x2(self, gfx10_sbase_1:reg_block, gfx10_soffset_2:reg_block, MODIFIERS:str=''):
		return self.ic_pb(smem_base('s_buffer_atomic_smax_x2', None, None, gfx10_sbase_1, gfx10_soffset_2, MODIFIERS))
	def s_buffer_atomic_smin(self, gfx10_sbase_1:reg_block, gfx10_soffset_2:reg_block, MODIFIERS:str=''):
		return self.ic_pb(smem_base('s_buffer_atomic_smin', None, None, gfx10_sbase_1, gfx10_soffset_2, MODIFIERS))
	def s_buffer_atomic_smin_x2(self, gfx10_sbase_1:reg_block, gfx10_soffset_2:reg_block, MODIFIERS:str=''):
		return self.ic_pb(smem_base('s_buffer_atomic_smin_x2', None, None, gfx10_sbase_1, gfx10_soffset_2, MODIFIERS))
	def s_buffer_atomic_sub(self, gfx10_sbase_1:reg_block, gfx10_soffset_2:reg_block, MODIFIERS:str=''):
		return self.ic_pb(smem_base('s_buffer_atomic_sub', None, None, gfx10_sbase_1, gfx10_soffset_2, MODIFIERS))
	def s_buffer_atomic_sub_x2(self, gfx10_sbase_1:reg_block, gfx10_soffset_2:reg_block, MODIFIERS:str=''):
		return self.ic_pb(smem_base('s_buffer_atomic_sub_x2', None, None, gfx10_sbase_1, gfx10_soffset_2, MODIFIERS))
	def s_buffer_atomic_swap(self, gfx10_sbase_1:reg_block, gfx10_soffset_2:reg_block, MODIFIERS:str=''):
		return self.ic_pb(smem_base('s_buffer_atomic_swap', None, None, gfx10_sbase_1, gfx10_soffset_2, MODIFIERS))
	def s_buffer_atomic_swap_x2(self, gfx10_sbase_1:reg_block, gfx10_soffset_2:reg_block, MODIFIERS:str=''):
		return self.ic_pb(smem_base('s_buffer_atomic_swap_x2', None, None, gfx10_sbase_1, gfx10_soffset_2, MODIFIERS))
	def s_buffer_atomic_umax(self, gfx10_sbase_1:reg_block, gfx10_soffset_2:reg_block, MODIFIERS:str=''):
		return self.ic_pb(smem_base('s_buffer_atomic_umax', None, None, gfx10_sbase_1, gfx10_soffset_2, MODIFIERS))
	def s_buffer_atomic_umax_x2(self, gfx10_sbase_1:reg_block, gfx10_soffset_2:reg_block, MODIFIERS:str=''):
		return self.ic_pb(smem_base('s_buffer_atomic_umax_x2', None, None, gfx10_sbase_1, gfx10_soffset_2, MODIFIERS))
	def s_buffer_atomic_umin(self, gfx10_sbase_1:reg_block, gfx10_soffset_2:reg_block, MODIFIERS:str=''):
		return self.ic_pb(smem_base('s_buffer_atomic_umin', None, None, gfx10_sbase_1, gfx10_soffset_2, MODIFIERS))
	def s_buffer_atomic_umin_x2(self, gfx10_sbase_1:reg_block, gfx10_soffset_2:reg_block, MODIFIERS:str=''):
		return self.ic_pb(smem_base('s_buffer_atomic_umin_x2', None, None, gfx10_sbase_1, gfx10_soffset_2, MODIFIERS))
	def s_buffer_atomic_xor(self, gfx10_sbase_1:reg_block, gfx10_soffset_2:reg_block, MODIFIERS:str=''):
		return self.ic_pb(smem_base('s_buffer_atomic_xor', None, None, gfx10_sbase_1, gfx10_soffset_2, MODIFIERS))
	def s_buffer_atomic_xor_x2(self, gfx10_sbase_1:reg_block, gfx10_soffset_2:reg_block, MODIFIERS:str=''):
		return self.ic_pb(smem_base('s_buffer_atomic_xor_x2', None, None, gfx10_sbase_1, gfx10_soffset_2, MODIFIERS))
	def s_buffer_load_dword(self, gfx10_sbase_1:reg_block, gfx10_soffset_2:reg_block, MODIFIERS:str=''):
		return self.ic_pb(smem_base('s_buffer_load_dword', None, gfx10_sbase_1, gfx10_soffset_2, None, MODIFIERS))
	def s_buffer_load_dwordx16(self, gfx10_sbase_1:reg_block, gfx10_soffset_2:reg_block, MODIFIERS:str=''):
		return self.ic_pb(smem_base('s_buffer_load_dwordx16', None, gfx10_sbase_1, gfx10_soffset_2, None, MODIFIERS))
	def s_buffer_load_dwordx2(self, gfx10_sbase_1:reg_block, gfx10_soffset_2:reg_block, MODIFIERS:str=''):
		return self.ic_pb(smem_base('s_buffer_load_dwordx2', None, gfx10_sbase_1, gfx10_soffset_2, None, MODIFIERS))
	def s_buffer_load_dwordx4(self, gfx10_sbase_1:reg_block, gfx10_soffset_2:reg_block, MODIFIERS:str=''):
		return self.ic_pb(smem_base('s_buffer_load_dwordx4', None, gfx10_sbase_1, gfx10_soffset_2, None, MODIFIERS))
	def s_buffer_load_dwordx8(self, gfx10_sbase_1:reg_block, gfx10_soffset_2:reg_block, MODIFIERS:str=''):
		return self.ic_pb(smem_base('s_buffer_load_dwordx8', None, gfx10_sbase_1, gfx10_soffset_2, None, MODIFIERS))
	def s_buffer_store_dword(self, gfx10_sbase_1:reg_block, gfx10_soffset_2:reg_block, MODIFIERS:str=''):
		return self.ic_pb(smem_base('s_buffer_store_dword', None, None, gfx10_sbase_1, gfx10_soffset_2, MODIFIERS))
	def s_buffer_store_dwordx2(self, gfx10_sbase_1:reg_block, gfx10_soffset_2:reg_block, MODIFIERS:str=''):
		return self.ic_pb(smem_base('s_buffer_store_dwordx2', None, None, gfx10_sbase_1, gfx10_soffset_2, MODIFIERS))
	def s_buffer_store_dwordx4(self, gfx10_sbase_1:reg_block, gfx10_soffset_2:reg_block, MODIFIERS:str=''):
		return self.ic_pb(smem_base('s_buffer_store_dwordx4', None, None, gfx10_sbase_1, gfx10_soffset_2, MODIFIERS))
	def s_dcache_discard(self, gfx10_soffset_1:reg_block):
		return self.ic_pb(smem_base('s_dcache_discard', None, None, gfx10_soffset_1, None))
	def s_dcache_discard_x2(self, gfx10_soffset_1:reg_block):
		return self.ic_pb(smem_base('s_dcache_discard_x2', None, None, gfx10_soffset_1, None))
	def s_dcache_inv(self):
		return self.ic_pb(smem_base('s_dcache_inv', None, None, None, None))
	def s_dcache_wb(self):
		return self.ic_pb(smem_base('s_dcache_wb', None, None, None, None))
	def s_get_waveid_in_workgroup(self):
		return self.ic_pb(smem_base('s_get_waveid_in_workgroup', None, None, None, None))
	def s_gl1_inv(self):
		return self.ic_pb(smem_base('s_gl1_inv', None, None, None, None))
	def s_load_dword(self, gfx10_sbase:reg_block, gfx10_soffset_1:reg_block, MODIFIERS:str=''):
		return self.ic_pb(smem_base('s_load_dword', None, gfx10_sbase, gfx10_soffset_1, None, MODIFIERS))
	def s_load_dwordx16(self, gfx10_sbase:reg_block, gfx10_soffset_1:reg_block, MODIFIERS:str=''):
		return self.ic_pb(smem_base('s_load_dwordx16', None, gfx10_sbase, gfx10_soffset_1, None, MODIFIERS))
	def s_load_dwordx2(self, gfx10_sbase:reg_block, gfx10_soffset_1:reg_block, MODIFIERS:str=''):
		return self.ic_pb(smem_base('s_load_dwordx2', None, gfx10_sbase, gfx10_soffset_1, None, MODIFIERS))
	def s_load_dwordx4(self, gfx10_sbase:reg_block, gfx10_soffset_1:reg_block, MODIFIERS:str=''):
		return self.ic_pb(smem_base('s_load_dwordx4', None, gfx10_sbase, gfx10_soffset_1, None, MODIFIERS))
	def s_load_dwordx8(self, gfx10_sbase:reg_block, gfx10_soffset_1:reg_block, MODIFIERS:str=''):
		return self.ic_pb(smem_base('s_load_dwordx8', None, gfx10_sbase, gfx10_soffset_1, None, MODIFIERS))
	def s_memrealtime(self):
		return self.ic_pb(smem_base('s_memrealtime', None, None, None, None))
	def s_memtime(self):
		return self.ic_pb(smem_base('s_memtime', None, None, None, None))
	def s_scratch_load_dword(self, gfx10_sbase_2:reg_block, gfx10_soffset_1:reg_block, MODIFIERS:str=''):
		return self.ic_pb(smem_base('s_scratch_load_dword', None, gfx10_sbase_2, gfx10_soffset_1, None, MODIFIERS))
	def s_scratch_load_dwordx2(self, gfx10_sbase_2:reg_block, gfx10_soffset_1:reg_block, MODIFIERS:str=''):
		return self.ic_pb(smem_base('s_scratch_load_dwordx2', None, gfx10_sbase_2, gfx10_soffset_1, None, MODIFIERS))
	def s_scratch_load_dwordx4(self, gfx10_sbase_2:reg_block, gfx10_soffset_1:reg_block, MODIFIERS:str=''):
		return self.ic_pb(smem_base('s_scratch_load_dwordx4', None, gfx10_sbase_2, gfx10_soffset_1, None, MODIFIERS))
	def s_scratch_store_dword(self, gfx10_sbase_2:reg_block, gfx10_soffset_1:reg_block, MODIFIERS:str=''):
		return self.ic_pb(smem_base('s_scratch_store_dword', None, None, gfx10_sbase_2, gfx10_soffset_1, MODIFIERS))
	def s_scratch_store_dwordx2(self, gfx10_sbase_2:reg_block, gfx10_soffset_1:reg_block, MODIFIERS:str=''):
		return self.ic_pb(smem_base('s_scratch_store_dwordx2', None, None, gfx10_sbase_2, gfx10_soffset_1, MODIFIERS))
	def s_scratch_store_dwordx4(self, gfx10_sbase_2:reg_block, gfx10_soffset_1:reg_block, MODIFIERS:str=''):
		return self.ic_pb(smem_base('s_scratch_store_dwordx4', None, None, gfx10_sbase_2, gfx10_soffset_1, MODIFIERS))
	def s_store_dword(self, gfx10_sbase:reg_block, gfx10_soffset_1:reg_block, MODIFIERS:str=''):
		return self.ic_pb(smem_base('s_store_dword', None, None, gfx10_sbase, gfx10_soffset_1, MODIFIERS))
	def s_store_dwordx2(self, gfx10_sbase:reg_block, gfx10_soffset_1:reg_block, MODIFIERS:str=''):
		return self.ic_pb(smem_base('s_store_dwordx2', None, None, gfx10_sbase, gfx10_soffset_1, MODIFIERS))
	def s_store_dwordx4(self, gfx10_sbase:reg_block, gfx10_soffset_1:reg_block, MODIFIERS:str=''):
		return self.ic_pb(smem_base('s_store_dwordx4', None, None, gfx10_sbase, gfx10_soffset_1, MODIFIERS))
class sop1_base(inst_base): 
	def __init__(self, INSTRUCTION:str, DST:Union[reg_block,None], SRC:Union[reg_block,None]): 
		super().__init__(instruction_type.SMEM, INSTRUCTION)
		self.DST = DST 
		self.SRC = SRC 
	def __str__(self): 
		args_l = filter(None.__ne__, [self.DST,self.SRC]) 
		return f"{self.label} {','.join(map(str, args_l))} " 
class sop1_instr_caller(inst_caller_base): 
	def __init__(self, insturction_container) -> None:
     		super().__init__(insturction_container)
	def s_abs_i32(self, gfx10_ssrc:reg_block):
		return self.ic_pb(sop1_base('s_abs_i32', None, gfx10_ssrc))
	def s_and_saveexec_b32(self, gfx10_ssrc:reg_block):
		return self.ic_pb(sop1_base('s_and_saveexec_b32', None, gfx10_ssrc))
	def s_and_saveexec_b64(self, gfx10_ssrc_1:reg_block):
		return self.ic_pb(sop1_base('s_and_saveexec_b64', None, gfx10_ssrc_1))
	def s_andn1_saveexec_b32(self, gfx10_ssrc:reg_block):
		return self.ic_pb(sop1_base('s_andn1_saveexec_b32', None, gfx10_ssrc))
	def s_andn1_saveexec_b64(self, gfx10_ssrc_1:reg_block):
		return self.ic_pb(sop1_base('s_andn1_saveexec_b64', None, gfx10_ssrc_1))
	def s_andn1_wrexec_b32(self, gfx10_ssrc:reg_block):
		return self.ic_pb(sop1_base('s_andn1_wrexec_b32', None, gfx10_ssrc))
	def s_andn1_wrexec_b64(self, gfx10_ssrc_1:reg_block):
		return self.ic_pb(sop1_base('s_andn1_wrexec_b64', None, gfx10_ssrc_1))
	def s_andn2_saveexec_b32(self, gfx10_ssrc:reg_block):
		return self.ic_pb(sop1_base('s_andn2_saveexec_b32', None, gfx10_ssrc))
	def s_andn2_saveexec_b64(self, gfx10_ssrc_1:reg_block):
		return self.ic_pb(sop1_base('s_andn2_saveexec_b64', None, gfx10_ssrc_1))
	def s_andn2_wrexec_b32(self, gfx10_ssrc:reg_block):
		return self.ic_pb(sop1_base('s_andn2_wrexec_b32', None, gfx10_ssrc))
	def s_andn2_wrexec_b64(self, gfx10_ssrc_1:reg_block):
		return self.ic_pb(sop1_base('s_andn2_wrexec_b64', None, gfx10_ssrc_1))
	def s_bcnt0_i32_b32(self, gfx10_ssrc:reg_block):
		return self.ic_pb(sop1_base('s_bcnt0_i32_b32', None, gfx10_ssrc))
	def s_bcnt0_i32_b64(self, gfx10_ssrc_1:reg_block):
		return self.ic_pb(sop1_base('s_bcnt0_i32_b64', None, gfx10_ssrc_1))
	def s_bcnt1_i32_b32(self, gfx10_ssrc:reg_block):
		return self.ic_pb(sop1_base('s_bcnt1_i32_b32', None, gfx10_ssrc))
	def s_bcnt1_i32_b64(self, gfx10_ssrc_1:reg_block):
		return self.ic_pb(sop1_base('s_bcnt1_i32_b64', None, gfx10_ssrc_1))
	def s_bitreplicate_b64_b32(self, gfx10_ssrc:reg_block):
		return self.ic_pb(sop1_base('s_bitreplicate_b64_b32', None, gfx10_ssrc))
	def s_bitset0_b32(self, gfx10_ssrc:reg_block):
		return self.ic_pb(sop1_base('s_bitset0_b32', None, gfx10_ssrc))
	def s_bitset0_b64(self, gfx10_ssrc:reg_block):
		return self.ic_pb(sop1_base('s_bitset0_b64', None, gfx10_ssrc))
	def s_bitset1_b32(self, gfx10_ssrc:reg_block):
		return self.ic_pb(sop1_base('s_bitset1_b32', None, gfx10_ssrc))
	def s_bitset1_b64(self, gfx10_ssrc:reg_block):
		return self.ic_pb(sop1_base('s_bitset1_b64', None, gfx10_ssrc))
	def s_brev_b32(self, gfx10_ssrc:reg_block):
		return self.ic_pb(sop1_base('s_brev_b32', None, gfx10_ssrc))
	def s_brev_b64(self, gfx10_ssrc_1:reg_block):
		return self.ic_pb(sop1_base('s_brev_b64', None, gfx10_ssrc_1))
	def s_cmov_b32(self, gfx10_ssrc:reg_block):
		return self.ic_pb(sop1_base('s_cmov_b32', None, gfx10_ssrc))
	def s_cmov_b64(self, gfx10_ssrc_1:reg_block):
		return self.ic_pb(sop1_base('s_cmov_b64', None, gfx10_ssrc_1))
	def s_ff0_i32_b32(self, gfx10_ssrc:reg_block):
		return self.ic_pb(sop1_base('s_ff0_i32_b32', None, gfx10_ssrc))
	def s_ff0_i32_b64(self, gfx10_ssrc_1:reg_block):
		return self.ic_pb(sop1_base('s_ff0_i32_b64', None, gfx10_ssrc_1))
	def s_ff1_i32_b32(self, gfx10_ssrc:reg_block):
		return self.ic_pb(sop1_base('s_ff1_i32_b32', None, gfx10_ssrc))
	def s_ff1_i32_b64(self, gfx10_ssrc_1:reg_block):
		return self.ic_pb(sop1_base('s_ff1_i32_b64', None, gfx10_ssrc_1))
	def s_flbit_i32(self, gfx10_ssrc:reg_block):
		return self.ic_pb(sop1_base('s_flbit_i32', None, gfx10_ssrc))
	def s_flbit_i32_b32(self, gfx10_ssrc:reg_block):
		return self.ic_pb(sop1_base('s_flbit_i32_b32', None, gfx10_ssrc))
	def s_flbit_i32_b64(self, gfx10_ssrc_1:reg_block):
		return self.ic_pb(sop1_base('s_flbit_i32_b64', None, gfx10_ssrc_1))
	def s_flbit_i32_i64(self, gfx10_ssrc_1:reg_block):
		return self.ic_pb(sop1_base('s_flbit_i32_i64', None, gfx10_ssrc_1))
	def s_getpc_b64(self):
		return self.ic_pb(sop1_base('s_getpc_b64', None, None))
	def s_mov_b32(self, gfx10_ssrc:reg_block):
		return self.ic_pb(sop1_base('s_mov_b32', None, gfx10_ssrc))
	def s_mov_b64(self, gfx10_ssrc_1:reg_block):
		return self.ic_pb(sop1_base('s_mov_b64', None, gfx10_ssrc_1))
	def s_movreld_b32(self, gfx10_ssrc:reg_block):
		return self.ic_pb(sop1_base('s_movreld_b32', None, gfx10_ssrc))
	def s_movreld_b64(self, gfx10_ssrc_1:reg_block):
		return self.ic_pb(sop1_base('s_movreld_b64', None, gfx10_ssrc_1))
	def s_movrels_b32(self, gfx10_ssrc_2:reg_block):
		return self.ic_pb(sop1_base('s_movrels_b32', None, gfx10_ssrc_2))
	def s_movrels_b64(self, gfx10_ssrc_3:reg_block):
		return self.ic_pb(sop1_base('s_movrels_b64', None, gfx10_ssrc_3))
	def s_movrelsd_2_b32(self, gfx10_ssrc_2:reg_block):
		return self.ic_pb(sop1_base('s_movrelsd_2_b32', None, gfx10_ssrc_2))
	def s_nand_saveexec_b32(self, gfx10_ssrc:reg_block):
		return self.ic_pb(sop1_base('s_nand_saveexec_b32', None, gfx10_ssrc))
	def s_nand_saveexec_b64(self, gfx10_ssrc_1:reg_block):
		return self.ic_pb(sop1_base('s_nand_saveexec_b64', None, gfx10_ssrc_1))
	def s_nor_saveexec_b32(self, gfx10_ssrc:reg_block):
		return self.ic_pb(sop1_base('s_nor_saveexec_b32', None, gfx10_ssrc))
	def s_nor_saveexec_b64(self, gfx10_ssrc_1:reg_block):
		return self.ic_pb(sop1_base('s_nor_saveexec_b64', None, gfx10_ssrc_1))
	def s_not_b32(self, gfx10_ssrc:reg_block):
		return self.ic_pb(sop1_base('s_not_b32', None, gfx10_ssrc))
	def s_not_b64(self, gfx10_ssrc_1:reg_block):
		return self.ic_pb(sop1_base('s_not_b64', None, gfx10_ssrc_1))
	def s_or_saveexec_b32(self, gfx10_ssrc:reg_block):
		return self.ic_pb(sop1_base('s_or_saveexec_b32', None, gfx10_ssrc))
	def s_or_saveexec_b64(self, gfx10_ssrc_1:reg_block):
		return self.ic_pb(sop1_base('s_or_saveexec_b64', None, gfx10_ssrc_1))
	def s_orn1_saveexec_b32(self, gfx10_ssrc:reg_block):
		return self.ic_pb(sop1_base('s_orn1_saveexec_b32', None, gfx10_ssrc))
	def s_orn1_saveexec_b64(self, gfx10_ssrc_1:reg_block):
		return self.ic_pb(sop1_base('s_orn1_saveexec_b64', None, gfx10_ssrc_1))
	def s_orn2_saveexec_b32(self, gfx10_ssrc:reg_block):
		return self.ic_pb(sop1_base('s_orn2_saveexec_b32', None, gfx10_ssrc))
	def s_orn2_saveexec_b64(self, gfx10_ssrc_1:reg_block):
		return self.ic_pb(sop1_base('s_orn2_saveexec_b64', None, gfx10_ssrc_1))
	def s_quadmask_b32(self, gfx10_ssrc:reg_block):
		return self.ic_pb(sop1_base('s_quadmask_b32', None, gfx10_ssrc))
	def s_quadmask_b64(self, gfx10_ssrc_1:reg_block):
		return self.ic_pb(sop1_base('s_quadmask_b64', None, gfx10_ssrc_1))
	def s_rfe_b64(self):
		return self.ic_pb(sop1_base('s_rfe_b64', None, None))
	def s_setpc_b64(self):
		return self.ic_pb(sop1_base('s_setpc_b64', None, None))
	def s_sext_i32_i16(self, gfx10_ssrc:reg_block):
		return self.ic_pb(sop1_base('s_sext_i32_i16', None, gfx10_ssrc))
	def s_sext_i32_i8(self, gfx10_ssrc:reg_block):
		return self.ic_pb(sop1_base('s_sext_i32_i8', None, gfx10_ssrc))
	def s_swappc_b64(self, gfx10_ssrc_3:reg_block):
		return self.ic_pb(sop1_base('s_swappc_b64', None, gfx10_ssrc_3))
	def s_wqm_b32(self, gfx10_ssrc:reg_block):
		return self.ic_pb(sop1_base('s_wqm_b32', None, gfx10_ssrc))
	def s_wqm_b64(self, gfx10_ssrc_1:reg_block):
		return self.ic_pb(sop1_base('s_wqm_b64', None, gfx10_ssrc_1))
	def s_xnor_saveexec_b32(self, gfx10_ssrc:reg_block):
		return self.ic_pb(sop1_base('s_xnor_saveexec_b32', None, gfx10_ssrc))
	def s_xnor_saveexec_b64(self, gfx10_ssrc_1:reg_block):
		return self.ic_pb(sop1_base('s_xnor_saveexec_b64', None, gfx10_ssrc_1))
	def s_xor_saveexec_b32(self, gfx10_ssrc:reg_block):
		return self.ic_pb(sop1_base('s_xor_saveexec_b32', None, gfx10_ssrc))
	def s_xor_saveexec_b64(self, gfx10_ssrc_1:reg_block):
		return self.ic_pb(sop1_base('s_xor_saveexec_b64', None, gfx10_ssrc_1))
class sop2_base(inst_base): 
	def __init__(self, INSTRUCTION:str, DST:Union[reg_block,None], SRC0:Union[reg_block,None], SRC1:Union[reg_block,None]): 
		super().__init__(instruction_type.SMEM, INSTRUCTION)
		self.DST = DST 
		self.SRC0 = SRC0 
		self.SRC1 = SRC1 
	def __str__(self): 
		args_l = filter(None.__ne__, [self.DST,self.SRC0,self.SRC1]) 
		return f"{self.label} {','.join(map(str, args_l))} " 
class sop2_instr_caller(inst_caller_base): 
	def __init__(self, insturction_container) -> None:
     		super().__init__(insturction_container)
	def s_absdiff_i32(self, gfx10_ssrc:reg_block, gfx10_ssrc:reg_block):
		return self.ic_pb(sop2_base('s_absdiff_i32', None, gfx10_ssrc, gfx10_ssrc))
	def s_add_i32(self, gfx10_ssrc:reg_block, gfx10_ssrc:reg_block):
		return self.ic_pb(sop2_base('s_add_i32', None, gfx10_ssrc, gfx10_ssrc))
	def s_add_u32(self, gfx10_ssrc:reg_block, gfx10_ssrc:reg_block):
		return self.ic_pb(sop2_base('s_add_u32', None, gfx10_ssrc, gfx10_ssrc))
	def s_addc_u32(self, gfx10_ssrc:reg_block, gfx10_ssrc:reg_block):
		return self.ic_pb(sop2_base('s_addc_u32', None, gfx10_ssrc, gfx10_ssrc))
	def s_and_b32(self, gfx10_ssrc:reg_block, gfx10_ssrc:reg_block):
		return self.ic_pb(sop2_base('s_and_b32', None, gfx10_ssrc, gfx10_ssrc))
	def s_and_b64(self, gfx10_ssrc_1:reg_block, gfx10_ssrc_1:reg_block):
		return self.ic_pb(sop2_base('s_and_b64', None, gfx10_ssrc_1, gfx10_ssrc_1))
	def s_andn2_b32(self, gfx10_ssrc:reg_block, gfx10_ssrc:reg_block):
		return self.ic_pb(sop2_base('s_andn2_b32', None, gfx10_ssrc, gfx10_ssrc))
	def s_andn2_b64(self, gfx10_ssrc_1:reg_block, gfx10_ssrc_1:reg_block):
		return self.ic_pb(sop2_base('s_andn2_b64', None, gfx10_ssrc_1, gfx10_ssrc_1))
	def s_ashr_i32(self, gfx10_ssrc:reg_block, gfx10_ssrc:reg_block):
		return self.ic_pb(sop2_base('s_ashr_i32', None, gfx10_ssrc, gfx10_ssrc))
	def s_ashr_i64(self, gfx10_ssrc_1:reg_block, gfx10_ssrc:reg_block):
		return self.ic_pb(sop2_base('s_ashr_i64', None, gfx10_ssrc_1, gfx10_ssrc))
	def s_bfe_i32(self, gfx10_ssrc:reg_block, gfx10_ssrc:reg_block):
		return self.ic_pb(sop2_base('s_bfe_i32', None, gfx10_ssrc, gfx10_ssrc))
	def s_bfe_i64(self, gfx10_ssrc_1:reg_block, gfx10_ssrc:reg_block):
		return self.ic_pb(sop2_base('s_bfe_i64', None, gfx10_ssrc_1, gfx10_ssrc))
	def s_bfe_u32(self, gfx10_ssrc:reg_block, gfx10_ssrc:reg_block):
		return self.ic_pb(sop2_base('s_bfe_u32', None, gfx10_ssrc, gfx10_ssrc))
	def s_bfe_u64(self, gfx10_ssrc_1:reg_block, gfx10_ssrc:reg_block):
		return self.ic_pb(sop2_base('s_bfe_u64', None, gfx10_ssrc_1, gfx10_ssrc))
	def s_bfm_b32(self, gfx10_ssrc:reg_block, gfx10_ssrc:reg_block):
		return self.ic_pb(sop2_base('s_bfm_b32', None, gfx10_ssrc, gfx10_ssrc))
	def s_bfm_b64(self, gfx10_ssrc:reg_block, gfx10_ssrc:reg_block):
		return self.ic_pb(sop2_base('s_bfm_b64', None, gfx10_ssrc, gfx10_ssrc))
	def s_cselect_b32(self, gfx10_ssrc:reg_block, gfx10_ssrc:reg_block):
		return self.ic_pb(sop2_base('s_cselect_b32', None, gfx10_ssrc, gfx10_ssrc))
	def s_cselect_b64(self, gfx10_ssrc_1:reg_block, gfx10_ssrc_1:reg_block):
		return self.ic_pb(sop2_base('s_cselect_b64', None, gfx10_ssrc_1, gfx10_ssrc_1))
	def s_lshl1_add_u32(self, gfx10_ssrc:reg_block, gfx10_ssrc:reg_block):
		return self.ic_pb(sop2_base('s_lshl1_add_u32', None, gfx10_ssrc, gfx10_ssrc))
	def s_lshl2_add_u32(self, gfx10_ssrc:reg_block, gfx10_ssrc:reg_block):
		return self.ic_pb(sop2_base('s_lshl2_add_u32', None, gfx10_ssrc, gfx10_ssrc))
	def s_lshl3_add_u32(self, gfx10_ssrc:reg_block, gfx10_ssrc:reg_block):
		return self.ic_pb(sop2_base('s_lshl3_add_u32', None, gfx10_ssrc, gfx10_ssrc))
	def s_lshl4_add_u32(self, gfx10_ssrc:reg_block, gfx10_ssrc:reg_block):
		return self.ic_pb(sop2_base('s_lshl4_add_u32', None, gfx10_ssrc, gfx10_ssrc))
	def s_lshl_b32(self, gfx10_ssrc:reg_block, gfx10_ssrc:reg_block):
		return self.ic_pb(sop2_base('s_lshl_b32', None, gfx10_ssrc, gfx10_ssrc))
	def s_lshl_b64(self, gfx10_ssrc_1:reg_block, gfx10_ssrc:reg_block):
		return self.ic_pb(sop2_base('s_lshl_b64', None, gfx10_ssrc_1, gfx10_ssrc))
	def s_lshr_b32(self, gfx10_ssrc:reg_block, gfx10_ssrc:reg_block):
		return self.ic_pb(sop2_base('s_lshr_b32', None, gfx10_ssrc, gfx10_ssrc))
	def s_lshr_b64(self, gfx10_ssrc_1:reg_block, gfx10_ssrc:reg_block):
		return self.ic_pb(sop2_base('s_lshr_b64', None, gfx10_ssrc_1, gfx10_ssrc))
	def s_max_i32(self, gfx10_ssrc:reg_block, gfx10_ssrc:reg_block):
		return self.ic_pb(sop2_base('s_max_i32', None, gfx10_ssrc, gfx10_ssrc))
	def s_max_u32(self, gfx10_ssrc:reg_block, gfx10_ssrc:reg_block):
		return self.ic_pb(sop2_base('s_max_u32', None, gfx10_ssrc, gfx10_ssrc))
	def s_min_i32(self, gfx10_ssrc:reg_block, gfx10_ssrc:reg_block):
		return self.ic_pb(sop2_base('s_min_i32', None, gfx10_ssrc, gfx10_ssrc))
	def s_min_u32(self, gfx10_ssrc:reg_block, gfx10_ssrc:reg_block):
		return self.ic_pb(sop2_base('s_min_u32', None, gfx10_ssrc, gfx10_ssrc))
	def s_mul_hi_i32(self, gfx10_ssrc:reg_block, gfx10_ssrc:reg_block):
		return self.ic_pb(sop2_base('s_mul_hi_i32', None, gfx10_ssrc, gfx10_ssrc))
	def s_mul_hi_u32(self, gfx10_ssrc:reg_block, gfx10_ssrc:reg_block):
		return self.ic_pb(sop2_base('s_mul_hi_u32', None, gfx10_ssrc, gfx10_ssrc))
	def s_mul_i32(self, gfx10_ssrc:reg_block, gfx10_ssrc:reg_block):
		return self.ic_pb(sop2_base('s_mul_i32', None, gfx10_ssrc, gfx10_ssrc))
	def s_nand_b32(self, gfx10_ssrc:reg_block, gfx10_ssrc:reg_block):
		return self.ic_pb(sop2_base('s_nand_b32', None, gfx10_ssrc, gfx10_ssrc))
	def s_nand_b64(self, gfx10_ssrc_1:reg_block, gfx10_ssrc_1:reg_block):
		return self.ic_pb(sop2_base('s_nand_b64', None, gfx10_ssrc_1, gfx10_ssrc_1))
	def s_nor_b32(self, gfx10_ssrc:reg_block, gfx10_ssrc:reg_block):
		return self.ic_pb(sop2_base('s_nor_b32', None, gfx10_ssrc, gfx10_ssrc))
	def s_nor_b64(self, gfx10_ssrc_1:reg_block, gfx10_ssrc_1:reg_block):
		return self.ic_pb(sop2_base('s_nor_b64', None, gfx10_ssrc_1, gfx10_ssrc_1))
	def s_or_b32(self, gfx10_ssrc:reg_block, gfx10_ssrc:reg_block):
		return self.ic_pb(sop2_base('s_or_b32', None, gfx10_ssrc, gfx10_ssrc))
	def s_or_b64(self, gfx10_ssrc_1:reg_block, gfx10_ssrc_1:reg_block):
		return self.ic_pb(sop2_base('s_or_b64', None, gfx10_ssrc_1, gfx10_ssrc_1))
	def s_orn2_b32(self, gfx10_ssrc:reg_block, gfx10_ssrc:reg_block):
		return self.ic_pb(sop2_base('s_orn2_b32', None, gfx10_ssrc, gfx10_ssrc))
	def s_orn2_b64(self, gfx10_ssrc_1:reg_block, gfx10_ssrc_1:reg_block):
		return self.ic_pb(sop2_base('s_orn2_b64', None, gfx10_ssrc_1, gfx10_ssrc_1))
	def s_pack_hh_b32_b16(self, gfx10_ssrc:reg_block, gfx10_ssrc:reg_block):
		return self.ic_pb(sop2_base('s_pack_hh_b32_b16', None, gfx10_ssrc, gfx10_ssrc))
	def s_pack_lh_b32_b16(self, gfx10_ssrc:reg_block, gfx10_ssrc:reg_block):
		return self.ic_pb(sop2_base('s_pack_lh_b32_b16', None, gfx10_ssrc, gfx10_ssrc))
	def s_pack_ll_b32_b16(self, gfx10_ssrc:reg_block, gfx10_ssrc:reg_block):
		return self.ic_pb(sop2_base('s_pack_ll_b32_b16', None, gfx10_ssrc, gfx10_ssrc))
	def s_sub_i32(self, gfx10_ssrc:reg_block, gfx10_ssrc:reg_block):
		return self.ic_pb(sop2_base('s_sub_i32', None, gfx10_ssrc, gfx10_ssrc))
	def s_sub_u32(self, gfx10_ssrc:reg_block, gfx10_ssrc:reg_block):
		return self.ic_pb(sop2_base('s_sub_u32', None, gfx10_ssrc, gfx10_ssrc))
	def s_subb_u32(self, gfx10_ssrc:reg_block, gfx10_ssrc:reg_block):
		return self.ic_pb(sop2_base('s_subb_u32', None, gfx10_ssrc, gfx10_ssrc))
	def s_xnor_b32(self, gfx10_ssrc:reg_block, gfx10_ssrc:reg_block):
		return self.ic_pb(sop2_base('s_xnor_b32', None, gfx10_ssrc, gfx10_ssrc))
	def s_xnor_b64(self, gfx10_ssrc_1:reg_block, gfx10_ssrc_1:reg_block):
		return self.ic_pb(sop2_base('s_xnor_b64', None, gfx10_ssrc_1, gfx10_ssrc_1))
	def s_xor_b32(self, gfx10_ssrc:reg_block, gfx10_ssrc:reg_block):
		return self.ic_pb(sop2_base('s_xor_b32', None, gfx10_ssrc, gfx10_ssrc))
	def s_xor_b64(self, gfx10_ssrc_1:reg_block, gfx10_ssrc_1:reg_block):
		return self.ic_pb(sop2_base('s_xor_b64', None, gfx10_ssrc_1, gfx10_ssrc_1))
class sopc_base(inst_base): 
	def __init__(self, INSTRUCTION:str, SRC0:Union[reg_block,None], SRC1:Union[reg_block,None]): 
		super().__init__(instruction_type.SMEM, INSTRUCTION)
		self.SRC0 = SRC0 
		self.SRC1 = SRC1 
	def __str__(self): 
		args_l = filter(None.__ne__, [self.SRC0,self.SRC1]) 
		return f"{self.label} {','.join(map(str, args_l))} " 
class sopc_instr_caller(inst_caller_base): 
	def __init__(self, insturction_container) -> None:
     		super().__init__(insturction_container)
	def s_bitcmp0_b32(self, gfx10_ssrc:reg_block):
		return self.ic_pb(sopc_base('s_bitcmp0_b32', None, gfx10_ssrc))
	def s_bitcmp0_b64(self, gfx10_ssrc:reg_block):
		return self.ic_pb(sopc_base('s_bitcmp0_b64', None, gfx10_ssrc))
	def s_bitcmp1_b32(self, gfx10_ssrc:reg_block):
		return self.ic_pb(sopc_base('s_bitcmp1_b32', None, gfx10_ssrc))
	def s_bitcmp1_b64(self, gfx10_ssrc:reg_block):
		return self.ic_pb(sopc_base('s_bitcmp1_b64', None, gfx10_ssrc))
	def s_cmp_eq_i32(self, gfx10_ssrc:reg_block):
		return self.ic_pb(sopc_base('s_cmp_eq_i32', None, gfx10_ssrc))
	def s_cmp_eq_u32(self, gfx10_ssrc:reg_block):
		return self.ic_pb(sopc_base('s_cmp_eq_u32', None, gfx10_ssrc))
	def s_cmp_eq_u64(self, gfx10_ssrc_1:reg_block):
		return self.ic_pb(sopc_base('s_cmp_eq_u64', None, gfx10_ssrc_1))
	def s_cmp_ge_i32(self, gfx10_ssrc:reg_block):
		return self.ic_pb(sopc_base('s_cmp_ge_i32', None, gfx10_ssrc))
	def s_cmp_ge_u32(self, gfx10_ssrc:reg_block):
		return self.ic_pb(sopc_base('s_cmp_ge_u32', None, gfx10_ssrc))
	def s_cmp_gt_i32(self, gfx10_ssrc:reg_block):
		return self.ic_pb(sopc_base('s_cmp_gt_i32', None, gfx10_ssrc))
	def s_cmp_gt_u32(self, gfx10_ssrc:reg_block):
		return self.ic_pb(sopc_base('s_cmp_gt_u32', None, gfx10_ssrc))
	def s_cmp_le_i32(self, gfx10_ssrc:reg_block):
		return self.ic_pb(sopc_base('s_cmp_le_i32', None, gfx10_ssrc))
	def s_cmp_le_u32(self, gfx10_ssrc:reg_block):
		return self.ic_pb(sopc_base('s_cmp_le_u32', None, gfx10_ssrc))
	def s_cmp_lg_i32(self, gfx10_ssrc:reg_block):
		return self.ic_pb(sopc_base('s_cmp_lg_i32', None, gfx10_ssrc))
	def s_cmp_lg_u32(self, gfx10_ssrc:reg_block):
		return self.ic_pb(sopc_base('s_cmp_lg_u32', None, gfx10_ssrc))
	def s_cmp_lg_u64(self, gfx10_ssrc_1:reg_block):
		return self.ic_pb(sopc_base('s_cmp_lg_u64', None, gfx10_ssrc_1))
	def s_cmp_lt_i32(self, gfx10_ssrc:reg_block):
		return self.ic_pb(sopc_base('s_cmp_lt_i32', None, gfx10_ssrc))
	def s_cmp_lt_u32(self, gfx10_ssrc:reg_block):
		return self.ic_pb(sopc_base('s_cmp_lt_u32', None, gfx10_ssrc))
class sopk_base(inst_base): 
	def __init__(self, INSTRUCTION:str, DST:Union[reg_block,None], SRC0:Union[reg_block,None], SRC1:Union[reg_block,None]): 
		super().__init__(instruction_type.SMEM, INSTRUCTION)
		self.DST = DST 
		self.SRC0 = SRC0 
		self.SRC1 = SRC1 
	def __str__(self): 
		args_l = filter(None.__ne__, [self.DST,self.SRC0,self.SRC1]) 
		return f"{self.label} {','.join(map(str, args_l))} " 
class sopk_instr_caller(inst_caller_base): 
	def __init__(self, insturction_container) -> None:
     		super().__init__(insturction_container)
	def s_addk_i32(self, gfx10_imm16:reg_block):
		return self.ic_pb(sopk_base('s_addk_i32', None, gfx10_imm16, None))
	def s_call_b64(self, gfx10_label:reg_block):
		return self.ic_pb(sopk_base('s_call_b64', None, gfx10_label, None))
	def s_cmovk_i32(self, gfx10_imm16:reg_block):
		return self.ic_pb(sopk_base('s_cmovk_i32', None, gfx10_imm16, None))
	def s_cmpk_eq_i32(self, gfx10_imm16:reg_block):
		return self.ic_pb(sopk_base('s_cmpk_eq_i32', None, None, gfx10_imm16))
	def s_cmpk_eq_u32(self, gfx10_imm16_1:reg_block):
		return self.ic_pb(sopk_base('s_cmpk_eq_u32', None, None, gfx10_imm16_1))
	def s_cmpk_ge_i32(self, gfx10_imm16:reg_block):
		return self.ic_pb(sopk_base('s_cmpk_ge_i32', None, None, gfx10_imm16))
	def s_cmpk_ge_u32(self, gfx10_imm16_1:reg_block):
		return self.ic_pb(sopk_base('s_cmpk_ge_u32', None, None, gfx10_imm16_1))
	def s_cmpk_gt_i32(self, gfx10_imm16:reg_block):
		return self.ic_pb(sopk_base('s_cmpk_gt_i32', None, None, gfx10_imm16))
	def s_cmpk_gt_u32(self, gfx10_imm16_1:reg_block):
		return self.ic_pb(sopk_base('s_cmpk_gt_u32', None, None, gfx10_imm16_1))
	def s_cmpk_le_i32(self, gfx10_imm16:reg_block):
		return self.ic_pb(sopk_base('s_cmpk_le_i32', None, None, gfx10_imm16))
	def s_cmpk_le_u32(self, gfx10_imm16_1:reg_block):
		return self.ic_pb(sopk_base('s_cmpk_le_u32', None, None, gfx10_imm16_1))
	def s_cmpk_lg_i32(self, gfx10_imm16:reg_block):
		return self.ic_pb(sopk_base('s_cmpk_lg_i32', None, None, gfx10_imm16))
	def s_cmpk_lg_u32(self, gfx10_imm16_1:reg_block):
		return self.ic_pb(sopk_base('s_cmpk_lg_u32', None, None, gfx10_imm16_1))
	def s_cmpk_lt_i32(self, gfx10_imm16:reg_block):
		return self.ic_pb(sopk_base('s_cmpk_lt_i32', None, None, gfx10_imm16))
	def s_cmpk_lt_u32(self, gfx10_imm16_1:reg_block):
		return self.ic_pb(sopk_base('s_cmpk_lt_u32', None, None, gfx10_imm16_1))
	def s_getreg_b32(self, gfx10_hwreg:reg_block):
		return self.ic_pb(sopk_base('s_getreg_b32', None, gfx10_hwreg, None))
	def s_movk_i32(self, gfx10_imm16:reg_block):
		return self.ic_pb(sopk_base('s_movk_i32', None, gfx10_imm16, None))
	def s_mulk_i32(self, gfx10_imm16:reg_block):
		return self.ic_pb(sopk_base('s_mulk_i32', None, gfx10_imm16, None))
	def s_setreg_b32(self, gfx10_ssrc_4:reg_block):
		return self.ic_pb(sopk_base('s_setreg_b32', None, gfx10_ssrc_4, None))
	def s_setreg_imm32_b32(self, gfx10_simm32:reg_block):
		return self.ic_pb(sopk_base('s_setreg_imm32_b32', None, gfx10_simm32, None))
	def s_subvector_loop_begin(self, gfx10_label:reg_block):
		return self.ic_pb(sopk_base('s_subvector_loop_begin', None, gfx10_label, None))
	def s_subvector_loop_end(self, gfx10_label:reg_block):
		return self.ic_pb(sopk_base('s_subvector_loop_end', None, gfx10_label, None))
	def s_version(self):
		return self.ic_pb(sopk_base('s_version', None, None, None))
	def s_waitcnt_expcnt(self, gfx10_imm16_2:reg_block):
		return self.ic_pb(sopk_base('s_waitcnt_expcnt', None, None, gfx10_imm16_2))
	def s_waitcnt_lgkmcnt(self, gfx10_imm16_2:reg_block):
		return self.ic_pb(sopk_base('s_waitcnt_lgkmcnt', None, None, gfx10_imm16_2))
	def s_waitcnt_vmcnt(self, gfx10_imm16_2:reg_block):
		return self.ic_pb(sopk_base('s_waitcnt_vmcnt', None, None, gfx10_imm16_2))
	def s_waitcnt_vscnt(self, gfx10_imm16_2:reg_block):
		return self.ic_pb(sopk_base('s_waitcnt_vscnt', None, None, gfx10_imm16_2))
class sopp_base(inst_base): 
	def __init__(self, INSTRUCTION:str, SRC:Union[reg_block,None]): 
		super().__init__(instruction_type.SMEM, INSTRUCTION)
		self.SRC = SRC 
	def __str__(self): 
		args_l = filter(None.__ne__, [self.SRC]) 
		return f"{self.label} {','.join(map(str, args_l))} " 
class sopp_instr_caller(inst_caller_base): 
	def __init__(self, insturction_container) -> None:
     		super().__init__(insturction_container)
	def s_barrier(self):
		return self.ic_pb(sopp_base('s_barrier', None))
	def s_branch(self):
		return self.ic_pb(sopp_base('s_branch', None))
	def s_cbranch_cdbgsys(self):
		return self.ic_pb(sopp_base('s_cbranch_cdbgsys', None))
	def s_cbranch_cdbgsys_and_user(self):
		return self.ic_pb(sopp_base('s_cbranch_cdbgsys_and_user', None))
	def s_cbranch_cdbgsys_or_user(self):
		return self.ic_pb(sopp_base('s_cbranch_cdbgsys_or_user', None))
	def s_cbranch_cdbguser(self):
		return self.ic_pb(sopp_base('s_cbranch_cdbguser', None))
	def s_cbranch_execnz(self):
		return self.ic_pb(sopp_base('s_cbranch_execnz', None))
	def s_cbranch_execz(self):
		return self.ic_pb(sopp_base('s_cbranch_execz', None))
	def s_cbranch_scc0(self):
		return self.ic_pb(sopp_base('s_cbranch_scc0', None))
	def s_cbranch_scc1(self):
		return self.ic_pb(sopp_base('s_cbranch_scc1', None))
	def s_cbranch_vccnz(self):
		return self.ic_pb(sopp_base('s_cbranch_vccnz', None))
	def s_cbranch_vccz(self):
		return self.ic_pb(sopp_base('s_cbranch_vccz', None))
	def s_clause(self):
		return self.ic_pb(sopp_base('s_clause', None))
	def s_code_end(self):
		return self.ic_pb(sopp_base('s_code_end', None))
	def s_decperflevel(self):
		return self.ic_pb(sopp_base('s_decperflevel', None))
	def s_denorm_mode(self):
		return self.ic_pb(sopp_base('s_denorm_mode', None))
	def s_endpgm(self):
		return self.ic_pb(sopp_base('s_endpgm', None))
	def s_endpgm_ordered_ps_done(self):
		return self.ic_pb(sopp_base('s_endpgm_ordered_ps_done', None))
	def s_endpgm_saved(self):
		return self.ic_pb(sopp_base('s_endpgm_saved', None))
	def s_icache_inv(self):
		return self.ic_pb(sopp_base('s_icache_inv', None))
	def s_incperflevel(self):
		return self.ic_pb(sopp_base('s_incperflevel', None))
	def s_inst_prefetch(self):
		return self.ic_pb(sopp_base('s_inst_prefetch', None))
	def s_nop(self):
		return self.ic_pb(sopp_base('s_nop', None))
	def s_round_mode(self):
		return self.ic_pb(sopp_base('s_round_mode', None))
	def s_sendmsg(self):
		return self.ic_pb(sopp_base('s_sendmsg', None))
	def s_sendmsghalt(self):
		return self.ic_pb(sopp_base('s_sendmsghalt', None))
	def s_sethalt(self):
		return self.ic_pb(sopp_base('s_sethalt', None))
	def s_setkill(self):
		return self.ic_pb(sopp_base('s_setkill', None))
	def s_setprio(self):
		return self.ic_pb(sopp_base('s_setprio', None))
	def s_sleep(self):
		return self.ic_pb(sopp_base('s_sleep', None))
	def s_trap(self):
		return self.ic_pb(sopp_base('s_trap', None))
	def s_ttracedata(self):
		return self.ic_pb(sopp_base('s_ttracedata', None))
	def s_ttracedata_imm(self):
		return self.ic_pb(sopp_base('s_ttracedata_imm', None))
	def s_waitcnt(self):
		return self.ic_pb(sopp_base('s_waitcnt', None))
	def s_wakeup(self):
		return self.ic_pb(sopp_base('s_wakeup', None))
class vintrp_base(inst_base): 
	def __init__(self, INSTRUCTION:str, DST:Union[reg_block,None], SRC0:Union[reg_block,None], SRC1:Union[reg_block,None]): 
		super().__init__(instruction_type.SMEM, INSTRUCTION)
		self.DST = DST 
		self.SRC0 = SRC0 
		self.SRC1 = SRC1 
	def __str__(self): 
		args_l = filter(None.__ne__, [self.DST,self.SRC0,self.SRC1]) 
		return f"{self.label} {','.join(map(str, args_l))} " 
class vintrp_instr_caller(inst_caller_base): 
	def __init__(self, insturction_container) -> None:
     		super().__init__(insturction_container)
	def v_interp_mov_f32(self, gfx10_param:reg_block, gfx10_attr:reg_block):
		return self.ic_pb(vintrp_base('v_interp_mov_f32', None, gfx10_param, gfx10_attr))
	def v_interp_p1_f32(self, gfx10_vsrc:reg_block, gfx10_attr:reg_block):
		return self.ic_pb(vintrp_base('v_interp_p1_f32', None, gfx10_vsrc, gfx10_attr))
	def v_interp_p2_f32(self, gfx10_vsrc:reg_block, gfx10_attr:reg_block):
		return self.ic_pb(vintrp_base('v_interp_p2_f32', None, gfx10_vsrc, gfx10_attr))
class vop1_base(inst_base): 
	def __init__(self, INSTRUCTION:str, DST:Union[reg_block,None], SRC:Union[reg_block,None]): 
		super().__init__(instruction_type.SMEM, INSTRUCTION)
		self.DST = DST 
		self.SRC = SRC 
	def __str__(self): 
		args_l = filter(None.__ne__, [self.DST,self.SRC]) 
		return f"{self.label} {','.join(map(str, args_l))} " 
class vop1_instr_caller(inst_caller_base): 
	def __init__(self, insturction_container) -> None:
     		super().__init__(insturction_container)
	def v_bfrev_b32(self, gfx10_src_2:reg_block):
		return self.ic_pb(vop1_base('v_bfrev_b32', None, gfx10_src_2))
	def v_ceil_f16(self, gfx10_src_2:reg_block):
		return self.ic_pb(vop1_base('v_ceil_f16', None, gfx10_src_2))
	def v_ceil_f32(self, gfx10_src_2:reg_block):
		return self.ic_pb(vop1_base('v_ceil_f32', None, gfx10_src_2))
	def v_ceil_f64(self, gfx10_src_3:reg_block):
		return self.ic_pb(vop1_base('v_ceil_f64', None, gfx10_src_3))
	def v_clrexcp(self):
		return self.ic_pb(vop1_base('v_clrexcp', None, None))
	def v_cos_f16(self, gfx10_src_2:reg_block):
		return self.ic_pb(vop1_base('v_cos_f16', None, gfx10_src_2))
	def v_cos_f32(self, gfx10_src_2:reg_block):
		return self.ic_pb(vop1_base('v_cos_f32', None, gfx10_src_2))
	def v_cvt_f16_f32(self, gfx10_src_2:reg_block):
		return self.ic_pb(vop1_base('v_cvt_f16_f32', None, gfx10_src_2))
	def v_cvt_f16_i16(self, gfx10_src_4:reg_block):
		return self.ic_pb(vop1_base('v_cvt_f16_i16', None, gfx10_src_4))
	def v_cvt_f16_u16(self, gfx10_src_4:reg_block):
		return self.ic_pb(vop1_base('v_cvt_f16_u16', None, gfx10_src_4))
	def v_cvt_f32_f16(self, gfx10_src_2:reg_block):
		return self.ic_pb(vop1_base('v_cvt_f32_f16', None, gfx10_src_2))
	def v_cvt_f32_f64(self, gfx10_src_3:reg_block):
		return self.ic_pb(vop1_base('v_cvt_f32_f64', None, gfx10_src_3))
	def v_cvt_f32_i32(self, gfx10_src_2:reg_block):
		return self.ic_pb(vop1_base('v_cvt_f32_i32', None, gfx10_src_2))
	def v_cvt_f32_u32(self, gfx10_src_2:reg_block):
		return self.ic_pb(vop1_base('v_cvt_f32_u32', None, gfx10_src_2))
	def v_cvt_f32_ubyte0(self, gfx10_src_2:reg_block):
		return self.ic_pb(vop1_base('v_cvt_f32_ubyte0', None, gfx10_src_2))
	def v_cvt_f32_ubyte1(self, gfx10_src_2:reg_block):
		return self.ic_pb(vop1_base('v_cvt_f32_ubyte1', None, gfx10_src_2))
	def v_cvt_f32_ubyte2(self, gfx10_src_2:reg_block):
		return self.ic_pb(vop1_base('v_cvt_f32_ubyte2', None, gfx10_src_2))
	def v_cvt_f32_ubyte3(self, gfx10_src_2:reg_block):
		return self.ic_pb(vop1_base('v_cvt_f32_ubyte3', None, gfx10_src_2))
	def v_cvt_f64_f32(self, gfx10_src_2:reg_block):
		return self.ic_pb(vop1_base('v_cvt_f64_f32', None, gfx10_src_2))
	def v_cvt_f64_i32(self, gfx10_src_2:reg_block):
		return self.ic_pb(vop1_base('v_cvt_f64_i32', None, gfx10_src_2))
	def v_cvt_f64_u32(self, gfx10_src_2:reg_block):
		return self.ic_pb(vop1_base('v_cvt_f64_u32', None, gfx10_src_2))
	def v_cvt_flr_i32_f32(self, gfx10_src_2:reg_block):
		return self.ic_pb(vop1_base('v_cvt_flr_i32_f32', None, gfx10_src_2))
	def v_cvt_i16_f16(self, gfx10_src_2:reg_block):
		return self.ic_pb(vop1_base('v_cvt_i16_f16', None, gfx10_src_2))
	def v_cvt_i32_f32(self, gfx10_src_2:reg_block):
		return self.ic_pb(vop1_base('v_cvt_i32_f32', None, gfx10_src_2))
	def v_cvt_i32_f64(self, gfx10_src_3:reg_block):
		return self.ic_pb(vop1_base('v_cvt_i32_f64', None, gfx10_src_3))
	def v_cvt_norm_i16_f16(self, gfx10_src_2:reg_block):
		return self.ic_pb(vop1_base('v_cvt_norm_i16_f16', None, gfx10_src_2))
	def v_cvt_norm_u16_f16(self, gfx10_src_2:reg_block):
		return self.ic_pb(vop1_base('v_cvt_norm_u16_f16', None, gfx10_src_2))
	def v_cvt_off_f32_i4(self, gfx10_src_2:reg_block):
		return self.ic_pb(vop1_base('v_cvt_off_f32_i4', None, gfx10_src_2))
	def v_cvt_rpi_i32_f32(self, gfx10_src_2:reg_block):
		return self.ic_pb(vop1_base('v_cvt_rpi_i32_f32', None, gfx10_src_2))
	def v_cvt_u16_f16(self, gfx10_src_2:reg_block):
		return self.ic_pb(vop1_base('v_cvt_u16_f16', None, gfx10_src_2))
	def v_cvt_u32_f32(self, gfx10_src_2:reg_block):
		return self.ic_pb(vop1_base('v_cvt_u32_f32', None, gfx10_src_2))
	def v_cvt_u32_f64(self, gfx10_src_3:reg_block):
		return self.ic_pb(vop1_base('v_cvt_u32_f64', None, gfx10_src_3))
	def v_exp_f16(self, gfx10_src_2:reg_block):
		return self.ic_pb(vop1_base('v_exp_f16', None, gfx10_src_2))
	def v_exp_f32(self, gfx10_src_2:reg_block):
		return self.ic_pb(vop1_base('v_exp_f32', None, gfx10_src_2))
	def v_ffbh_i32(self, gfx10_src_2:reg_block):
		return self.ic_pb(vop1_base('v_ffbh_i32', None, gfx10_src_2))
	def v_ffbh_u32(self, gfx10_src_2:reg_block):
		return self.ic_pb(vop1_base('v_ffbh_u32', None, gfx10_src_2))
	def v_ffbl_b32(self, gfx10_src_2:reg_block):
		return self.ic_pb(vop1_base('v_ffbl_b32', None, gfx10_src_2))
	def v_floor_f16(self, gfx10_src_2:reg_block):
		return self.ic_pb(vop1_base('v_floor_f16', None, gfx10_src_2))
	def v_floor_f32(self, gfx10_src_2:reg_block):
		return self.ic_pb(vop1_base('v_floor_f32', None, gfx10_src_2))
	def v_floor_f64(self, gfx10_src_3:reg_block):
		return self.ic_pb(vop1_base('v_floor_f64', None, gfx10_src_3))
	def v_fract_f16(self, gfx10_src_2:reg_block):
		return self.ic_pb(vop1_base('v_fract_f16', None, gfx10_src_2))
	def v_fract_f32(self, gfx10_src_2:reg_block):
		return self.ic_pb(vop1_base('v_fract_f32', None, gfx10_src_2))
	def v_fract_f64(self, gfx10_src_3:reg_block):
		return self.ic_pb(vop1_base('v_fract_f64', None, gfx10_src_3))
	def v_frexp_exp_i16_f16(self, gfx10_src_2:reg_block):
		return self.ic_pb(vop1_base('v_frexp_exp_i16_f16', None, gfx10_src_2))
	def v_frexp_exp_i32_f32(self, gfx10_src_2:reg_block):
		return self.ic_pb(vop1_base('v_frexp_exp_i32_f32', None, gfx10_src_2))
	def v_frexp_exp_i32_f64(self, gfx10_src_3:reg_block):
		return self.ic_pb(vop1_base('v_frexp_exp_i32_f64', None, gfx10_src_3))
	def v_frexp_mant_f16(self, gfx10_src_2:reg_block):
		return self.ic_pb(vop1_base('v_frexp_mant_f16', None, gfx10_src_2))
	def v_frexp_mant_f32(self, gfx10_src_2:reg_block):
		return self.ic_pb(vop1_base('v_frexp_mant_f32', None, gfx10_src_2))
	def v_frexp_mant_f64(self, gfx10_src_3:reg_block):
		return self.ic_pb(vop1_base('v_frexp_mant_f64', None, gfx10_src_3))
	def v_log_f16(self, gfx10_src_2:reg_block):
		return self.ic_pb(vop1_base('v_log_f16', None, gfx10_src_2))
	def v_log_f32(self, gfx10_src_2:reg_block):
		return self.ic_pb(vop1_base('v_log_f32', None, gfx10_src_2))
	def v_mov_b32(self, gfx10_src_2:reg_block):
		return self.ic_pb(vop1_base('v_mov_b32', None, gfx10_src_2))
	def v_movreld_b32(self, gfx10_src_2:reg_block):
		return self.ic_pb(vop1_base('v_movreld_b32', None, gfx10_src_2))
	def v_movrels_b32(self, gfx10_vsrc:reg_block):
		return self.ic_pb(vop1_base('v_movrels_b32', None, gfx10_vsrc))
	def v_movrelsd_2_b32(self, gfx10_vsrc:reg_block):
		return self.ic_pb(vop1_base('v_movrelsd_2_b32', None, gfx10_vsrc))
	def v_movrelsd_b32(self, gfx10_vsrc:reg_block):
		return self.ic_pb(vop1_base('v_movrelsd_b32', None, gfx10_vsrc))
	def v_nop(self):
		return self.ic_pb(vop1_base('v_nop', None, None))
	def v_not_b32(self, gfx10_src_2:reg_block):
		return self.ic_pb(vop1_base('v_not_b32', None, gfx10_src_2))
	def v_pipeflush(self):
		return self.ic_pb(vop1_base('v_pipeflush', None, None))
	def v_rcp_f16(self, gfx10_src_2:reg_block):
		return self.ic_pb(vop1_base('v_rcp_f16', None, gfx10_src_2))
	def v_rcp_f32(self, gfx10_src_2:reg_block):
		return self.ic_pb(vop1_base('v_rcp_f32', None, gfx10_src_2))
	def v_rcp_f64(self, gfx10_src_3:reg_block):
		return self.ic_pb(vop1_base('v_rcp_f64', None, gfx10_src_3))
	def v_rcp_iflag_f32(self, gfx10_src_2:reg_block):
		return self.ic_pb(vop1_base('v_rcp_iflag_f32', None, gfx10_src_2))
	def v_readfirstlane_b32(self, gfx10_src_5:reg_block):
		return self.ic_pb(vop1_base('v_readfirstlane_b32', None, gfx10_src_5))
	def v_rndne_f16(self, gfx10_src_2:reg_block):
		return self.ic_pb(vop1_base('v_rndne_f16', None, gfx10_src_2))
	def v_rndne_f32(self, gfx10_src_2:reg_block):
		return self.ic_pb(vop1_base('v_rndne_f32', None, gfx10_src_2))
	def v_rndne_f64(self, gfx10_src_3:reg_block):
		return self.ic_pb(vop1_base('v_rndne_f64', None, gfx10_src_3))
	def v_rsq_f16(self, gfx10_src_2:reg_block):
		return self.ic_pb(vop1_base('v_rsq_f16', None, gfx10_src_2))
	def v_rsq_f32(self, gfx10_src_2:reg_block):
		return self.ic_pb(vop1_base('v_rsq_f32', None, gfx10_src_2))
	def v_rsq_f64(self, gfx10_src_3:reg_block):
		return self.ic_pb(vop1_base('v_rsq_f64', None, gfx10_src_3))
	def v_sat_pk_u8_i16(self, gfx10_src_2:reg_block):
		return self.ic_pb(vop1_base('v_sat_pk_u8_i16', None, gfx10_src_2))
	def v_sin_f16(self, gfx10_src_2:reg_block):
		return self.ic_pb(vop1_base('v_sin_f16', None, gfx10_src_2))
	def v_sin_f32(self, gfx10_src_2:reg_block):
		return self.ic_pb(vop1_base('v_sin_f32', None, gfx10_src_2))
	def v_sqrt_f16(self, gfx10_src_2:reg_block):
		return self.ic_pb(vop1_base('v_sqrt_f16', None, gfx10_src_2))
	def v_sqrt_f32(self, gfx10_src_2:reg_block):
		return self.ic_pb(vop1_base('v_sqrt_f32', None, gfx10_src_2))
	def v_sqrt_f64(self, gfx10_src_3:reg_block):
		return self.ic_pb(vop1_base('v_sqrt_f64', None, gfx10_src_3))
	def v_swap_b32(self, gfx10_vsrc:reg_block):
		return self.ic_pb(vop1_base('v_swap_b32', None, gfx10_vsrc))
	def v_swaprel_b32(self, gfx10_vsrc:reg_block):
		return self.ic_pb(vop1_base('v_swaprel_b32', None, gfx10_vsrc))
	def v_trunc_f16(self, gfx10_src_2:reg_block):
		return self.ic_pb(vop1_base('v_trunc_f16', None, gfx10_src_2))
	def v_trunc_f32(self, gfx10_src_2:reg_block):
		return self.ic_pb(vop1_base('v_trunc_f32', None, gfx10_src_2))
	def v_trunc_f64(self, gfx10_src_3:reg_block):
		return self.ic_pb(vop1_base('v_trunc_f64', None, gfx10_src_3))
class vop2_base(inst_base): 
	def __init__(self, INSTRUCTION:str, DST0:Union[reg_block,None], DST1:Union[reg_block,None], SRC0:Union[reg_block,None], SRC1:Union[reg_block,None], SRC2:Union[reg_block,None]): 
		super().__init__(instruction_type.SMEM, INSTRUCTION)
		self.DST0 = DST0 
		self.DST1 = DST1 
		self.SRC0 = SRC0 
		self.SRC1 = SRC1 
		self.SRC2 = SRC2 
	def __str__(self): 
		args_l = filter(None.__ne__, [self.DST0,self.DST1,self.SRC0,self.SRC1,self.SRC2]) 
		return f"{self.label} {','.join(map(str, args_l))} " 
class vop2_instr_caller(inst_caller_base): 
	def __init__(self, insturction_container) -> None:
     		super().__init__(insturction_container)
	def v_add_co_ci_u32(self, gfx10_vcc:reg_block, gfx10_src_2:reg_block, gfx10_vsrc:reg_block, gfx10_vcc:reg_block):
		return self.ic_pb(vop2_base('v_add_co_ci_u32', None, gfx10_vcc, gfx10_src_2, gfx10_vsrc, gfx10_vcc))
	def v_add_f16(self, gfx10_src_2:reg_block, gfx10_vsrc:reg_block):
		return self.ic_pb(vop2_base('v_add_f16', None, None, gfx10_src_2, gfx10_vsrc, None))
	def v_add_f32(self, gfx10_src_2:reg_block, gfx10_vsrc:reg_block):
		return self.ic_pb(vop2_base('v_add_f32', None, None, gfx10_src_2, gfx10_vsrc, None))
	def v_add_nc_u32(self, gfx10_src_2:reg_block, gfx10_vsrc:reg_block):
		return self.ic_pb(vop2_base('v_add_nc_u32', None, None, gfx10_src_2, gfx10_vsrc, None))
	def v_and_b32(self, gfx10_src_2:reg_block, gfx10_vsrc:reg_block):
		return self.ic_pb(vop2_base('v_and_b32', None, None, gfx10_src_2, gfx10_vsrc, None))
	def v_ashrrev_i32(self, gfx10_src_6:reg_block, gfx10_vsrc:reg_block):
		return self.ic_pb(vop2_base('v_ashrrev_i32', None, None, gfx10_src_6, gfx10_vsrc, None))
	def v_cndmask_b32(self, gfx10_src_2:reg_block, gfx10_vsrc:reg_block, gfx10_vcc:reg_block):
		return self.ic_pb(vop2_base('v_cndmask_b32', None, None, gfx10_src_2, gfx10_vsrc, gfx10_vcc))
	def v_cvt_pkrtz_f16_f32(self, gfx10_src_2:reg_block, gfx10_vsrc:reg_block):
		return self.ic_pb(vop2_base('v_cvt_pkrtz_f16_f32', None, None, gfx10_src_2, gfx10_vsrc, None))
	def v_fmaak_f16(self, gfx10_src_7:reg_block, gfx10_vsrc:reg_block, gfx10_simm32_1:reg_block):
		return self.ic_pb(vop2_base('v_fmaak_f16', None, None, gfx10_src_7, gfx10_vsrc, gfx10_simm32_1))
	def v_fmaak_f32(self, gfx10_src_7:reg_block, gfx10_vsrc:reg_block, gfx10_simm32_2:reg_block):
		return self.ic_pb(vop2_base('v_fmaak_f32', None, None, gfx10_src_7, gfx10_vsrc, gfx10_simm32_2))
	def v_fmac_f16(self, gfx10_src_2:reg_block, gfx10_vsrc:reg_block):
		return self.ic_pb(vop2_base('v_fmac_f16', None, None, gfx10_src_2, gfx10_vsrc, None))
	def v_fmac_f32(self, gfx10_src_2:reg_block, gfx10_vsrc:reg_block):
		return self.ic_pb(vop2_base('v_fmac_f32', None, None, gfx10_src_2, gfx10_vsrc, None))
	def v_fmamk_f16(self, gfx10_src_7:reg_block, gfx10_simm32_1:reg_block, gfx10_vsrc:reg_block):
		return self.ic_pb(vop2_base('v_fmamk_f16', None, None, gfx10_src_7, gfx10_simm32_1, gfx10_vsrc))
	def v_fmamk_f32(self, gfx10_src_7:reg_block, gfx10_simm32_2:reg_block, gfx10_vsrc:reg_block):
		return self.ic_pb(vop2_base('v_fmamk_f32', None, None, gfx10_src_7, gfx10_simm32_2, gfx10_vsrc))
	def v_ldexp_f16(self, gfx10_src_2:reg_block, gfx10_vsrc:reg_block):
		return self.ic_pb(vop2_base('v_ldexp_f16', None, None, gfx10_src_2, gfx10_vsrc, None))
	def v_lshlrev_b32(self, gfx10_src_6:reg_block, gfx10_vsrc:reg_block):
		return self.ic_pb(vop2_base('v_lshlrev_b32', None, None, gfx10_src_6, gfx10_vsrc, None))
	def v_lshrrev_b32(self, gfx10_src_6:reg_block, gfx10_vsrc:reg_block):
		return self.ic_pb(vop2_base('v_lshrrev_b32', None, None, gfx10_src_6, gfx10_vsrc, None))
	def v_mac_f32(self, gfx10_src_2:reg_block, gfx10_vsrc:reg_block):
		return self.ic_pb(vop2_base('v_mac_f32', None, None, gfx10_src_2, gfx10_vsrc, None))
	def v_mac_legacy_f32(self, gfx10_src_2:reg_block, gfx10_vsrc:reg_block):
		return self.ic_pb(vop2_base('v_mac_legacy_f32', None, None, gfx10_src_2, gfx10_vsrc, None))
	def v_madak_f32(self, gfx10_src_7:reg_block, gfx10_vsrc:reg_block, gfx10_simm32_2:reg_block):
		return self.ic_pb(vop2_base('v_madak_f32', None, None, gfx10_src_7, gfx10_vsrc, gfx10_simm32_2))
	def v_madmk_f32(self, gfx10_src_7:reg_block, gfx10_simm32_2:reg_block, gfx10_vsrc:reg_block):
		return self.ic_pb(vop2_base('v_madmk_f32', None, None, gfx10_src_7, gfx10_simm32_2, gfx10_vsrc))
	def v_max_f16(self, gfx10_src_2:reg_block, gfx10_vsrc:reg_block):
		return self.ic_pb(vop2_base('v_max_f16', None, None, gfx10_src_2, gfx10_vsrc, None))
	def v_max_f32(self, gfx10_src_2:reg_block, gfx10_vsrc:reg_block):
		return self.ic_pb(vop2_base('v_max_f32', None, None, gfx10_src_2, gfx10_vsrc, None))
	def v_max_i32(self, gfx10_src_2:reg_block, gfx10_vsrc:reg_block):
		return self.ic_pb(vop2_base('v_max_i32', None, None, gfx10_src_2, gfx10_vsrc, None))
	def v_max_u32(self, gfx10_src_2:reg_block, gfx10_vsrc:reg_block):
		return self.ic_pb(vop2_base('v_max_u32', None, None, gfx10_src_2, gfx10_vsrc, None))
	def v_min_f16(self, gfx10_src_2:reg_block, gfx10_vsrc:reg_block):
		return self.ic_pb(vop2_base('v_min_f16', None, None, gfx10_src_2, gfx10_vsrc, None))
	def v_min_f32(self, gfx10_src_2:reg_block, gfx10_vsrc:reg_block):
		return self.ic_pb(vop2_base('v_min_f32', None, None, gfx10_src_2, gfx10_vsrc, None))
	def v_min_i32(self, gfx10_src_2:reg_block, gfx10_vsrc:reg_block):
		return self.ic_pb(vop2_base('v_min_i32', None, None, gfx10_src_2, gfx10_vsrc, None))
	def v_min_u32(self, gfx10_src_2:reg_block, gfx10_vsrc:reg_block):
		return self.ic_pb(vop2_base('v_min_u32', None, None, gfx10_src_2, gfx10_vsrc, None))
	def v_mul_f16(self, gfx10_src_2:reg_block, gfx10_vsrc:reg_block):
		return self.ic_pb(vop2_base('v_mul_f16', None, None, gfx10_src_2, gfx10_vsrc, None))
	def v_mul_f32(self, gfx10_src_2:reg_block, gfx10_vsrc:reg_block):
		return self.ic_pb(vop2_base('v_mul_f32', None, None, gfx10_src_2, gfx10_vsrc, None))
	def v_mul_hi_i32_i24(self, gfx10_src_2:reg_block, gfx10_vsrc:reg_block):
		return self.ic_pb(vop2_base('v_mul_hi_i32_i24', None, None, gfx10_src_2, gfx10_vsrc, None))
	def v_mul_hi_u32_u24(self, gfx10_src_2:reg_block, gfx10_vsrc:reg_block):
		return self.ic_pb(vop2_base('v_mul_hi_u32_u24', None, None, gfx10_src_2, gfx10_vsrc, None))
	def v_mul_i32_i24(self, gfx10_src_2:reg_block, gfx10_vsrc:reg_block):
		return self.ic_pb(vop2_base('v_mul_i32_i24', None, None, gfx10_src_2, gfx10_vsrc, None))
	def v_mul_legacy_f32(self, gfx10_src_2:reg_block, gfx10_vsrc:reg_block):
		return self.ic_pb(vop2_base('v_mul_legacy_f32', None, None, gfx10_src_2, gfx10_vsrc, None))
	def v_mul_u32_u24(self, gfx10_src_2:reg_block, gfx10_vsrc:reg_block):
		return self.ic_pb(vop2_base('v_mul_u32_u24', None, None, gfx10_src_2, gfx10_vsrc, None))
	def v_or_b32(self, gfx10_src_2:reg_block, gfx10_vsrc:reg_block):
		return self.ic_pb(vop2_base('v_or_b32', None, None, gfx10_src_2, gfx10_vsrc, None))
	def v_pk_fmac_f16(self, gfx10_src_2:reg_block, gfx10_vsrc:reg_block):
		return self.ic_pb(vop2_base('v_pk_fmac_f16', None, None, gfx10_src_2, gfx10_vsrc, None))
	def v_sub_co_ci_u32(self, gfx10_vcc:reg_block, gfx10_src_2:reg_block, gfx10_vsrc:reg_block, gfx10_vcc:reg_block):
		return self.ic_pb(vop2_base('v_sub_co_ci_u32', None, gfx10_vcc, gfx10_src_2, gfx10_vsrc, gfx10_vcc))
	def v_sub_f16(self, gfx10_src_2:reg_block, gfx10_vsrc:reg_block):
		return self.ic_pb(vop2_base('v_sub_f16', None, None, gfx10_src_2, gfx10_vsrc, None))
	def v_sub_f32(self, gfx10_src_2:reg_block, gfx10_vsrc:reg_block):
		return self.ic_pb(vop2_base('v_sub_f32', None, None, gfx10_src_2, gfx10_vsrc, None))
	def v_sub_nc_u32(self, gfx10_src_2:reg_block, gfx10_vsrc:reg_block):
		return self.ic_pb(vop2_base('v_sub_nc_u32', None, None, gfx10_src_2, gfx10_vsrc, None))
	def v_subrev_co_ci_u32(self, gfx10_vcc:reg_block, gfx10_src_6:reg_block, gfx10_vsrc:reg_block, gfx10_vcc:reg_block):
		return self.ic_pb(vop2_base('v_subrev_co_ci_u32', None, gfx10_vcc, gfx10_src_6, gfx10_vsrc, gfx10_vcc))
	def v_subrev_f16(self, gfx10_src_6:reg_block, gfx10_vsrc:reg_block):
		return self.ic_pb(vop2_base('v_subrev_f16', None, None, gfx10_src_6, gfx10_vsrc, None))
	def v_subrev_f32(self, gfx10_src_6:reg_block, gfx10_vsrc:reg_block):
		return self.ic_pb(vop2_base('v_subrev_f32', None, None, gfx10_src_6, gfx10_vsrc, None))
	def v_subrev_nc_u32(self, gfx10_src_6:reg_block, gfx10_vsrc:reg_block):
		return self.ic_pb(vop2_base('v_subrev_nc_u32', None, None, gfx10_src_6, gfx10_vsrc, None))
	def v_xnor_b32(self, gfx10_src_2:reg_block, gfx10_vsrc:reg_block):
		return self.ic_pb(vop2_base('v_xnor_b32', None, None, gfx10_src_2, gfx10_vsrc, None))
	def v_xor_b32(self, gfx10_src_2:reg_block, gfx10_vsrc:reg_block):
		return self.ic_pb(vop2_base('v_xor_b32', None, None, gfx10_src_2, gfx10_vsrc, None))
class vop3_base(inst_base): 
	def __init__(self, INSTRUCTION:str, DST0:Union[reg_block,None], DST1:Union[reg_block,None], SRC0:Union[reg_block,None], SRC1:Union[reg_block,None], SRC2:Union[reg_block,None], MODIFIERS:str): 
		super().__init__(instruction_type.SMEM, INSTRUCTION)
		self.DST0 = DST0 
		self.DST1 = DST1 
		self.SRC0 = SRC0 
		self.SRC1 = SRC1 
		self.SRC2 = SRC2 
		self.MODIFIERS = MODIFIERS 
	def __str__(self): 
		args_l = filter(None.__ne__, [self.DST0,self.DST1,self.SRC0,self.SRC1,self.SRC2]) 
		return f"{self.label} {','.join(map(str, args_l))} {self.MODIFIERS}" 
class vop3_instr_caller(inst_caller_base): 
	def __init__(self, insturction_container) -> None:
     		super().__init__(insturction_container)
	def v_add3_u32(self, gfx10_src_2:reg_block, gfx10_src_6:reg_block, gfx10_src_6:reg_block):
		return self.ic_pb(vop3_base('v_add3_u32', None, None, gfx10_src_2, gfx10_src_6, gfx10_src_6))
	def v_add_co_ci_u32_e64(self, gfx10_sdst:reg_block, gfx10_src_2:reg_block, gfx10_src_6:reg_block, gfx10_ssrc_5:reg_block, MODIFIERS:str=''):
		return self.ic_pb(vop3_base('v_add_co_ci_u32_e64', None, gfx10_sdst, gfx10_src_2, gfx10_src_6, gfx10_ssrc_5, MODIFIERS))
	def v_add_co_u32(self, gfx10_sdst:reg_block, gfx10_src_2:reg_block, gfx10_src_6:reg_block, MODIFIERS:str=''):
		return self.ic_pb(vop3_base('v_add_co_u32', None, gfx10_sdst, gfx10_src_2, gfx10_src_6, None, MODIFIERS))
	def v_add_f16_e64(self, gfx10_src_2:reg_block, gfx10_src_6:reg_block, MODIFIERS:str=''):
		return self.ic_pb(vop3_base('v_add_f16_e64', None, None, gfx10_src_2, gfx10_src_6, None, MODIFIERS))
	def v_add_f32_e64(self, gfx10_src_2:reg_block, gfx10_src_6:reg_block, MODIFIERS:str=''):
		return self.ic_pb(vop3_base('v_add_f32_e64', None, None, gfx10_src_2, gfx10_src_6, None, MODIFIERS))
	def v_add_f64(self, gfx10_src_3:reg_block, gfx10_src_3:reg_block, MODIFIERS:str=''):
		return self.ic_pb(vop3_base('v_add_f64', None, None, gfx10_src_3, gfx10_src_3, None, MODIFIERS))
	def v_add_lshl_u32(self, gfx10_src_2:reg_block, gfx10_src_6:reg_block, gfx10_src_6:reg_block):
		return self.ic_pb(vop3_base('v_add_lshl_u32', None, None, gfx10_src_2, gfx10_src_6, gfx10_src_6))
	def v_add_nc_i16(self, gfx10_src_4:reg_block, gfx10_src_8:reg_block, MODIFIERS:str=''):
		return self.ic_pb(vop3_base('v_add_nc_i16', None, None, gfx10_src_4, gfx10_src_8, None, MODIFIERS))
	def v_add_nc_i32(self, gfx10_src_2:reg_block, gfx10_src_6:reg_block, MODIFIERS:str=''):
		return self.ic_pb(vop3_base('v_add_nc_i32', None, None, gfx10_src_2, gfx10_src_6, None, MODIFIERS))
	def v_add_nc_u16(self, gfx10_src_4:reg_block, gfx10_src_8:reg_block, MODIFIERS:str=''):
		return self.ic_pb(vop3_base('v_add_nc_u16', None, None, gfx10_src_4, gfx10_src_8, None, MODIFIERS))
	def v_add_nc_u32_e64(self, gfx10_src_2:reg_block, gfx10_src_6:reg_block, MODIFIERS:str=''):
		return self.ic_pb(vop3_base('v_add_nc_u32_e64', None, None, gfx10_src_2, gfx10_src_6, None, MODIFIERS))
	def v_alignbit_b32(self, gfx10_src_2:reg_block, gfx10_src_6:reg_block, gfx10_src_8:reg_block):
		return self.ic_pb(vop3_base('v_alignbit_b32', None, None, gfx10_src_2, gfx10_src_6, gfx10_src_8))
	def v_alignbyte_b32(self, gfx10_src_2:reg_block, gfx10_src_6:reg_block, gfx10_src_8:reg_block):
		return self.ic_pb(vop3_base('v_alignbyte_b32', None, None, gfx10_src_2, gfx10_src_6, gfx10_src_8))
	def v_and_b32_e64(self, gfx10_src_2:reg_block, gfx10_src_6:reg_block):
		return self.ic_pb(vop3_base('v_and_b32_e64', None, None, gfx10_src_2, gfx10_src_6, None))
	def v_and_or_b32(self, gfx10_src_2:reg_block, gfx10_src_6:reg_block, gfx10_src_6:reg_block):
		return self.ic_pb(vop3_base('v_and_or_b32', None, None, gfx10_src_2, gfx10_src_6, gfx10_src_6))
	def v_ashrrev_i16(self, gfx10_src_8:reg_block, gfx10_src_8:reg_block):
		return self.ic_pb(vop3_base('v_ashrrev_i16', None, None, gfx10_src_8, gfx10_src_8, None))
	def v_ashrrev_i32_e64(self, gfx10_src_6:reg_block, gfx10_src_6:reg_block):
		return self.ic_pb(vop3_base('v_ashrrev_i32_e64', None, None, gfx10_src_6, gfx10_src_6, None))
	def v_ashrrev_i64(self, gfx10_src_6:reg_block, gfx10_src_3:reg_block):
		return self.ic_pb(vop3_base('v_ashrrev_i64', None, None, gfx10_src_6, gfx10_src_3, None))
	def v_bcnt_u32_b32(self, gfx10_src_2:reg_block, gfx10_src_6:reg_block):
		return self.ic_pb(vop3_base('v_bcnt_u32_b32', None, None, gfx10_src_2, gfx10_src_6, None))
	def v_bfe_i32(self, gfx10_src_2:reg_block, gfx10_src_6:reg_block, gfx10_src_6:reg_block):
		return self.ic_pb(vop3_base('v_bfe_i32', None, None, gfx10_src_2, gfx10_src_6, gfx10_src_6))
	def v_bfe_u32(self, gfx10_src_2:reg_block, gfx10_src_6:reg_block, gfx10_src_6:reg_block):
		return self.ic_pb(vop3_base('v_bfe_u32', None, None, gfx10_src_2, gfx10_src_6, gfx10_src_6))
	def v_bfi_b32(self, gfx10_src_2:reg_block, gfx10_src_6:reg_block, gfx10_src_6:reg_block):
		return self.ic_pb(vop3_base('v_bfi_b32', None, None, gfx10_src_2, gfx10_src_6, gfx10_src_6))
	def v_bfm_b32(self, gfx10_src_2:reg_block, gfx10_src_6:reg_block):
		return self.ic_pb(vop3_base('v_bfm_b32', None, None, gfx10_src_2, gfx10_src_6, None))
	def v_bfrev_b32_e64(self, gfx10_src_2:reg_block):
		return self.ic_pb(vop3_base('v_bfrev_b32_e64', None, None, gfx10_src_2, None, None))
	def v_ceil_f16_e64(self, gfx10_src_2:reg_block, MODIFIERS:str=''):
		return self.ic_pb(vop3_base('v_ceil_f16_e64', None, None, gfx10_src_2, None, None, MODIFIERS))
	def v_ceil_f32_e64(self, gfx10_src_2:reg_block, MODIFIERS:str=''):
		return self.ic_pb(vop3_base('v_ceil_f32_e64', None, None, gfx10_src_2, None, None, MODIFIERS))
	def v_ceil_f64_e64(self, gfx10_src_3:reg_block, MODIFIERS:str=''):
		return self.ic_pb(vop3_base('v_ceil_f64_e64', None, None, gfx10_src_3, None, None, MODIFIERS))
	def v_clrexcp_e64(self):
		return self.ic_pb(vop3_base('v_clrexcp_e64', None, None, None, None, None))
	def v_cmp_class_f16_e64(self, gfx10_src_2:reg_block, gfx10_src_6:reg_block):
		return self.ic_pb(vop3_base('v_cmp_class_f16_e64', None, None, gfx10_src_2, gfx10_src_6, None))
	def v_cmp_class_f32_e64(self, gfx10_src_2:reg_block, gfx10_src_6:reg_block):
		return self.ic_pb(vop3_base('v_cmp_class_f32_e64', None, None, gfx10_src_2, gfx10_src_6, None))
	def v_cmp_class_f64_e64(self, gfx10_src_3:reg_block, gfx10_src_6:reg_block):
		return self.ic_pb(vop3_base('v_cmp_class_f64_e64', None, None, gfx10_src_3, gfx10_src_6, None))
	def v_cmp_eq_f16_e64(self, gfx10_src_2:reg_block, gfx10_src_6:reg_block, MODIFIERS:str=''):
		return self.ic_pb(vop3_base('v_cmp_eq_f16_e64', None, None, gfx10_src_2, gfx10_src_6, None, MODIFIERS))
	def v_cmp_eq_f32_e64(self, gfx10_src_2:reg_block, gfx10_src_6:reg_block, MODIFIERS:str=''):
		return self.ic_pb(vop3_base('v_cmp_eq_f32_e64', None, None, gfx10_src_2, gfx10_src_6, None, MODIFIERS))
	def v_cmp_eq_f64_e64(self, gfx10_src_3:reg_block, gfx10_src_3:reg_block, MODIFIERS:str=''):
		return self.ic_pb(vop3_base('v_cmp_eq_f64_e64', None, None, gfx10_src_3, gfx10_src_3, None, MODIFIERS))
	def v_cmp_eq_i16_e64(self, gfx10_src_4:reg_block, gfx10_src_8:reg_block):
		return self.ic_pb(vop3_base('v_cmp_eq_i16_e64', None, None, gfx10_src_4, gfx10_src_8, None))
	def v_cmp_eq_i32_e64(self, gfx10_src_2:reg_block, gfx10_src_6:reg_block):
		return self.ic_pb(vop3_base('v_cmp_eq_i32_e64', None, None, gfx10_src_2, gfx10_src_6, None))
	def v_cmp_eq_i64_e64(self, gfx10_src_3:reg_block, gfx10_src_3:reg_block):
		return self.ic_pb(vop3_base('v_cmp_eq_i64_e64', None, None, gfx10_src_3, gfx10_src_3, None))
	def v_cmp_eq_u16_e64(self, gfx10_src_4:reg_block, gfx10_src_8:reg_block):
		return self.ic_pb(vop3_base('v_cmp_eq_u16_e64', None, None, gfx10_src_4, gfx10_src_8, None))
	def v_cmp_eq_u32_e64(self, gfx10_src_2:reg_block, gfx10_src_6:reg_block):
		return self.ic_pb(vop3_base('v_cmp_eq_u32_e64', None, None, gfx10_src_2, gfx10_src_6, None))
	def v_cmp_eq_u64_e64(self, gfx10_src_3:reg_block, gfx10_src_3:reg_block):
		return self.ic_pb(vop3_base('v_cmp_eq_u64_e64', None, None, gfx10_src_3, gfx10_src_3, None))
	def v_cmp_f_f16_e64(self, gfx10_src_2:reg_block, gfx10_src_6:reg_block, MODIFIERS:str=''):
		return self.ic_pb(vop3_base('v_cmp_f_f16_e64', None, None, gfx10_src_2, gfx10_src_6, None, MODIFIERS))
	def v_cmp_f_f32_e64(self, gfx10_src_2:reg_block, gfx10_src_6:reg_block, MODIFIERS:str=''):
		return self.ic_pb(vop3_base('v_cmp_f_f32_e64', None, None, gfx10_src_2, gfx10_src_6, None, MODIFIERS))
	def v_cmp_f_f64_e64(self, gfx10_src_3:reg_block, gfx10_src_3:reg_block, MODIFIERS:str=''):
		return self.ic_pb(vop3_base('v_cmp_f_f64_e64', None, None, gfx10_src_3, gfx10_src_3, None, MODIFIERS))
	def v_cmp_f_i32_e64(self, gfx10_src_2:reg_block, gfx10_src_6:reg_block):
		return self.ic_pb(vop3_base('v_cmp_f_i32_e64', None, None, gfx10_src_2, gfx10_src_6, None))
	def v_cmp_f_i64_e64(self, gfx10_src_3:reg_block, gfx10_src_3:reg_block):
		return self.ic_pb(vop3_base('v_cmp_f_i64_e64', None, None, gfx10_src_3, gfx10_src_3, None))
	def v_cmp_f_u32_e64(self, gfx10_src_2:reg_block, gfx10_src_6:reg_block):
		return self.ic_pb(vop3_base('v_cmp_f_u32_e64', None, None, gfx10_src_2, gfx10_src_6, None))
	def v_cmp_f_u64_e64(self, gfx10_src_3:reg_block, gfx10_src_3:reg_block):
		return self.ic_pb(vop3_base('v_cmp_f_u64_e64', None, None, gfx10_src_3, gfx10_src_3, None))
	def v_cmp_ge_f16_e64(self, gfx10_src_2:reg_block, gfx10_src_6:reg_block, MODIFIERS:str=''):
		return self.ic_pb(vop3_base('v_cmp_ge_f16_e64', None, None, gfx10_src_2, gfx10_src_6, None, MODIFIERS))
	def v_cmp_ge_f32_e64(self, gfx10_src_2:reg_block, gfx10_src_6:reg_block, MODIFIERS:str=''):
		return self.ic_pb(vop3_base('v_cmp_ge_f32_e64', None, None, gfx10_src_2, gfx10_src_6, None, MODIFIERS))
	def v_cmp_ge_f64_e64(self, gfx10_src_3:reg_block, gfx10_src_3:reg_block, MODIFIERS:str=''):
		return self.ic_pb(vop3_base('v_cmp_ge_f64_e64', None, None, gfx10_src_3, gfx10_src_3, None, MODIFIERS))
	def v_cmp_ge_i16_e64(self, gfx10_src_4:reg_block, gfx10_src_8:reg_block):
		return self.ic_pb(vop3_base('v_cmp_ge_i16_e64', None, None, gfx10_src_4, gfx10_src_8, None))
	def v_cmp_ge_i32_e64(self, gfx10_src_2:reg_block, gfx10_src_6:reg_block):
		return self.ic_pb(vop3_base('v_cmp_ge_i32_e64', None, None, gfx10_src_2, gfx10_src_6, None))
	def v_cmp_ge_i64_e64(self, gfx10_src_3:reg_block, gfx10_src_3:reg_block):
		return self.ic_pb(vop3_base('v_cmp_ge_i64_e64', None, None, gfx10_src_3, gfx10_src_3, None))
	def v_cmp_ge_u16_e64(self, gfx10_src_4:reg_block, gfx10_src_8:reg_block):
		return self.ic_pb(vop3_base('v_cmp_ge_u16_e64', None, None, gfx10_src_4, gfx10_src_8, None))
	def v_cmp_ge_u32_e64(self, gfx10_src_2:reg_block, gfx10_src_6:reg_block):
		return self.ic_pb(vop3_base('v_cmp_ge_u32_e64', None, None, gfx10_src_2, gfx10_src_6, None))
	def v_cmp_ge_u64_e64(self, gfx10_src_3:reg_block, gfx10_src_3:reg_block):
		return self.ic_pb(vop3_base('v_cmp_ge_u64_e64', None, None, gfx10_src_3, gfx10_src_3, None))
	def v_cmp_gt_f16_e64(self, gfx10_src_2:reg_block, gfx10_src_6:reg_block, MODIFIERS:str=''):
		return self.ic_pb(vop3_base('v_cmp_gt_f16_e64', None, None, gfx10_src_2, gfx10_src_6, None, MODIFIERS))
	def v_cmp_gt_f32_e64(self, gfx10_src_2:reg_block, gfx10_src_6:reg_block, MODIFIERS:str=''):
		return self.ic_pb(vop3_base('v_cmp_gt_f32_e64', None, None, gfx10_src_2, gfx10_src_6, None, MODIFIERS))
	def v_cmp_gt_f64_e64(self, gfx10_src_3:reg_block, gfx10_src_3:reg_block, MODIFIERS:str=''):
		return self.ic_pb(vop3_base('v_cmp_gt_f64_e64', None, None, gfx10_src_3, gfx10_src_3, None, MODIFIERS))
	def v_cmp_gt_i16_e64(self, gfx10_src_4:reg_block, gfx10_src_8:reg_block):
		return self.ic_pb(vop3_base('v_cmp_gt_i16_e64', None, None, gfx10_src_4, gfx10_src_8, None))
	def v_cmp_gt_i32_e64(self, gfx10_src_2:reg_block, gfx10_src_6:reg_block):
		return self.ic_pb(vop3_base('v_cmp_gt_i32_e64', None, None, gfx10_src_2, gfx10_src_6, None))
	def v_cmp_gt_i64_e64(self, gfx10_src_3:reg_block, gfx10_src_3:reg_block):
		return self.ic_pb(vop3_base('v_cmp_gt_i64_e64', None, None, gfx10_src_3, gfx10_src_3, None))
	def v_cmp_gt_u16_e64(self, gfx10_src_4:reg_block, gfx10_src_8:reg_block):
		return self.ic_pb(vop3_base('v_cmp_gt_u16_e64', None, None, gfx10_src_4, gfx10_src_8, None))
	def v_cmp_gt_u32_e64(self, gfx10_src_2:reg_block, gfx10_src_6:reg_block):
		return self.ic_pb(vop3_base('v_cmp_gt_u32_e64', None, None, gfx10_src_2, gfx10_src_6, None))
	def v_cmp_gt_u64_e64(self, gfx10_src_3:reg_block, gfx10_src_3:reg_block):
		return self.ic_pb(vop3_base('v_cmp_gt_u64_e64', None, None, gfx10_src_3, gfx10_src_3, None))
	def v_cmp_le_f16_e64(self, gfx10_src_2:reg_block, gfx10_src_6:reg_block, MODIFIERS:str=''):
		return self.ic_pb(vop3_base('v_cmp_le_f16_e64', None, None, gfx10_src_2, gfx10_src_6, None, MODIFIERS))
	def v_cmp_le_f32_e64(self, gfx10_src_2:reg_block, gfx10_src_6:reg_block, MODIFIERS:str=''):
		return self.ic_pb(vop3_base('v_cmp_le_f32_e64', None, None, gfx10_src_2, gfx10_src_6, None, MODIFIERS))
	def v_cmp_le_f64_e64(self, gfx10_src_3:reg_block, gfx10_src_3:reg_block, MODIFIERS:str=''):
		return self.ic_pb(vop3_base('v_cmp_le_f64_e64', None, None, gfx10_src_3, gfx10_src_3, None, MODIFIERS))
	def v_cmp_le_i16_e64(self, gfx10_src_4:reg_block, gfx10_src_8:reg_block):
		return self.ic_pb(vop3_base('v_cmp_le_i16_e64', None, None, gfx10_src_4, gfx10_src_8, None))
	def v_cmp_le_i32_e64(self, gfx10_src_2:reg_block, gfx10_src_6:reg_block):
		return self.ic_pb(vop3_base('v_cmp_le_i32_e64', None, None, gfx10_src_2, gfx10_src_6, None))
	def v_cmp_le_i64_e64(self, gfx10_src_3:reg_block, gfx10_src_3:reg_block):
		return self.ic_pb(vop3_base('v_cmp_le_i64_e64', None, None, gfx10_src_3, gfx10_src_3, None))
	def v_cmp_le_u16_e64(self, gfx10_src_4:reg_block, gfx10_src_8:reg_block):
		return self.ic_pb(vop3_base('v_cmp_le_u16_e64', None, None, gfx10_src_4, gfx10_src_8, None))
	def v_cmp_le_u32_e64(self, gfx10_src_2:reg_block, gfx10_src_6:reg_block):
		return self.ic_pb(vop3_base('v_cmp_le_u32_e64', None, None, gfx10_src_2, gfx10_src_6, None))
	def v_cmp_le_u64_e64(self, gfx10_src_3:reg_block, gfx10_src_3:reg_block):
		return self.ic_pb(vop3_base('v_cmp_le_u64_e64', None, None, gfx10_src_3, gfx10_src_3, None))
	def v_cmp_lg_f16_e64(self, gfx10_src_2:reg_block, gfx10_src_6:reg_block, MODIFIERS:str=''):
		return self.ic_pb(vop3_base('v_cmp_lg_f16_e64', None, None, gfx10_src_2, gfx10_src_6, None, MODIFIERS))
	def v_cmp_lg_f32_e64(self, gfx10_src_2:reg_block, gfx10_src_6:reg_block, MODIFIERS:str=''):
		return self.ic_pb(vop3_base('v_cmp_lg_f32_e64', None, None, gfx10_src_2, gfx10_src_6, None, MODIFIERS))
	def v_cmp_lg_f64_e64(self, gfx10_src_3:reg_block, gfx10_src_3:reg_block, MODIFIERS:str=''):
		return self.ic_pb(vop3_base('v_cmp_lg_f64_e64', None, None, gfx10_src_3, gfx10_src_3, None, MODIFIERS))
	def v_cmp_lt_f16_e64(self, gfx10_src_2:reg_block, gfx10_src_6:reg_block, MODIFIERS:str=''):
		return self.ic_pb(vop3_base('v_cmp_lt_f16_e64', None, None, gfx10_src_2, gfx10_src_6, None, MODIFIERS))
	def v_cmp_lt_f32_e64(self, gfx10_src_2:reg_block, gfx10_src_6:reg_block, MODIFIERS:str=''):
		return self.ic_pb(vop3_base('v_cmp_lt_f32_e64', None, None, gfx10_src_2, gfx10_src_6, None, MODIFIERS))
	def v_cmp_lt_f64_e64(self, gfx10_src_3:reg_block, gfx10_src_3:reg_block, MODIFIERS:str=''):
		return self.ic_pb(vop3_base('v_cmp_lt_f64_e64', None, None, gfx10_src_3, gfx10_src_3, None, MODIFIERS))
	def v_cmp_lt_i16_e64(self, gfx10_src_4:reg_block, gfx10_src_8:reg_block):
		return self.ic_pb(vop3_base('v_cmp_lt_i16_e64', None, None, gfx10_src_4, gfx10_src_8, None))
	def v_cmp_lt_i32_e64(self, gfx10_src_2:reg_block, gfx10_src_6:reg_block):
		return self.ic_pb(vop3_base('v_cmp_lt_i32_e64', None, None, gfx10_src_2, gfx10_src_6, None))
	def v_cmp_lt_i64_e64(self, gfx10_src_3:reg_block, gfx10_src_3:reg_block):
		return self.ic_pb(vop3_base('v_cmp_lt_i64_e64', None, None, gfx10_src_3, gfx10_src_3, None))
	def v_cmp_lt_u16_e64(self, gfx10_src_4:reg_block, gfx10_src_8:reg_block):
		return self.ic_pb(vop3_base('v_cmp_lt_u16_e64', None, None, gfx10_src_4, gfx10_src_8, None))
	def v_cmp_lt_u32_e64(self, gfx10_src_2:reg_block, gfx10_src_6:reg_block):
		return self.ic_pb(vop3_base('v_cmp_lt_u32_e64', None, None, gfx10_src_2, gfx10_src_6, None))
	def v_cmp_lt_u64_e64(self, gfx10_src_3:reg_block, gfx10_src_3:reg_block):
		return self.ic_pb(vop3_base('v_cmp_lt_u64_e64', None, None, gfx10_src_3, gfx10_src_3, None))
	def v_cmp_ne_i16_e64(self, gfx10_src_4:reg_block, gfx10_src_8:reg_block):
		return self.ic_pb(vop3_base('v_cmp_ne_i16_e64', None, None, gfx10_src_4, gfx10_src_8, None))
	def v_cmp_ne_i32_e64(self, gfx10_src_2:reg_block, gfx10_src_6:reg_block):
		return self.ic_pb(vop3_base('v_cmp_ne_i32_e64', None, None, gfx10_src_2, gfx10_src_6, None))
	def v_cmp_ne_i64_e64(self, gfx10_src_3:reg_block, gfx10_src_3:reg_block):
		return self.ic_pb(vop3_base('v_cmp_ne_i64_e64', None, None, gfx10_src_3, gfx10_src_3, None))
	def v_cmp_ne_u16_e64(self, gfx10_src_4:reg_block, gfx10_src_8:reg_block):
		return self.ic_pb(vop3_base('v_cmp_ne_u16_e64', None, None, gfx10_src_4, gfx10_src_8, None))
	def v_cmp_ne_u32_e64(self, gfx10_src_2:reg_block, gfx10_src_6:reg_block):
		return self.ic_pb(vop3_base('v_cmp_ne_u32_e64', None, None, gfx10_src_2, gfx10_src_6, None))
	def v_cmp_ne_u64_e64(self, gfx10_src_3:reg_block, gfx10_src_3:reg_block):
		return self.ic_pb(vop3_base('v_cmp_ne_u64_e64', None, None, gfx10_src_3, gfx10_src_3, None))
	def v_cmp_neq_f16_e64(self, gfx10_src_2:reg_block, gfx10_src_6:reg_block, MODIFIERS:str=''):
		return self.ic_pb(vop3_base('v_cmp_neq_f16_e64', None, None, gfx10_src_2, gfx10_src_6, None, MODIFIERS))
	def v_cmp_neq_f32_e64(self, gfx10_src_2:reg_block, gfx10_src_6:reg_block, MODIFIERS:str=''):
		return self.ic_pb(vop3_base('v_cmp_neq_f32_e64', None, None, gfx10_src_2, gfx10_src_6, None, MODIFIERS))
	def v_cmp_neq_f64_e64(self, gfx10_src_3:reg_block, gfx10_src_3:reg_block, MODIFIERS:str=''):
		return self.ic_pb(vop3_base('v_cmp_neq_f64_e64', None, None, gfx10_src_3, gfx10_src_3, None, MODIFIERS))
	def v_cmp_nge_f16_e64(self, gfx10_src_2:reg_block, gfx10_src_6:reg_block, MODIFIERS:str=''):
		return self.ic_pb(vop3_base('v_cmp_nge_f16_e64', None, None, gfx10_src_2, gfx10_src_6, None, MODIFIERS))
	def v_cmp_nge_f32_e64(self, gfx10_src_2:reg_block, gfx10_src_6:reg_block, MODIFIERS:str=''):
		return self.ic_pb(vop3_base('v_cmp_nge_f32_e64', None, None, gfx10_src_2, gfx10_src_6, None, MODIFIERS))
	def v_cmp_nge_f64_e64(self, gfx10_src_3:reg_block, gfx10_src_3:reg_block, MODIFIERS:str=''):
		return self.ic_pb(vop3_base('v_cmp_nge_f64_e64', None, None, gfx10_src_3, gfx10_src_3, None, MODIFIERS))
	def v_cmp_ngt_f16_e64(self, gfx10_src_2:reg_block, gfx10_src_6:reg_block, MODIFIERS:str=''):
		return self.ic_pb(vop3_base('v_cmp_ngt_f16_e64', None, None, gfx10_src_2, gfx10_src_6, None, MODIFIERS))
	def v_cmp_ngt_f32_e64(self, gfx10_src_2:reg_block, gfx10_src_6:reg_block, MODIFIERS:str=''):
		return self.ic_pb(vop3_base('v_cmp_ngt_f32_e64', None, None, gfx10_src_2, gfx10_src_6, None, MODIFIERS))
	def v_cmp_ngt_f64_e64(self, gfx10_src_3:reg_block, gfx10_src_3:reg_block, MODIFIERS:str=''):
		return self.ic_pb(vop3_base('v_cmp_ngt_f64_e64', None, None, gfx10_src_3, gfx10_src_3, None, MODIFIERS))
	def v_cmp_nle_f16_e64(self, gfx10_src_2:reg_block, gfx10_src_6:reg_block, MODIFIERS:str=''):
		return self.ic_pb(vop3_base('v_cmp_nle_f16_e64', None, None, gfx10_src_2, gfx10_src_6, None, MODIFIERS))
	def v_cmp_nle_f32_e64(self, gfx10_src_2:reg_block, gfx10_src_6:reg_block, MODIFIERS:str=''):
		return self.ic_pb(vop3_base('v_cmp_nle_f32_e64', None, None, gfx10_src_2, gfx10_src_6, None, MODIFIERS))
	def v_cmp_nle_f64_e64(self, gfx10_src_3:reg_block, gfx10_src_3:reg_block, MODIFIERS:str=''):
		return self.ic_pb(vop3_base('v_cmp_nle_f64_e64', None, None, gfx10_src_3, gfx10_src_3, None, MODIFIERS))
	def v_cmp_nlg_f16_e64(self, gfx10_src_2:reg_block, gfx10_src_6:reg_block, MODIFIERS:str=''):
		return self.ic_pb(vop3_base('v_cmp_nlg_f16_e64', None, None, gfx10_src_2, gfx10_src_6, None, MODIFIERS))
	def v_cmp_nlg_f32_e64(self, gfx10_src_2:reg_block, gfx10_src_6:reg_block, MODIFIERS:str=''):
		return self.ic_pb(vop3_base('v_cmp_nlg_f32_e64', None, None, gfx10_src_2, gfx10_src_6, None, MODIFIERS))
	def v_cmp_nlg_f64_e64(self, gfx10_src_3:reg_block, gfx10_src_3:reg_block, MODIFIERS:str=''):
		return self.ic_pb(vop3_base('v_cmp_nlg_f64_e64', None, None, gfx10_src_3, gfx10_src_3, None, MODIFIERS))
	def v_cmp_nlt_f16_e64(self, gfx10_src_2:reg_block, gfx10_src_6:reg_block, MODIFIERS:str=''):
		return self.ic_pb(vop3_base('v_cmp_nlt_f16_e64', None, None, gfx10_src_2, gfx10_src_6, None, MODIFIERS))
	def v_cmp_nlt_f32_e64(self, gfx10_src_2:reg_block, gfx10_src_6:reg_block, MODIFIERS:str=''):
		return self.ic_pb(vop3_base('v_cmp_nlt_f32_e64', None, None, gfx10_src_2, gfx10_src_6, None, MODIFIERS))
	def v_cmp_nlt_f64_e64(self, gfx10_src_3:reg_block, gfx10_src_3:reg_block, MODIFIERS:str=''):
		return self.ic_pb(vop3_base('v_cmp_nlt_f64_e64', None, None, gfx10_src_3, gfx10_src_3, None, MODIFIERS))
	def v_cmp_o_f16_e64(self, gfx10_src_2:reg_block, gfx10_src_6:reg_block, MODIFIERS:str=''):
		return self.ic_pb(vop3_base('v_cmp_o_f16_e64', None, None, gfx10_src_2, gfx10_src_6, None, MODIFIERS))
	def v_cmp_o_f32_e64(self, gfx10_src_2:reg_block, gfx10_src_6:reg_block, MODIFIERS:str=''):
		return self.ic_pb(vop3_base('v_cmp_o_f32_e64', None, None, gfx10_src_2, gfx10_src_6, None, MODIFIERS))
	def v_cmp_o_f64_e64(self, gfx10_src_3:reg_block, gfx10_src_3:reg_block, MODIFIERS:str=''):
		return self.ic_pb(vop3_base('v_cmp_o_f64_e64', None, None, gfx10_src_3, gfx10_src_3, None, MODIFIERS))
	def v_cmp_t_i32_e64(self, gfx10_src_2:reg_block, gfx10_src_6:reg_block):
		return self.ic_pb(vop3_base('v_cmp_t_i32_e64', None, None, gfx10_src_2, gfx10_src_6, None))
	def v_cmp_t_i64_e64(self, gfx10_src_3:reg_block, gfx10_src_3:reg_block):
		return self.ic_pb(vop3_base('v_cmp_t_i64_e64', None, None, gfx10_src_3, gfx10_src_3, None))
	def v_cmp_t_u32_e64(self, gfx10_src_2:reg_block, gfx10_src_6:reg_block):
		return self.ic_pb(vop3_base('v_cmp_t_u32_e64', None, None, gfx10_src_2, gfx10_src_6, None))
	def v_cmp_t_u64_e64(self, gfx10_src_3:reg_block, gfx10_src_3:reg_block):
		return self.ic_pb(vop3_base('v_cmp_t_u64_e64', None, None, gfx10_src_3, gfx10_src_3, None))
	def v_cmp_tru_f16_e64(self, gfx10_src_2:reg_block, gfx10_src_6:reg_block, MODIFIERS:str=''):
		return self.ic_pb(vop3_base('v_cmp_tru_f16_e64', None, None, gfx10_src_2, gfx10_src_6, None, MODIFIERS))
	def v_cmp_tru_f32_e64(self, gfx10_src_2:reg_block, gfx10_src_6:reg_block, MODIFIERS:str=''):
		return self.ic_pb(vop3_base('v_cmp_tru_f32_e64', None, None, gfx10_src_2, gfx10_src_6, None, MODIFIERS))
	def v_cmp_tru_f64_e64(self, gfx10_src_3:reg_block, gfx10_src_3:reg_block, MODIFIERS:str=''):
		return self.ic_pb(vop3_base('v_cmp_tru_f64_e64', None, None, gfx10_src_3, gfx10_src_3, None, MODIFIERS))
	def v_cmp_u_f16_e64(self, gfx10_src_2:reg_block, gfx10_src_6:reg_block, MODIFIERS:str=''):
		return self.ic_pb(vop3_base('v_cmp_u_f16_e64', None, None, gfx10_src_2, gfx10_src_6, None, MODIFIERS))
	def v_cmp_u_f32_e64(self, gfx10_src_2:reg_block, gfx10_src_6:reg_block, MODIFIERS:str=''):
		return self.ic_pb(vop3_base('v_cmp_u_f32_e64', None, None, gfx10_src_2, gfx10_src_6, None, MODIFIERS))
	def v_cmp_u_f64_e64(self, gfx10_src_3:reg_block, gfx10_src_3:reg_block, MODIFIERS:str=''):
		return self.ic_pb(vop3_base('v_cmp_u_f64_e64', None, None, gfx10_src_3, gfx10_src_3, None, MODIFIERS))
	def v_cmpx_class_f16_e64(self, gfx10_src_6:reg_block):
		return self.ic_pb(vop3_base('v_cmpx_class_f16_e64', None, None, None, gfx10_src_6, None))
	def v_cmpx_class_f32_e64(self, gfx10_src_6:reg_block):
		return self.ic_pb(vop3_base('v_cmpx_class_f32_e64', None, None, None, gfx10_src_6, None))
	def v_cmpx_class_f64_e64(self, gfx10_src_6:reg_block):
		return self.ic_pb(vop3_base('v_cmpx_class_f64_e64', None, None, None, gfx10_src_6, None))
	def v_cmpx_eq_f16_e64(self, gfx10_src_6:reg_block, MODIFIERS:str=''):
		return self.ic_pb(vop3_base('v_cmpx_eq_f16_e64', None, None, None, gfx10_src_6, None, MODIFIERS))
	def v_cmpx_eq_f32_e64(self, gfx10_src_6:reg_block, MODIFIERS:str=''):
		return self.ic_pb(vop3_base('v_cmpx_eq_f32_e64', None, None, None, gfx10_src_6, None, MODIFIERS))
	def v_cmpx_eq_f64_e64(self, gfx10_src_3:reg_block, MODIFIERS:str=''):
		return self.ic_pb(vop3_base('v_cmpx_eq_f64_e64', None, None, None, gfx10_src_3, None, MODIFIERS))
	def v_cmpx_eq_i16_e64(self, gfx10_src_8:reg_block):
		return self.ic_pb(vop3_base('v_cmpx_eq_i16_e64', None, None, None, gfx10_src_8, None))
	def v_cmpx_eq_i32_e64(self, gfx10_src_6:reg_block):
		return self.ic_pb(vop3_base('v_cmpx_eq_i32_e64', None, None, None, gfx10_src_6, None))
	def v_cmpx_eq_i64_e64(self, gfx10_src_3:reg_block):
		return self.ic_pb(vop3_base('v_cmpx_eq_i64_e64', None, None, None, gfx10_src_3, None))
	def v_cmpx_eq_u16_e64(self, gfx10_src_8:reg_block):
		return self.ic_pb(vop3_base('v_cmpx_eq_u16_e64', None, None, None, gfx10_src_8, None))
	def v_cmpx_eq_u32_e64(self, gfx10_src_6:reg_block):
		return self.ic_pb(vop3_base('v_cmpx_eq_u32_e64', None, None, None, gfx10_src_6, None))
	def v_cmpx_eq_u64_e64(self, gfx10_src_3:reg_block):
		return self.ic_pb(vop3_base('v_cmpx_eq_u64_e64', None, None, None, gfx10_src_3, None))
	def v_cmpx_f_f16_e64(self, gfx10_src_6:reg_block, MODIFIERS:str=''):
		return self.ic_pb(vop3_base('v_cmpx_f_f16_e64', None, None, None, gfx10_src_6, None, MODIFIERS))
	def v_cmpx_f_f32_e64(self, gfx10_src_6:reg_block, MODIFIERS:str=''):
		return self.ic_pb(vop3_base('v_cmpx_f_f32_e64', None, None, None, gfx10_src_6, None, MODIFIERS))
	def v_cmpx_f_f64_e64(self, gfx10_src_3:reg_block, MODIFIERS:str=''):
		return self.ic_pb(vop3_base('v_cmpx_f_f64_e64', None, None, None, gfx10_src_3, None, MODIFIERS))
	def v_cmpx_f_i32_e64(self, gfx10_src_6:reg_block):
		return self.ic_pb(vop3_base('v_cmpx_f_i32_e64', None, None, None, gfx10_src_6, None))
	def v_cmpx_f_i64_e64(self, gfx10_src_3:reg_block):
		return self.ic_pb(vop3_base('v_cmpx_f_i64_e64', None, None, None, gfx10_src_3, None))
	def v_cmpx_f_u32_e64(self, gfx10_src_6:reg_block):
		return self.ic_pb(vop3_base('v_cmpx_f_u32_e64', None, None, None, gfx10_src_6, None))
	def v_cmpx_f_u64_e64(self, gfx10_src_3:reg_block):
		return self.ic_pb(vop3_base('v_cmpx_f_u64_e64', None, None, None, gfx10_src_3, None))
	def v_cmpx_ge_f16_e64(self, gfx10_src_6:reg_block, MODIFIERS:str=''):
		return self.ic_pb(vop3_base('v_cmpx_ge_f16_e64', None, None, None, gfx10_src_6, None, MODIFIERS))
	def v_cmpx_ge_f32_e64(self, gfx10_src_6:reg_block, MODIFIERS:str=''):
		return self.ic_pb(vop3_base('v_cmpx_ge_f32_e64', None, None, None, gfx10_src_6, None, MODIFIERS))
	def v_cmpx_ge_f64_e64(self, gfx10_src_3:reg_block, MODIFIERS:str=''):
		return self.ic_pb(vop3_base('v_cmpx_ge_f64_e64', None, None, None, gfx10_src_3, None, MODIFIERS))
	def v_cmpx_ge_i16_e64(self, gfx10_src_8:reg_block):
		return self.ic_pb(vop3_base('v_cmpx_ge_i16_e64', None, None, None, gfx10_src_8, None))
	def v_cmpx_ge_i32_e64(self, gfx10_src_6:reg_block):
		return self.ic_pb(vop3_base('v_cmpx_ge_i32_e64', None, None, None, gfx10_src_6, None))
	def v_cmpx_ge_i64_e64(self, gfx10_src_3:reg_block):
		return self.ic_pb(vop3_base('v_cmpx_ge_i64_e64', None, None, None, gfx10_src_3, None))
	def v_cmpx_ge_u16_e64(self, gfx10_src_8:reg_block):
		return self.ic_pb(vop3_base('v_cmpx_ge_u16_e64', None, None, None, gfx10_src_8, None))
	def v_cmpx_ge_u32_e64(self, gfx10_src_6:reg_block):
		return self.ic_pb(vop3_base('v_cmpx_ge_u32_e64', None, None, None, gfx10_src_6, None))
	def v_cmpx_ge_u64_e64(self, gfx10_src_3:reg_block):
		return self.ic_pb(vop3_base('v_cmpx_ge_u64_e64', None, None, None, gfx10_src_3, None))
	def v_cmpx_gt_f16_e64(self, gfx10_src_6:reg_block, MODIFIERS:str=''):
		return self.ic_pb(vop3_base('v_cmpx_gt_f16_e64', None, None, None, gfx10_src_6, None, MODIFIERS))
	def v_cmpx_gt_f32_e64(self, gfx10_src_6:reg_block, MODIFIERS:str=''):
		return self.ic_pb(vop3_base('v_cmpx_gt_f32_e64', None, None, None, gfx10_src_6, None, MODIFIERS))
	def v_cmpx_gt_f64_e64(self, gfx10_src_3:reg_block, MODIFIERS:str=''):
		return self.ic_pb(vop3_base('v_cmpx_gt_f64_e64', None, None, None, gfx10_src_3, None, MODIFIERS))
	def v_cmpx_gt_i16_e64(self, gfx10_src_8:reg_block):
		return self.ic_pb(vop3_base('v_cmpx_gt_i16_e64', None, None, None, gfx10_src_8, None))
	def v_cmpx_gt_i32_e64(self, gfx10_src_6:reg_block):
		return self.ic_pb(vop3_base('v_cmpx_gt_i32_e64', None, None, None, gfx10_src_6, None))
	def v_cmpx_gt_i64_e64(self, gfx10_src_3:reg_block):
		return self.ic_pb(vop3_base('v_cmpx_gt_i64_e64', None, None, None, gfx10_src_3, None))
	def v_cmpx_gt_u16_e64(self, gfx10_src_8:reg_block):
		return self.ic_pb(vop3_base('v_cmpx_gt_u16_e64', None, None, None, gfx10_src_8, None))
	def v_cmpx_gt_u32_e64(self, gfx10_src_6:reg_block):
		return self.ic_pb(vop3_base('v_cmpx_gt_u32_e64', None, None, None, gfx10_src_6, None))
	def v_cmpx_gt_u64_e64(self, gfx10_src_3:reg_block):
		return self.ic_pb(vop3_base('v_cmpx_gt_u64_e64', None, None, None, gfx10_src_3, None))
	def v_cmpx_le_f16_e64(self, gfx10_src_6:reg_block, MODIFIERS:str=''):
		return self.ic_pb(vop3_base('v_cmpx_le_f16_e64', None, None, None, gfx10_src_6, None, MODIFIERS))
	def v_cmpx_le_f32_e64(self, gfx10_src_6:reg_block, MODIFIERS:str=''):
		return self.ic_pb(vop3_base('v_cmpx_le_f32_e64', None, None, None, gfx10_src_6, None, MODIFIERS))
	def v_cmpx_le_f64_e64(self, gfx10_src_3:reg_block, MODIFIERS:str=''):
		return self.ic_pb(vop3_base('v_cmpx_le_f64_e64', None, None, None, gfx10_src_3, None, MODIFIERS))
	def v_cmpx_le_i16_e64(self, gfx10_src_8:reg_block):
		return self.ic_pb(vop3_base('v_cmpx_le_i16_e64', None, None, None, gfx10_src_8, None))
	def v_cmpx_le_i32_e64(self, gfx10_src_6:reg_block):
		return self.ic_pb(vop3_base('v_cmpx_le_i32_e64', None, None, None, gfx10_src_6, None))
	def v_cmpx_le_i64_e64(self, gfx10_src_3:reg_block):
		return self.ic_pb(vop3_base('v_cmpx_le_i64_e64', None, None, None, gfx10_src_3, None))
	def v_cmpx_le_u16_e64(self, gfx10_src_8:reg_block):
		return self.ic_pb(vop3_base('v_cmpx_le_u16_e64', None, None, None, gfx10_src_8, None))
	def v_cmpx_le_u32_e64(self, gfx10_src_6:reg_block):
		return self.ic_pb(vop3_base('v_cmpx_le_u32_e64', None, None, None, gfx10_src_6, None))
	def v_cmpx_le_u64_e64(self, gfx10_src_3:reg_block):
		return self.ic_pb(vop3_base('v_cmpx_le_u64_e64', None, None, None, gfx10_src_3, None))
	def v_cmpx_lg_f16_e64(self, gfx10_src_6:reg_block, MODIFIERS:str=''):
		return self.ic_pb(vop3_base('v_cmpx_lg_f16_e64', None, None, None, gfx10_src_6, None, MODIFIERS))
	def v_cmpx_lg_f32_e64(self, gfx10_src_6:reg_block, MODIFIERS:str=''):
		return self.ic_pb(vop3_base('v_cmpx_lg_f32_e64', None, None, None, gfx10_src_6, None, MODIFIERS))
	def v_cmpx_lg_f64_e64(self, gfx10_src_3:reg_block, MODIFIERS:str=''):
		return self.ic_pb(vop3_base('v_cmpx_lg_f64_e64', None, None, None, gfx10_src_3, None, MODIFIERS))
	def v_cmpx_lt_f16_e64(self, gfx10_src_6:reg_block, MODIFIERS:str=''):
		return self.ic_pb(vop3_base('v_cmpx_lt_f16_e64', None, None, None, gfx10_src_6, None, MODIFIERS))
	def v_cmpx_lt_f32_e64(self, gfx10_src_6:reg_block, MODIFIERS:str=''):
		return self.ic_pb(vop3_base('v_cmpx_lt_f32_e64', None, None, None, gfx10_src_6, None, MODIFIERS))
	def v_cmpx_lt_f64_e64(self, gfx10_src_3:reg_block, MODIFIERS:str=''):
		return self.ic_pb(vop3_base('v_cmpx_lt_f64_e64', None, None, None, gfx10_src_3, None, MODIFIERS))
	def v_cmpx_lt_i16_e64(self, gfx10_src_8:reg_block):
		return self.ic_pb(vop3_base('v_cmpx_lt_i16_e64', None, None, None, gfx10_src_8, None))
	def v_cmpx_lt_i32_e64(self, gfx10_src_6:reg_block):
		return self.ic_pb(vop3_base('v_cmpx_lt_i32_e64', None, None, None, gfx10_src_6, None))
	def v_cmpx_lt_i64_e64(self, gfx10_src_3:reg_block):
		return self.ic_pb(vop3_base('v_cmpx_lt_i64_e64', None, None, None, gfx10_src_3, None))
	def v_cmpx_lt_u16_e64(self, gfx10_src_8:reg_block):
		return self.ic_pb(vop3_base('v_cmpx_lt_u16_e64', None, None, None, gfx10_src_8, None))
	def v_cmpx_lt_u32_e64(self, gfx10_src_6:reg_block):
		return self.ic_pb(vop3_base('v_cmpx_lt_u32_e64', None, None, None, gfx10_src_6, None))
	def v_cmpx_lt_u64_e64(self, gfx10_src_3:reg_block):
		return self.ic_pb(vop3_base('v_cmpx_lt_u64_e64', None, None, None, gfx10_src_3, None))
	def v_cmpx_ne_i16_e64(self, gfx10_src_8:reg_block):
		return self.ic_pb(vop3_base('v_cmpx_ne_i16_e64', None, None, None, gfx10_src_8, None))
	def v_cmpx_ne_i32_e64(self, gfx10_src_6:reg_block):
		return self.ic_pb(vop3_base('v_cmpx_ne_i32_e64', None, None, None, gfx10_src_6, None))
	def v_cmpx_ne_i64_e64(self, gfx10_src_3:reg_block):
		return self.ic_pb(vop3_base('v_cmpx_ne_i64_e64', None, None, None, gfx10_src_3, None))
	def v_cmpx_ne_u16_e64(self, gfx10_src_8:reg_block):
		return self.ic_pb(vop3_base('v_cmpx_ne_u16_e64', None, None, None, gfx10_src_8, None))
	def v_cmpx_ne_u32_e64(self, gfx10_src_6:reg_block):
		return self.ic_pb(vop3_base('v_cmpx_ne_u32_e64', None, None, None, gfx10_src_6, None))
	def v_cmpx_ne_u64_e64(self, gfx10_src_3:reg_block):
		return self.ic_pb(vop3_base('v_cmpx_ne_u64_e64', None, None, None, gfx10_src_3, None))
	def v_cmpx_neq_f16_e64(self, gfx10_src_6:reg_block, MODIFIERS:str=''):
		return self.ic_pb(vop3_base('v_cmpx_neq_f16_e64', None, None, None, gfx10_src_6, None, MODIFIERS))
	def v_cmpx_neq_f32_e64(self, gfx10_src_6:reg_block, MODIFIERS:str=''):
		return self.ic_pb(vop3_base('v_cmpx_neq_f32_e64', None, None, None, gfx10_src_6, None, MODIFIERS))
	def v_cmpx_neq_f64_e64(self, gfx10_src_3:reg_block, MODIFIERS:str=''):
		return self.ic_pb(vop3_base('v_cmpx_neq_f64_e64', None, None, None, gfx10_src_3, None, MODIFIERS))
	def v_cmpx_nge_f16_e64(self, gfx10_src_6:reg_block, MODIFIERS:str=''):
		return self.ic_pb(vop3_base('v_cmpx_nge_f16_e64', None, None, None, gfx10_src_6, None, MODIFIERS))
	def v_cmpx_nge_f32_e64(self, gfx10_src_6:reg_block, MODIFIERS:str=''):
		return self.ic_pb(vop3_base('v_cmpx_nge_f32_e64', None, None, None, gfx10_src_6, None, MODIFIERS))
	def v_cmpx_nge_f64_e64(self, gfx10_src_3:reg_block, MODIFIERS:str=''):
		return self.ic_pb(vop3_base('v_cmpx_nge_f64_e64', None, None, None, gfx10_src_3, None, MODIFIERS))
	def v_cmpx_ngt_f16_e64(self, gfx10_src_6:reg_block, MODIFIERS:str=''):
		return self.ic_pb(vop3_base('v_cmpx_ngt_f16_e64', None, None, None, gfx10_src_6, None, MODIFIERS))
	def v_cmpx_ngt_f32_e64(self, gfx10_src_6:reg_block, MODIFIERS:str=''):
		return self.ic_pb(vop3_base('v_cmpx_ngt_f32_e64', None, None, None, gfx10_src_6, None, MODIFIERS))
	def v_cmpx_ngt_f64_e64(self, gfx10_src_3:reg_block, MODIFIERS:str=''):
		return self.ic_pb(vop3_base('v_cmpx_ngt_f64_e64', None, None, None, gfx10_src_3, None, MODIFIERS))
	def v_cmpx_nle_f16_e64(self, gfx10_src_6:reg_block, MODIFIERS:str=''):
		return self.ic_pb(vop3_base('v_cmpx_nle_f16_e64', None, None, None, gfx10_src_6, None, MODIFIERS))
	def v_cmpx_nle_f32_e64(self, gfx10_src_6:reg_block, MODIFIERS:str=''):
		return self.ic_pb(vop3_base('v_cmpx_nle_f32_e64', None, None, None, gfx10_src_6, None, MODIFIERS))
	def v_cmpx_nle_f64_e64(self, gfx10_src_3:reg_block, MODIFIERS:str=''):
		return self.ic_pb(vop3_base('v_cmpx_nle_f64_e64', None, None, None, gfx10_src_3, None, MODIFIERS))
	def v_cmpx_nlg_f16_e64(self, gfx10_src_6:reg_block, MODIFIERS:str=''):
		return self.ic_pb(vop3_base('v_cmpx_nlg_f16_e64', None, None, None, gfx10_src_6, None, MODIFIERS))
	def v_cmpx_nlg_f32_e64(self, gfx10_src_6:reg_block, MODIFIERS:str=''):
		return self.ic_pb(vop3_base('v_cmpx_nlg_f32_e64', None, None, None, gfx10_src_6, None, MODIFIERS))
	def v_cmpx_nlg_f64_e64(self, gfx10_src_3:reg_block, MODIFIERS:str=''):
		return self.ic_pb(vop3_base('v_cmpx_nlg_f64_e64', None, None, None, gfx10_src_3, None, MODIFIERS))
	def v_cmpx_nlt_f16_e64(self, gfx10_src_6:reg_block, MODIFIERS:str=''):
		return self.ic_pb(vop3_base('v_cmpx_nlt_f16_e64', None, None, None, gfx10_src_6, None, MODIFIERS))
	def v_cmpx_nlt_f32_e64(self, gfx10_src_6:reg_block, MODIFIERS:str=''):
		return self.ic_pb(vop3_base('v_cmpx_nlt_f32_e64', None, None, None, gfx10_src_6, None, MODIFIERS))
	def v_cmpx_nlt_f64_e64(self, gfx10_src_3:reg_block, MODIFIERS:str=''):
		return self.ic_pb(vop3_base('v_cmpx_nlt_f64_e64', None, None, None, gfx10_src_3, None, MODIFIERS))
	def v_cmpx_o_f16_e64(self, gfx10_src_6:reg_block, MODIFIERS:str=''):
		return self.ic_pb(vop3_base('v_cmpx_o_f16_e64', None, None, None, gfx10_src_6, None, MODIFIERS))
	def v_cmpx_o_f32_e64(self, gfx10_src_6:reg_block, MODIFIERS:str=''):
		return self.ic_pb(vop3_base('v_cmpx_o_f32_e64', None, None, None, gfx10_src_6, None, MODIFIERS))
	def v_cmpx_o_f64_e64(self, gfx10_src_3:reg_block, MODIFIERS:str=''):
		return self.ic_pb(vop3_base('v_cmpx_o_f64_e64', None, None, None, gfx10_src_3, None, MODIFIERS))
	def v_cmpx_t_i32_e64(self, gfx10_src_6:reg_block):
		return self.ic_pb(vop3_base('v_cmpx_t_i32_e64', None, None, None, gfx10_src_6, None))
	def v_cmpx_t_i64_e64(self, gfx10_src_3:reg_block):
		return self.ic_pb(vop3_base('v_cmpx_t_i64_e64', None, None, None, gfx10_src_3, None))
	def v_cmpx_t_u32_e64(self, gfx10_src_6:reg_block):
		return self.ic_pb(vop3_base('v_cmpx_t_u32_e64', None, None, None, gfx10_src_6, None))
	def v_cmpx_t_u64_e64(self, gfx10_src_3:reg_block):
		return self.ic_pb(vop3_base('v_cmpx_t_u64_e64', None, None, None, gfx10_src_3, None))
	def v_cmpx_tru_f16_e64(self, gfx10_src_6:reg_block, MODIFIERS:str=''):
		return self.ic_pb(vop3_base('v_cmpx_tru_f16_e64', None, None, None, gfx10_src_6, None, MODIFIERS))
	def v_cmpx_tru_f32_e64(self, gfx10_src_6:reg_block, MODIFIERS:str=''):
		return self.ic_pb(vop3_base('v_cmpx_tru_f32_e64', None, None, None, gfx10_src_6, None, MODIFIERS))
	def v_cmpx_tru_f64_e64(self, gfx10_src_3:reg_block, MODIFIERS:str=''):
		return self.ic_pb(vop3_base('v_cmpx_tru_f64_e64', None, None, None, gfx10_src_3, None, MODIFIERS))
	def v_cmpx_u_f16_e64(self, gfx10_src_6:reg_block, MODIFIERS:str=''):
		return self.ic_pb(vop3_base('v_cmpx_u_f16_e64', None, None, None, gfx10_src_6, None, MODIFIERS))
	def v_cmpx_u_f32_e64(self, gfx10_src_6:reg_block, MODIFIERS:str=''):
		return self.ic_pb(vop3_base('v_cmpx_u_f32_e64', None, None, None, gfx10_src_6, None, MODIFIERS))
	def v_cmpx_u_f64_e64(self, gfx10_src_3:reg_block, MODIFIERS:str=''):
		return self.ic_pb(vop3_base('v_cmpx_u_f64_e64', None, None, None, gfx10_src_3, None, MODIFIERS))
	def v_cndmask_b32_e64(self, gfx10_src_2:reg_block, gfx10_src_6:reg_block, gfx10_ssrc_5:reg_block):
		return self.ic_pb(vop3_base('v_cndmask_b32_e64', None, None, gfx10_src_2, gfx10_src_6, gfx10_ssrc_5))
	def v_cos_f16_e64(self, gfx10_src_2:reg_block, MODIFIERS:str=''):
		return self.ic_pb(vop3_base('v_cos_f16_e64', None, None, gfx10_src_2, None, None, MODIFIERS))
	def v_cos_f32_e64(self, gfx10_src_2:reg_block, MODIFIERS:str=''):
		return self.ic_pb(vop3_base('v_cos_f32_e64', None, None, gfx10_src_2, None, None, MODIFIERS))
	def v_cubeid_f32(self, gfx10_src_2:reg_block, gfx10_src_6:reg_block, gfx10_src_6:reg_block, MODIFIERS:str=''):
		return self.ic_pb(vop3_base('v_cubeid_f32', None, None, gfx10_src_2, gfx10_src_6, gfx10_src_6, MODIFIERS))
	def v_cubema_f32(self, gfx10_src_2:reg_block, gfx10_src_6:reg_block, gfx10_src_6:reg_block, MODIFIERS:str=''):
		return self.ic_pb(vop3_base('v_cubema_f32', None, None, gfx10_src_2, gfx10_src_6, gfx10_src_6, MODIFIERS))
	def v_cubesc_f32(self, gfx10_src_2:reg_block, gfx10_src_6:reg_block, gfx10_src_6:reg_block, MODIFIERS:str=''):
		return self.ic_pb(vop3_base('v_cubesc_f32', None, None, gfx10_src_2, gfx10_src_6, gfx10_src_6, MODIFIERS))
	def v_cubetc_f32(self, gfx10_src_2:reg_block, gfx10_src_6:reg_block, gfx10_src_6:reg_block, MODIFIERS:str=''):
		return self.ic_pb(vop3_base('v_cubetc_f32', None, None, gfx10_src_2, gfx10_src_6, gfx10_src_6, MODIFIERS))
	def v_cvt_f16_f32_e64(self, gfx10_src_2:reg_block, MODIFIERS:str=''):
		return self.ic_pb(vop3_base('v_cvt_f16_f32_e64', None, None, gfx10_src_2, None, None, MODIFIERS))
	def v_cvt_f16_i16_e64(self, gfx10_src_4:reg_block, MODIFIERS:str=''):
		return self.ic_pb(vop3_base('v_cvt_f16_i16_e64', None, None, gfx10_src_4, None, None, MODIFIERS))
	def v_cvt_f16_u16_e64(self, gfx10_src_4:reg_block, MODIFIERS:str=''):
		return self.ic_pb(vop3_base('v_cvt_f16_u16_e64', None, None, gfx10_src_4, None, None, MODIFIERS))
	def v_cvt_f32_f16_e64(self, gfx10_src_2:reg_block, MODIFIERS:str=''):
		return self.ic_pb(vop3_base('v_cvt_f32_f16_e64', None, None, gfx10_src_2, None, None, MODIFIERS))
	def v_cvt_f32_f64_e64(self, gfx10_src_3:reg_block, MODIFIERS:str=''):
		return self.ic_pb(vop3_base('v_cvt_f32_f64_e64', None, None, gfx10_src_3, None, None, MODIFIERS))
	def v_cvt_f32_i32_e64(self, gfx10_src_2:reg_block, MODIFIERS:str=''):
		return self.ic_pb(vop3_base('v_cvt_f32_i32_e64', None, None, gfx10_src_2, None, None, MODIFIERS))
	def v_cvt_f32_u32_e64(self, gfx10_src_2:reg_block, MODIFIERS:str=''):
		return self.ic_pb(vop3_base('v_cvt_f32_u32_e64', None, None, gfx10_src_2, None, None, MODIFIERS))
	def v_cvt_f32_ubyte0_e64(self, gfx10_src_2:reg_block, MODIFIERS:str=''):
		return self.ic_pb(vop3_base('v_cvt_f32_ubyte0_e64', None, None, gfx10_src_2, None, None, MODIFIERS))
	def v_cvt_f32_ubyte1_e64(self, gfx10_src_2:reg_block, MODIFIERS:str=''):
		return self.ic_pb(vop3_base('v_cvt_f32_ubyte1_e64', None, None, gfx10_src_2, None, None, MODIFIERS))
	def v_cvt_f32_ubyte2_e64(self, gfx10_src_2:reg_block, MODIFIERS:str=''):
		return self.ic_pb(vop3_base('v_cvt_f32_ubyte2_e64', None, None, gfx10_src_2, None, None, MODIFIERS))
	def v_cvt_f32_ubyte3_e64(self, gfx10_src_2:reg_block, MODIFIERS:str=''):
		return self.ic_pb(vop3_base('v_cvt_f32_ubyte3_e64', None, None, gfx10_src_2, None, None, MODIFIERS))
	def v_cvt_f64_f32_e64(self, gfx10_src_2:reg_block, MODIFIERS:str=''):
		return self.ic_pb(vop3_base('v_cvt_f64_f32_e64', None, None, gfx10_src_2, None, None, MODIFIERS))
	def v_cvt_f64_i32_e64(self, gfx10_src_2:reg_block, MODIFIERS:str=''):
		return self.ic_pb(vop3_base('v_cvt_f64_i32_e64', None, None, gfx10_src_2, None, None, MODIFIERS))
	def v_cvt_f64_u32_e64(self, gfx10_src_2:reg_block, MODIFIERS:str=''):
		return self.ic_pb(vop3_base('v_cvt_f64_u32_e64', None, None, gfx10_src_2, None, None, MODIFIERS))
	def v_cvt_flr_i32_f32_e64(self, gfx10_src_2:reg_block):
		return self.ic_pb(vop3_base('v_cvt_flr_i32_f32_e64', None, None, gfx10_src_2, None, None))
	def v_cvt_i16_f16_e64(self, gfx10_src_2:reg_block, MODIFIERS:str=''):
		return self.ic_pb(vop3_base('v_cvt_i16_f16_e64', None, None, gfx10_src_2, None, None, MODIFIERS))
	def v_cvt_i32_f32_e64(self, gfx10_src_2:reg_block, MODIFIERS:str=''):
		return self.ic_pb(vop3_base('v_cvt_i32_f32_e64', None, None, gfx10_src_2, None, None, MODIFIERS))
	def v_cvt_i32_f64_e64(self, gfx10_src_3:reg_block, MODIFIERS:str=''):
		return self.ic_pb(vop3_base('v_cvt_i32_f64_e64', None, None, gfx10_src_3, None, None, MODIFIERS))
	def v_cvt_norm_i16_f16_e64(self, gfx10_src_2:reg_block, MODIFIERS:str=''):
		return self.ic_pb(vop3_base('v_cvt_norm_i16_f16_e64', None, None, gfx10_src_2, None, None, MODIFIERS))
	def v_cvt_norm_u16_f16_e64(self, gfx10_src_2:reg_block, MODIFIERS:str=''):
		return self.ic_pb(vop3_base('v_cvt_norm_u16_f16_e64', None, None, gfx10_src_2, None, None, MODIFIERS))
	def v_cvt_off_f32_i4_e64(self, gfx10_src_2:reg_block, MODIFIERS:str=''):
		return self.ic_pb(vop3_base('v_cvt_off_f32_i4_e64', None, None, gfx10_src_2, None, None, MODIFIERS))
	def v_cvt_pk_i16_i32(self, gfx10_src_2:reg_block, gfx10_src_6:reg_block):
		return self.ic_pb(vop3_base('v_cvt_pk_i16_i32', None, None, gfx10_src_2, gfx10_src_6, None))
	def v_cvt_pk_u16_u32(self, gfx10_src_2:reg_block, gfx10_src_6:reg_block):
		return self.ic_pb(vop3_base('v_cvt_pk_u16_u32', None, None, gfx10_src_2, gfx10_src_6, None))
	def v_cvt_pk_u8_f32(self, gfx10_src_2:reg_block, gfx10_src_6:reg_block, gfx10_src_6:reg_block):
		return self.ic_pb(vop3_base('v_cvt_pk_u8_f32', None, None, gfx10_src_2, gfx10_src_6, gfx10_src_6))
	def v_cvt_pknorm_i16_f16(self, gfx10_src_2:reg_block, gfx10_src_6:reg_block, MODIFIERS:str=''):
		return self.ic_pb(vop3_base('v_cvt_pknorm_i16_f16', None, None, gfx10_src_2, gfx10_src_6, None, MODIFIERS))
	def v_cvt_pknorm_i16_f32(self, gfx10_src_2:reg_block, gfx10_src_6:reg_block):
		return self.ic_pb(vop3_base('v_cvt_pknorm_i16_f32', None, None, gfx10_src_2, gfx10_src_6, None))
	def v_cvt_pknorm_u16_f16(self, gfx10_src_2:reg_block, gfx10_src_6:reg_block, MODIFIERS:str=''):
		return self.ic_pb(vop3_base('v_cvt_pknorm_u16_f16', None, None, gfx10_src_2, gfx10_src_6, None, MODIFIERS))
	def v_cvt_pknorm_u16_f32(self, gfx10_src_2:reg_block, gfx10_src_6:reg_block):
		return self.ic_pb(vop3_base('v_cvt_pknorm_u16_f32', None, None, gfx10_src_2, gfx10_src_6, None))
	def v_cvt_pkrtz_f16_f32_e64(self, gfx10_src_2:reg_block, gfx10_src_6:reg_block, MODIFIERS:str=''):
		return self.ic_pb(vop3_base('v_cvt_pkrtz_f16_f32_e64', None, None, gfx10_src_2, gfx10_src_6, None, MODIFIERS))
	def v_cvt_rpi_i32_f32_e64(self, gfx10_src_2:reg_block):
		return self.ic_pb(vop3_base('v_cvt_rpi_i32_f32_e64', None, None, gfx10_src_2, None, None))
	def v_cvt_u16_f16_e64(self, gfx10_src_2:reg_block, MODIFIERS:str=''):
		return self.ic_pb(vop3_base('v_cvt_u16_f16_e64', None, None, gfx10_src_2, None, None, MODIFIERS))
	def v_cvt_u32_f32_e64(self, gfx10_src_2:reg_block, MODIFIERS:str=''):
		return self.ic_pb(vop3_base('v_cvt_u32_f32_e64', None, None, gfx10_src_2, None, None, MODIFIERS))
	def v_cvt_u32_f64_e64(self, gfx10_src_3:reg_block, MODIFIERS:str=''):
		return self.ic_pb(vop3_base('v_cvt_u32_f64_e64', None, None, gfx10_src_3, None, None, MODIFIERS))
	def v_div_fixup_f16(self, gfx10_src_2:reg_block, gfx10_src_6:reg_block, gfx10_src_6:reg_block, MODIFIERS:str=''):
		return self.ic_pb(vop3_base('v_div_fixup_f16', None, None, gfx10_src_2, gfx10_src_6, gfx10_src_6, MODIFIERS))
	def v_div_fixup_f32(self, gfx10_src_2:reg_block, gfx10_src_6:reg_block, gfx10_src_6:reg_block, MODIFIERS:str=''):
		return self.ic_pb(vop3_base('v_div_fixup_f32', None, None, gfx10_src_2, gfx10_src_6, gfx10_src_6, MODIFIERS))
	def v_div_fixup_f64(self, gfx10_src_3:reg_block, gfx10_src_3:reg_block, gfx10_src_3:reg_block, MODIFIERS:str=''):
		return self.ic_pb(vop3_base('v_div_fixup_f64', None, None, gfx10_src_3, gfx10_src_3, gfx10_src_3, MODIFIERS))
	def v_div_fmas_f32(self, gfx10_src_2:reg_block, gfx10_src_6:reg_block, gfx10_src_6:reg_block, MODIFIERS:str=''):
		return self.ic_pb(vop3_base('v_div_fmas_f32', None, None, gfx10_src_2, gfx10_src_6, gfx10_src_6, MODIFIERS))
	def v_div_fmas_f64(self, gfx10_src_3:reg_block, gfx10_src_3:reg_block, gfx10_src_3:reg_block, MODIFIERS:str=''):
		return self.ic_pb(vop3_base('v_div_fmas_f64', None, None, gfx10_src_3, gfx10_src_3, gfx10_src_3, MODIFIERS))
	def v_div_scale_f32(self, gfx10_vcc:reg_block, gfx10_src_2:reg_block, gfx10_src_6:reg_block, gfx10_src_6:reg_block):
		return self.ic_pb(vop3_base('v_div_scale_f32', None, gfx10_vcc, gfx10_src_2, gfx10_src_6, gfx10_src_6))
	def v_div_scale_f64(self, gfx10_vcc:reg_block, gfx10_src_3:reg_block, gfx10_src_3:reg_block, gfx10_src_3:reg_block):
		return self.ic_pb(vop3_base('v_div_scale_f64', None, gfx10_vcc, gfx10_src_3, gfx10_src_3, gfx10_src_3))
	def v_exp_f16_e64(self, gfx10_src_2:reg_block, MODIFIERS:str=''):
		return self.ic_pb(vop3_base('v_exp_f16_e64', None, None, gfx10_src_2, None, None, MODIFIERS))
	def v_exp_f32_e64(self, gfx10_src_2:reg_block, MODIFIERS:str=''):
		return self.ic_pb(vop3_base('v_exp_f32_e64', None, None, gfx10_src_2, None, None, MODIFIERS))
	def v_ffbh_i32_e64(self, gfx10_src_2:reg_block):
		return self.ic_pb(vop3_base('v_ffbh_i32_e64', None, None, gfx10_src_2, None, None))
	def v_ffbh_u32_e64(self, gfx10_src_2:reg_block):
		return self.ic_pb(vop3_base('v_ffbh_u32_e64', None, None, gfx10_src_2, None, None))
	def v_ffbl_b32_e64(self, gfx10_src_2:reg_block):
		return self.ic_pb(vop3_base('v_ffbl_b32_e64', None, None, gfx10_src_2, None, None))
	def v_floor_f16_e64(self, gfx10_src_2:reg_block, MODIFIERS:str=''):
		return self.ic_pb(vop3_base('v_floor_f16_e64', None, None, gfx10_src_2, None, None, MODIFIERS))
	def v_floor_f32_e64(self, gfx10_src_2:reg_block, MODIFIERS:str=''):
		return self.ic_pb(vop3_base('v_floor_f32_e64', None, None, gfx10_src_2, None, None, MODIFIERS))
	def v_floor_f64_e64(self, gfx10_src_3:reg_block, MODIFIERS:str=''):
		return self.ic_pb(vop3_base('v_floor_f64_e64', None, None, gfx10_src_3, None, None, MODIFIERS))
	def v_fma_f16(self, gfx10_src_2:reg_block, gfx10_src_6:reg_block, gfx10_src_6:reg_block, MODIFIERS:str=''):
		return self.ic_pb(vop3_base('v_fma_f16', None, None, gfx10_src_2, gfx10_src_6, gfx10_src_6, MODIFIERS))
	def v_fma_f32(self, gfx10_src_2:reg_block, gfx10_src_6:reg_block, gfx10_src_6:reg_block, MODIFIERS:str=''):
		return self.ic_pb(vop3_base('v_fma_f32', None, None, gfx10_src_2, gfx10_src_6, gfx10_src_6, MODIFIERS))
	def v_fma_f64(self, gfx10_src_3:reg_block, gfx10_src_3:reg_block, gfx10_src_3:reg_block, MODIFIERS:str=''):
		return self.ic_pb(vop3_base('v_fma_f64', None, None, gfx10_src_3, gfx10_src_3, gfx10_src_3, MODIFIERS))
	def v_fmac_f16_e64(self, gfx10_src_2:reg_block, gfx10_src_6:reg_block, MODIFIERS:str=''):
		return self.ic_pb(vop3_base('v_fmac_f16_e64', None, None, gfx10_src_2, gfx10_src_6, None, MODIFIERS))
	def v_fmac_f32_e64(self, gfx10_src_2:reg_block, gfx10_src_6:reg_block, MODIFIERS:str=''):
		return self.ic_pb(vop3_base('v_fmac_f32_e64', None, None, gfx10_src_2, gfx10_src_6, None, MODIFIERS))
	def v_fract_f16_e64(self, gfx10_src_2:reg_block, MODIFIERS:str=''):
		return self.ic_pb(vop3_base('v_fract_f16_e64', None, None, gfx10_src_2, None, None, MODIFIERS))
	def v_fract_f32_e64(self, gfx10_src_2:reg_block, MODIFIERS:str=''):
		return self.ic_pb(vop3_base('v_fract_f32_e64', None, None, gfx10_src_2, None, None, MODIFIERS))
	def v_fract_f64_e64(self, gfx10_src_3:reg_block, MODIFIERS:str=''):
		return self.ic_pb(vop3_base('v_fract_f64_e64', None, None, gfx10_src_3, None, None, MODIFIERS))
	def v_frexp_exp_i16_f16_e64(self, gfx10_src_2:reg_block):
		return self.ic_pb(vop3_base('v_frexp_exp_i16_f16_e64', None, None, gfx10_src_2, None, None))
	def v_frexp_exp_i32_f32_e64(self, gfx10_src_2:reg_block):
		return self.ic_pb(vop3_base('v_frexp_exp_i32_f32_e64', None, None, gfx10_src_2, None, None))
	def v_frexp_exp_i32_f64_e64(self, gfx10_src_3:reg_block):
		return self.ic_pb(vop3_base('v_frexp_exp_i32_f64_e64', None, None, gfx10_src_3, None, None))
	def v_frexp_mant_f16_e64(self, gfx10_src_2:reg_block, MODIFIERS:str=''):
		return self.ic_pb(vop3_base('v_frexp_mant_f16_e64', None, None, gfx10_src_2, None, None, MODIFIERS))
	def v_frexp_mant_f32_e64(self, gfx10_src_2:reg_block, MODIFIERS:str=''):
		return self.ic_pb(vop3_base('v_frexp_mant_f32_e64', None, None, gfx10_src_2, None, None, MODIFIERS))
	def v_frexp_mant_f64_e64(self, gfx10_src_3:reg_block, MODIFIERS:str=''):
		return self.ic_pb(vop3_base('v_frexp_mant_f64_e64', None, None, gfx10_src_3, None, None, MODIFIERS))
	def v_interp_mov_f32_e64(self, gfx10_param:reg_block, gfx10_attr:reg_block, MODIFIERS:str=''):
		return self.ic_pb(vop3_base('v_interp_mov_f32_e64', None, None, gfx10_param, gfx10_attr, None, MODIFIERS))
	def v_interp_p1_f32_e64(self, gfx10_vsrc:reg_block, gfx10_attr:reg_block, MODIFIERS:str=''):
		return self.ic_pb(vop3_base('v_interp_p1_f32_e64', None, None, gfx10_vsrc, gfx10_attr, None, MODIFIERS))
	def v_interp_p1ll_f16(self, gfx10_vsrc:reg_block, gfx10_attr:reg_block, MODIFIERS:str=''):
		return self.ic_pb(vop3_base('v_interp_p1ll_f16', None, None, gfx10_vsrc, gfx10_attr, None, MODIFIERS))
	def v_interp_p1lv_f16(self, gfx10_vsrc:reg_block, gfx10_attr:reg_block, gfx10_vsrc:reg_block, MODIFIERS:str=''):
		return self.ic_pb(vop3_base('v_interp_p1lv_f16', None, None, gfx10_vsrc, gfx10_attr, gfx10_vsrc, MODIFIERS))
	def v_interp_p2_f16(self, gfx10_vsrc:reg_block, gfx10_attr:reg_block, gfx10_vsrc:reg_block, MODIFIERS:str=''):
		return self.ic_pb(vop3_base('v_interp_p2_f16', None, None, gfx10_vsrc, gfx10_attr, gfx10_vsrc, MODIFIERS))
	def v_interp_p2_f32_e64(self, gfx10_vsrc:reg_block, gfx10_attr:reg_block, MODIFIERS:str=''):
		return self.ic_pb(vop3_base('v_interp_p2_f32_e64', None, None, gfx10_vsrc, gfx10_attr, None, MODIFIERS))
	def v_ldexp_f16_e64(self, gfx10_src_2:reg_block, gfx10_src_6:reg_block, MODIFIERS:str=''):
		return self.ic_pb(vop3_base('v_ldexp_f16_e64', None, None, gfx10_src_2, gfx10_src_6, None, MODIFIERS))
	def v_ldexp_f32(self, gfx10_src_2:reg_block, gfx10_src_6:reg_block, MODIFIERS:str=''):
		return self.ic_pb(vop3_base('v_ldexp_f32', None, None, gfx10_src_2, gfx10_src_6, None, MODIFIERS))
	def v_ldexp_f64(self, gfx10_src_3:reg_block, gfx10_src_6:reg_block, MODIFIERS:str=''):
		return self.ic_pb(vop3_base('v_ldexp_f64', None, None, gfx10_src_3, gfx10_src_6, None, MODIFIERS))
	def v_lerp_u8(self, gfx10_src_2:reg_block, gfx10_src_6:reg_block, gfx10_src_6:reg_block):
		return self.ic_pb(vop3_base('v_lerp_u8', None, None, gfx10_src_2, gfx10_src_6, gfx10_src_6))
	def v_log_f16_e64(self, gfx10_src_2:reg_block, MODIFIERS:str=''):
		return self.ic_pb(vop3_base('v_log_f16_e64', None, None, gfx10_src_2, None, None, MODIFIERS))
	def v_log_f32_e64(self, gfx10_src_2:reg_block, MODIFIERS:str=''):
		return self.ic_pb(vop3_base('v_log_f32_e64', None, None, gfx10_src_2, None, None, MODIFIERS))
	def v_lshl_add_u32(self, gfx10_src_2:reg_block, gfx10_src_6:reg_block, gfx10_src_6:reg_block):
		return self.ic_pb(vop3_base('v_lshl_add_u32', None, None, gfx10_src_2, gfx10_src_6, gfx10_src_6))
	def v_lshl_or_b32(self, gfx10_src_2:reg_block, gfx10_src_6:reg_block, gfx10_src_6:reg_block):
		return self.ic_pb(vop3_base('v_lshl_or_b32', None, None, gfx10_src_2, gfx10_src_6, gfx10_src_6))
	def v_lshlrev_b16(self, gfx10_src_8:reg_block, gfx10_src_8:reg_block):
		return self.ic_pb(vop3_base('v_lshlrev_b16', None, None, gfx10_src_8, gfx10_src_8, None))
	def v_lshlrev_b32_e64(self, gfx10_src_6:reg_block, gfx10_src_6:reg_block):
		return self.ic_pb(vop3_base('v_lshlrev_b32_e64', None, None, gfx10_src_6, gfx10_src_6, None))
	def v_lshlrev_b64(self, gfx10_src_6:reg_block, gfx10_src_3:reg_block):
		return self.ic_pb(vop3_base('v_lshlrev_b64', None, None, gfx10_src_6, gfx10_src_3, None))
	def v_lshrrev_b16(self, gfx10_src_8:reg_block, gfx10_src_8:reg_block):
		return self.ic_pb(vop3_base('v_lshrrev_b16', None, None, gfx10_src_8, gfx10_src_8, None))
	def v_lshrrev_b32_e64(self, gfx10_src_6:reg_block, gfx10_src_6:reg_block):
		return self.ic_pb(vop3_base('v_lshrrev_b32_e64', None, None, gfx10_src_6, gfx10_src_6, None))
	def v_lshrrev_b64(self, gfx10_src_6:reg_block, gfx10_src_3:reg_block):
		return self.ic_pb(vop3_base('v_lshrrev_b64', None, None, gfx10_src_6, gfx10_src_3, None))
	def v_mac_f32_e64(self, gfx10_src_2:reg_block, gfx10_src_6:reg_block, MODIFIERS:str=''):
		return self.ic_pb(vop3_base('v_mac_f32_e64', None, None, gfx10_src_2, gfx10_src_6, None, MODIFIERS))
	def v_mac_legacy_f32_e64(self, gfx10_src_2:reg_block, gfx10_src_6:reg_block, MODIFIERS:str=''):
		return self.ic_pb(vop3_base('v_mac_legacy_f32_e64', None, None, gfx10_src_2, gfx10_src_6, None, MODIFIERS))
	def v_mad_f32(self, gfx10_src_2:reg_block, gfx10_src_6:reg_block, gfx10_src_6:reg_block, MODIFIERS:str=''):
		return self.ic_pb(vop3_base('v_mad_f32', None, None, gfx10_src_2, gfx10_src_6, gfx10_src_6, MODIFIERS))
	def v_mad_i16(self, gfx10_src_4:reg_block, gfx10_src_8:reg_block, gfx10_src_8:reg_block, MODIFIERS:str=''):
		return self.ic_pb(vop3_base('v_mad_i16', None, None, gfx10_src_4, gfx10_src_8, gfx10_src_8, MODIFIERS))
	def v_mad_i32_i16(self, gfx10_src_4:reg_block, gfx10_src_8:reg_block, gfx10_src_6:reg_block, MODIFIERS:str=''):
		return self.ic_pb(vop3_base('v_mad_i32_i16', None, None, gfx10_src_4, gfx10_src_8, gfx10_src_6, MODIFIERS))
	def v_mad_i32_i24(self, gfx10_src_2:reg_block, gfx10_src_6:reg_block, gfx10_src_6:reg_block, MODIFIERS:str=''):
		return self.ic_pb(vop3_base('v_mad_i32_i24', None, None, gfx10_src_2, gfx10_src_6, gfx10_src_6, MODIFIERS))
	def v_mad_i64_i32(self, gfx10_sdst:reg_block, gfx10_src_2:reg_block, gfx10_src_6:reg_block, gfx10_src_3:reg_block, MODIFIERS:str=''):
		return self.ic_pb(vop3_base('v_mad_i64_i32', None, gfx10_sdst, gfx10_src_2, gfx10_src_6, gfx10_src_3, MODIFIERS))
	def v_mad_legacy_f32(self, gfx10_src_2:reg_block, gfx10_src_6:reg_block, gfx10_src_6:reg_block, MODIFIERS:str=''):
		return self.ic_pb(vop3_base('v_mad_legacy_f32', None, None, gfx10_src_2, gfx10_src_6, gfx10_src_6, MODIFIERS))
	def v_mad_u16(self, gfx10_src_4:reg_block, gfx10_src_8:reg_block, gfx10_src_8:reg_block, MODIFIERS:str=''):
		return self.ic_pb(vop3_base('v_mad_u16', None, None, gfx10_src_4, gfx10_src_8, gfx10_src_8, MODIFIERS))
	def v_mad_u32_u16(self, gfx10_src_4:reg_block, gfx10_src_8:reg_block, gfx10_src_6:reg_block, MODIFIERS:str=''):
		return self.ic_pb(vop3_base('v_mad_u32_u16', None, None, gfx10_src_4, gfx10_src_8, gfx10_src_6, MODIFIERS))
	def v_mad_u32_u24(self, gfx10_src_2:reg_block, gfx10_src_6:reg_block, gfx10_src_6:reg_block, MODIFIERS:str=''):
		return self.ic_pb(vop3_base('v_mad_u32_u24', None, None, gfx10_src_2, gfx10_src_6, gfx10_src_6, MODIFIERS))
	def v_mad_u64_u32(self, gfx10_sdst:reg_block, gfx10_src_2:reg_block, gfx10_src_6:reg_block, gfx10_src_3:reg_block, MODIFIERS:str=''):
		return self.ic_pb(vop3_base('v_mad_u64_u32', None, gfx10_sdst, gfx10_src_2, gfx10_src_6, gfx10_src_3, MODIFIERS))
	def v_max3_f16(self, gfx10_src_2:reg_block, gfx10_src_6:reg_block, gfx10_src_6:reg_block, MODIFIERS:str=''):
		return self.ic_pb(vop3_base('v_max3_f16', None, None, gfx10_src_2, gfx10_src_6, gfx10_src_6, MODIFIERS))
	def v_max3_f32(self, gfx10_src_2:reg_block, gfx10_src_6:reg_block, gfx10_src_6:reg_block, MODIFIERS:str=''):
		return self.ic_pb(vop3_base('v_max3_f32', None, None, gfx10_src_2, gfx10_src_6, gfx10_src_6, MODIFIERS))
	def v_max3_i16(self, gfx10_src_4:reg_block, gfx10_src_8:reg_block, gfx10_src_8:reg_block, MODIFIERS:str=''):
		return self.ic_pb(vop3_base('v_max3_i16', None, None, gfx10_src_4, gfx10_src_8, gfx10_src_8, MODIFIERS))
	def v_max3_i32(self, gfx10_src_2:reg_block, gfx10_src_6:reg_block, gfx10_src_6:reg_block):
		return self.ic_pb(vop3_base('v_max3_i32', None, None, gfx10_src_2, gfx10_src_6, gfx10_src_6))
	def v_max3_u16(self, gfx10_src_4:reg_block, gfx10_src_8:reg_block, gfx10_src_8:reg_block, MODIFIERS:str=''):
		return self.ic_pb(vop3_base('v_max3_u16', None, None, gfx10_src_4, gfx10_src_8, gfx10_src_8, MODIFIERS))
	def v_max3_u32(self, gfx10_src_2:reg_block, gfx10_src_6:reg_block, gfx10_src_6:reg_block):
		return self.ic_pb(vop3_base('v_max3_u32', None, None, gfx10_src_2, gfx10_src_6, gfx10_src_6))
	def v_max_f16_e64(self, gfx10_src_2:reg_block, gfx10_src_6:reg_block, MODIFIERS:str=''):
		return self.ic_pb(vop3_base('v_max_f16_e64', None, None, gfx10_src_2, gfx10_src_6, None, MODIFIERS))
	def v_max_f32_e64(self, gfx10_src_2:reg_block, gfx10_src_6:reg_block, MODIFIERS:str=''):
		return self.ic_pb(vop3_base('v_max_f32_e64', None, None, gfx10_src_2, gfx10_src_6, None, MODIFIERS))
	def v_max_f64(self, gfx10_src_3:reg_block, gfx10_src_3:reg_block, MODIFIERS:str=''):
		return self.ic_pb(vop3_base('v_max_f64', None, None, gfx10_src_3, gfx10_src_3, None, MODIFIERS))
	def v_max_i16(self, gfx10_src_4:reg_block, gfx10_src_8:reg_block):
		return self.ic_pb(vop3_base('v_max_i16', None, None, gfx10_src_4, gfx10_src_8, None))
	def v_max_i32_e64(self, gfx10_src_2:reg_block, gfx10_src_6:reg_block):
		return self.ic_pb(vop3_base('v_max_i32_e64', None, None, gfx10_src_2, gfx10_src_6, None))
	def v_max_u16(self, gfx10_src_4:reg_block, gfx10_src_8:reg_block):
		return self.ic_pb(vop3_base('v_max_u16', None, None, gfx10_src_4, gfx10_src_8, None))
	def v_max_u32_e64(self, gfx10_src_2:reg_block, gfx10_src_6:reg_block):
		return self.ic_pb(vop3_base('v_max_u32_e64', None, None, gfx10_src_2, gfx10_src_6, None))
	def v_mbcnt_hi_u32_b32(self, gfx10_src_2:reg_block, gfx10_src_6:reg_block):
		return self.ic_pb(vop3_base('v_mbcnt_hi_u32_b32', None, None, gfx10_src_2, gfx10_src_6, None))
	def v_mbcnt_lo_u32_b32(self, gfx10_src_2:reg_block, gfx10_src_6:reg_block):
		return self.ic_pb(vop3_base('v_mbcnt_lo_u32_b32', None, None, gfx10_src_2, gfx10_src_6, None))
	def v_med3_f16(self, gfx10_src_2:reg_block, gfx10_src_6:reg_block, gfx10_src_6:reg_block, MODIFIERS:str=''):
		return self.ic_pb(vop3_base('v_med3_f16', None, None, gfx10_src_2, gfx10_src_6, gfx10_src_6, MODIFIERS))
	def v_med3_f32(self, gfx10_src_2:reg_block, gfx10_src_6:reg_block, gfx10_src_6:reg_block, MODIFIERS:str=''):
		return self.ic_pb(vop3_base('v_med3_f32', None, None, gfx10_src_2, gfx10_src_6, gfx10_src_6, MODIFIERS))
	def v_med3_i16(self, gfx10_src_4:reg_block, gfx10_src_8:reg_block, gfx10_src_8:reg_block, MODIFIERS:str=''):
		return self.ic_pb(vop3_base('v_med3_i16', None, None, gfx10_src_4, gfx10_src_8, gfx10_src_8, MODIFIERS))
	def v_med3_i32(self, gfx10_src_2:reg_block, gfx10_src_6:reg_block, gfx10_src_6:reg_block):
		return self.ic_pb(vop3_base('v_med3_i32', None, None, gfx10_src_2, gfx10_src_6, gfx10_src_6))
	def v_med3_u16(self, gfx10_src_4:reg_block, gfx10_src_8:reg_block, gfx10_src_8:reg_block, MODIFIERS:str=''):
		return self.ic_pb(vop3_base('v_med3_u16', None, None, gfx10_src_4, gfx10_src_8, gfx10_src_8, MODIFIERS))
	def v_med3_u32(self, gfx10_src_2:reg_block, gfx10_src_6:reg_block, gfx10_src_6:reg_block):
		return self.ic_pb(vop3_base('v_med3_u32', None, None, gfx10_src_2, gfx10_src_6, gfx10_src_6))
	def v_min3_f16(self, gfx10_src_2:reg_block, gfx10_src_6:reg_block, gfx10_src_6:reg_block, MODIFIERS:str=''):
		return self.ic_pb(vop3_base('v_min3_f16', None, None, gfx10_src_2, gfx10_src_6, gfx10_src_6, MODIFIERS))
	def v_min3_f32(self, gfx10_src_2:reg_block, gfx10_src_6:reg_block, gfx10_src_6:reg_block, MODIFIERS:str=''):
		return self.ic_pb(vop3_base('v_min3_f32', None, None, gfx10_src_2, gfx10_src_6, gfx10_src_6, MODIFIERS))
	def v_min3_i16(self, gfx10_src_4:reg_block, gfx10_src_8:reg_block, gfx10_src_8:reg_block, MODIFIERS:str=''):
		return self.ic_pb(vop3_base('v_min3_i16', None, None, gfx10_src_4, gfx10_src_8, gfx10_src_8, MODIFIERS))
	def v_min3_i32(self, gfx10_src_2:reg_block, gfx10_src_6:reg_block, gfx10_src_6:reg_block):
		return self.ic_pb(vop3_base('v_min3_i32', None, None, gfx10_src_2, gfx10_src_6, gfx10_src_6))
	def v_min3_u16(self, gfx10_src_4:reg_block, gfx10_src_8:reg_block, gfx10_src_8:reg_block, MODIFIERS:str=''):
		return self.ic_pb(vop3_base('v_min3_u16', None, None, gfx10_src_4, gfx10_src_8, gfx10_src_8, MODIFIERS))
	def v_min3_u32(self, gfx10_src_2:reg_block, gfx10_src_6:reg_block, gfx10_src_6:reg_block):
		return self.ic_pb(vop3_base('v_min3_u32', None, None, gfx10_src_2, gfx10_src_6, gfx10_src_6))
	def v_min_f16_e64(self, gfx10_src_2:reg_block, gfx10_src_6:reg_block, MODIFIERS:str=''):
		return self.ic_pb(vop3_base('v_min_f16_e64', None, None, gfx10_src_2, gfx10_src_6, None, MODIFIERS))
	def v_min_f32_e64(self, gfx10_src_2:reg_block, gfx10_src_6:reg_block, MODIFIERS:str=''):
		return self.ic_pb(vop3_base('v_min_f32_e64', None, None, gfx10_src_2, gfx10_src_6, None, MODIFIERS))
	def v_min_f64(self, gfx10_src_3:reg_block, gfx10_src_3:reg_block, MODIFIERS:str=''):
		return self.ic_pb(vop3_base('v_min_f64', None, None, gfx10_src_3, gfx10_src_3, None, MODIFIERS))
	def v_min_i16(self, gfx10_src_4:reg_block, gfx10_src_8:reg_block):
		return self.ic_pb(vop3_base('v_min_i16', None, None, gfx10_src_4, gfx10_src_8, None))
	def v_min_i32_e64(self, gfx10_src_2:reg_block, gfx10_src_6:reg_block):
		return self.ic_pb(vop3_base('v_min_i32_e64', None, None, gfx10_src_2, gfx10_src_6, None))
	def v_min_u16(self, gfx10_src_4:reg_block, gfx10_src_8:reg_block):
		return self.ic_pb(vop3_base('v_min_u16', None, None, gfx10_src_4, gfx10_src_8, None))
	def v_min_u32_e64(self, gfx10_src_2:reg_block, gfx10_src_6:reg_block):
		return self.ic_pb(vop3_base('v_min_u32_e64', None, None, gfx10_src_2, gfx10_src_6, None))
	def v_mov_b32_e64(self, gfx10_src_2:reg_block):
		return self.ic_pb(vop3_base('v_mov_b32_e64', None, None, gfx10_src_2, None, None))
	def v_movreld_b32_e64(self, gfx10_src_2:reg_block):
		return self.ic_pb(vop3_base('v_movreld_b32_e64', None, None, gfx10_src_2, None, None))
	def v_movrels_b32_e64(self, gfx10_vsrc:reg_block):
		return self.ic_pb(vop3_base('v_movrels_b32_e64', None, None, gfx10_vsrc, None, None))
	def v_movrelsd_2_b32_e64(self, gfx10_vsrc:reg_block):
		return self.ic_pb(vop3_base('v_movrelsd_2_b32_e64', None, None, gfx10_vsrc, None, None))
	def v_movrelsd_b32_e64(self, gfx10_vsrc:reg_block):
		return self.ic_pb(vop3_base('v_movrelsd_b32_e64', None, None, gfx10_vsrc, None, None))
	def v_mqsad_pk_u16_u8(self, gfx10_src_3:reg_block, gfx10_src_6:reg_block, gfx10_src_3:reg_block, MODIFIERS:str=''):
		return self.ic_pb(vop3_base('v_mqsad_pk_u16_u8', None, None, gfx10_src_3, gfx10_src_6, gfx10_src_3, MODIFIERS))
	def v_mqsad_u32_u8(self, gfx10_src_3:reg_block, gfx10_src_6:reg_block, gfx10_vsrc_2:reg_block, MODIFIERS:str=''):
		return self.ic_pb(vop3_base('v_mqsad_u32_u8', None, None, gfx10_src_3, gfx10_src_6, gfx10_vsrc_2, MODIFIERS))
	def v_msad_u8(self, gfx10_src_2:reg_block, gfx10_src_6:reg_block, gfx10_src_6:reg_block, MODIFIERS:str=''):
		return self.ic_pb(vop3_base('v_msad_u8', None, None, gfx10_src_2, gfx10_src_6, gfx10_src_6, MODIFIERS))
	def v_mul_f16_e64(self, gfx10_src_2:reg_block, gfx10_src_6:reg_block, MODIFIERS:str=''):
		return self.ic_pb(vop3_base('v_mul_f16_e64', None, None, gfx10_src_2, gfx10_src_6, None, MODIFIERS))
	def v_mul_f32_e64(self, gfx10_src_2:reg_block, gfx10_src_6:reg_block, MODIFIERS:str=''):
		return self.ic_pb(vop3_base('v_mul_f32_e64', None, None, gfx10_src_2, gfx10_src_6, None, MODIFIERS))
	def v_mul_f64(self, gfx10_src_3:reg_block, gfx10_src_3:reg_block, MODIFIERS:str=''):
		return self.ic_pb(vop3_base('v_mul_f64', None, None, gfx10_src_3, gfx10_src_3, None, MODIFIERS))
	def v_mul_hi_i32(self, gfx10_src_2:reg_block, gfx10_src_6:reg_block):
		return self.ic_pb(vop3_base('v_mul_hi_i32', None, None, gfx10_src_2, gfx10_src_6, None))
	def v_mul_hi_i32_i24_e64(self, gfx10_src_2:reg_block, gfx10_src_6:reg_block):
		return self.ic_pb(vop3_base('v_mul_hi_i32_i24_e64', None, None, gfx10_src_2, gfx10_src_6, None))
	def v_mul_hi_u32(self, gfx10_src_2:reg_block, gfx10_src_6:reg_block):
		return self.ic_pb(vop3_base('v_mul_hi_u32', None, None, gfx10_src_2, gfx10_src_6, None))
	def v_mul_hi_u32_u24_e64(self, gfx10_src_2:reg_block, gfx10_src_6:reg_block):
		return self.ic_pb(vop3_base('v_mul_hi_u32_u24_e64', None, None, gfx10_src_2, gfx10_src_6, None))
	def v_mul_i32_i24_e64(self, gfx10_src_2:reg_block, gfx10_src_6:reg_block, MODIFIERS:str=''):
		return self.ic_pb(vop3_base('v_mul_i32_i24_e64', None, None, gfx10_src_2, gfx10_src_6, None, MODIFIERS))
	def v_mul_legacy_f32_e64(self, gfx10_src_2:reg_block, gfx10_src_6:reg_block, MODIFIERS:str=''):
		return self.ic_pb(vop3_base('v_mul_legacy_f32_e64', None, None, gfx10_src_2, gfx10_src_6, None, MODIFIERS))
	def v_mul_lo_u16(self, gfx10_src_4:reg_block, gfx10_src_8:reg_block):
		return self.ic_pb(vop3_base('v_mul_lo_u16', None, None, gfx10_src_4, gfx10_src_8, None))
	def v_mul_lo_u32(self, gfx10_src_2:reg_block, gfx10_src_6:reg_block):
		return self.ic_pb(vop3_base('v_mul_lo_u32', None, None, gfx10_src_2, gfx10_src_6, None))
	def v_mul_u32_u24_e64(self, gfx10_src_2:reg_block, gfx10_src_6:reg_block, MODIFIERS:str=''):
		return self.ic_pb(vop3_base('v_mul_u32_u24_e64', None, None, gfx10_src_2, gfx10_src_6, None, MODIFIERS))
	def v_mullit_f32(self, gfx10_src_2:reg_block, gfx10_src_6:reg_block, gfx10_src_6:reg_block, MODIFIERS:str=''):
		return self.ic_pb(vop3_base('v_mullit_f32', None, None, gfx10_src_2, gfx10_src_6, gfx10_src_6, MODIFIERS))
	def v_nop_e64(self):
		return self.ic_pb(vop3_base('v_nop_e64', None, None, None, None, None))
	def v_not_b32_e64(self, gfx10_src_2:reg_block):
		return self.ic_pb(vop3_base('v_not_b32_e64', None, None, gfx10_src_2, None, None))
	def v_or3_b32(self, gfx10_src_2:reg_block, gfx10_src_6:reg_block, gfx10_src_6:reg_block):
		return self.ic_pb(vop3_base('v_or3_b32', None, None, gfx10_src_2, gfx10_src_6, gfx10_src_6))
	def v_or_b32_e64(self, gfx10_src_2:reg_block, gfx10_src_6:reg_block):
		return self.ic_pb(vop3_base('v_or_b32_e64', None, None, gfx10_src_2, gfx10_src_6, None))
	def v_pack_b32_f16(self, gfx10_src_2:reg_block, gfx10_src_6:reg_block, MODIFIERS:str=''):
		return self.ic_pb(vop3_base('v_pack_b32_f16', None, None, gfx10_src_2, gfx10_src_6, None, MODIFIERS))
	def v_perm_b32(self, gfx10_src_2:reg_block, gfx10_src_6:reg_block, gfx10_src_6:reg_block):
		return self.ic_pb(vop3_base('v_perm_b32', None, None, gfx10_src_2, gfx10_src_6, gfx10_src_6))
	def v_permlane16_b32(self, gfx10_vdata:reg_block, gfx10_ssrc_6:reg_block, gfx10_ssrc_6:reg_block, MODIFIERS:str=''):
		return self.ic_pb(vop3_base('v_permlane16_b32', None, None, gfx10_vdata, gfx10_ssrc_6, gfx10_ssrc_6, MODIFIERS))
	def v_permlanex16_b32(self, gfx10_vdata:reg_block, gfx10_ssrc_6:reg_block, gfx10_ssrc_6:reg_block, MODIFIERS:str=''):
		return self.ic_pb(vop3_base('v_permlanex16_b32', None, None, gfx10_vdata, gfx10_ssrc_6, gfx10_ssrc_6, MODIFIERS))
	def v_pipeflush_e64(self):
		return self.ic_pb(vop3_base('v_pipeflush_e64', None, None, None, None, None))
	def v_qsad_pk_u16_u8(self, gfx10_src_3:reg_block, gfx10_src_6:reg_block, gfx10_src_3:reg_block, MODIFIERS:str=''):
		return self.ic_pb(vop3_base('v_qsad_pk_u16_u8', None, None, gfx10_src_3, gfx10_src_6, gfx10_src_3, MODIFIERS))
	def v_rcp_f16_e64(self, gfx10_src_2:reg_block, MODIFIERS:str=''):
		return self.ic_pb(vop3_base('v_rcp_f16_e64', None, None, gfx10_src_2, None, None, MODIFIERS))
	def v_rcp_f32_e64(self, gfx10_src_2:reg_block, MODIFIERS:str=''):
		return self.ic_pb(vop3_base('v_rcp_f32_e64', None, None, gfx10_src_2, None, None, MODIFIERS))
	def v_rcp_f64_e64(self, gfx10_src_3:reg_block, MODIFIERS:str=''):
		return self.ic_pb(vop3_base('v_rcp_f64_e64', None, None, gfx10_src_3, None, None, MODIFIERS))
	def v_rcp_iflag_f32_e64(self, gfx10_src_2:reg_block, MODIFIERS:str=''):
		return self.ic_pb(vop3_base('v_rcp_iflag_f32_e64', None, None, gfx10_src_2, None, None, MODIFIERS))
	def v_readlane_b32(self, gfx10_src_5:reg_block, gfx10_ssrc_7:reg_block):
		return self.ic_pb(vop3_base('v_readlane_b32', None, None, gfx10_src_5, gfx10_ssrc_7, None))
	def v_rndne_f16_e64(self, gfx10_src_2:reg_block, MODIFIERS:str=''):
		return self.ic_pb(vop3_base('v_rndne_f16_e64', None, None, gfx10_src_2, None, None, MODIFIERS))
	def v_rndne_f32_e64(self, gfx10_src_2:reg_block, MODIFIERS:str=''):
		return self.ic_pb(vop3_base('v_rndne_f32_e64', None, None, gfx10_src_2, None, None, MODIFIERS))
	def v_rndne_f64_e64(self, gfx10_src_3:reg_block, MODIFIERS:str=''):
		return self.ic_pb(vop3_base('v_rndne_f64_e64', None, None, gfx10_src_3, None, None, MODIFIERS))
	def v_rsq_f16_e64(self, gfx10_src_2:reg_block, MODIFIERS:str=''):
		return self.ic_pb(vop3_base('v_rsq_f16_e64', None, None, gfx10_src_2, None, None, MODIFIERS))
	def v_rsq_f32_e64(self, gfx10_src_2:reg_block, MODIFIERS:str=''):
		return self.ic_pb(vop3_base('v_rsq_f32_e64', None, None, gfx10_src_2, None, None, MODIFIERS))
	def v_rsq_f64_e64(self, gfx10_src_3:reg_block, MODIFIERS:str=''):
		return self.ic_pb(vop3_base('v_rsq_f64_e64', None, None, gfx10_src_3, None, None, MODIFIERS))
	def v_sad_hi_u8(self, gfx10_src_2:reg_block, gfx10_src_6:reg_block, gfx10_src_6:reg_block, MODIFIERS:str=''):
		return self.ic_pb(vop3_base('v_sad_hi_u8', None, None, gfx10_src_2, gfx10_src_6, gfx10_src_6, MODIFIERS))
	def v_sad_u16(self, gfx10_src_2:reg_block, gfx10_src_6:reg_block, gfx10_src_6:reg_block, MODIFIERS:str=''):
		return self.ic_pb(vop3_base('v_sad_u16', None, None, gfx10_src_2, gfx10_src_6, gfx10_src_6, MODIFIERS))
	def v_sad_u32(self, gfx10_src_2:reg_block, gfx10_src_6:reg_block, gfx10_src_6:reg_block, MODIFIERS:str=''):
		return self.ic_pb(vop3_base('v_sad_u32', None, None, gfx10_src_2, gfx10_src_6, gfx10_src_6, MODIFIERS))
	def v_sad_u8(self, gfx10_src_2:reg_block, gfx10_src_6:reg_block, gfx10_src_6:reg_block, MODIFIERS:str=''):
		return self.ic_pb(vop3_base('v_sad_u8', None, None, gfx10_src_2, gfx10_src_6, gfx10_src_6, MODIFIERS))
	def v_sat_pk_u8_i16_e64(self, gfx10_src_2:reg_block):
		return self.ic_pb(vop3_base('v_sat_pk_u8_i16_e64', None, None, gfx10_src_2, None, None))
	def v_sin_f16_e64(self, gfx10_src_2:reg_block, MODIFIERS:str=''):
		return self.ic_pb(vop3_base('v_sin_f16_e64', None, None, gfx10_src_2, None, None, MODIFIERS))
	def v_sin_f32_e64(self, gfx10_src_2:reg_block, MODIFIERS:str=''):
		return self.ic_pb(vop3_base('v_sin_f32_e64', None, None, gfx10_src_2, None, None, MODIFIERS))
	def v_sqrt_f16_e64(self, gfx10_src_2:reg_block, MODIFIERS:str=''):
		return self.ic_pb(vop3_base('v_sqrt_f16_e64', None, None, gfx10_src_2, None, None, MODIFIERS))
	def v_sqrt_f32_e64(self, gfx10_src_2:reg_block, MODIFIERS:str=''):
		return self.ic_pb(vop3_base('v_sqrt_f32_e64', None, None, gfx10_src_2, None, None, MODIFIERS))
	def v_sqrt_f64_e64(self, gfx10_src_3:reg_block, MODIFIERS:str=''):
		return self.ic_pb(vop3_base('v_sqrt_f64_e64', None, None, gfx10_src_3, None, None, MODIFIERS))
	def v_sub_co_ci_u32_e64(self, gfx10_sdst:reg_block, gfx10_src_2:reg_block, gfx10_src_6:reg_block, gfx10_ssrc_5:reg_block, MODIFIERS:str=''):
		return self.ic_pb(vop3_base('v_sub_co_ci_u32_e64', None, gfx10_sdst, gfx10_src_2, gfx10_src_6, gfx10_ssrc_5, MODIFIERS))
	def v_sub_co_u32(self, gfx10_sdst:reg_block, gfx10_src_2:reg_block, gfx10_src_6:reg_block, MODIFIERS:str=''):
		return self.ic_pb(vop3_base('v_sub_co_u32', None, gfx10_sdst, gfx10_src_2, gfx10_src_6, None, MODIFIERS))
	def v_sub_f16_e64(self, gfx10_src_2:reg_block, gfx10_src_6:reg_block, MODIFIERS:str=''):
		return self.ic_pb(vop3_base('v_sub_f16_e64', None, None, gfx10_src_2, gfx10_src_6, None, MODIFIERS))
	def v_sub_f32_e64(self, gfx10_src_2:reg_block, gfx10_src_6:reg_block, MODIFIERS:str=''):
		return self.ic_pb(vop3_base('v_sub_f32_e64', None, None, gfx10_src_2, gfx10_src_6, None, MODIFIERS))
	def v_sub_nc_i16(self, gfx10_src_4:reg_block, gfx10_src_8:reg_block, MODIFIERS:str=''):
		return self.ic_pb(vop3_base('v_sub_nc_i16', None, None, gfx10_src_4, gfx10_src_8, None, MODIFIERS))
	def v_sub_nc_i32(self, gfx10_src_2:reg_block, gfx10_src_6:reg_block, MODIFIERS:str=''):
		return self.ic_pb(vop3_base('v_sub_nc_i32', None, None, gfx10_src_2, gfx10_src_6, None, MODIFIERS))
	def v_sub_nc_u16(self, gfx10_src_4:reg_block, gfx10_src_8:reg_block, MODIFIERS:str=''):
		return self.ic_pb(vop3_base('v_sub_nc_u16', None, None, gfx10_src_4, gfx10_src_8, None, MODIFIERS))
	def v_sub_nc_u32_e64(self, gfx10_src_2:reg_block, gfx10_src_6:reg_block, MODIFIERS:str=''):
		return self.ic_pb(vop3_base('v_sub_nc_u32_e64', None, None, gfx10_src_2, gfx10_src_6, None, MODIFIERS))
	def v_subrev_co_ci_u32_e64(self, gfx10_sdst:reg_block, gfx10_src_6:reg_block, gfx10_src_6:reg_block, gfx10_ssrc_5:reg_block, MODIFIERS:str=''):
		return self.ic_pb(vop3_base('v_subrev_co_ci_u32_e64', None, gfx10_sdst, gfx10_src_6, gfx10_src_6, gfx10_ssrc_5, MODIFIERS))
	def v_subrev_co_u32(self, gfx10_sdst:reg_block, gfx10_src_6:reg_block, gfx10_src_6:reg_block, MODIFIERS:str=''):
		return self.ic_pb(vop3_base('v_subrev_co_u32', None, gfx10_sdst, gfx10_src_6, gfx10_src_6, None, MODIFIERS))
	def v_subrev_f16_e64(self, gfx10_src_6:reg_block, gfx10_src_6:reg_block, MODIFIERS:str=''):
		return self.ic_pb(vop3_base('v_subrev_f16_e64', None, None, gfx10_src_6, gfx10_src_6, None, MODIFIERS))
	def v_subrev_f32_e64(self, gfx10_src_6:reg_block, gfx10_src_6:reg_block, MODIFIERS:str=''):
		return self.ic_pb(vop3_base('v_subrev_f32_e64', None, None, gfx10_src_6, gfx10_src_6, None, MODIFIERS))
	def v_subrev_nc_u32_e64(self, gfx10_src_6:reg_block, gfx10_src_6:reg_block, MODIFIERS:str=''):
		return self.ic_pb(vop3_base('v_subrev_nc_u32_e64', None, None, gfx10_src_6, gfx10_src_6, None, MODIFIERS))
	def v_trig_preop_f64(self, gfx10_src_3:reg_block, gfx10_src_6:reg_block, MODIFIERS:str=''):
		return self.ic_pb(vop3_base('v_trig_preop_f64', None, None, gfx10_src_3, gfx10_src_6, None, MODIFIERS))
	def v_trunc_f16_e64(self, gfx10_src_2:reg_block, MODIFIERS:str=''):
		return self.ic_pb(vop3_base('v_trunc_f16_e64', None, None, gfx10_src_2, None, None, MODIFIERS))
	def v_trunc_f32_e64(self, gfx10_src_2:reg_block, MODIFIERS:str=''):
		return self.ic_pb(vop3_base('v_trunc_f32_e64', None, None, gfx10_src_2, None, None, MODIFIERS))
	def v_trunc_f64_e64(self, gfx10_src_3:reg_block, MODIFIERS:str=''):
		return self.ic_pb(vop3_base('v_trunc_f64_e64', None, None, gfx10_src_3, None, None, MODIFIERS))
	def v_writelane_b32(self, gfx10_ssrc_8:reg_block, gfx10_ssrc_7:reg_block):
		return self.ic_pb(vop3_base('v_writelane_b32', None, None, gfx10_ssrc_8, gfx10_ssrc_7, None))
	def v_xad_u32(self, gfx10_src_2:reg_block, gfx10_src_6:reg_block, gfx10_src_6:reg_block):
		return self.ic_pb(vop3_base('v_xad_u32', None, None, gfx10_src_2, gfx10_src_6, gfx10_src_6))
	def v_xnor_b32_e64(self, gfx10_src_2:reg_block, gfx10_src_6:reg_block):
		return self.ic_pb(vop3_base('v_xnor_b32_e64', None, None, gfx10_src_2, gfx10_src_6, None))
	def v_xor3_b32(self, gfx10_src_2:reg_block, gfx10_src_6:reg_block, gfx10_src_6:reg_block):
		return self.ic_pb(vop3_base('v_xor3_b32', None, None, gfx10_src_2, gfx10_src_6, gfx10_src_6))
	def v_xor_b32_e64(self, gfx10_src_2:reg_block, gfx10_src_6:reg_block):
		return self.ic_pb(vop3_base('v_xor_b32_e64', None, None, gfx10_src_2, gfx10_src_6, None))
class vop3p_base(inst_base): 
	def __init__(self, INSTRUCTION:str, DST:Union[reg_block,None], SRC0:Union[reg_block,None], SRC1:Union[reg_block,None], SRC2:Union[reg_block,None], MODIFIERS:str): 
		super().__init__(instruction_type.SMEM, INSTRUCTION)
		self.DST = DST 
		self.SRC0 = SRC0 
		self.SRC1 = SRC1 
		self.SRC2 = SRC2 
		self.MODIFIERS = MODIFIERS 
	def __str__(self): 
		args_l = filter(None.__ne__, [self.DST,self.SRC0,self.SRC1,self.SRC2]) 
		return f"{self.label} {','.join(map(str, args_l))} {self.MODIFIERS}" 
class vop3p_instr_caller(inst_caller_base): 
	def __init__(self, insturction_container) -> None:
     		super().__init__(insturction_container)
	def v_fma_mix_f32(self, gfx10_src_2:reg_block, gfx10_src_6:reg_block, gfx10_src_6:reg_block, MODIFIERS:str=''):
		return self.ic_pb(vop3p_base('v_fma_mix_f32', None, gfx10_src_2, gfx10_src_6, gfx10_src_6, MODIFIERS))
	def v_fma_mixhi_f16(self, gfx10_src_2:reg_block, gfx10_src_6:reg_block, gfx10_src_6:reg_block, MODIFIERS:str=''):
		return self.ic_pb(vop3p_base('v_fma_mixhi_f16', None, gfx10_src_2, gfx10_src_6, gfx10_src_6, MODIFIERS))
	def v_fma_mixlo_f16(self, gfx10_src_2:reg_block, gfx10_src_6:reg_block, gfx10_src_6:reg_block, MODIFIERS:str=''):
		return self.ic_pb(vop3p_base('v_fma_mixlo_f16', None, gfx10_src_2, gfx10_src_6, gfx10_src_6, MODIFIERS))
	def v_pk_add_f16(self, gfx10_src_2:reg_block, gfx10_src_6:reg_block, MODIFIERS:str=''):
		return self.ic_pb(vop3p_base('v_pk_add_f16', None, gfx10_src_2, gfx10_src_6, None, MODIFIERS))
	def v_pk_add_i16(self, gfx10_src_4:reg_block, gfx10_src_8:reg_block, MODIFIERS:str=''):
		return self.ic_pb(vop3p_base('v_pk_add_i16', None, gfx10_src_4, gfx10_src_8, None, MODIFIERS))
	def v_pk_add_u16(self, gfx10_src_4:reg_block, gfx10_src_8:reg_block, MODIFIERS:str=''):
		return self.ic_pb(vop3p_base('v_pk_add_u16', None, gfx10_src_4, gfx10_src_8, None, MODIFIERS))
	def v_pk_ashrrev_i16(self, gfx10_src_8:reg_block, gfx10_src_8:reg_block, MODIFIERS:str=''):
		return self.ic_pb(vop3p_base('v_pk_ashrrev_i16', None, gfx10_src_8, gfx10_src_8, None, MODIFIERS))
	def v_pk_fma_f16(self, gfx10_src_2:reg_block, gfx10_src_6:reg_block, gfx10_src_6:reg_block, MODIFIERS:str=''):
		return self.ic_pb(vop3p_base('v_pk_fma_f16', None, gfx10_src_2, gfx10_src_6, gfx10_src_6, MODIFIERS))
	def v_pk_lshlrev_b16(self, gfx10_src_8:reg_block, gfx10_src_8:reg_block, MODIFIERS:str=''):
		return self.ic_pb(vop3p_base('v_pk_lshlrev_b16', None, gfx10_src_8, gfx10_src_8, None, MODIFIERS))
	def v_pk_lshrrev_b16(self, gfx10_src_8:reg_block, gfx10_src_8:reg_block, MODIFIERS:str=''):
		return self.ic_pb(vop3p_base('v_pk_lshrrev_b16', None, gfx10_src_8, gfx10_src_8, None, MODIFIERS))
	def v_pk_mad_i16(self, gfx10_src_4:reg_block, gfx10_src_8:reg_block, gfx10_src_8:reg_block, MODIFIERS:str=''):
		return self.ic_pb(vop3p_base('v_pk_mad_i16', None, gfx10_src_4, gfx10_src_8, gfx10_src_8, MODIFIERS))
	def v_pk_mad_u16(self, gfx10_src_4:reg_block, gfx10_src_8:reg_block, gfx10_src_8:reg_block, MODIFIERS:str=''):
		return self.ic_pb(vop3p_base('v_pk_mad_u16', None, gfx10_src_4, gfx10_src_8, gfx10_src_8, MODIFIERS))
	def v_pk_max_f16(self, gfx10_src_2:reg_block, gfx10_src_6:reg_block, MODIFIERS:str=''):
		return self.ic_pb(vop3p_base('v_pk_max_f16', None, gfx10_src_2, gfx10_src_6, None, MODIFIERS))
	def v_pk_max_i16(self, gfx10_src_4:reg_block, gfx10_src_8:reg_block, MODIFIERS:str=''):
		return self.ic_pb(vop3p_base('v_pk_max_i16', None, gfx10_src_4, gfx10_src_8, None, MODIFIERS))
	def v_pk_max_u16(self, gfx10_src_4:reg_block, gfx10_src_8:reg_block, MODIFIERS:str=''):
		return self.ic_pb(vop3p_base('v_pk_max_u16', None, gfx10_src_4, gfx10_src_8, None, MODIFIERS))
	def v_pk_min_f16(self, gfx10_src_2:reg_block, gfx10_src_6:reg_block, MODIFIERS:str=''):
		return self.ic_pb(vop3p_base('v_pk_min_f16', None, gfx10_src_2, gfx10_src_6, None, MODIFIERS))
	def v_pk_min_i16(self, gfx10_src_4:reg_block, gfx10_src_8:reg_block, MODIFIERS:str=''):
		return self.ic_pb(vop3p_base('v_pk_min_i16', None, gfx10_src_4, gfx10_src_8, None, MODIFIERS))
	def v_pk_min_u16(self, gfx10_src_4:reg_block, gfx10_src_8:reg_block, MODIFIERS:str=''):
		return self.ic_pb(vop3p_base('v_pk_min_u16', None, gfx10_src_4, gfx10_src_8, None, MODIFIERS))
	def v_pk_mul_f16(self, gfx10_src_2:reg_block, gfx10_src_6:reg_block, MODIFIERS:str=''):
		return self.ic_pb(vop3p_base('v_pk_mul_f16', None, gfx10_src_2, gfx10_src_6, None, MODIFIERS))
	def v_pk_mul_lo_u16(self, gfx10_src_4:reg_block, gfx10_src_8:reg_block, MODIFIERS:str=''):
		return self.ic_pb(vop3p_base('v_pk_mul_lo_u16', None, gfx10_src_4, gfx10_src_8, None, MODIFIERS))
	def v_pk_sub_i16(self, gfx10_src_4:reg_block, gfx10_src_8:reg_block, MODIFIERS:str=''):
		return self.ic_pb(vop3p_base('v_pk_sub_i16', None, gfx10_src_4, gfx10_src_8, None, MODIFIERS))
	def v_pk_sub_u16(self, gfx10_src_4:reg_block, gfx10_src_8:reg_block, MODIFIERS:str=''):
		return self.ic_pb(vop3p_base('v_pk_sub_u16', None, gfx10_src_4, gfx10_src_8, None, MODIFIERS))
class vopc_base(inst_base): 
	def __init__(self, INSTRUCTION:str, DST:Union[reg_block,None], SRC0:Union[reg_block,None], SRC1:Union[reg_block,None]): 
		super().__init__(instruction_type.SMEM, INSTRUCTION)
		self.DST = DST 
		self.SRC0 = SRC0 
		self.SRC1 = SRC1 
	def __str__(self): 
		args_l = filter(None.__ne__, [self.DST,self.SRC0,self.SRC1]) 
		return f"{self.label} {','.join(map(str, args_l))} " 
class vopc_instr_caller(inst_caller_base): 
	def __init__(self, insturction_container) -> None:
     		super().__init__(insturction_container)
	def v_cmp_class_f16(self, gfx10_src_2:reg_block, gfx10_vsrc:reg_block):
		return self.ic_pb(vopc_base('v_cmp_class_f16', None, gfx10_src_2, gfx10_vsrc))
	def v_cmp_class_f32(self, gfx10_src_2:reg_block, gfx10_vsrc:reg_block):
		return self.ic_pb(vopc_base('v_cmp_class_f32', None, gfx10_src_2, gfx10_vsrc))
	def v_cmp_class_f64(self, gfx10_src_3:reg_block, gfx10_vsrc:reg_block):
		return self.ic_pb(vopc_base('v_cmp_class_f64', None, gfx10_src_3, gfx10_vsrc))
	def v_cmp_eq_f16(self, gfx10_src_2:reg_block, gfx10_vsrc:reg_block):
		return self.ic_pb(vopc_base('v_cmp_eq_f16', None, gfx10_src_2, gfx10_vsrc))
	def v_cmp_eq_f32(self, gfx10_src_2:reg_block, gfx10_vsrc:reg_block):
		return self.ic_pb(vopc_base('v_cmp_eq_f32', None, gfx10_src_2, gfx10_vsrc))
	def v_cmp_eq_f64(self, gfx10_src_3:reg_block, gfx10_vsrc_3:reg_block):
		return self.ic_pb(vopc_base('v_cmp_eq_f64', None, gfx10_src_3, gfx10_vsrc_3))
	def v_cmp_eq_i16(self, gfx10_src_4:reg_block, gfx10_vsrc:reg_block):
		return self.ic_pb(vopc_base('v_cmp_eq_i16', None, gfx10_src_4, gfx10_vsrc))
	def v_cmp_eq_i32(self, gfx10_src_2:reg_block, gfx10_vsrc:reg_block):
		return self.ic_pb(vopc_base('v_cmp_eq_i32', None, gfx10_src_2, gfx10_vsrc))
	def v_cmp_eq_i64(self, gfx10_src_3:reg_block, gfx10_vsrc_3:reg_block):
		return self.ic_pb(vopc_base('v_cmp_eq_i64', None, gfx10_src_3, gfx10_vsrc_3))
	def v_cmp_eq_u16(self, gfx10_src_4:reg_block, gfx10_vsrc:reg_block):
		return self.ic_pb(vopc_base('v_cmp_eq_u16', None, gfx10_src_4, gfx10_vsrc))
	def v_cmp_eq_u32(self, gfx10_src_2:reg_block, gfx10_vsrc:reg_block):
		return self.ic_pb(vopc_base('v_cmp_eq_u32', None, gfx10_src_2, gfx10_vsrc))
	def v_cmp_eq_u64(self, gfx10_src_3:reg_block, gfx10_vsrc_3:reg_block):
		return self.ic_pb(vopc_base('v_cmp_eq_u64', None, gfx10_src_3, gfx10_vsrc_3))
	def v_cmp_f_f16(self, gfx10_src_2:reg_block, gfx10_vsrc:reg_block):
		return self.ic_pb(vopc_base('v_cmp_f_f16', None, gfx10_src_2, gfx10_vsrc))
	def v_cmp_f_f32(self, gfx10_src_2:reg_block, gfx10_vsrc:reg_block):
		return self.ic_pb(vopc_base('v_cmp_f_f32', None, gfx10_src_2, gfx10_vsrc))
	def v_cmp_f_f64(self, gfx10_src_3:reg_block, gfx10_vsrc_3:reg_block):
		return self.ic_pb(vopc_base('v_cmp_f_f64', None, gfx10_src_3, gfx10_vsrc_3))
	def v_cmp_f_i32(self, gfx10_src_2:reg_block, gfx10_vsrc:reg_block):
		return self.ic_pb(vopc_base('v_cmp_f_i32', None, gfx10_src_2, gfx10_vsrc))
	def v_cmp_f_i64(self, gfx10_src_3:reg_block, gfx10_vsrc_3:reg_block):
		return self.ic_pb(vopc_base('v_cmp_f_i64', None, gfx10_src_3, gfx10_vsrc_3))
	def v_cmp_f_u32(self, gfx10_src_2:reg_block, gfx10_vsrc:reg_block):
		return self.ic_pb(vopc_base('v_cmp_f_u32', None, gfx10_src_2, gfx10_vsrc))
	def v_cmp_f_u64(self, gfx10_src_3:reg_block, gfx10_vsrc_3:reg_block):
		return self.ic_pb(vopc_base('v_cmp_f_u64', None, gfx10_src_3, gfx10_vsrc_3))
	def v_cmp_ge_f16(self, gfx10_src_2:reg_block, gfx10_vsrc:reg_block):
		return self.ic_pb(vopc_base('v_cmp_ge_f16', None, gfx10_src_2, gfx10_vsrc))
	def v_cmp_ge_f32(self, gfx10_src_2:reg_block, gfx10_vsrc:reg_block):
		return self.ic_pb(vopc_base('v_cmp_ge_f32', None, gfx10_src_2, gfx10_vsrc))
	def v_cmp_ge_f64(self, gfx10_src_3:reg_block, gfx10_vsrc_3:reg_block):
		return self.ic_pb(vopc_base('v_cmp_ge_f64', None, gfx10_src_3, gfx10_vsrc_3))
	def v_cmp_ge_i16(self, gfx10_src_4:reg_block, gfx10_vsrc:reg_block):
		return self.ic_pb(vopc_base('v_cmp_ge_i16', None, gfx10_src_4, gfx10_vsrc))
	def v_cmp_ge_i32(self, gfx10_src_2:reg_block, gfx10_vsrc:reg_block):
		return self.ic_pb(vopc_base('v_cmp_ge_i32', None, gfx10_src_2, gfx10_vsrc))
	def v_cmp_ge_i64(self, gfx10_src_3:reg_block, gfx10_vsrc_3:reg_block):
		return self.ic_pb(vopc_base('v_cmp_ge_i64', None, gfx10_src_3, gfx10_vsrc_3))
	def v_cmp_ge_u16(self, gfx10_src_4:reg_block, gfx10_vsrc:reg_block):
		return self.ic_pb(vopc_base('v_cmp_ge_u16', None, gfx10_src_4, gfx10_vsrc))
	def v_cmp_ge_u32(self, gfx10_src_2:reg_block, gfx10_vsrc:reg_block):
		return self.ic_pb(vopc_base('v_cmp_ge_u32', None, gfx10_src_2, gfx10_vsrc))
	def v_cmp_ge_u64(self, gfx10_src_3:reg_block, gfx10_vsrc_3:reg_block):
		return self.ic_pb(vopc_base('v_cmp_ge_u64', None, gfx10_src_3, gfx10_vsrc_3))
	def v_cmp_gt_f16(self, gfx10_src_2:reg_block, gfx10_vsrc:reg_block):
		return self.ic_pb(vopc_base('v_cmp_gt_f16', None, gfx10_src_2, gfx10_vsrc))
	def v_cmp_gt_f32(self, gfx10_src_2:reg_block, gfx10_vsrc:reg_block):
		return self.ic_pb(vopc_base('v_cmp_gt_f32', None, gfx10_src_2, gfx10_vsrc))
	def v_cmp_gt_f64(self, gfx10_src_3:reg_block, gfx10_vsrc_3:reg_block):
		return self.ic_pb(vopc_base('v_cmp_gt_f64', None, gfx10_src_3, gfx10_vsrc_3))
	def v_cmp_gt_i16(self, gfx10_src_4:reg_block, gfx10_vsrc:reg_block):
		return self.ic_pb(vopc_base('v_cmp_gt_i16', None, gfx10_src_4, gfx10_vsrc))
	def v_cmp_gt_i32(self, gfx10_src_2:reg_block, gfx10_vsrc:reg_block):
		return self.ic_pb(vopc_base('v_cmp_gt_i32', None, gfx10_src_2, gfx10_vsrc))
	def v_cmp_gt_i64(self, gfx10_src_3:reg_block, gfx10_vsrc_3:reg_block):
		return self.ic_pb(vopc_base('v_cmp_gt_i64', None, gfx10_src_3, gfx10_vsrc_3))
	def v_cmp_gt_u16(self, gfx10_src_4:reg_block, gfx10_vsrc:reg_block):
		return self.ic_pb(vopc_base('v_cmp_gt_u16', None, gfx10_src_4, gfx10_vsrc))
	def v_cmp_gt_u32(self, gfx10_src_2:reg_block, gfx10_vsrc:reg_block):
		return self.ic_pb(vopc_base('v_cmp_gt_u32', None, gfx10_src_2, gfx10_vsrc))
	def v_cmp_gt_u64(self, gfx10_src_3:reg_block, gfx10_vsrc_3:reg_block):
		return self.ic_pb(vopc_base('v_cmp_gt_u64', None, gfx10_src_3, gfx10_vsrc_3))
	def v_cmp_le_f16(self, gfx10_src_2:reg_block, gfx10_vsrc:reg_block):
		return self.ic_pb(vopc_base('v_cmp_le_f16', None, gfx10_src_2, gfx10_vsrc))
	def v_cmp_le_f32(self, gfx10_src_2:reg_block, gfx10_vsrc:reg_block):
		return self.ic_pb(vopc_base('v_cmp_le_f32', None, gfx10_src_2, gfx10_vsrc))
	def v_cmp_le_f64(self, gfx10_src_3:reg_block, gfx10_vsrc_3:reg_block):
		return self.ic_pb(vopc_base('v_cmp_le_f64', None, gfx10_src_3, gfx10_vsrc_3))
	def v_cmp_le_i16(self, gfx10_src_4:reg_block, gfx10_vsrc:reg_block):
		return self.ic_pb(vopc_base('v_cmp_le_i16', None, gfx10_src_4, gfx10_vsrc))
	def v_cmp_le_i32(self, gfx10_src_2:reg_block, gfx10_vsrc:reg_block):
		return self.ic_pb(vopc_base('v_cmp_le_i32', None, gfx10_src_2, gfx10_vsrc))
	def v_cmp_le_i64(self, gfx10_src_3:reg_block, gfx10_vsrc_3:reg_block):
		return self.ic_pb(vopc_base('v_cmp_le_i64', None, gfx10_src_3, gfx10_vsrc_3))
	def v_cmp_le_u16(self, gfx10_src_4:reg_block, gfx10_vsrc:reg_block):
		return self.ic_pb(vopc_base('v_cmp_le_u16', None, gfx10_src_4, gfx10_vsrc))
	def v_cmp_le_u32(self, gfx10_src_2:reg_block, gfx10_vsrc:reg_block):
		return self.ic_pb(vopc_base('v_cmp_le_u32', None, gfx10_src_2, gfx10_vsrc))
	def v_cmp_le_u64(self, gfx10_src_3:reg_block, gfx10_vsrc_3:reg_block):
		return self.ic_pb(vopc_base('v_cmp_le_u64', None, gfx10_src_3, gfx10_vsrc_3))
	def v_cmp_lg_f16(self, gfx10_src_2:reg_block, gfx10_vsrc:reg_block):
		return self.ic_pb(vopc_base('v_cmp_lg_f16', None, gfx10_src_2, gfx10_vsrc))
	def v_cmp_lg_f32(self, gfx10_src_2:reg_block, gfx10_vsrc:reg_block):
		return self.ic_pb(vopc_base('v_cmp_lg_f32', None, gfx10_src_2, gfx10_vsrc))
	def v_cmp_lg_f64(self, gfx10_src_3:reg_block, gfx10_vsrc_3:reg_block):
		return self.ic_pb(vopc_base('v_cmp_lg_f64', None, gfx10_src_3, gfx10_vsrc_3))
	def v_cmp_lt_f16(self, gfx10_src_2:reg_block, gfx10_vsrc:reg_block):
		return self.ic_pb(vopc_base('v_cmp_lt_f16', None, gfx10_src_2, gfx10_vsrc))
	def v_cmp_lt_f32(self, gfx10_src_2:reg_block, gfx10_vsrc:reg_block):
		return self.ic_pb(vopc_base('v_cmp_lt_f32', None, gfx10_src_2, gfx10_vsrc))
	def v_cmp_lt_f64(self, gfx10_src_3:reg_block, gfx10_vsrc_3:reg_block):
		return self.ic_pb(vopc_base('v_cmp_lt_f64', None, gfx10_src_3, gfx10_vsrc_3))
	def v_cmp_lt_i16(self, gfx10_src_4:reg_block, gfx10_vsrc:reg_block):
		return self.ic_pb(vopc_base('v_cmp_lt_i16', None, gfx10_src_4, gfx10_vsrc))
	def v_cmp_lt_i32(self, gfx10_src_2:reg_block, gfx10_vsrc:reg_block):
		return self.ic_pb(vopc_base('v_cmp_lt_i32', None, gfx10_src_2, gfx10_vsrc))
	def v_cmp_lt_i64(self, gfx10_src_3:reg_block, gfx10_vsrc_3:reg_block):
		return self.ic_pb(vopc_base('v_cmp_lt_i64', None, gfx10_src_3, gfx10_vsrc_3))
	def v_cmp_lt_u16(self, gfx10_src_4:reg_block, gfx10_vsrc:reg_block):
		return self.ic_pb(vopc_base('v_cmp_lt_u16', None, gfx10_src_4, gfx10_vsrc))
	def v_cmp_lt_u32(self, gfx10_src_2:reg_block, gfx10_vsrc:reg_block):
		return self.ic_pb(vopc_base('v_cmp_lt_u32', None, gfx10_src_2, gfx10_vsrc))
	def v_cmp_lt_u64(self, gfx10_src_3:reg_block, gfx10_vsrc_3:reg_block):
		return self.ic_pb(vopc_base('v_cmp_lt_u64', None, gfx10_src_3, gfx10_vsrc_3))
	def v_cmp_ne_i16(self, gfx10_src_4:reg_block, gfx10_vsrc:reg_block):
		return self.ic_pb(vopc_base('v_cmp_ne_i16', None, gfx10_src_4, gfx10_vsrc))
	def v_cmp_ne_i32(self, gfx10_src_2:reg_block, gfx10_vsrc:reg_block):
		return self.ic_pb(vopc_base('v_cmp_ne_i32', None, gfx10_src_2, gfx10_vsrc))
	def v_cmp_ne_i64(self, gfx10_src_3:reg_block, gfx10_vsrc_3:reg_block):
		return self.ic_pb(vopc_base('v_cmp_ne_i64', None, gfx10_src_3, gfx10_vsrc_3))
	def v_cmp_ne_u16(self, gfx10_src_4:reg_block, gfx10_vsrc:reg_block):
		return self.ic_pb(vopc_base('v_cmp_ne_u16', None, gfx10_src_4, gfx10_vsrc))
	def v_cmp_ne_u32(self, gfx10_src_2:reg_block, gfx10_vsrc:reg_block):
		return self.ic_pb(vopc_base('v_cmp_ne_u32', None, gfx10_src_2, gfx10_vsrc))
	def v_cmp_ne_u64(self, gfx10_src_3:reg_block, gfx10_vsrc_3:reg_block):
		return self.ic_pb(vopc_base('v_cmp_ne_u64', None, gfx10_src_3, gfx10_vsrc_3))
	def v_cmp_neq_f16(self, gfx10_src_2:reg_block, gfx10_vsrc:reg_block):
		return self.ic_pb(vopc_base('v_cmp_neq_f16', None, gfx10_src_2, gfx10_vsrc))
	def v_cmp_neq_f32(self, gfx10_src_2:reg_block, gfx10_vsrc:reg_block):
		return self.ic_pb(vopc_base('v_cmp_neq_f32', None, gfx10_src_2, gfx10_vsrc))
	def v_cmp_neq_f64(self, gfx10_src_3:reg_block, gfx10_vsrc_3:reg_block):
		return self.ic_pb(vopc_base('v_cmp_neq_f64', None, gfx10_src_3, gfx10_vsrc_3))
	def v_cmp_nge_f16(self, gfx10_src_2:reg_block, gfx10_vsrc:reg_block):
		return self.ic_pb(vopc_base('v_cmp_nge_f16', None, gfx10_src_2, gfx10_vsrc))
	def v_cmp_nge_f32(self, gfx10_src_2:reg_block, gfx10_vsrc:reg_block):
		return self.ic_pb(vopc_base('v_cmp_nge_f32', None, gfx10_src_2, gfx10_vsrc))
	def v_cmp_nge_f64(self, gfx10_src_3:reg_block, gfx10_vsrc_3:reg_block):
		return self.ic_pb(vopc_base('v_cmp_nge_f64', None, gfx10_src_3, gfx10_vsrc_3))
	def v_cmp_ngt_f16(self, gfx10_src_2:reg_block, gfx10_vsrc:reg_block):
		return self.ic_pb(vopc_base('v_cmp_ngt_f16', None, gfx10_src_2, gfx10_vsrc))
	def v_cmp_ngt_f32(self, gfx10_src_2:reg_block, gfx10_vsrc:reg_block):
		return self.ic_pb(vopc_base('v_cmp_ngt_f32', None, gfx10_src_2, gfx10_vsrc))
	def v_cmp_ngt_f64(self, gfx10_src_3:reg_block, gfx10_vsrc_3:reg_block):
		return self.ic_pb(vopc_base('v_cmp_ngt_f64', None, gfx10_src_3, gfx10_vsrc_3))
	def v_cmp_nle_f16(self, gfx10_src_2:reg_block, gfx10_vsrc:reg_block):
		return self.ic_pb(vopc_base('v_cmp_nle_f16', None, gfx10_src_2, gfx10_vsrc))
	def v_cmp_nle_f32(self, gfx10_src_2:reg_block, gfx10_vsrc:reg_block):
		return self.ic_pb(vopc_base('v_cmp_nle_f32', None, gfx10_src_2, gfx10_vsrc))
	def v_cmp_nle_f64(self, gfx10_src_3:reg_block, gfx10_vsrc_3:reg_block):
		return self.ic_pb(vopc_base('v_cmp_nle_f64', None, gfx10_src_3, gfx10_vsrc_3))
	def v_cmp_nlg_f16(self, gfx10_src_2:reg_block, gfx10_vsrc:reg_block):
		return self.ic_pb(vopc_base('v_cmp_nlg_f16', None, gfx10_src_2, gfx10_vsrc))
	def v_cmp_nlg_f32(self, gfx10_src_2:reg_block, gfx10_vsrc:reg_block):
		return self.ic_pb(vopc_base('v_cmp_nlg_f32', None, gfx10_src_2, gfx10_vsrc))
	def v_cmp_nlg_f64(self, gfx10_src_3:reg_block, gfx10_vsrc_3:reg_block):
		return self.ic_pb(vopc_base('v_cmp_nlg_f64', None, gfx10_src_3, gfx10_vsrc_3))
	def v_cmp_nlt_f16(self, gfx10_src_2:reg_block, gfx10_vsrc:reg_block):
		return self.ic_pb(vopc_base('v_cmp_nlt_f16', None, gfx10_src_2, gfx10_vsrc))
	def v_cmp_nlt_f32(self, gfx10_src_2:reg_block, gfx10_vsrc:reg_block):
		return self.ic_pb(vopc_base('v_cmp_nlt_f32', None, gfx10_src_2, gfx10_vsrc))
	def v_cmp_nlt_f64(self, gfx10_src_3:reg_block, gfx10_vsrc_3:reg_block):
		return self.ic_pb(vopc_base('v_cmp_nlt_f64', None, gfx10_src_3, gfx10_vsrc_3))
	def v_cmp_o_f16(self, gfx10_src_2:reg_block, gfx10_vsrc:reg_block):
		return self.ic_pb(vopc_base('v_cmp_o_f16', None, gfx10_src_2, gfx10_vsrc))
	def v_cmp_o_f32(self, gfx10_src_2:reg_block, gfx10_vsrc:reg_block):
		return self.ic_pb(vopc_base('v_cmp_o_f32', None, gfx10_src_2, gfx10_vsrc))
	def v_cmp_o_f64(self, gfx10_src_3:reg_block, gfx10_vsrc_3:reg_block):
		return self.ic_pb(vopc_base('v_cmp_o_f64', None, gfx10_src_3, gfx10_vsrc_3))
	def v_cmp_t_i32(self, gfx10_src_2:reg_block, gfx10_vsrc:reg_block):
		return self.ic_pb(vopc_base('v_cmp_t_i32', None, gfx10_src_2, gfx10_vsrc))
	def v_cmp_t_i64(self, gfx10_src_3:reg_block, gfx10_vsrc_3:reg_block):
		return self.ic_pb(vopc_base('v_cmp_t_i64', None, gfx10_src_3, gfx10_vsrc_3))
	def v_cmp_t_u32(self, gfx10_src_2:reg_block, gfx10_vsrc:reg_block):
		return self.ic_pb(vopc_base('v_cmp_t_u32', None, gfx10_src_2, gfx10_vsrc))
	def v_cmp_t_u64(self, gfx10_src_3:reg_block, gfx10_vsrc_3:reg_block):
		return self.ic_pb(vopc_base('v_cmp_t_u64', None, gfx10_src_3, gfx10_vsrc_3))
	def v_cmp_tru_f16(self, gfx10_src_2:reg_block, gfx10_vsrc:reg_block):
		return self.ic_pb(vopc_base('v_cmp_tru_f16', None, gfx10_src_2, gfx10_vsrc))
	def v_cmp_tru_f32(self, gfx10_src_2:reg_block, gfx10_vsrc:reg_block):
		return self.ic_pb(vopc_base('v_cmp_tru_f32', None, gfx10_src_2, gfx10_vsrc))
	def v_cmp_tru_f64(self, gfx10_src_3:reg_block, gfx10_vsrc_3:reg_block):
		return self.ic_pb(vopc_base('v_cmp_tru_f64', None, gfx10_src_3, gfx10_vsrc_3))
	def v_cmp_u_f16(self, gfx10_src_2:reg_block, gfx10_vsrc:reg_block):
		return self.ic_pb(vopc_base('v_cmp_u_f16', None, gfx10_src_2, gfx10_vsrc))
	def v_cmp_u_f32(self, gfx10_src_2:reg_block, gfx10_vsrc:reg_block):
		return self.ic_pb(vopc_base('v_cmp_u_f32', None, gfx10_src_2, gfx10_vsrc))
	def v_cmp_u_f64(self, gfx10_src_3:reg_block, gfx10_vsrc_3:reg_block):
		return self.ic_pb(vopc_base('v_cmp_u_f64', None, gfx10_src_3, gfx10_vsrc_3))
	def v_cmpx_class_f16(self, gfx10_vsrc:reg_block):
		return self.ic_pb(vopc_base('v_cmpx_class_f16', None, None, gfx10_vsrc))
	def v_cmpx_class_f32(self, gfx10_vsrc:reg_block):
		return self.ic_pb(vopc_base('v_cmpx_class_f32', None, None, gfx10_vsrc))
	def v_cmpx_class_f64(self, gfx10_vsrc:reg_block):
		return self.ic_pb(vopc_base('v_cmpx_class_f64', None, None, gfx10_vsrc))
	def v_cmpx_eq_f16(self, gfx10_vsrc:reg_block):
		return self.ic_pb(vopc_base('v_cmpx_eq_f16', None, None, gfx10_vsrc))
	def v_cmpx_eq_f32(self, gfx10_vsrc:reg_block):
		return self.ic_pb(vopc_base('v_cmpx_eq_f32', None, None, gfx10_vsrc))
	def v_cmpx_eq_f64(self, gfx10_vsrc_3:reg_block):
		return self.ic_pb(vopc_base('v_cmpx_eq_f64', None, None, gfx10_vsrc_3))
	def v_cmpx_eq_i16(self, gfx10_vsrc:reg_block):
		return self.ic_pb(vopc_base('v_cmpx_eq_i16', None, None, gfx10_vsrc))
	def v_cmpx_eq_i32(self, gfx10_vsrc:reg_block):
		return self.ic_pb(vopc_base('v_cmpx_eq_i32', None, None, gfx10_vsrc))
	def v_cmpx_eq_i64(self, gfx10_vsrc_3:reg_block):
		return self.ic_pb(vopc_base('v_cmpx_eq_i64', None, None, gfx10_vsrc_3))
	def v_cmpx_eq_u16(self, gfx10_vsrc:reg_block):
		return self.ic_pb(vopc_base('v_cmpx_eq_u16', None, None, gfx10_vsrc))
	def v_cmpx_eq_u32(self, gfx10_vsrc:reg_block):
		return self.ic_pb(vopc_base('v_cmpx_eq_u32', None, None, gfx10_vsrc))
	def v_cmpx_eq_u64(self, gfx10_vsrc_3:reg_block):
		return self.ic_pb(vopc_base('v_cmpx_eq_u64', None, None, gfx10_vsrc_3))
	def v_cmpx_f_f16(self, gfx10_vsrc:reg_block):
		return self.ic_pb(vopc_base('v_cmpx_f_f16', None, None, gfx10_vsrc))
	def v_cmpx_f_f32(self, gfx10_vsrc:reg_block):
		return self.ic_pb(vopc_base('v_cmpx_f_f32', None, None, gfx10_vsrc))
	def v_cmpx_f_f64(self, gfx10_vsrc_3:reg_block):
		return self.ic_pb(vopc_base('v_cmpx_f_f64', None, None, gfx10_vsrc_3))
	def v_cmpx_f_i32(self, gfx10_vsrc:reg_block):
		return self.ic_pb(vopc_base('v_cmpx_f_i32', None, None, gfx10_vsrc))
	def v_cmpx_f_i64(self, gfx10_vsrc_3:reg_block):
		return self.ic_pb(vopc_base('v_cmpx_f_i64', None, None, gfx10_vsrc_3))
	def v_cmpx_f_u32(self, gfx10_vsrc:reg_block):
		return self.ic_pb(vopc_base('v_cmpx_f_u32', None, None, gfx10_vsrc))
	def v_cmpx_f_u64(self, gfx10_vsrc_3:reg_block):
		return self.ic_pb(vopc_base('v_cmpx_f_u64', None, None, gfx10_vsrc_3))
	def v_cmpx_ge_f16(self, gfx10_vsrc:reg_block):
		return self.ic_pb(vopc_base('v_cmpx_ge_f16', None, None, gfx10_vsrc))
	def v_cmpx_ge_f32(self, gfx10_vsrc:reg_block):
		return self.ic_pb(vopc_base('v_cmpx_ge_f32', None, None, gfx10_vsrc))
	def v_cmpx_ge_f64(self, gfx10_vsrc_3:reg_block):
		return self.ic_pb(vopc_base('v_cmpx_ge_f64', None, None, gfx10_vsrc_3))
	def v_cmpx_ge_i16(self, gfx10_vsrc:reg_block):
		return self.ic_pb(vopc_base('v_cmpx_ge_i16', None, None, gfx10_vsrc))
	def v_cmpx_ge_i32(self, gfx10_vsrc:reg_block):
		return self.ic_pb(vopc_base('v_cmpx_ge_i32', None, None, gfx10_vsrc))
	def v_cmpx_ge_i64(self, gfx10_vsrc_3:reg_block):
		return self.ic_pb(vopc_base('v_cmpx_ge_i64', None, None, gfx10_vsrc_3))
	def v_cmpx_ge_u16(self, gfx10_vsrc:reg_block):
		return self.ic_pb(vopc_base('v_cmpx_ge_u16', None, None, gfx10_vsrc))
	def v_cmpx_ge_u32(self, gfx10_vsrc:reg_block):
		return self.ic_pb(vopc_base('v_cmpx_ge_u32', None, None, gfx10_vsrc))
	def v_cmpx_ge_u64(self, gfx10_vsrc_3:reg_block):
		return self.ic_pb(vopc_base('v_cmpx_ge_u64', None, None, gfx10_vsrc_3))
	def v_cmpx_gt_f16(self, gfx10_vsrc:reg_block):
		return self.ic_pb(vopc_base('v_cmpx_gt_f16', None, None, gfx10_vsrc))
	def v_cmpx_gt_f32(self, gfx10_vsrc:reg_block):
		return self.ic_pb(vopc_base('v_cmpx_gt_f32', None, None, gfx10_vsrc))
	def v_cmpx_gt_f64(self, gfx10_vsrc_3:reg_block):
		return self.ic_pb(vopc_base('v_cmpx_gt_f64', None, None, gfx10_vsrc_3))
	def v_cmpx_gt_i16(self, gfx10_vsrc:reg_block):
		return self.ic_pb(vopc_base('v_cmpx_gt_i16', None, None, gfx10_vsrc))
	def v_cmpx_gt_i32(self, gfx10_vsrc:reg_block):
		return self.ic_pb(vopc_base('v_cmpx_gt_i32', None, None, gfx10_vsrc))
	def v_cmpx_gt_i64(self, gfx10_vsrc_3:reg_block):
		return self.ic_pb(vopc_base('v_cmpx_gt_i64', None, None, gfx10_vsrc_3))
	def v_cmpx_gt_u16(self, gfx10_vsrc:reg_block):
		return self.ic_pb(vopc_base('v_cmpx_gt_u16', None, None, gfx10_vsrc))
	def v_cmpx_gt_u32(self, gfx10_vsrc:reg_block):
		return self.ic_pb(vopc_base('v_cmpx_gt_u32', None, None, gfx10_vsrc))
	def v_cmpx_gt_u64(self, gfx10_vsrc_3:reg_block):
		return self.ic_pb(vopc_base('v_cmpx_gt_u64', None, None, gfx10_vsrc_3))
	def v_cmpx_le_f16(self, gfx10_vsrc:reg_block):
		return self.ic_pb(vopc_base('v_cmpx_le_f16', None, None, gfx10_vsrc))
	def v_cmpx_le_f32(self, gfx10_vsrc:reg_block):
		return self.ic_pb(vopc_base('v_cmpx_le_f32', None, None, gfx10_vsrc))
	def v_cmpx_le_f64(self, gfx10_vsrc_3:reg_block):
		return self.ic_pb(vopc_base('v_cmpx_le_f64', None, None, gfx10_vsrc_3))
	def v_cmpx_le_i16(self, gfx10_vsrc:reg_block):
		return self.ic_pb(vopc_base('v_cmpx_le_i16', None, None, gfx10_vsrc))
	def v_cmpx_le_i32(self, gfx10_vsrc:reg_block):
		return self.ic_pb(vopc_base('v_cmpx_le_i32', None, None, gfx10_vsrc))
	def v_cmpx_le_i64(self, gfx10_vsrc_3:reg_block):
		return self.ic_pb(vopc_base('v_cmpx_le_i64', None, None, gfx10_vsrc_3))
	def v_cmpx_le_u16(self, gfx10_vsrc:reg_block):
		return self.ic_pb(vopc_base('v_cmpx_le_u16', None, None, gfx10_vsrc))
	def v_cmpx_le_u32(self, gfx10_vsrc:reg_block):
		return self.ic_pb(vopc_base('v_cmpx_le_u32', None, None, gfx10_vsrc))
	def v_cmpx_le_u64(self, gfx10_vsrc_3:reg_block):
		return self.ic_pb(vopc_base('v_cmpx_le_u64', None, None, gfx10_vsrc_3))
	def v_cmpx_lg_f16(self, gfx10_vsrc:reg_block):
		return self.ic_pb(vopc_base('v_cmpx_lg_f16', None, None, gfx10_vsrc))
	def v_cmpx_lg_f32(self, gfx10_vsrc:reg_block):
		return self.ic_pb(vopc_base('v_cmpx_lg_f32', None, None, gfx10_vsrc))
	def v_cmpx_lg_f64(self, gfx10_vsrc_3:reg_block):
		return self.ic_pb(vopc_base('v_cmpx_lg_f64', None, None, gfx10_vsrc_3))
	def v_cmpx_lt_f16(self, gfx10_vsrc:reg_block):
		return self.ic_pb(vopc_base('v_cmpx_lt_f16', None, None, gfx10_vsrc))
	def v_cmpx_lt_f32(self, gfx10_vsrc:reg_block):
		return self.ic_pb(vopc_base('v_cmpx_lt_f32', None, None, gfx10_vsrc))
	def v_cmpx_lt_f64(self, gfx10_vsrc_3:reg_block):
		return self.ic_pb(vopc_base('v_cmpx_lt_f64', None, None, gfx10_vsrc_3))
	def v_cmpx_lt_i16(self, gfx10_vsrc:reg_block):
		return self.ic_pb(vopc_base('v_cmpx_lt_i16', None, None, gfx10_vsrc))
	def v_cmpx_lt_i32(self, gfx10_vsrc:reg_block):
		return self.ic_pb(vopc_base('v_cmpx_lt_i32', None, None, gfx10_vsrc))
	def v_cmpx_lt_i64(self, gfx10_vsrc_3:reg_block):
		return self.ic_pb(vopc_base('v_cmpx_lt_i64', None, None, gfx10_vsrc_3))
	def v_cmpx_lt_u16(self, gfx10_vsrc:reg_block):
		return self.ic_pb(vopc_base('v_cmpx_lt_u16', None, None, gfx10_vsrc))
	def v_cmpx_lt_u32(self, gfx10_vsrc:reg_block):
		return self.ic_pb(vopc_base('v_cmpx_lt_u32', None, None, gfx10_vsrc))
	def v_cmpx_lt_u64(self, gfx10_vsrc_3:reg_block):
		return self.ic_pb(vopc_base('v_cmpx_lt_u64', None, None, gfx10_vsrc_3))
	def v_cmpx_ne_i16(self, gfx10_vsrc:reg_block):
		return self.ic_pb(vopc_base('v_cmpx_ne_i16', None, None, gfx10_vsrc))
	def v_cmpx_ne_i32(self, gfx10_vsrc:reg_block):
		return self.ic_pb(vopc_base('v_cmpx_ne_i32', None, None, gfx10_vsrc))
	def v_cmpx_ne_i64(self, gfx10_vsrc_3:reg_block):
		return self.ic_pb(vopc_base('v_cmpx_ne_i64', None, None, gfx10_vsrc_3))
	def v_cmpx_ne_u16(self, gfx10_vsrc:reg_block):
		return self.ic_pb(vopc_base('v_cmpx_ne_u16', None, None, gfx10_vsrc))
	def v_cmpx_ne_u32(self, gfx10_vsrc:reg_block):
		return self.ic_pb(vopc_base('v_cmpx_ne_u32', None, None, gfx10_vsrc))
	def v_cmpx_ne_u64(self, gfx10_vsrc_3:reg_block):
		return self.ic_pb(vopc_base('v_cmpx_ne_u64', None, None, gfx10_vsrc_3))
	def v_cmpx_neq_f16(self, gfx10_vsrc:reg_block):
		return self.ic_pb(vopc_base('v_cmpx_neq_f16', None, None, gfx10_vsrc))
	def v_cmpx_neq_f32(self, gfx10_vsrc:reg_block):
		return self.ic_pb(vopc_base('v_cmpx_neq_f32', None, None, gfx10_vsrc))
	def v_cmpx_neq_f64(self, gfx10_vsrc_3:reg_block):
		return self.ic_pb(vopc_base('v_cmpx_neq_f64', None, None, gfx10_vsrc_3))
	def v_cmpx_nge_f16(self, gfx10_vsrc:reg_block):
		return self.ic_pb(vopc_base('v_cmpx_nge_f16', None, None, gfx10_vsrc))
	def v_cmpx_nge_f32(self, gfx10_vsrc:reg_block):
		return self.ic_pb(vopc_base('v_cmpx_nge_f32', None, None, gfx10_vsrc))
	def v_cmpx_nge_f64(self, gfx10_vsrc_3:reg_block):
		return self.ic_pb(vopc_base('v_cmpx_nge_f64', None, None, gfx10_vsrc_3))
	def v_cmpx_ngt_f16(self, gfx10_vsrc:reg_block):
		return self.ic_pb(vopc_base('v_cmpx_ngt_f16', None, None, gfx10_vsrc))
	def v_cmpx_ngt_f32(self, gfx10_vsrc:reg_block):
		return self.ic_pb(vopc_base('v_cmpx_ngt_f32', None, None, gfx10_vsrc))
	def v_cmpx_ngt_f64(self, gfx10_vsrc_3:reg_block):
		return self.ic_pb(vopc_base('v_cmpx_ngt_f64', None, None, gfx10_vsrc_3))
	def v_cmpx_nle_f16(self, gfx10_vsrc:reg_block):
		return self.ic_pb(vopc_base('v_cmpx_nle_f16', None, None, gfx10_vsrc))
	def v_cmpx_nle_f32(self, gfx10_vsrc:reg_block):
		return self.ic_pb(vopc_base('v_cmpx_nle_f32', None, None, gfx10_vsrc))
	def v_cmpx_nle_f64(self, gfx10_vsrc_3:reg_block):
		return self.ic_pb(vopc_base('v_cmpx_nle_f64', None, None, gfx10_vsrc_3))
	def v_cmpx_nlg_f16(self, gfx10_vsrc:reg_block):
		return self.ic_pb(vopc_base('v_cmpx_nlg_f16', None, None, gfx10_vsrc))
	def v_cmpx_nlg_f32(self, gfx10_vsrc:reg_block):
		return self.ic_pb(vopc_base('v_cmpx_nlg_f32', None, None, gfx10_vsrc))
	def v_cmpx_nlg_f64(self, gfx10_vsrc_3:reg_block):
		return self.ic_pb(vopc_base('v_cmpx_nlg_f64', None, None, gfx10_vsrc_3))
	def v_cmpx_nlt_f16(self, gfx10_vsrc:reg_block):
		return self.ic_pb(vopc_base('v_cmpx_nlt_f16', None, None, gfx10_vsrc))
	def v_cmpx_nlt_f32(self, gfx10_vsrc:reg_block):
		return self.ic_pb(vopc_base('v_cmpx_nlt_f32', None, None, gfx10_vsrc))
	def v_cmpx_nlt_f64(self, gfx10_vsrc_3:reg_block):
		return self.ic_pb(vopc_base('v_cmpx_nlt_f64', None, None, gfx10_vsrc_3))
	def v_cmpx_o_f16(self, gfx10_vsrc:reg_block):
		return self.ic_pb(vopc_base('v_cmpx_o_f16', None, None, gfx10_vsrc))
	def v_cmpx_o_f32(self, gfx10_vsrc:reg_block):
		return self.ic_pb(vopc_base('v_cmpx_o_f32', None, None, gfx10_vsrc))
	def v_cmpx_o_f64(self, gfx10_vsrc_3:reg_block):
		return self.ic_pb(vopc_base('v_cmpx_o_f64', None, None, gfx10_vsrc_3))
	def v_cmpx_t_i32(self, gfx10_vsrc:reg_block):
		return self.ic_pb(vopc_base('v_cmpx_t_i32', None, None, gfx10_vsrc))
	def v_cmpx_t_i64(self, gfx10_vsrc_3:reg_block):
		return self.ic_pb(vopc_base('v_cmpx_t_i64', None, None, gfx10_vsrc_3))
	def v_cmpx_t_u32(self, gfx10_vsrc:reg_block):
		return self.ic_pb(vopc_base('v_cmpx_t_u32', None, None, gfx10_vsrc))
	def v_cmpx_t_u64(self, gfx10_vsrc_3:reg_block):
		return self.ic_pb(vopc_base('v_cmpx_t_u64', None, None, gfx10_vsrc_3))
	def v_cmpx_tru_f16(self, gfx10_vsrc:reg_block):
		return self.ic_pb(vopc_base('v_cmpx_tru_f16', None, None, gfx10_vsrc))
	def v_cmpx_tru_f32(self, gfx10_vsrc:reg_block):
		return self.ic_pb(vopc_base('v_cmpx_tru_f32', None, None, gfx10_vsrc))
	def v_cmpx_tru_f64(self, gfx10_vsrc_3:reg_block):
		return self.ic_pb(vopc_base('v_cmpx_tru_f64', None, None, gfx10_vsrc_3))
	def v_cmpx_u_f16(self, gfx10_vsrc:reg_block):
		return self.ic_pb(vopc_base('v_cmpx_u_f16', None, None, gfx10_vsrc))
	def v_cmpx_u_f32(self, gfx10_vsrc:reg_block):
		return self.ic_pb(vopc_base('v_cmpx_u_f32', None, None, gfx10_vsrc))
	def v_cmpx_u_f64(self, gfx10_vsrc_3:reg_block):
		return self.ic_pb(vopc_base('v_cmpx_u_f64', None, None, gfx10_vsrc_3))
