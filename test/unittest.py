from igemm import *

def get_default_mc():
    return mc_asm_printer_t(mc_emit_to_string_t(), amdgpu_arch_config_t(None))

def unittest_share_memory():
    v_dst = sym_t('v_dst')
    v_sld = sym_t('v_sld')
    v_src = sym_t('v_src')
    v_sst = sym_t('v_sst')

    mc = get_default_mc()
    sldx2 = inst_ds_read2_likely_t(mc, 4, 16, 1030)
    mc.emit(sldx2(v_dst(), v_sld()))
    #print(mc.emitter.get_buffer())

    sstx2 = inst_ds_write2_likely_t(mc, 4, 8, 512)
    mc.emit(sstx2(v_sst(), v_src(), 256))
    print(mc.emitter.get_buffer())

def unittest_coalescing_store():

    mc = get_default_mc()
    ctrl = ctrl_coalescing_store_t()

    ctrl.coalescing_groups = 2
    ctrl.gemm_m_repeat = 2
    ctrl.gemm_m_per_thread = 4
    ctrl.gemm_n_repeat = 2
    ctrl.gemm_n_per_thread = 4

    ctrl.gemm_m_length = 128  # in unit of dword
    ctrl.gemm_n_length = 128  # in unit of dword
    ctrl.data_byte = 4

    ctrl.vector_write_out = 1
    ctrl.block_size = 256


    coalescing_store = igemm_coalescing_store_t(mc, ctrl)

    mc.emit(coalescing_store('v_c', 'v_co_sst', 'v_co_sld', 's_p_out', 'v_out_offset', 's_out_offset', 's_gemm_m_stride', 's_tmp'))
    mc.emit(coalescing_store('v_c', 'v_co_sst', 'v_co_sld', 's_p_out', 'v_out_offset', 's_out_offset', 's_gemm_m_stride', 's_tmp'))
    print(mc.emitter.get_buffer())


def run_all_unittest():
    # unittest_share_memory()
    unittest_coalescing_store()

if __name__ == '__main__':
    run_all_unittest()