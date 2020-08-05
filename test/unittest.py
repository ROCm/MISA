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
    ctm = ctrl_thread_mapping_t()
    ctm.thread_lengths = [2,2,1,1,4,4]
    ctm.cluster_lengths = [1,1,4,4,4,4]

    ctrl = ctrl_coalescing_store_t()
    ctrl.ctm = ctm
    ctrl.coalescing_groups = 4
    ctrl.data_byte = 4

    ctrl.vector_write_out = 1
    ctrl.block_size = 256

    coalescing_store = igemm_coalescing_store_t(mc, ctrl)


    mc.emit(coalescing_store.init_co_lds_offset('v_co_sst', 'v_co_sld', 'v_gemm_im', 'v_gemm_in', 'v0', 'v_tmp'))
    mc.emit(coalescing_store.init_co_sub_m_index('v_co_sub_m_index', 'v_tid', 'v_tmp'))
    mc.emit(coalescing_store('v_c', 'v_co_sst', 'v_co_sld', 's_p_out', 'v_out_offset', 's_out_offset', 's_gemm_m_stride', 's_tmp'))
    print(mc.emitter.get_buffer())



def unittest_coalescing_store_m1_m0():
    mc = get_default_mc()
    ctm = ctrl_thread_mapping_t()
    ctm.thread_lengths = [2,2,1,1,4,4]
    ctm.cluster_lengths = [1,1,4,4,4,4]

    ctrl = ctrl_coalescing_store_t()
    ctrl.ctm = ctm
    ctrl.coalescing_groups = 4
    ctrl.data_byte = 4

    ctrl.vector_write_out = 1
    ctrl.block_size = 256
    ctrl.gemm_m_order = IGEMM_COALESCING_GEMM_M_ORDER_M1_M0
    ctrl.gemm_m_m0_m1 = [4, 32]

    ctrl.adjust_optimal_coalescing_groups()

    m_index_per_group       = ctrl.get_m_index_per_group()
    m_index_per_group_m1_m0 = ctrl.get_m_index_per_group_m1_m0()
    for ig in range(len(m_index_per_group)):
        for ic in range(len(m_index_per_group[ig])):
            print(f"ig:{ig} ic:{ic}, m0_m1: {m_index_per_group[ig][ic]}")
            print("    |" + " ".join( f"{ctrl.get_m0_m1_index(x)}" for x in m_index_per_group[ig][ic]))
    print("")
    for ig in range(len(m_index_per_group)):
        for ic in range(len(m_index_per_group[ig])):
            print(f"ig:{ig} ic:{ic}, m1_m0: {m_index_per_group_m1_m0[ig][ic]}")
            print("    |" + " ".join( f"{ctrl.get_m0_m1_index(x)}" for x in m_index_per_group_m1_m0[ig][ic]))


    coalescing_store = igemm_coalescing_store_t(mc, ctrl)

    mc.emit(coalescing_store.init_co_lds_offset('v_co_sst', 'v_co_sld', 'v_gemm_im', 'v_gemm_in', 'v0', 'v_tmp'))
    mc.emit(coalescing_store.init_co_sub_m_index('v_co_sub_m_index', 'v_tid', 'v_tmp'))
    mc.emit(coalescing_store('v_c', 'v_co_sst', 'v_co_sld', 's_p_out', 'v_out_offset', 's_out_offset', 's_gemm_m0_stride', 's_gemm_m1_stride', 's_tmp'))
    print(mc.emitter.get_buffer())

def unittest_thread_mapping():
    mc = get_default_mc()
    ctm = ctrl_thread_mapping_t()
    ctm.thread_lengths = [2,2,1,1,4,4]
    ctm.cluster_lengths = [1,1,4,4,4,4]
    thread_mapping = igemm_thread_mapping_t(mc, ctm)
    mc.emit(thread_mapping( 'v_gemm_in', 'v_gemm_im', 'v_tid_shifter', 'v_tmp'))
    print(mc.emitter.get_buffer())

def run_all_unittest():
    # unittest_share_memory()
    #unittest_coalescing_store()
    unittest_coalescing_store_m1_m0()
    # unittest_thread_mapping()

if __name__ == '__main__':
    run_all_unittest()