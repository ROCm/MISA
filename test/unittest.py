from python import *
import math

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

def unittest_coalescing_store_m1_m0_xdlops():
    macro_tile_m = 128
    macro_tile_n = 256
    block_size = 256

    mc = get_default_mc()
    cxm = get_ctrl_xdlops_mapping_fp32(macro_tile_m, macro_tile_n, block_size // 64)

    ctrl = ctrl_coalescing_store_xdlops_t()
    ctrl.cxm = cxm
    ctrl.coalescing_groups = 2
    ctrl.data_byte = 4

    ctrl.vector_write_out = 1
    ctrl.block_size = block_size
    ctrl.gemm_m_order = IGEMM_COALESCING_GEMM_M_ORDER_M1_M0
    ctrl.gemm_m_m0_m1 = [4, macro_tile_m // 4] # similar to non-xdlops

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


    #coalescing_store = igemm_coalescing_store_t(mc, ctrl)

    #mc.emit(coalescing_store.init_co_lds_offset('v_co_sst', 'v_co_sld', 'v_gemm_im', 'v_gemm_in', 'v0', 'v_tmp'))
    #mc.emit(coalescing_store.init_co_sub_m_index('v_co_sub_m_index', 'v_tid', 'v_tmp'))
    #mc.emit(coalescing_store('v_c', 'v_co_sst', 'v_co_sld', 's_p_out', 'v_out_offset', 's_out_offset', 's_gemm_m0_stride', 's_gemm_m1_stride', 's_tmp'))
    #print(mc.emitter.get_buffer())

def unittest_xdlops_mapping():
    for xdlops_mapping in ctrl_xdlops_mapping_fp32:
        print(xdlops_mapping.serialize())


def unittest_coalescing_store_m1_m0_xdlops_iterate():
    for xdlops_mapping in ctrl_xdlops_mapping_fp32:
        # max_possible_groups = xdlops_mapping.wave_repeat_m * xdlops_mapping.wave_step_m * xdlops_mapping.lanegroup_m_per_wave() * xdlops_mapping.lanegroup_m_per_block() * xdlops_mapping.lanegroup_m_per_thread()
        max_possible_groups = xdlops_mapping.wave_repeat_m * xdlops_mapping.wave_step_m * xdlops_mapping.lanegroup_m_per_wave() * \
                xdlops_mapping.lanegroup_m_per_block() * xdlops_mapping.lanegroup_m_per_thread() \
                    // 4
        cgroup_list = [2**x for x in range(0, int(math.log2(max_possible_groups)) + 1)]
        print(f"[<<<<<<]max_possible_groups:{max_possible_groups}, cgroup_list:{cgroup_list}, {xdlops_mapping.serialize()}")
        for cgroups in cgroup_list:
            mc = get_default_mc()
            print(f"[------] groups:{cgroups}")
            ctrl = ctrl_coalescing_store_xdlops_t()
            ctrl.cxm = xdlops_mapping
            ctrl.coalescing_groups = cgroups
            ctrl.data_byte = 4

            ctrl.vector_write_out = 1
            ctrl.block_size = xdlops_mapping.block_size()
            # ctrl.gemm_m_order = IGEMM_COALESCING_GEMM_M_ORDER_M1_M0
            ctrl.gemm_m_m0_m1 = [4, xdlops_mapping.macro_tile_m // 4] # similar to non-xdlops

            ctrl.adjust_optimal_coalescing_groups()



            coalescing_store = igemm_coalescing_store_xdlops_t(mc, ctrl)

            m_index_per_group       = ctrl.get_m_index_per_group()
            m_index_per_group_m1_m0 = ctrl.get_m_index_per_group_m1_m0()

            # def get_co_sub_m():
            #     ttm = ctrl.get_transposed_thread_mapping()
            #     g_mr, g_ms, g_mw, g_mb, g_mt = ctrl.get_subgroups()
            #     l_mr, l_ms, l_mw, l_mb, l_mt = ctrl.get_subgroup_length()
            #     n_mc = ctrl.cxm.lanegroup_m_per_cluster()       # this iteration is among different thread
            #     n_ml = ctrl.cxm.block_m_per_lanegroup()         # this iteration is among different thread
            #     n_mv = ctrl.cxm.waves_per_m()                   # this iteration is among different thread
            #     #print("<+> g_mr:{}, g_ms:{}, g_mw:{}, g_mb:{}, g_mt:{}  ==  l_mr:{}, l_ms:{}, l_mw:{}, l_mb:{}, l_mt:{} | n_mc:{}, n_ml:{}, n_mv:{}".format(
            #     #        g_mr, g_ms, g_mw, g_mb, g_mt, l_mr, l_ms, l_mw, l_mb, l_mt, n_mc, n_ml, n_mv))
            #     sub_m_index = [0] * ttm.c_m0()
            #     for ic in range(ttm.c_m0()):
            #         nic = ic
            #         x_mc = nic % n_mc; nic = nic // n_mc
            #         x_ml = nic % n_ml; nic = nic // n_ml
            #         x_mb = nic % l_mb; nic = nic // l_mb
            #         x_mw = nic % l_mw; nic = nic // l_mw
            #         x_ms = nic % l_ms; nic = nic // l_ms
            #         x_mv = nic % n_mv; nic = nic // n_mv
            #         x_mr = nic % l_mr
            #         # print("    +-> x_mc:{}, x_ml:{}, x_mb:{}, x_mw:{}, x_ms:{}, x_mv:{}, x_mr:{}".format(x_mc, x_ml, x_mb, x_mw, x_ms, x_mv, x_mr))
            #         sub_m = x_mr * n_mv + x_mv
            #         sub_m = sub_m * (g_ms * l_ms) + x_ms
            #         sub_m = sub_m * (l_mw * g_mw) + x_mw
            #         sub_m = sub_m * (g_mb * l_mb) + x_mb
            #         sub_m = sub_m * n_ml + x_ml
            #         sub_m = sub_m * n_mc + x_mc
            #         sub_m = sub_m * 4
            #         sub_m_index[ic] = sub_m
            #     return sub_m_index

            def get_sliced_sub_m_list():
                # from m_index_per_group[0]
                sliced_sub_m_list = list()
                m_start_index = 0
                for ic in range(len(m_index_per_group[0])):
                    sliced_sub_m_list.append(m_index_per_group[0][ic][m_start_index])
                return sliced_sub_m_list

            print(f"<sub_m_index>:{ctrl.get_co_sub_m_index()}")
            for ig in range(len(m_index_per_group)):
                for ic in range(len(m_index_per_group[ig])):
                    print(f"ig:{ig} ic:{ic}, m0_m1: {m_index_per_group[ig][ic]}")
                    print("    |" + " ".join( f"{ctrl.get_m0_m1_index(x)}" for x in m_index_per_group[ig][ic]))

            mc.emit(coalescing_store.init_co_sub_m_index('v_co_sub_m_index', 'v_tid', 'v_tmp6'))
            mc.emit(';@@@@ ------------------------------------------------')
            mc.emit(coalescing_store('a_c', 'v_c', 'v_co_sst', 'v_co_sld',
                    's_p_out', 'v_out_offset', 's_out_offset' if 0 else None, None, 's_gemm_m1_stride', 's_tmp4', 'v_store_flag'))
            mc.emit(';XXXX ------------------------------------------------')
            print(mc.emitter.get_buffer())
            print("------------------------------------------------------------")


            # assert co_sub_m along returned m_index_per_group[0], aka first group.
            assert ctrl.get_co_sub_m_index() == get_sliced_sub_m_list()

            # print("")
            # for ig in range(len(m_index_per_group)):
            #     for ic in range(len(m_index_per_group[ig])):
            #         print(f"ig:{ig} ic:{ic}, m1_m0: {m_index_per_group_m1_m0[ig][ic]}")
            #         print("    |" + " ".join( f"{ctrl.get_m0_m1_index(x)}" for x in m_index_per_group_m1_m0[ig][ic]))

def unittest_macro():
    class macro_igemm_bwd_gtc_out_update_hw_t(macro_base_t):
        def __init__(self, mc):
            macro_base_t.__init__(self, mc)
            self.declare_arg("v_out_iho")
            self.declare_arg("v_out_iwo")
            self.declare_arg("v_out_dslice_ih")
            self.declare_arg("v_out_dslice_iw")
            self.declare_arg("v_out_dslice_iy")
            self.declare_arg("v_out_dslice_ix")
            self.declare_arg("s_dtile_dy_neg")
            self.declare_arg("s_dtile_dx_neg")

        def name(self):
            return '.v_bwd_gtc_out_update_hw'
        # def __call__(self, v_out_iho, v_out_iwo, v_out_dslice_ih, v_out_dslice_iw, v_out_dslice_iy, v_out_dslice_ix, s_dtile_dy_neg, s_dtile_dx_neg):
        #     return '{} {}, {}, {}, {}, {}, {}, {}, {}'.format(self.name(),
        #         v_out_iho, v_out_iwo, v_out_dslice_ih, v_out_dslice_iw, v_out_dslice_iy, v_out_dslice_ix, s_dtile_dy_neg, s_dtile_dx_neg)
        # def emit(self):
        #     with self._emit_macro_indented('.macro {} v_out_iho, v_out_iwo, v_out_dslice_ih, v_out_dslice_iw, v_out_dslice_iy, v_out_dslice_ix, s_dtile_dy_neg, s_dtile_dx_neg'.format(self.name())):
        #         self._emit(f"; dslice_y,dslice_h -> oh, dslice_x,dslice_w -> ow")
        #         self._emit(f"v_mad_i32_i24 v[\\v_out_iho], s[\\s_dtile_dy_neg], v[\\v_out_dslice_iy], v[\\v_out_dslice_ih]")
        #         self._emit(f"v_mad_i32_i24 v[\\v_out_iwo], s[\\s_dtile_dx_neg], v[\\v_out_dslice_ix], v[\\v_out_dslice_iw]")

        #def is_inline(self):
        #    return True
        def expr(self):
            self._emit(f"; dslice_y,dslice_h -> oh, dslice_x,dslice_w -> ow")
            self._emit(f"v_mad_i32_i24 v[{self.v_out_iho()}], s[{self.s_dtile_dy_neg()}], v[{self.v_out_dslice_iy()}], v[{self.v_out_dslice_ih()}]")
            self._emit(f"v_mad_i32_i24 v[{self.v_out_iwo()}], s[{self.s_dtile_dx_neg()}], v[{self.v_out_dslice_ix()}], v[{self.v_out_dslice_iw()}]")

    mc = get_default_mc()
    igemm_bwd_gtc_out_update_hw = macro_igemm_bwd_gtc_out_update_hw_t(mc)

    igemm_bwd_gtc_out_update_hw.emit()
    mc.emit(igemm_bwd_gtc_out_update_hw('v_out_iho', 'v_out_iwo', 'v_out_dslice_ih222', 'v_out_dslice_iw222', 'v_out_dslice_iy', 'v_out_dslice_ix', 's_dtile_dy_neg', 's_dtile_dx_neg'))


    mc.emit(igemm_bwd_gtc_out_update_hw(sym_t('v_out_ihoabab')(), 'v_out_iwo', 'v_out_dslice_ih222', 'v_out_dslice_iw222', 'v_out_dslice_iy', 'v_out_dslice_ix', 's_dtile_dy_neg', 's_dtile_dx_neg'))
    print(mc.emitter.get_buffer())

def unittest_thread_mapping():
    mc = get_default_mc()
    ctm = ctrl_thread_mapping_t()
    ctm.thread_lengths = [2,2,1,1,4,4]
    ctm.cluster_lengths = [1,1,4,4,4,4]
    thread_mapping = igemm_thread_mapping_t(mc, ctm)
    mc.emit(thread_mapping( 'v_gemm_in', 'v_gemm_im', 'v_tid_shifter', 'v_tmp'))
    print(mc.emitter.get_buffer())

def unittest_dotx_mapping():
    for ctrl in ctrl_dotx_mapping_fp16:
        print(ctrl.serialize())

def unittest_dotx_coalescing_store():
    mc = get_default_mc()
    mc_set_current(mc)

    # cdm = ctrl_dotx_mapping_t(128, 128,   8,   8,   2,   4,   4,   2,   4, v_dot2c_f32_f16)
    cdm = ctrl_dotx_mapping_t(128, 128,   8,   8,   4,   2,   4,   2,   4, v_dot2c_f32_f16)
    precision = 'fp16'

    coalescing_store_groups = 2

    ctrl = ctrl_coalescing_store_dotx_t()
    ctrl.cdm = cdm                     # ctrl_dotx_mapping_t
    ctrl.coalescing_groups = coalescing_store_groups
    ctrl.block_size = cdm.block_size()
    ctrl.vector_store_m = 8             # global vector store in m/n
    ctrl.vector_fold_m = 8
    ctrl.vector_store_n = 1             # ... m, n can't be non-1 at the same time
    ctrl.precision = 'fp16'             # dotx only support fp16 & int8
    ctrl.gemm_k_global_split = False
    ctrl.feat_vgpr_collapse = True
    ctrl.co_m_update_os_functor = None  # update offset based on current i_m. otherwise use sgpr to update offset

    ctrl.feat_co_m_flag_check = False   # custom flag check, not using internal check
    ctrl.co_m_flag_check_start_functor = None
    ctrl.co_m_flag_check_reset_functor = None

    coalescing_store = igemm_coalescing_store_dotx_t(mc, ctrl)

    mc.emit(coalescing_store.init_co_sub_m_index('v_co_sub_m_index', 'v_tid', 'v_tmp'))
    mc.emit(';-----------------------------------------------')
    mc.emit(coalescing_store.init_co_sub_n_index('v_co_sub_n_index', 'v_tid', 'v_tmp'))
    mc.emit(';-----------------------------------------------')
    mc.emit(coalescing_store.init_co_lds_offset('v_co_sst', 'v_co_sld', 'v_gemm_im', 'v_gemm_in', 'v_tid', 'v_tmp'))
    mc.emit(';-----------------------------------------------')

    mc.emit(coalescing_store('v_tmp', 'v_c', 'v_co_sst', 'v_co_sld', 's_p_out', 'v_out_os', None,
                    None, 's_out_stride_gemm_m', 's_tmp', 'v_out_flag'))

    print(mc.emitter.get_buffer())

def run_all_unittest():
    # unittest_share_memory()
    #unittest_coalescing_store()
    #unittest_coalescing_store_m1_m0()
    #unittest_coalescing_store_m1_m0_xdlops()
    #unittest_xdlops_mapping()
    #unittest_coalescing_store_m1_m0_xdlops_iterate()
    # unittest_thread_mapping()
    #unittest_macro()
    #unittest_dotx_mapping()
    unittest_dotx_coalescing_store()

if __name__ == '__main__':
    run_all_unittest()