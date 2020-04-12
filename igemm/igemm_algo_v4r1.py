# pylint: disable=maybe-no-member
from .igemm_base import *
from .amdgpu import *
from .codegen import *
from .conv import *
class igemm_v4r1_dynamic_t(object):
    def __init__(self, mc, tunable):
        self.mc = mc
        self.tunable = tunable
        mc.inject(self)

class emit_fma_subtile_t(igemm_v4r1_dynamic_t):
    def name(self):
        return self.fma.name()
    def __init__(self, mc, tunable):
        igemm_v4r1_dynamic_t.__init__(self, mc, tunable)
        self.fma = emit_fma_mxn_t(mc, tunable.thread_sub_tile_m, tunable.thread_sub_tile_n, tunable.thread_tile_n)
    def __call__(self, c, a, b):
        return self.fma(c, a, b)
    def emit(self):
        self.fma.emit()

class emit_in_set_flag_t(igemm_v4r1_dynamic_t):
    '''
    update v_flag
    '''
    def name(self):
        return '.v_in_set_flag'
    def __init__(self, mc, tunable):
        igemm_v4r1_dynamic_t.__init__(self, mc, tunable)
    def __call__(self, v_flag, v_in_ihi, v_in_iwi, s_hi, s_wi, s_tmp2):
        return '{} {}, {}, {}, {}, {}, {}'.format(self.name(),
                        v_flag, v_in_ihi, v_in_iwi, s_hi, s_wi, s_tmp2)
    def emit(self):
        self._emit_macro_desc()
        with self._emit_macro_indented('.macro {} v_flag, v_in_ihi, v_in_iwi, s_hi, s_wi, s_tmp2'.format(self.name())):
            self._emit(';   flag: 0<= * <wi')
            self._emit('v_cmp_le_i32 vcc, 0, v[\\v_in_ihi]')
            self._emit('v_cmp_gt_i32 s[\\s_tmp2:\\s_tmp2+1], s[\\s_hi], v[\\v_in_ihi]')
            self._emit('s_and_b64 vcc, vcc, s[\\s_tmp2:\\s_tmp2+1]')
            self._emit('v_cndmask_b32 v[\\v_flag], 0, 1, vcc')
            self._emit(';   flag: 0<= * <wi')
            self._emit('v_cmp_le_i32 vcc, 0, v[\\v_in_iwi]')
            self._emit('v_cmp_gt_i32 s[\\s_tmp2:\\s_tmp2+1], s[\\s_wi], v[\\v_in_iwi]')
            self._emit('s_and_b64 vcc, vcc, s[\\s_tmp2:\\s_tmp2+1]')
            self._emit('v_cndmask_b32 v[\\v_flag], 0, v[\\v_flag], vcc')

class emit_in_load_e_n1_b_n2_t(igemm_v4r1_dynamic_t):
    '''
    load input from global.
    '''
    def name(self):
        return '.v_in_load_e_n1_b_n2_' + '1_{}_1_{}'.format(
                                self.t_n1,
                                self.t_n2)
    def __init__(self, mc, tunable):
        igemm_v4r1_dynamic_t.__init__(self, mc, tunable)
        self.t_n1 = tunable.in_block_copy_sub_lengths_n1
        self.t_n2 = tunable.in_block_copy_sub_lengths_n2
        assert self.t_n2 != 1, "currently t_n2 should not be 1"
    def __call__(self, v_dst, s_p_buf_in, v_in_os, s_in_stride_n1, s_in_stride_n2, v_flag, s_tmp4):
        return '{} {}, {}, {}, {}, {}, {}, {}'.format(self.name(), v_dst, s_p_buf_in, v_in_os, s_in_stride_n1, s_in_stride_n2, v_flag, s_tmp4)
    def emit(self):
        m_v_clear_nc = emit_c_clear_t(self.mc)
        self._emit_macro_desc('{{e,n1,b,n2}}:{{{},{},{},{}}}'.format(1,self.t_n1,1,self.t_n2))
        with self._emit_macro_indented(".macro {} v_dst, s_p_buf_in, v_in_os, s_in_stride_n1, s_in_stride_n2, v_flag, s_tmp4".format(self.name())):
            self._emit(m_v_clear_nc('\\v_dst', self.t_n1 * self.t_n2))
            self._emit('v_cmp_eq_u32 vcc, 1, v[\\v_flag]')
            self._emit('s_and_saveexec_b64 s[\\s_tmp4+2:\\s_tmp4+3], vcc')
            idst = 0
            for itr_n1 in range(self.t_n1):
                for itr_n2 in range(self.t_n2):
                    if idst == 0:
                        self._emit('buffer_load_dword v[\\v_dst+{}], v[\\v_in_os], s[\\s_p_buf_in:\\s_p_buf_in+3], 0 offen'.format(idst))
                    else:
                        self._emit('buffer_load_dword v[\\v_dst+{}], v[\\v_in_os], s[\\s_p_buf_in:\\s_p_buf_in+3], s[\\s_tmp4] offen'.format(idst))
                    if itr_n2 == 0 and itr_n1 == 0:
                        self._emit('s_mov_b32 s[\\s_tmp4], s[\\s_in_stride_n2]')
                    elif itr_n2 != self.t_n2 - 1:
                        self._emit('s_add_u32 s[\\s_tmp4], s[\\s_tmp4], s[\\s_in_stride_n2]')

                    idst = idst + 1
                if self.t_n1 != 1:
                    if itr_n1 != self.t_n1 - 1:
                        self._emit('s_mul_i32 s[\\s_tmp4], {}, s[\\s_in_stride_n1]'.format(itr_n1+1))
            self._emit('s_or_b64 exec, exec, s[\\s_tmp4+2:\\s_tmp4+3]')

class emit_wei_load_e_k_t(igemm_v4r1_dynamic_t):
    '''
    load weight from global.
    '''
    def name(self):
        return '.v_wei_load_e_k_'+ '{}_{}_ev{}'.format(
                        self.t_e,
                        self.t_k,
                        self.t_e_vec_size)
    def __init__(self, mc, tunable):
        igemm_v4r1_dynamic_t.__init__(self, mc, tunable)
        self.t_e = tunable.wei_block_copy_sub_lengths_e
        self.t_k = tunable.wei_block_copy_sub_lengths_k
        self.t_e_vec_size = tunable.wei_block_copy_src_data_per_read_e
        assert self.t_e_vec_size==1 or self.t_e_vec_size==2 or self.t_e_vec_size==4
        assert self.t_e == self.t_e_vec_size, 'currenly only implemented e equal to e_vector_size'
    def __call__(self, v_dst, s_p_buf_wei, v_wei_os, s_wei_stride_k, s_tmp2):
        return '{} {}, {}, {}, {}, {}'.format(self.name(), v_dst, s_p_buf_wei, v_wei_os, s_wei_stride_k, s_tmp2)
    def emit(self):
        class wei_buffer_load_t:
            def __init__(self, vector_size):
                self.vector_size = vector_size
            def __call__(self, start_v_index, s_offset):
                if self.vector_size == 1:
                    return 'buffer_load_dword v[\\v_dst+{}], v[\\v_wei_os], s[\\s_p_buf_wei:\\s_p_buf_wei+3], {} offen'.format(start_v_index, s_offset)
                if self.vector_size == 2:
                    return 'buffer_load_dwordx2 v[\\v_dst+{}:\\v_dst+{}], v[\\v_wei_os], s[\\s_p_buf_wei:\\s_p_buf_wei+3], {} offen'.format(start_v_index, start_v_index+1, s_offset)
                if self.vector_size == 4:
                    return 'buffer_load_dwordx4 v[\\v_dst+{}:\\v_dst+{}], v[\\v_wei_os], s[\\s_p_buf_wei:\\s_p_buf_wei+3], {} offen'.format(start_v_index, start_v_index+3, s_offset)
                return 'invalid'

        # assert t_k<=4, "k only support 1,2,4, other size to be implemented"
        self._emit_macro_desc('{{e,k}}:{{{},{}}}, vector_e:{}'.format(self.t_e, self.t_k, self.t_e_vec_size))
        with self._emit_macro_indented('.macro {} v_dst, s_p_buf_wei, v_wei_os, s_wei_stride_k, s_tmp2'.format(self.name())):
            buffer_loader = wei_buffer_load_t(self.t_e_vec_size)
            for itk in range(self.t_k):
                if itk == 0:
                    self._emit('{}'.format(buffer_loader(itk*self.t_e_vec_size, '0')))
                elif itk == 1:
                    self._emit('{}'.format(buffer_loader(itk*self.t_e_vec_size, 's[\\s_wei_stride_k]')))
                    if itk != self.t_k - 1:
                        self._emit('s_lshl_b32 s[\\s_tmp2], s[\\s_wei_stride_k], 1')
                else:
                    self._emit('{}'.format(buffer_loader(itk*self.t_e_vec_size, 's[\\s_tmp2]')))
                    if itk != self.t_k - 1:
                        self._emit('s_add_u32 s[\\s_tmp2], s[\\s_tmp2], s[\\s_wei_stride_k]')

class emit_in_sst_e_n1_b_n2_t(igemm_v4r1_dynamic_t):
    '''
    store input to LDS.
    '''
    def name(self):
        return '.v_in_sst_e_n1_b_n2_1_{}_1_{}_n1s{}_n2v{}'.format(
                    self.t_n1,
                    self.t_n2,
                    self.t_n1_stride,
                    self.t_n2_vec_size)
    def __init__(self, mc, tunable):
        igemm_v4r1_dynamic_t.__init__(self, mc, tunable)
        self.t_n1 = tunable.in_block_copy_sub_lengths_n1
        self.t_n2 = tunable.in_block_copy_sub_lengths_n2
        self.t_n1_stride = tunable.gemm_n_per_thread_subc * tunable.b_per_block * 4
        self.t_n2_vec_size = tunable.in_block_copy_dst_data_per_write_n2
        assert self.t_n2 == self.t_n2_vec_size, "currently only implemented n2 equal n2_vector_size"
    def __call__(self, v_src, v_sst_os):
        return '{} {}, {}'.format(self.name(), v_src, v_sst_os)
    def emit(self):
        self._emit_macro_desc('{{e,n1,b,n2}}:{{{},{},{},{}}}, stride_n1:{}, vector_n2:{}, offset:{}'.format(1,self.t_n1,1,self.t_n2,self.t_n1_stride,self.t_n2_vec_size,0))
        with self._emit_macro_indented('.macro {} v_src, v_sst_os'.format(self.name())):
            ds_write = ds_write_t(self.t_n2_vec_size * 4)
            for itn1 in range(self.t_n1):
                self._emit('{}'.format(ds_write('\\v_sst_os', gpr_t('\\v_src')(itn1*self.t_n2_vec_size), itn1 * self.t_n1_stride)))
    def get_issues(self):
        return self.t_n1

class emit_wei_ds_write2_likely_t(igemm_v4r1_dynamic_t):
    '''
    generate ds_write2 if possible. otherwise fallback to ds_write.
    Design this not as macro, but inlined into other LDS store operation like emit_wei_sst_e_k_t
    So need upper caller to make sure the uniqueness

    For wei load from global is {k, e}, and store to LDS is {e, k}, so need consider swap
    '''
    
    def name(self):
        return ''
    def __init__(self, mc, tunable, t_n_vec, t_vec_size, t_vec_stride, t_sst_base):
        igemm_v4r1_dynamic_t.__init__(self, mc, tunable)
        self.t_n_vec        = t_n_vec
        self.t_vec_size     = t_vec_size
        self.t_vec_stride   = t_vec_stride
        self.t_sst_base     = t_sst_base
    def likely_write2_b32(self):
        if self.t_n_vec % 2 != 0:
            return False
        if (self.t_sst_base % 4 == 0) and (self.t_vec_stride % 4 == 0):
            if (self.t_sst_base // 4) + (self.t_vec_stride // 4) * (self.t_n_vec - 1) < 256:
                return True
        return False
    def likely_write2st64_b32(self):
        if self.t_n_vec % 2 != 0:
            return False
        if (self.t_sst_base % (4*64) == 0) and (self.t_vec_stride % 4 == 0):
            if (self.t_sst_base // (4*64)) + (self.t_vec_stride // (4*64)) * (self.t_n_vec - 1) < 256:
                return True
        return False
    def likely_write2_b64(self):
        if self.t_n_vec % 2 != 0:
            return False
        if (self.t_sst_base % 8 == 0) and (self.t_vec_stride % 8 == 0):
            if (self.t_sst_base // 8) + (self.t_vec_stride // 8) * (self.t_n_vec - 1) < 256:
                return True
        return False
    def likely_write2st64_b64(self):
        if self.t_n_vec % 2 != 0:
            return False
        if (self.t_sst_base % (8*64) == 0) and (self.t_vec_stride % (8*64) == 0):
            if (self.t_sst_base // (8*64)) + (self.t_vec_stride // (8*64)) * (self.t_n_vec - 1) < 256:
                return True
        return False
    def __call__(self, v_src, v_sst):
        g_src = gpr_t(v_src)
        g_sst = gpr_t(v_sst)
        def emit_write2_fallback():
            with self._deferred_context():
                if self.t_vec_size == 1:
                    for n in range(self.t_n_vec):
                        self._emit('ds_write_b32 v[{}], v[{}] offset:{}'.format(g_sst(), g_src(n), self.t_sst_base + n * self.t_vec_stride))
                elif self.t_vec_size == 2:
                    if self.t_n_vec == 1:
                        self._emit('ds_write_b64 v[{}], v[{}:{}] offset:{}'.format(g_sst(), g_src(), g_src(1), self.t_sst_base ))
                    else:
                        swap_start = (self.t_n_vec*self.t_vec_size) // 2
                        for n in range(self.t_n_vec // 2):
                            self._emit('v_swap_b32 v[{}], v[{}]'.format(g_src(2*n + 1), g_src(2*n + swap_start)))
                            self._emit('ds_write_b64 v[{}], v[{}:{}] offset:{}'.format(g_sst(), g_src(2*n), g_src(2*n + 1), self.t_sst_base + 2*n * self.t_vec_stride))
                            self._emit('ds_write_b64 v[{}], v[{}:{}] offset:{}'.format(g_sst(), g_src(2*n + swap_start) , g_src(2*n + swap_start + 1), self.t_sst_base + (2*n+1) * self.t_vec_stride))
                elif self.t_vec_size == 4:
                    if self.t_n_vec == 1:
                        self._emit('ds_write_b128 v[{}], v[{}:{}] offset:{}'.format(g_sst(), g_src(), g_src(3), self.t_sst_base ))
                    else:
                        # though we use algorithm in swap_seq to interleave swap with ds_write, but it is still wise to use extra tmp register for swap is half speed
                        swap_list = amdgpu_swap_sequencer_t(self.t_n_vec , self.t_vec_size)()
                        # print('self.t_n_vec:{}, self.t_vec_size:{}, {}'.format(self.t_n_vec , self.t_vec_size, swap_list))
                        for n in range(self.t_n_vec):
                            sw = swap_list[n]
                            if type(sw) is str:
                                pass
                            else:
                                for sw_item in sw:
                                    self._emit('v_swap_b32 v[{}], v[{}]'.format(g_src(sw_item[0]) , g_src(sw_item[1]) ))
                            self._emit('ds_write_b128 v[{}], v[{}:{}] offset:{}'.format(g_sst(), g_src(4*n), g_src(4*n + 3), self.t_sst_base + n * self.t_vec_stride))
                else:
                    assert False, 'unsupported vector size'
            return self._get_deferred()

        def emit_write2_b32():
            with self._deferred_context():
                for n in range(self.t_n_vec // 2):
                    self._emit('ds_write2_b32 v[{}], v[{}], v[{}], offset0:{}, offset1:{}'.format(g_sst(),
                                g_src(2*n), g_src(2*n+1),
                                (self.t_sst_base//4)+2*n*(self.t_vec_stride//4), (self.t_sst_base//4)+(2*n+1)*(self.t_vec_stride//4)))
            return self._get_deferred()

        def emit_write2st64_b32():
            with self._deferred_context():
                for n in range(self.t_n_vec // 2):
                    self._emit('ds_write2st64_b32 v[{}], v[{}], v[{}], offset0:{}, offset1:{}'.format(g_sst(),
                                g_src(2*n), g_src(2*n+1),
                                (self.t_sst_base//(4*64))+2*n*(self.t_vec_stride//(4*64)), (self.t_sst_base//(4*64))+(2*n+1)*(self.t_vec_stride//(4*64))))
            return self._get_deferred()

        def emit_write2_b64():
            swap_start = (self.t_n_vec*self.t_vec_size) // 2
            with self._deferred_context():
                for n in range(self.t_n_vec // 2):
                    self._emit('v_swap_b32 v[{}], v[{}]'.format(g_src(2*n+1), g_src(2*n+swap_start)))
                    self._emit('ds_write2_b64 v[{}], v[{}:{}], v[{}:{}], offset0:{}, offset1:{}'.format(g_sst(),
                            g_src(2*n), g_src(2*n+1), g_src(2*n+swap_start), g_src(2*n+swap_start+1),
                            (self.t_sst_base//8)+2*n*(self.t_vec_stride//8), (self.t_sst_base//8)+(2*n+1)*(self.t_vec_stride//8)))
            return self._get_deferred()

        def emit_write2st64_b64():
            swap_start = (self.t_n_vec*self.t_vec_size) // 2
            with self._deferred_context():
                for n in range(self.t_n_vec // 2):
                    self._emit('v_swap_b32 v[{}], v[{}]'.format(g_src(2*n+1), g_src(2*n+swap_start)))
                    self._emit('ds_write2st64_b64 v[{}], v[{}:{}], v[{}:{}], offset0:{}, offset1:{}'.format(g_sst(),
                            g_src(2*n), g_src(2*n+1), g_src(2*n+swap_start), g_src(2*n+swap_start+1),
                            (self.t_sst_base//(8*64))+2*n*(self.t_vec_stride//(8*64)), (self.t_sst_base//(8*64))+(2*n+1)*(self.t_vec_stride//(8*64))))
            return self._get_deferred()

        def likely_emit():
            if self.t_vec_size == 1:
                if self.likely_write2_b32():
                    return emit_write2_b32()
                if self.likely_write2st64_b32():
                    return emit_write2st64_b32()
                return emit_write2_fallback()
            if self.t_vec_size == 2:
                if self.likely_write2_b64():
                    return emit_write2_b64()
                if self.likely_write2st64_b64():
                    return emit_write2st64_b64()
                return emit_write2_fallback()
            return emit_write2_fallback()

        return likely_emit()
    def emit(self):
        assert False, 'dont use emit of this'
    def get_issues(self):
        if self.t_vec_size == 1:
            if self.likely_write2_b32() or self.likely_write2st64_b32():
                return self.t_n_vec // 2
        if self.t_vec_size == 2:
            if self.likely_write2_b64() or self.likely_write2st64_b64():
                return self.t_n_vec // 2
        return self.t_n_vec

class emit_wei_sst_e_k_t(igemm_v4r1_dynamic_t):
    '''
    store weight to LDS.
    '''
    def name(self):
        return '.v_wei_sst_e_k_{}_{}_es{}_kv{}'.format(
            self.t_e,
            self.t_k,
            self.t_e_stride,
            self.t_k_vec_size)
    def __init__(self, mc, tunable):
        igemm_v4r1_dynamic_t.__init__(self, mc, tunable)
        self.t_e = tunable.wei_block_copy_sub_lengths_e
        self.t_k = tunable.wei_block_copy_sub_lengths_k
        self.t_e_stride = tunable.k_per_block * 4
        self.t_k_vec_size = tunable.wei_block_copy_src_data_per_write_k
        assert self.t_k == self.t_k_vec_size
        #self.t_sst_base = tunable.byte_lds_b_np2
        self.t_sst_base = 0
        self.write2_likely = emit_wei_ds_write2_likely_t(self.mc, self.tunable, self.t_e, self.t_k_vec_size, self.t_e_stride, self.t_sst_base)
    def __call__(self, v_src, v_sst_os):
        return '{} {}, {}'.format(self.name(), v_src, v_sst_os)
    def emit(self):
        self._emit_macro_desc('{{e,k}}:{{{},{}}}, stride_e:{}, vector_k:{}, offset:{}'.format(
                self.t_e, self.t_k, self.t_e_stride, self.t_k_vec_size, self.t_sst_base))

        with self._emit_macro_indented('.macro {} v_src, v_sst_os'.format(self.name())):
            self._emit(self.write2_likely('\\v_src', '\\v_sst_os'))
    def get_issues(self):
        return self.write2_likely.get_issues()

class emit_out_write_k0_k1_n1_b_n2_t(igemm_v4r1_dynamic_t):
    '''
    store output to global. s_dst_os_4 need be zero
    '''
    def name(self):
        return '.v_out_write_k0_k1_n1_b_n2_{}_{}_{}_1_{}'.format(
                        self.t_k0,
                        self.t_k1,
                        self.t_n1,
                        self.t_n2)
    def __init__(self, mc, tunable):
        igemm_v4r1_dynamic_t.__init__(self, mc, tunable)
        self.t_k0 = tunable.gemm_m_repeat
        self.t_k1 = tunable.gemm_m_per_thread_subc
        self.t_n1 = tunable.gemm_n_repeat
        self.t_n2 = tunable.gemm_n_per_thread_subc
    def __call__(self, v_src, s_p_out, v_out_os, s_out_stride_k0, s_out_stride_k1, s_out_stride_n1, s_out_stride_n2, s_dst_os_4):
        return '{} {}, {}, {}, {}, {}, {}, {}, {}'.format(self.name(),
                    v_src, s_p_out, v_out_os,
                    s_out_stride_k0, s_out_stride_k1, s_out_stride_n1, s_out_stride_n2,
                    s_dst_os_4)
    def emit(self):
        m_write4d = emit_write_4d_strided_t(self.mc)
        self._emit_macro_desc('{{k0,k1,n1,b,n2}}:{{{},{},{},1,{}}}'.format(self.t_k0, self.t_k1, self.t_n1, self.t_n2))
        with self._emit_macro_indented('.macro {} v_src, s_p_out, v_out_os, s_out_stride_k0, s_out_stride_k1, s_out_stride_n1, s_out_stride_n2, s_dst_os_4, t_k0, t_k1, t_n1, t_n2'.format(self.name())):
            self._emit(m_write4d('\\v_src', '\\s_p_out', '\\v_out_os',
                        '\\s_out_stride_n2', '\\s_out_stride_n1', '\\s_out_stride_k1', '\\s_out_stride_k0',
                        '\\s_dst_os_4', self.t_n2, self.t_n1, self.t_k1, self.t_k0))

class emit_in_move_slice_window_t(igemm_v4r1_dynamic_t):
    '''
    move input slice window. unified for all tunable along e=c*y*x
    '''
    def name(self):
        return '.v_in_move_slice_window'
    def __init__(self, mc, tunable):
        igemm_v4r1_dynamic_t.__init__(self, mc, tunable)
    def __call__(self, v_in_os, v_in_ic, v_in_iy, v_in_ix, v_in_ihi, v_in_iwi, v_flag,
                        s_hi, s_wi, s_y, s_x, s_in_stride_c, s_dilation_h, s_dilation_w, s_in_ic, s_in_iy, s_in_ix,
                        v_idc, v_idy, v_idx, s_tmp2):
        return '{} {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}'.format(self.name(),
                        v_in_os, v_in_ic, v_in_iy, v_in_ix, v_in_ihi, v_in_iwi, v_flag,
                        s_hi, s_wi, s_y, s_x, s_in_stride_c, s_dilation_h, s_dilation_w, s_in_ic, s_in_iy, s_in_ix,
                        v_idc, v_idy, v_idx, s_tmp2)
    def emit(self):
        self._emit_macro_desc('\n; update v_in_os, v_flag, update v_in_ic, v_in_iy, v_in_ix (zero or possitive), v_in_ihi, v_in_iwi (negative, zero, possitive)')
        with self._emit_macro_indented('.macro {} v_in_os, v_in_ic, v_in_iy, v_in_ix, v_in_ihi, v_in_iwi, v_flag, s_hi, s_wi, s_y, s_x, s_in_stride_c, s_dilation_h, s_dilation_w, s_in_ic, s_in_iy, s_in_ix, v_idc, v_idy, v_idx, s_tmp2'.format(self.name())):
            self._emit('; record old ic, iy, ix')
            self._emit('v_mov_b32 v[\\v_idx], v[\\v_in_ix]')
            self._emit('v_mov_b32 v[\\v_idy], v[\\v_in_iy]')
            self._emit('v_mov_b32 v[\\v_idc], v[\\v_in_ic]')
            self._emit_empty_line()
            self._emit('; update ix, calculate idx, carry-out to iy')
            self._emit('v_add_u32 v[\\v_in_ix], s[\\s_in_ix], v[\\v_in_ix]')
            self._emit('v_cmp_le_u32 vcc, s[\\s_x], v[\\v_in_ix]')
            self._emit('s_and_saveexec_b64 s[\\s_tmp2:\\s_tmp2+1], vcc')
            self._emit('v_subrev_u32 v[\\v_in_ix], s[\\s_x], v[\\v_in_ix]')
            self._emit('v_add_u32 v[\\v_in_iy], 1, v[\\v_in_iy]')
            self._emit('s_or_b64 exec, exec, s[\\s_tmp2:\\s_tmp2+1]')
            self._emit('v_sub_i32 v[\\v_idx], v[\\v_in_ix], v[\\v_idx]')
            self._emit_empty_line()
            self._emit('; update iy, calculate idy, carry-out to ic')
            self._emit('v_add_u32 v[\\v_in_iy], s[\\s_in_iy], v[\\v_in_iy]')
            self._emit('v_cmp_le_u32 vcc, s[\\s_y], v[\\v_in_iy]')
            self._emit('s_and_saveexec_b64 s[\\s_tmp2:\\s_tmp2+1], vcc')
            self._emit('v_subrev_u32 v[\\v_in_iy], s[\\s_y], v[\\v_in_iy]')
            self._emit('v_add_u32 v[\\v_in_ic], 1, v[\\v_in_ic]')
            self._emit('s_or_b64 exec, exec, s[\\s_tmp2:\\s_tmp2+1]')
            self._emit('v_sub_i32 v[\\v_idy], v[\\v_in_iy], v[\\v_idy]')
            self._emit_empty_line()
            self._emit('; update ic, calculate idc, ignore overflow check')
            self._emit('v_add_u32 v[\\v_in_ic], s[\\s_in_ic], v[\\v_in_ic]')
            self._emit('v_sub_u32 v[\\v_idc], v[\\v_in_ic], v[\\v_idc]')
            self._emit_empty_line()
            self._emit('; calculate offset: idc*(s_hi*s_wi) + idy*s_dilation_h*s_wi + idx*s_dilation_w')
            self._emit('; we use i24 as multiplier, for 24bit(-8388607 ~ 8388608) is enough for index')
            self._emit('; also, update ihi, iwi here')
            self._emit('v_mul_i32_i24 v[\\v_idy], s[\\s_dilation_h], v[\\v_idy]')
            self._emit('v_mul_i32_i24 v[\\v_idx], s[\\s_dilation_w], v[\\v_idx]')
            self._emit('v_add_i32 v[\\v_in_ihi], v[\\v_idy], v[\\v_in_ihi]')
            self._emit('v_add_i32 v[\\v_in_iwi], v[\\v_idx], v[\\v_in_iwi]')
            self._emit('v_mul_i32_i24 v[\\v_idy], s[\\s_wi], v[\\v_idy]')
            self._emit_empty_line()
            self._emit('v_add_i32 v[\\v_idx], v[\\v_idx], v[\\v_idy]')
            self._emit('v_mul_lo_u32 v[\\v_idc], s[\\s_in_stride_c], v[\\v_idc]')
            self._emit('v_add_i32 v[\\v_idc], v[\\v_idc], v[\\v_idx]')
            self._emit('v_lshl_add_u32 v[\\v_in_os], v[\\v_idc], 2, v[\\v_in_os]   ; indeed, v_idc here must be possitive')
            self._emit_empty_line()
            self._emit('; update v_flag')
            self._emit('.v_in_set_flag \\v_flag, \\v_in_ihi, \\v_in_iwi, \\s_hi, \\s_wi, \\s_tmp2')

class emit_wei_move_slice_window_t(igemm_v4r1_dynamic_t):
    '''
    move weight slice window. unified for all tunable along e=c*y*x
    '''
    def name(self):
        return '.v_wei_move_slice_window'
    def __init__(self, mc, tunable):
        igemm_v4r1_dynamic_t.__init__(self, mc, tunable)
    def __call__(self, v_wei_os, v_wei_ic, v_wei_iy, v_wei_ix,
                        s_y, s_x, s_wei_stride_c, s_wei_ic, s_wei_iy, s_wei_ix,
                        v_idc, v_idy, v_idx, s_tmp2):
        return '{} {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}'.format(self.name(),
                        v_wei_os, v_wei_ic, v_wei_iy, v_wei_ix,
                        s_y, s_x, s_wei_stride_c, s_wei_ic, s_wei_iy, s_wei_ix,
                        v_idc, v_idy, v_idx, s_tmp2)
    def emit(self):
        self._emit_macro_desc('\n; update v_wei_os, update v_wei_ic, v_wei_iy, v_wei_ix (zero or possitive)')
        with self._emit_macro_indented('.macro {} v_wei_os, v_wei_ic, v_wei_iy, v_wei_ix, s_y, s_x, s_wei_stride_c, s_wei_ic, s_wei_iy, s_wei_ix, v_idc, v_idy, v_idx, s_tmp2'.format(self.name())):
            self._emit('; record old ic, iy, ix')
            self._emit('v_mov_b32 v[\\v_idx], v[\\v_wei_ix]')
            self._emit('v_mov_b32 v[\\v_idy], v[\\v_wei_iy]')
            self._emit('v_mov_b32 v[\\v_idc], v[\\v_wei_ic]')
            self._emit_empty_line()
            self._emit('; update ix, calculate idx, carry-out to iy')
            self._emit('v_add_u32 v[\\v_wei_ix], s[\\s_wei_ix], v[\\v_wei_ix]')
            self._emit('v_cmp_le_u32 vcc, s[\\s_x], v[\\v_wei_ix]')
            self._emit('s_and_saveexec_b64 s[\\s_tmp2:\\s_tmp2+1], vcc')
            self._emit('v_subrev_u32 v[\\v_wei_ix], s[\\s_x], v[\\v_wei_ix]')
            self._emit('v_add_u32 v[\\v_wei_iy], 1, v[\\v_wei_iy]')
            self._emit('s_or_b64 exec, exec, s[\\s_tmp2:\\s_tmp2+1]')
            self._emit('v_sub_i32 v[\\v_idx], v[\\v_wei_ix], v[\\v_idx]')
            self._emit_empty_line()
            self._emit('; update iy, calculate idy, carry-out to ic')
            self._emit('v_add_u32 v[\\v_wei_iy], s[\\s_wei_iy], v[\\v_wei_iy]')
            self._emit('v_cmp_le_u32 vcc, s[\\s_y], v[\\v_wei_iy]')
            self._emit('s_and_saveexec_b64 s[\\s_tmp2:\\s_tmp2+1], vcc')
            self._emit('v_subrev_u32 v[\\v_wei_iy], s[\\s_y], v[\\v_wei_iy]')
            self._emit('v_add_u32 v[\\v_wei_ic], 1, v[\\v_wei_ic]')
            self._emit('s_or_b64 exec, exec, s[\\s_tmp2:\\s_tmp2+1]')
            self._emit('v_sub_i32 v[\\v_idy], v[\\v_wei_iy], v[\\v_idy]')
            self._emit_empty_line()
            self._emit('; update ic, calculate idc, ignore overflow check')
            self._emit('v_add_u32 v[\\v_wei_ic], s[\\s_wei_ic], v[\\v_wei_ic]')
            self._emit('v_sub_u32 v[\\v_idc], v[\\v_wei_ic], v[\\v_idc]')
            self._emit_empty_line()
            self._emit('; calculate offset: idc*(s_y*s_x) + idy*s_x + idx')
            self._emit('; we use i24 as multiplier, for 24bit(-8388607 ~ 8388608) is enough for index')
            self._emit('v_mad_i32_i24 v[\\v_idy], s[\\s_x], v[\\v_idy], v[\\v_idx]')
            self._emit('v_mul_lo_u32 v[\\v_idc], s[\\s_wei_stride_c], v[\\v_idc]')
            self._emit('v_add_i32 v[\\v_idc], v[\\v_idc], v[\\v_idy]')
            self._emit('v_lshl_add_u32 v[\\v_wei_os], v[\\v_idc], 2, v[\\v_wei_os]  ; indeed, idc here must be possitive')

class emit_v4r1_dynamic_kernel_t(igemm_v4r1_dynamic_t):
    class kernel_karg_t(igemm_v4r1_dynamic_t):
        def __init__(self, mc, tunable):
            igemm_v4r1_dynamic_t.__init__(self, mc, tunable)
        def __call__(self):
            with self._deferred_context():
                # Note here, in this implementation, all kernel should be the same
                # TODO: 1x1 is different
                self._emit('; kernarg offset')
                self._emit('.set k_p_in,                0')
                self._emit('.set k_p_wei,               8')
                self._emit('.set k_p_out,               16')
                self._emit('.set k_hi,                  24')
                self._emit('.set k_wi,                  28')
                self._emit('.set k_n,                   32')
                self._emit('.set k_k,                   36')
                self._emit('.set k_c,                   40')
                self._emit('.set k_ho,                  44')
                self._emit('.set k_wo,                  48')
                self._emit('.set k_stride_h,            52')
                self._emit('.set k_stride_w,            56')
                self._emit('.set k_dilation_h,          60')
                self._emit('.set k_dilation_w,          64')
                self._emit('.set k_pad_h,               68')
                self._emit('.set k_pad_w,               72')
                self._emit('.set k_y,                   76')
                self._emit('.set k_x,                   80')
                self._emit('.set k_end,                 84')
                self._emit_empty_line()
            return self._get_deferred(), igemm_next_mul(84, 8)   # TODO: karg alignment
        def get_count(self):
            _, cnt = self()
            return cnt
        def emit(self):
            code, _ = self()
            self._emit(code)

    class kernel_sgpr_t(igemm_v4r1_dynamic_t):
        def __init__(self, mc, tunable):
            igemm_v4r1_dynamic_t.__init__(self, mc, tunable)
        def __call__(self):
            with self._deferred_context():
                self._emit('; sgpr')
                self._emit('.set s_ka,                  0')
                self._emit('.set s_bx,                  2')
                self._emit('.set s_p_in,                4')
                self._emit('.set s_p_wei,               6')
                self._emit('.set s_hi,                  8')
                self._emit('.set s_wi,                  9')
                self._emit('.set s_n,                   10')
                self._emit('.set s_k,                   11')
                self._emit('.set s_c,                   12')
                self._emit('.set s_ho,                  13')
                self._emit('.set s_wo,                  14')
                self._emit('.set s_stride_h,            15')
                self._emit('.set s_stride_w,            16')
                self._emit('.set s_dilation_h,          17')
                self._emit('.set s_dilation_w,          18')
                self._emit('.set s_pad_h,               19')
                self._emit('.set s_pad_w,               20')
                self._emit('.set s_y,                   21')
                self._emit('.set s_x,                   22')
                self._emit('.set s_p_out,               24')
                self._emit('.set s_block_ik,            26')
                self._emit('.set s_block_ib,            27')
                self._emit('.set s_in_stride_c,         28')
                self._emit('.set s_in_stride_n2,        29')
                self._emit('.set s_in_stride_n1,        30')
                self._emit('.set s_in_ic,               31')
                self._emit('.set s_in_iy,               32')
                self._emit('.set s_in_ix,               33')
                self._emit('.set s_wei_stride_c,        34')
                self._emit('.set s_wei_stride_k,        35')
                self._emit('.set s_wei_ic,              s_in_ic     ; weight&input ic, iy, ix from EPerBlock is the same')
                self._emit('.set s_wei_iy,              s_in_iy')
                self._emit('.set s_wei_ix,              s_in_ix')
                self._emit('.set s_out_stride_k0,       36')
                self._emit('.set s_out_stride_k1,       37')
                self._emit('.set s_out_stride_n1,       38')
                self._emit('.set s_out_stride_n2,       39')
                self._emit('.set s_kitr,                0')
                self._emit('.set s_tmp,                 40')
                self._emit('.set s_p_buf_in,            s_p_in      ; 4 sgpr used for MUBUF')
                self._emit('.set s_p_buf_wei,           44')
                self._emit('.set s_p_buf_out,           s_p_out')
                self._emit('.set s_end,                 48')
                self._emit_empty_line()
            return self._get_deferred(), 48
        def get_count(self):
            _, cnt = self()
            return cnt
        def emit(self):
            code, _ = self()
            self._emit(code)

    class kernel_vgpr_t(igemm_v4r1_dynamic_t):
        def __init__(self, mc, tunable):
            igemm_v4r1_dynamic_t.__init__(self, mc, tunable)
        def __call__(self):
            with self._deferred_context():
                vseq = gpr_sequencer_t()
                self._emit('; vgpr')
                self._emit('.set v_c,                   {}'.format(vseq(self.tunable.num_accumulate_c_vgpr)))
                self._emit('.set v_a,                   {}'.format(vseq(self.tunable.num_accumulate_a_vgpr)))
                self._emit('.set v_b,                   {}'.format(vseq(self.tunable.num_accumulate_b_vgpr)))
                self._emit('.set v_gld_a,               {}'.format(vseq(self.tunable.num_global_load_a_vgpr)))
                self._emit('.set v_gld_b,               {}'.format(vseq(self.tunable.num_global_load_b_vgpr)))
                self._emit('.set v_in_os,               {}'.format(vseq(1)))
                self._emit('.set v_wei_os,              {}'.format(vseq(1)))
                self._emit('.set v_sst_a_os,            {}'.format(vseq(1)))
                self._emit('.set v_sst_b_os,            {}'.format(vseq(1)))
                self._emit('.set v_sld_a_os,            {}'.format(vseq(1)))
                self._emit('.set v_sld_b_os,            {}'.format(vseq(1)))
                self._emit('.set v_out_os,              {}'.format(vseq(1)))
                self._emit('.set v_flag,                {}'.format(vseq(1)))
                self._emit('.set v_in_ic,               {}'.format(vseq(1)))
                self._emit('.set v_in_iy,               {}'.format(vseq(1)))
                self._emit('.set v_in_ix,               {}'.format(vseq(1)))
                self._emit('.set v_in_ihi,              {}'.format(vseq(1)))
                self._emit('.set v_in_iwi,              {}'.format(vseq(1)))
                self._emit('.set v_wei_ic,              {}'.format(vseq(1)))
                self._emit('.set v_wei_iy,              {}'.format(vseq(1)))
                self._emit('.set v_wei_ix,              {}'.format(vseq(1)))
                if self.tunable.num_accumulate_c_vgpr <= 6:
                    self._emit('.set v_in_in0,              {}'.format(vseq(1)))
                    self._emit('.set v_in_iho,              {}'.format(vseq(1)))
                    self._emit('.set v_in_iwo,              {}'.format(vseq(1)))
                    self._emit('.set v_in_ie,               {}'.format(vseq(1)))
                else:
                    self._emit('.set v_in_in0,              {}'.format(self.tunable.num_accumulate_c_vgpr - 1))
                    self._emit('.set v_in_iho,              {}'.format(self.tunable.num_accumulate_c_vgpr - 2))
                    self._emit('.set v_in_iwo,              {}'.format(self.tunable.num_accumulate_c_vgpr - 3))
                    self._emit('.set v_in_ie,               {}'.format(self.tunable.num_accumulate_c_vgpr - 4))

                if self.tunable.num_accumulate_c_vgpr <= 9:
                    self._emit('.set v_in_in1,              {}'.format(vseq(1)))
                    self._emit('.set v_in_ib,               {}'.format(vseq(1)))
                    self._emit('.set v_in_in2,              {}'.format(vseq(1)))
                else:
                    self._emit('.set v_in_in1,              {}'.format(self.tunable.num_accumulate_c_vgpr - 5))
                    self._emit('.set v_in_ib,               {}'.format(self.tunable.num_accumulate_c_vgpr - 6))
                    self._emit('.set v_in_in2,              {}'.format(self.tunable.num_accumulate_c_vgpr - 7))

                if self.tunable.num_accumulate_c_vgpr <= 12:
                    self._emit('.set v_wei_ie,              {}'.format(vseq(1)))
                    self._emit('.set v_wei_ik,              {}'.format(vseq(1)))
                    self._emit('.set v_out_ik0,             {}'.format(vseq(1)))
                else:
                    self._emit('.set v_wei_ie,              {}'.format(self.tunable.num_accumulate_c_vgpr - 8))
                    self._emit('.set v_wei_ik,              {}'.format(self.tunable.num_accumulate_c_vgpr - 9))
                    self._emit('.set v_out_ik0,             {}'.format(self.tunable.num_accumulate_c_vgpr - 10))

                if self.tunable.num_accumulate_c_vgpr <= 16:
                    self._emit('.set v_out_ik1,             {}'.format(vseq(1)))
                    self._emit('.set v_out_ib,              {}'.format(vseq(1)))
                    self._emit('.set v_gemm_in,             {}'.format(vseq(1)))
                    self._emit('.set v_gemm_im,             {}'.format(vseq(1)))
                else:
                    self._emit('.set v_out_ik1,             {}'.format(self.tunable.num_accumulate_c_vgpr - 11))
                    self._emit('.set v_out_ib,              {}'.format(self.tunable.num_accumulate_c_vgpr - 12))
                    self._emit('.set v_gemm_in,             {}'.format(self.tunable.num_accumulate_c_vgpr - 13))
                    self._emit('.set v_gemm_im,             {}'.format(self.tunable.num_accumulate_c_vgpr - 14))

                self._emit('.set v_idc,                 {}'.format(vseq(1)))
                self._emit('.set v_idy,                 {}'.format(vseq(1)))
                self._emit('.set v_idx,                 {}'.format(vseq(1)))
                if self.tunable.num_accumulate_c_vgpr <= 24:
                    self._emit('.set v_tmp,                 {}'.format(vseq(6)))
                    self._emit('.set v_end,                 {}'.format(vseq()))
                else:
                    self._emit('.set v_tmp,                 {}'.format(self.tunable.num_accumulate_c_vgpr - 22))
                    self._emit('.set v_end,                 {}'.format(vseq()))
                self._emit_empty_line()
            return self._get_deferred(), vseq()
        def get_count(self):
            _, cnt = self()
            return cnt
        def emit(self):
            code, _ = self()
            self._emit(code) 

    def name(self):
        return igemm_encode_v4r1_kernel_name(self.tunable)
    def __init__(self, mc, tunable, emit_unified_define = False):
        igemm_v4r1_dynamic_t.__init__(self, mc, tunable)
        self.emit_unified_define = emit_unified_define
        self.kernel_karg = self.kernel_karg_t(mc, tunable)
        self.kernel_sgpr = self.kernel_sgpr_t(mc, tunable)
        self.kernel_vgpr = self.kernel_vgpr_t(mc, tunable)

    def emit_kernel_define_karg(self):
        # Note here, in this implementation, all kernel should be the same
        # TODO: 1x1 is different
        self._emit('; kernarg offset')
        self._emit('.set k_p_in,                0')
        self._emit('.set k_p_wei,               8')
        self._emit('.set k_p_out,               16')
        self._emit('.set k_hi,                  24')
        self._emit('.set k_wi,                  28')
        self._emit('.set k_n,                   32')
        self._emit('.set k_k,                   36')
        self._emit('.set k_c,                   40')
        self._emit('.set k_ho,                  44')
        self._emit('.set k_wo,                  48')
        self._emit('.set k_stride_h,            52')
        self._emit('.set k_stride_w,            56')
        self._emit('.set k_dilation_h,          60')
        self._emit('.set k_dilation_w,          64')
        self._emit('.set k_pad_h,               68')
        self._emit('.set k_pad_w,               72')
        self._emit('.set k_y,                   76')
        self._emit('.set k_x,                   80')
        self._emit('.set k_end,                 84')
        self._emit_empty_line()


    def emit_kernel_define_sgpr(self):
        # Note here, in this implementation, all kernel should be the same
        # TODO: 1x1 is different
        self._emit('; sgpr')
        self._emit('.set s_ka,                  0')
        self._emit('.set s_bx,                  2')
        self._emit('.set s_p_in,                4')
        self._emit('.set s_p_wei,               6')
        self._emit('.set s_hi,                  8')
        self._emit('.set s_wi,                  9')
        self._emit('.set s_n,                   10')
        self._emit('.set s_k,                   11')
        self._emit('.set s_c,                   12')
        self._emit('.set s_ho,                  13')
        self._emit('.set s_wo,                  14')
        self._emit('.set s_stride_h,            15')
        self._emit('.set s_stride_w,            16')
        self._emit('.set s_dilation_h,          17')
        self._emit('.set s_dilation_w,          18')
        self._emit('.set s_pad_h,               19')
        self._emit('.set s_pad_w,               20')
        self._emit('.set s_y,                   21')
        self._emit('.set s_x,                   22')
        self._emit('.set s_p_out,               24')
        self._emit('.set s_block_ik,            26')
        self._emit('.set s_block_ib,            27')
        self._emit('.set s_in_stride_c,         28')
        self._emit('.set s_in_stride_n2,        29')
        self._emit('.set s_in_stride_n1,        30')
        self._emit('.set s_in_ic,               31')
        self._emit('.set s_in_iy,               32')
        self._emit('.set s_in_ix,               33')
        self._emit('.set s_wei_stride_c,        34')
        self._emit('.set s_wei_stride_k,        35')
        self._emit('.set s_wei_ic,              s_in_ic     ; weight&input ic, iy, ix from EPerBlock is the same')
        self._emit('.set s_wei_iy,              s_in_iy')
        self._emit('.set s_wei_ix,              s_in_ix')
        self._emit('.set s_out_stride_k0,       36')
        self._emit('.set s_out_stride_k1,       37')
        self._emit('.set s_out_stride_n1,       38')
        self._emit('.set s_out_stride_n2,       39')
        self._emit('.set s_kitr,                0')
        self._emit('.set s_tmp,                 40')
        self._emit('.set s_p_buf_in,            s_p_in      ; 4 sgpr used for MUBUF')
        self._emit('.set s_p_buf_wei,           44')
        self._emit('.set s_p_buf_out,           s_p_out')
        self._emit('.set s_end,                 48')
        self._emit_empty_line()

    def emit_kernel_define_vgpr(self):
        vseq = gpr_sequencer_t()
        self._emit('; vgpr')
        self._emit('.set v_c,                   {}'.format(vseq(self.tunable.num_accumulate_c_vgpr)))
        self._emit('.set v_a,                   {}'.format(vseq(self.tunable.num_accumulate_a_vgpr)))
        self._emit('.set v_b,                   {}'.format(vseq(self.tunable.num_accumulate_b_vgpr)))
        self._emit('.set v_gld_a,               {}'.format(vseq(self.tunable.num_global_load_a_vgpr)))
        self._emit('.set v_gld_b,               {}'.format(vseq(self.tunable.num_global_load_b_vgpr)))
        self._emit('.set v_in_os,               {}'.format(vseq(1)))
        self._emit('.set v_wei_os,              {}'.format(vseq(1)))
        self._emit('.set v_sst_a_os,            {}'.format(vseq(1)))
        self._emit('.set v_sst_b_os,            {}'.format(vseq(1)))
        self._emit('.set v_sld_a_os,            {}'.format(vseq(1)))
        self._emit('.set v_sld_b_os,            {}'.format(vseq(1)))
        self._emit('.set v_out_os,              {}'.format(vseq(1)))
        self._emit('.set v_flag,                {}'.format(vseq(1)))
        self._emit('.set v_in_ic,               {}'.format(vseq(1)))
        self._emit('.set v_in_iy,               {}'.format(vseq(1)))
        self._emit('.set v_in_ix,               {}'.format(vseq(1)))
        self._emit('.set v_in_ihi,              {}'.format(vseq(1)))
        self._emit('.set v_in_iwi,              {}'.format(vseq(1)))
        self._emit('.set v_wei_ic,              {}'.format(vseq(1)))
        self._emit('.set v_wei_iy,              {}'.format(vseq(1)))
        self._emit('.set v_wei_ix,              {}'.format(vseq(1)))
        if self.tunable.num_accumulate_c_vgpr <= 6:
            self._emit('.set v_in_in0,              {}'.format(vseq(1)))
            self._emit('.set v_in_iho,              {}'.format(vseq(1)))
            self._emit('.set v_in_iwo,              {}'.format(vseq(1)))
            self._emit('.set v_in_ie,               {}'.format(vseq(1)))
        else:
            self._emit('.set v_in_in0,              {}'.format(self.tunable.num_accumulate_c_vgpr - 1))
            self._emit('.set v_in_iho,              {}'.format(self.tunable.num_accumulate_c_vgpr - 2))
            self._emit('.set v_in_iwo,              {}'.format(self.tunable.num_accumulate_c_vgpr - 3))
            self._emit('.set v_in_ie,               {}'.format(self.tunable.num_accumulate_c_vgpr - 4))

        if self.tunable.num_accumulate_c_vgpr <= 9:
            self._emit('.set v_in_in1,              {}'.format(vseq(1)))
            self._emit('.set v_in_ib,               {}'.format(vseq(1)))
            self._emit('.set v_in_in2,              {}'.format(vseq(1)))
        else:
            self._emit('.set v_in_in1,              {}'.format(self.tunable.num_accumulate_c_vgpr - 5))
            self._emit('.set v_in_ib,               {}'.format(self.tunable.num_accumulate_c_vgpr - 6))
            self._emit('.set v_in_in2,              {}'.format(self.tunable.num_accumulate_c_vgpr - 7))

        if self.tunable.num_accumulate_c_vgpr <= 12:
            self._emit('.set v_wei_ie,              {}'.format(vseq(1)))
            self._emit('.set v_wei_ik,              {}'.format(vseq(1)))
            self._emit('.set v_out_ik0,             {}'.format(vseq(1)))
        else:
            self._emit('.set v_wei_ie,              {}'.format(self.tunable.num_accumulate_c_vgpr - 8))
            self._emit('.set v_wei_ik,              {}'.format(self.tunable.num_accumulate_c_vgpr - 9))
            self._emit('.set v_out_ik0,             {}'.format(self.tunable.num_accumulate_c_vgpr - 10))

        if self.tunable.num_accumulate_c_vgpr <= 16:
            self._emit('.set v_out_ik1,             {}'.format(vseq(1)))
            self._emit('.set v_out_ib,              {}'.format(vseq(1)))
            self._emit('.set v_gemm_in,             {}'.format(vseq(1)))
            self._emit('.set v_gemm_im,             {}'.format(vseq(1)))
        else:
            self._emit('.set v_out_ik1,             {}'.format(self.tunable.num_accumulate_c_vgpr - 11))
            self._emit('.set v_out_ib,              {}'.format(self.tunable.num_accumulate_c_vgpr - 12))
            self._emit('.set v_gemm_in,             {}'.format(self.tunable.num_accumulate_c_vgpr - 13))
            self._emit('.set v_gemm_im,             {}'.format(self.tunable.num_accumulate_c_vgpr - 14))

        self._emit('.set v_idc,                 {}'.format(vseq(1)))
        self._emit('.set v_idy,                 {}'.format(vseq(1)))
        self._emit('.set v_idx,                 {}'.format(vseq(1)))
        if self.tunable.num_accumulate_c_vgpr <= 24:
            self._emit('.set v_tmp,                 {}'.format(vseq(6)))
            self._emit('.set v_end,                 {}'.format(vseq()))
        else:
            self._emit('.set v_tmp,                 {}'.format(self.tunable.num_accumulate_c_vgpr - 22))
            self._emit('.set v_end,                 {}'.format(vseq()))
        self._emit_empty_line()
        # TODO out of occupancy check

    def get_kernel_code(self):
        kernel_code = amdgpu_kernel_code_t({
                'enable_sgpr_kernarg_segment_ptr'   :   1,
                'enable_sgpr_workgroup_id_x'        :   1,
                'enable_vgpr_workitem_id'           :   0,
                'workgroup_group_segment_byte_size' :   self.tunable.byte_lds_total,
                'kernarg_segment_byte_size'         :   self.kernel_karg.get_count(),
                'wavefront_sgpr_count'              :   self.kernel_sgpr.get_count() + 2*3,
                'workitem_vgpr_count'               :   self.kernel_vgpr.get_count()
                })
        return kernel_code

    def get_kernel_info(self):
        kernel_code = self.get_kernel_code()
        kernel_info = amdgpu_kernel_info_t(kernel_code, self.name(), v4r1_dynamic_get_block_size(self.tunable))
        return kernel_info

    def emit_kernel_amd_kernel_code_t(self):
        emit_amd_kernel_code_t(self.mc, self.get_kernel_info()).emit()

    def emit_kernel_header(self):
        kernel_name = self.name()
        self._emit('.text')
        if self.mc.arch_config.code_object == AMDGPU_CODEOBJECT_V3:
            self._emit('.globl {}'.format(kernel_name))
        self._emit('.p2align 8')
        if self.mc.arch_config.code_object == AMDGPU_CODEOBJECT_V3:
            self._emit('.type {},@function'.format(kernel_name))
        if self.mc.arch_config.code_object == AMDGPU_CODEOBJECT_V2:
            self._emit('.amdgpu_hsa_kernel {}'.format(kernel_name))
        self._emit('{}:'.format(kernel_name))
    def emit_kernel_end(self):
        self._emit('s_endpgm')
    def emit_kernel_footer(self):
        self._emit_empty_line()

    def emit_kernel_prepare_phase(self):
        in_load = emit_in_load_e_n1_b_n2_t(self.mc, self.tunable)
        wei_load = emit_wei_load_e_k_t(self.mc, self.tunable)
        '''
        star from enter kernel, to right before FMA body
        '''
        self._emit('s_load_dwordx4  s[s_p_in:s_p_in+3],         s[s_ka:s_ka+1],     0+k_p_in')
        self._emit('s_load_dwordx2  s[s_p_out:s_p_out+1],       s[s_ka:s_ka+1],     0+k_p_out')
        self._emit('s_load_dwordx8  s[s_hi:s_hi+7],             s[s_ka:s_ka+1],     0+k_hi')
        self._emit('s_load_dwordx4  s[s_stride_w:s_stride_w+3], s[s_ka:s_ka+1],     0+k_stride_w')
        self._emit('s_load_dwordx2  s[s_pad_w:s_pad_w+1],       s[s_ka:s_ka+1],     0+k_pad_w')
        self._emit('s_load_dword    s[s_x],                     s[s_ka:s_ka+1],     0+k_x')
        self._emit_empty_line()

        # calculate cluster pattern of input, -> ib, in2, in1, ie
        self._emit('; in e_n1_b_n2 cluster_lengths:{{{},{},{},{}}}, sub_lengths:{{{},{},{},{}}}, order:{{0,1,3,2}}'.format(
                    self.tunable.in_block_copy_cluster_lengths_e, self.tunable.in_block_copy_cluster_lengths_n1,self.tunable.in_block_copy_cluster_lengths_b,self.tunable.in_block_copy_cluster_lengths_n2,
                    self.tunable.in_block_copy_sub_lengths_e,self.tunable.in_block_copy_sub_lengths_n1,self.tunable.in_block_copy_sub_lengths_b,self.tunable.in_block_copy_sub_lengths_n2))
        # ->b
        if self.tunable.in_block_copy_cluster_lengths_b != 1:
            self._emit('v_and_b32 v[v_in_ib], {}, v0'.format(self.tunable.in_block_copy_cluster_lengths_b - 1))
            self._emit('v_lshrrev_b32 v[v_tmp], {}, v0'.format(igemm_log2(self.tunable.in_block_copy_cluster_lengths_b)))
            if self.tunable.in_block_copy_sub_lengths_b != 1:
                self._emit('v_lshlrev_b32 v[v_in_ib], {}, v[v_in_ib]'.format(igemm_log2(self.tunable.in_block_copy_sub_lengths_b)))
        else:
            self._emit('v_mov_b32 v[v_in_ib], 0')
            self._emit('v_mov_b32 v[v_tmp], v0')

        # ->n2
        if self.tunable.in_block_copy_cluster_lengths_n2 != 1:
            self._emit('v_and_b32 v[v_in_in2], {}, v[v_tmp]'.format(self.tunable.in_block_copy_cluster_lengths_n2 - 1))
            self._emit('v_lshrrev_b32 v[v_tmp], {}, v[v_tmp]'.format(igemm_log2(self.tunable.in_block_copy_cluster_lengths_n2)))
            if self.tunable.in_block_copy_sub_lengths_n2 != 1:
                self._emit('v_lshlrev_b32 v[v_in_in2], {}, v[v_in_in2]'.format(igemm_log2(self.tunable.in_block_copy_sub_lengths_n2)))
        else:
            self._emit('v_mov_b32 v[v_in_in2], 0')

        # ->n1
        if self.tunable.in_block_copy_cluster_lengths_n1 != 1:
            self._emit('v_and_b32 v[v_in_in1], {}, v[v_tmp]'.format(self.tunable.in_block_copy_cluster_lengths_n1 - 1))
            self._emit('v_lshrrev_b32 v[v_tmp], {}, v[v_tmp]'.format(igemm_log2(self.tunable.in_block_copy_cluster_lengths_n1)))
            if self.tunable.in_block_copy_sub_lengths_n1 != 1:
                self._emit('v_lshlrev_b32 v[v_in_in1], {}, v[v_in_in1]'.format(igemm_log2(self.tunable.in_block_copy_sub_lengths_n1)))
        else:
                self._emit('v_mov_b32 v[v_in_in1], 0')

        # -> e
        if self.tunable.in_block_copy_cluster_lengths_e != 1:
            self._emit('v_and_b32 v[v_in_ie], {}, v[v_tmp]'.format(self.tunable.in_block_copy_cluster_lengths_e - 1))
            if self.tunable.in_block_copy_sub_lengths_e != 1:
                self._emit('v_lshlrev_b32 v[v_in_ie], {}, v[v_in_ie]'.format(igemm_log2(self.tunable.in_block_copy_sub_lengths_e)))
        else:
            self._emit('v_mov_b32 v[v_in_ie], 0')

        # calculate cluster pattern of weight, -> ie, ib
        self._emit('; wei e_k cluster_lengths:{{{},{}}}, sub_lengths:{{{},{}}}, order:{{1,0}}'.format(
                self.tunable.wei_block_copy_cluster_lengths_e, self.tunable.wei_block_copy_cluster_lengths_k,
                self.tunable.wei_block_copy_sub_lengths_e, self.tunable.wei_block_copy_sub_lengths_k))
        # -> e
        if self.tunable.wei_block_copy_cluster_lengths_e != 1:
            self._emit('v_and_b32 v[v_wei_ie], {}, v0'.format(self.tunable.wei_block_copy_cluster_lengths_e - 1))
            self._emit('v_lshrrev_b32 v[v_tmp], {}, v0'.format(igemm_log2(self.tunable.wei_block_copy_cluster_lengths_e)))
            if self.tunable.wei_block_copy_sub_lengths_e != 1:
                self._emit('v_lshlrev_b32 v[v_wei_ie], {}, v[v_wei_ie]'.format(igemm_log2(self.tunable.wei_block_copy_sub_lengths_e)))
        else:
            self._emit('v_mov_b32 v[v_wei_ie], 0')
            self._emit('v_mov_b32 v[v_tmp], v0')
        # ->k
        if self.tunable.wei_block_copy_cluster_lengths_k != 1:
            self._emit('v_and_b32 v[v_wei_ik], {}, v[v_tmp]'.format(self.tunable.wei_block_copy_cluster_lengths_k - 1))
            if self.tunable.wei_block_copy_sub_lengths_k != 1:
                self._emit('v_lshlrev_b32 v[v_wei_ik], {}, v[v_wei_ik]'.format(igemm_log2(self.tunable.wei_block_copy_sub_lengths_k)))
        else:
            self._emit('v_mov_b32 v[v_wei_ik], 0')

        self._emit('s_waitcnt lgkmcnt(0)')
        self._emit_empty_line()
        self._emit('; calculate index')
        self._emit('s_mul_i32 s[s_out_stride_k1], s[s_ho], s[s_wo]')
        self._emit('s_lshl_b32 s[s_out_stride_k0], s[s_out_stride_k1], {}'.format(
                                                        igemm_log2(self.tunable.gemm_m_per_thread_subc)+
                                                        igemm_log2(self.tunable.gemm_m_level0_cluster)+
                                                        igemm_log2(self.tunable.gemm_m_level1_cluster)) )
        self._emit('s_mul_i32 s[s_out_stride_n2], s[s_k], s[s_out_stride_k1]')
        self._emit('s_lshl_b32 s[s_out_stride_n1], s[s_out_stride_n2], {}'.format(igemm_log2(self.tunable.gemm_n_per_thread_subc)))
        self._emit('s_mul_i32 s[s_in_stride_c], s[s_hi], s[s_wi]')
        self._emit('s_mul_i32 s[s_in_stride_n2], s[s_c], s[s_in_stride_c]')
        self._emit('s_mul_i32 s[s_wei_stride_c], s[s_y], s[s_x]')
        self._emit('s_mul_i32 s[s_wei_stride_k], s[s_c], s[s_wei_stride_c]')
        self._emit('s_mov_b64 s[s_p_buf_wei:s_p_buf_wei+1], s[s_p_wei:s_p_wei+1]')
        self._emit('s_mov_b32 s[s_p_buf_in+2], 0xffffffff')
        self._emit('s_mov_b32 s[s_p_buf_in+3], 0x27000')
        self._emit('s_mov_b32 s[s_p_buf_wei+2], 0xffffffff')
        self._emit('s_mov_b32 s[s_p_buf_wei+3], 0x27000')
        self._emit_empty_line()
        self._emit('; block k, b index on global')
        # N0 = N / (N1 * N2)
        self._emit('s_lshr_b32 s[s_tmp], s[s_n], {}'.format(
                            igemm_log2(self.tunable.gemm_n_repeat) + igemm_log2(self.tunable.gemm_n_per_thread_subc)))
        # B = N0 * Ho * Wo')
        self._emit('s_mul_i32 s[s_tmp+1], s[s_out_stride_k1], s[s_tmp]')
        # BBlockWork = B / BPerBlock')
        self._emit('s_lshr_b32 s[0], s[s_tmp+1], {}'.format(igemm_log2(self.tunable.b_per_block)) )
        # KBlockID, BBlockID')
        self._emit('.v_u32_div_ss v_tmp+5, s_bx, 0, v_tmp, s_tmp')
        self._emit('v_readfirstlane_b32 s[s_tmp], v[v_tmp+5]')
        self._emit('s_mul_i32 s[s_tmp+2], s[s_tmp], s[0]')
        self._emit('s_sub_i32 s[s_tmp+1], s[s_bx], s[s_tmp+2]')
        self._emit('s_lshl_b32 s[s_block_ik], s[s_tmp], {}'.format(igemm_log2(self.tunable.k_per_block)))
        self._emit('s_lshl_b32 s[s_block_ib], s[s_tmp+1], {}'.format(igemm_log2(self.tunable.b_per_block)))
        self._emit_empty_line()
        '''
        input

        from tensor transform, input tensor is divided into folowing dim iterator (current position):
                    ic, iy, ix, in0, iho, iwo
        from iy, ix, iho, iwo, can get the input width, height iterator:
                    -> ihi = iho * s_stride_h + iy * s_dilation_h - s_pad_h
                    -> iwi = iwo * s_stride_w + ix * s_dilation_w - s_pad_w
        hence, can calculate input offset from above iterator:
            in_offset: in0 * (8*s_c*s_hi*s_wi) + ic * (s_hi*s_wi) + ihi * s_wi + iwi

        for each MoveSliceWindow, need move <EPerBlock, 0, 0, 0>, the diff can be divided into:
                    dc, dy, dx      (e=c*y*x, can all be sgpr)
        new c, y, x iterator:
                    ix_new = (ix+dx)%s_x
                    iy_new = (iy+dy+(ix+dx)/s_x)%s_y
                    ic_new = (ic+dc+(iy+dy+(ix+dx)/s_x)/s_y)   (no check overflow)
        hence the iterator diff (may have negative):
                    idx = ix_new - ix
                    idy = iy_new - iy
                    idc = ic_new - ic

        hence for offset, the diff should be:
        in_offset_diff: idc*(s_hi*s_wi) + idy*s_dilation_h*s_wi + idx*s_dilation_w

        note here:
            1) idc can only be 0 or possitive, idx, idy can be negative, possitive, 0
            2) ic, iy, ix need be updated to ic_new, iy_new, ix_new
        '''
        self._emit('; calculate input transform')
        self._emit('; e_n1_b_n2:b, transform: b -> n0*ho*wo')
        self._emit('v_add_u32 v[v_tmp+4], s[s_block_ib], v[v_in_ib]')
        self._emit('.v_u32_div_vs v_in_in0, v_tmp+4, s_out_stride_k1, v_tmp, s_tmp')
        self._emit('v_mul_lo_u32 v[v_tmp], s[s_out_stride_k1], v[v_in_in0]')
        self._emit('v_sub_u32 v[v_tmp+4], v[v_tmp+4], v[v_tmp]')
        self._emit('.v_u32_div_vs v_in_iho, v_tmp+4, s_wo, v_tmp, s_tmp')
        self._emit('v_mul_lo_u32 v[v_tmp], s[s_wo], v[v_in_iho]')
        self._emit('v_sub_u32 v[v_in_iwo], v[v_tmp+4], v[v_tmp]')
        self._emit_empty_line()
        self._emit('; e_n1_b_n2:e')
        self._emit(';   1) transform e -> c*y*x')
        self._emit('.v_u32_div_vs v_in_ic, v_in_ie, s_wei_stride_c, v_tmp, s_tmp')
        self._emit('v_mul_lo_u32 v[v_tmp], s[s_wei_stride_c], v[v_in_ic]')
        self._emit('v_sub_u32 v[v_tmp+4], v[v_in_ie], v[v_tmp]')
        self._emit('.v_u32_div_vs v_in_iy, v_tmp+4, s_x, v_tmp, s_tmp')
        self._emit('v_mul_lo_u32 v[v_tmp], s[s_x], v[v_in_iy]')
        self._emit('v_sub_u32 v[v_in_ix], v[v_tmp+4], v[v_tmp]')
        self._emit_empty_line()
        self._emit(';   2) transform iho, iwo, iy, ix -> hip, wip')
        self._emit('v_mul_lo_u32 v[v_tmp], s[s_stride_h], v[v_in_iho]')
        self._emit('v_mul_lo_u32 v[v_tmp+1], s[s_stride_w], v[v_in_iwo]')
        self._emit('v_mul_lo_u32 v[v_tmp+2], s[s_dilation_h], v[v_in_iy]')
        self._emit('v_mul_lo_u32 v[v_tmp+3], s[s_dilation_w], v[v_in_ix]')
        self._emit_empty_line()
        self._emit(';   3) transform hip, wip -> hi, wi')
        self._emit('v_add_u32 v[v_tmp], v[v_tmp], v[v_tmp+2]')
        self._emit('v_add_u32 v[v_tmp+1], v[v_tmp+1], v[v_tmp+3]')
        self._emit('v_sub_i32 v[v_in_ihi], v[v_tmp], s[s_pad_h]')
        self._emit('v_sub_i32 v[v_in_iwi], v[v_tmp+1], s[s_pad_w]')
        self._emit_empty_line()
        self._emit('; set input flag')
        self._emit('.v_in_set_flag v_flag, v_in_ihi, v_in_iwi, s_hi, s_wi, s_tmp')
        self._emit_empty_line()
        self._emit('; in offset: from ihi, iwi, ic, in, calculate v_in_os')
        self._emit('v_mul_lo_u32 v[v_tmp], s[s_wi], v[v_in_ihi]')
        self._emit('v_add_u32 v[v_tmp], v[v_tmp], v[v_in_iwi]')
        self._emit('v_mul_lo_u32 v[v_tmp+1], s[s_in_stride_c], v[v_in_ic]')
        self._emit('v_add_u32 v[v_tmp], v[v_tmp], v[v_tmp+1]')
        self._emit('v_lshl_add_u32 v[v_tmp+1], v[v_in_in0], {}, v[v_in_in2]'.format(igemm_log2(self.tunable.gemm_n_repeat) + igemm_log2(self.tunable.gemm_n_per_thread_subc)))
        self._emit('v_lshl_add_u32 v[v_tmp+1], v[v_in_in1], {}, v[v_tmp+1]'.format(igemm_log2(self.tunable.gemm_n_per_thread_subc)))
        self._emit('v_mul_lo_u32 v[v_tmp+1], s[s_in_stride_n2], v[v_tmp+1]')
        self._emit('v_add_lshl_u32 v[v_in_os], v[v_tmp], v[v_tmp+1], 2')
        self._emit_empty_line()
        self._emit('s_lshl_b32 s[s_in_stride_n2], s[s_in_stride_n2], 2')
        self._emit('s_lshl_b32 s[s_in_stride_n1], s[s_in_stride_n2], {}'.format(igemm_log2(self.tunable.gemm_n_per_thread_subc)))

        #; load input from global
        self._emit('; load input from global')
        self._emit(in_load('v_gld_b', 's_p_buf_in', 'v_in_os', 's_in_stride_n1', 's_in_stride_n2', 'v_flag', 's_tmp'))
        self._emit_empty_line()

        self._emit('; calculate SliceWindow e=c*y*x. this is same for both input/weight')
        self._emit('s_mov_b32 s[1], {}'.format(self.tunable.e_per_block))
        self._emit('.v_u32_div_ss v_tmp+4, 1, s_wei_stride_c, v_tmp, s_tmp')
        self._emit('v_readfirstlane_b32 s[s_in_ic], v[v_tmp+4]')
        self._emit('s_mul_i32 s[s_tmp], s[s_wei_stride_c], s[s_in_ic]')
        self._emit('s_sub_i32 s[1], s[1], s[s_tmp]')
        self._emit('.v_u32_div_ss v_tmp+4, 1, s_x, v_tmp, s_tmp')
        self._emit('v_readfirstlane_b32 s[s_in_iy], v[v_tmp+4]')
        self._emit('s_mul_i32 s[s_tmp], s[s_x], s[s_in_iy]')
        self._emit('s_sub_i32 s[s_in_ix], s[1], s[s_tmp]')
        self._emit_empty_line()
        self._emit('; c thread mapping')
        self._emit('v_and_b32 v[v_tmp+4], {}, v0'.format((self.tunable.gemm_m_level0_cluster * self.tunable.gemm_n_level0_cluster) - 1))
        self._emit('v_and_b32 v[v_tmp], {}, v[v_tmp+4]'.format(self.tunable.gemm_n_level0_cluster - 1))
        self._emit('v_lshrrev_b32 v[v_tmp+1], {}, v[v_tmp+4]'.format(igemm_log2(self.tunable.gemm_n_level0_cluster)))
        self._emit_empty_line()
        self._emit('v_lshrrev_b32 v[v_tmp+4], {}, v0'.format(igemm_log2(self.tunable.gemm_m_level0_cluster) + igemm_log2(self.tunable.gemm_n_level0_cluster)))
        self._emit('v_and_b32 v[v_tmp+2], {}, v[v_tmp+4]'.format(self.tunable.gemm_n_level1_cluster - 1))
        self._emit('v_lshrrev_b32 v[v_tmp+3], {}, v[v_tmp+4]'.format(igemm_log2(self.tunable.gemm_n_level1_cluster)))
        self._emit_empty_line()
        self._emit('v_lshl_or_b32 v[v_gemm_in], v[v_tmp+2], {}, v[v_tmp]               ; in'.format(igemm_log2(self.tunable.gemm_n_level0_cluster)))
        self._emit('v_lshl_or_b32 v[v_gemm_im], v[v_tmp+3], {}, v[v_tmp+1]             ; im'.format(igemm_log2(self.tunable.gemm_m_level0_cluster)))
        self._emit('v_lshlrev_b32 v[v_sld_b_os], {}, v[v_gemm_in]'.format(igemm_log2(self.tunable.gemm_n_per_thread_subc)+2))
        self._emit('v_lshlrev_b32 v[v_sld_a_os], {}, v[v_gemm_im]'.format(igemm_log2(self.tunable.gemm_m_per_thread_subc)+2))
        self._emit('v_add_u32 v[v_sld_a_os], {}, v[v_sld_a_os]'.format(self.tunable.byte_lds_b_np2))
        self._emit_empty_line()
        '''
        weight
    
        from tensor transform, weight tensor is divided into following dim iterator (current position):
                    ic, iy, ix, ik
        hence, can calculate weight offset from above iterator:
            in_offset: ik*(s_c*s_y*s_x) + ic*(s_y*s_x) + iy*s_x + ix
    
        for each MoveSliceWindow, need move <EPerBlock, 0>, the diff can be divided into:
                    dc, dy, dx      (e=c*y*x, can all be sgpr)
        new c, y, x iterator:
                    ix_new = (ix+dx)%s_x
                    iy_new = (iy+dy+(ix+dx)/s_x)%s_y
                    ic_new = (ic+dc+(iy+dy+(ix+dx)/s_x)/s_y)   (no check overflow)
        hence the iterator diff (may have negative):
                    idx = ix_new - ix
                    idy = iy_new - iy
                    idc = ic_new - ic
    
        hence for offset, the diff should be:
        wei_offset_diff: idc*(s_y*s_x) + idy*s_x + idx
    
        note here:
            1) idx can only be 0 or possitive, idx, idy can be negative, possitive, 0
            2) ic, iy, ix need be updated to ic_new, iy_new, ix_new
        '''
        self._emit('; calculate weight transform')
        self._emit('; e_k: e->c*y*x')
        self._emit('.v_u32_div_vs v_wei_ic, v_wei_ie, s_wei_stride_c, v_tmp, s_tmp')
        self._emit('v_mul_lo_u32 v[v_tmp], s[s_wei_stride_c], v[v_wei_ic]')
        self._emit('v_sub_u32 v[v_tmp+4], v[v_wei_ie], v[v_tmp]')
        self._emit('.v_u32_div_vs v_wei_iy, v_tmp+4, s_x, v_tmp, s_tmp')
        self._emit('v_mul_lo_u32 v[v_tmp], s[s_x], v[v_wei_iy]')
        self._emit('v_sub_u32 v[v_wei_ix], v[v_tmp+4], v[v_tmp]')
        self._emit_empty_line()
        self._emit('; wei offset: from ic, iy, ix, ik, calculate v_wei_os')
        self._emit('v_mul_lo_u32 v[v_tmp], s[s_wei_stride_c], v[v_wei_ic]')
        self._emit('v_mul_lo_u32 v[v_tmp+1], s[s_x], v[v_wei_iy]')
        self._emit('v_add3_u32 v[v_wei_os], v[v_tmp], v[v_tmp+1], v[v_wei_ix]')
        self._emit_empty_line()
        self._emit('v_add_u32 v[v_tmp], s[s_block_ik], v[v_wei_ik]')
        self._emit('v_mul_lo_u32 v[v_tmp+1], s[s_wei_stride_k], v[v_tmp]')
        self._emit('v_add_lshl_u32 v[v_wei_os], v[v_wei_os], v[v_tmp+1], 2')
        self._emit_empty_line()
        self._emit('s_lshl_b32 s[s_wei_stride_k], s[s_wei_stride_k], 2')
        self._emit_empty_line()
        self._emit('; load wei from global')
        self._emit(wei_load('v_gld_a', 's_p_buf_wei', 'v_wei_os', 's_wei_stride_k', 's_tmp'))
        self._emit_empty_line()
        '''
        ; out diff
        ; k_thread_data_on_global = k_block_data_on_global + c_thread_mtx_on_block.row
        ; k_thread_data_on_global / K1
        ; k_thread_data_on_global % K1
        '''
        self._emit('; calculate out index ik0, ik1, ib')
        self._emit('v_lshlrev_b32 v[v_tmp+1], {}, v[v_gemm_im]'.format(igemm_log2(self.tunable.gemm_m_per_thread_subc)))
        self._emit('v_add_u32 v[v_tmp], s[s_block_ik], v[v_tmp+1]')
        self._emit('v_lshrrev_b32 v[v_out_ik0], {}, v[v_tmp]'.format(igemm_log2(self.tunable.gemm_m_per_thread_subc) + igemm_log2(self.tunable.gemm_m_level0_cluster) + igemm_log2(self.tunable.gemm_m_level1_cluster)))
        self._emit('v_and_b32 v[v_out_ik1], {}, v[v_tmp]'.format((self.tunable.gemm_m_per_thread_subc * self.tunable.gemm_m_level0_cluster * self.tunable.gemm_m_level1_cluster) - 1))
        self._emit_empty_line()
        # b_thread_data_on_global = b_block_data_on_global + c_thread_mtx_on_block.col / N2
        self._emit('v_add_u32 v[v_out_ib], s[s_block_ib], v[v_gemm_in]')
        self._emit('.v_u32_div_vs v_tmp+4, v_out_ib, s_out_stride_k1, v_tmp, s_tmp')
        self._emit('v_mul_lo_u32 v[v_tmp+1], s[s_out_stride_k1], v[v_tmp+4]')
        self._emit('v_sub_u32 v[v_tmp+5], v[v_out_ib], v[v_tmp+1]')
        self._emit('.v_u32_div_vs v_tmp+6, v_tmp+5, s_wo, v_tmp, s_tmp')
        self._emit('v_mul_lo_u32 v[v_tmp+1], s[s_wo], v[v_tmp+6]')
        self._emit('v_sub_u32 v[v_tmp+5], v[v_tmp+5], v[v_tmp+1]')
        self._emit('; v_tmp+4:n0, v_tmp+6:ho, v_tmp+5:wo')
        self._emit_empty_line()
        self._emit('v_mul_lo_u32 v[v_tmp], s[s_wo], v[v_tmp+6]')
        self._emit('s_mul_i32 s[s_tmp], s[s_k], s[s_out_stride_k1]')
        self._emit('v_add_u32 v[v_out_os], v[v_tmp], v[v_tmp+5]')
        self._emit('s_lshl_b32 s[s_tmp+1], s[s_tmp], {}'.format(igemm_log2(self.tunable.gemm_n_repeat) + igemm_log2(self.tunable.gemm_n_per_thread_subc)))
        self._emit('v_mul_lo_u32 v[v_tmp], s[s_tmp+1], v[v_tmp+4]')
        self._emit('v_add_u32 v[v_out_os], v[v_out_os], v[v_tmp]')
        self._emit_empty_line()
        self._emit('s_lshl_b32 s[s_out_stride_k0], s[s_out_stride_k0], 2')
        self._emit('v_lshl_or_b32 v[v_tmp], v[v_out_ik0], {}, v[v_out_ik1]'.format(igemm_log2(self.tunable.gemm_m_per_thread_subc) + igemm_log2(self.tunable.gemm_m_level0_cluster) + igemm_log2(self.tunable.gemm_m_level1_cluster)))
        self._emit('s_lshl_b32 s[s_out_stride_n1], s[s_out_stride_n1], 2')
        self._emit('v_mul_lo_u32 v[v_tmp+1], s[s_out_stride_k1], v[v_tmp]')
        self._emit('s_lshl_b32 s[s_out_stride_n2], s[s_out_stride_n2], 2')
        self._emit('v_add_u32 v[v_out_os], v[v_out_os], v[v_tmp+1]')
        self._emit('s_lshl_b32 s[s_out_stride_k1], s[s_out_stride_k1], 2')
        self._emit('v_lshlrev_b32 v[v_out_os], 2, v[v_out_os]')
        self._emit_empty_line()
        self._emit('; in lds offset block e_n1_b_n2')
        self._emit('v_lshlrev_b32 v[v_tmp], {}, v[v_in_ie]'.format(igemm_log2(self.tunable.gemm_n_repeat) + igemm_log2(self.tunable.b_per_block) + igemm_log2(self.tunable.gemm_n_per_thread_subc)))
        # TODO, e must greater than 1
        if self.tunable.in_block_copy_cluster_lengths_n1 != 1:
            self._emit('v_lshl_or_b32 v[v_tmp], v[v_in_in1], {}, v[v_tmp]'.format(igemm_log2(self.tunable.b_per_block) + igemm_log2(self.tunable.gemm_n_per_thread_subc)))
        if self.tunable.in_block_copy_cluster_lengths_b != 1:
            self._emit('v_lshl_or_b32 v[v_tmp], v[v_in_ib], {}, v[v_tmp]'.format(igemm_log2(self.tunable.gemm_n_per_thread_subc)))
        if self.tunable.in_block_copy_cluster_lengths_n2 != 1:
            self._emit('v_or_b32 v[v_tmp], v[v_tmp], v[v_in_in2]')
        self._emit('v_lshlrev_b32 v[v_sst_b_os], 2, v[v_tmp]')
        self._emit_empty_line()
        self._emit('; wei lds offset block e_k')
        self._emit('v_lshl_or_b32 v[v_tmp], v[v_wei_ie], {}, v[v_wei_ik]'.format(igemm_log2(self.tunable.k_per_block)))
        self._emit('v_lshlrev_b32 v[v_sst_a_os], 2, v[v_tmp]')
        self._emit('v_add_u32 v[v_sst_a_os], {}, v[v_sst_a_os]'.format(self.tunable.byte_lds_b_np2))
        self._emit_empty_line()
        self._emit('s_mov_b32 s[s_p_buf_out+2], 0xffffffff')
        self._emit('s_mov_b32 s[s_p_buf_out+3], 0x27000')
        self._emit('.v_clear_nc v_c, {}'.format(self.tunable.num_accumulate_c_vgpr))
        self._emit_empty_line()

    def emit_kernel_fma_body(self):
        def fma_main_loop_sub_2x2_double_buffer():
            '''
            implement fma main loop with 2x2 sub buffer
            4x4, 4x6, 4x8, 6x4, 6x6, 6x8, 8x4, 8x6, 8x8
            other tile size may also useful, but can't form 2x2 sub buffer
            '''
            kernel_name = self.name()
            label_fma_body = 'L_{}_fma_body'.format(kernel_name)
            label_fma_finishing = 'L_{}_fma_finishing'.format(kernel_name)
            label_fma_end = 'L_{}_end'.format(kernel_name)
            wei_issues = self.tunable.wei_block_copy_sub_lengths_k
            in_sst = emit_in_sst_e_n1_b_n2_t(self.mc, self.tunable)
            wei_sst = emit_wei_sst_e_k_t(self.mc, self.tunable)
            in_move_slice_window = emit_in_move_slice_window_t(self.mc, self.tunable)
            wei_move_slice_window = emit_wei_move_slice_window_t(self.mc, self.tunable)
            in_load = emit_in_load_e_n1_b_n2_t(self.mc, self.tunable)
            wei_load = emit_wei_load_e_k_t(self.mc, self.tunable)
            lds_width_m = 4 * self.tunable.gemm_m_repeat * self.tunable.gemm_m_per_thread_subc * self.tunable.gemm_m_level0_cluster * self.tunable.gemm_m_level1_cluster
            lds_width_n = 4 * self.tunable.gemm_n_repeat * self.tunable.gemm_n_per_thread_subc * self.tunable.gemm_n_level0_cluster * self.tunable.gemm_n_level1_cluster
            # lds_base_m = self.tunable.byte_lds_b_np2
            lds_base_m = 0
            lds_base_n = 0
            unroll_k = self.tunable.e_per_block

            tile_m = self.tunable.thread_tile_m
            tile_n = self.tunable.thread_tile_n
            sub_tile_m = self.tunable.thread_sub_tile_m
            sub_tile_n = self.tunable.thread_sub_tile_n
            local_a = gpr_t('v_a')
            local_b = gpr_t('v_b')
            local_c = gpr_t('v_c')
            lds_single = self.tunable.byte_lds_single

            fma_sub_tile = emit_fma_mxn_t(self.mc, self.tunable.thread_sub_tile_m, self.tunable.thread_sub_tile_n, self.tunable.thread_tile_n)

            assert tile_m == 4 or tile_m == 6 or tile_m == 8
            assert tile_n == 4 or tile_n == 6 or tile_n == 8
            assert tile_m == sub_tile_m * 2
            assert tile_n == sub_tile_n * 2

            ds_read_a = ds_read_t(sub_tile_m * 4)
            ds_read_b = ds_read_t(sub_tile_n * 4)

            # start emit
            self._emit('; start FMA loop, {}x{} thread tile with {}x{} sub-tile'.format(
                                tile_m, tile_n, sub_tile_m, sub_tile_n))
            self._emit('s_waitcnt vmcnt({})'.format(wei_issues))

            self._emit(in_sst('v_gld_b', 'v_sst_b_os'))
            self._emit_empty_line()
            self._emit('s_waitcnt vmcnt(0)')
            self._emit(wei_sst('v_gld_a', 'v_sst_a_os'))
            self._emit_empty_line()
            self._emit('; E = C * Y * X')
            self._emit('s_mul_i32 s[s_tmp], s[s_c], s[s_wei_stride_c]')
            self._emit('s_sub_i32 s[s_kitr], s[s_tmp], {}'.format(unroll_k))
            self._emit('s_cmp_gt_i32 s[s_kitr], 0')
            self._emit('s_cbranch_scc0 {}'.format(label_fma_end))
            self._emit_empty_line()
            self._emit(in_move_slice_window('v_in_os', 'v_in_ic', 'v_in_iy', 'v_in_ix', 'v_in_ihi', 'v_in_iwi', 'v_flag',
                        's_hi', 's_wi', 's_y', 's_x', 's_in_stride_c', 's_dilation_h', 's_dilation_w', 's_in_ic', 's_in_iy', 's_in_ix', 'v_idc', 'v_idy', 'v_idx', 's_tmp'))
            self._emit(wei_move_slice_window('v_wei_os', 'v_wei_ic', 'v_wei_iy', 'v_wei_ix',
                        's_y', 's_x', 's_wei_stride_c', 's_wei_ic', 's_wei_iy', 's_wei_ix', 'v_idc', 'v_idy', 'v_idx', 's_tmp'))
            self._emit('v_xor_b32 v[v_sst_b_os], {}, v[v_sst_b_os] ; switch double buffer b store'.format(hex(lds_single)))
            self._emit('v_xor_b32 v[v_sst_a_os], {}, v[v_sst_a_os] ; switch double buffer a store'.format(hex(lds_single)))
            self._emit('s_waitcnt lgkmcnt(0)')
            self._emit('s_barrier')
            self._emit_empty_line()
            self._emit(in_load('v_gld_b', 's_p_buf_in', 'v_in_os', 's_in_stride_n1', 's_in_stride_n2', 'v_flag', 's_tmp'))
            self._emit(wei_load('v_gld_a', 's_p_buf_wei', 'v_wei_os', 's_wei_stride_k', 's_tmp'))
            self._emit_empty_line()

            # Label: start of fma body
            self._emit_front('{}:'.format(label_fma_body))
            self._emit('; do fma accumulate with unroll {}'.format(unroll_k))
            self._emit(ds_read_a(local_a(), 'v_sld_a_os', lds_base_m))
            self._emit(ds_read_b(local_b(), 'v_sld_b_os', lds_base_n))
            self._emit(ds_read_b(local_b(sub_tile_n), 'v_sld_b_os', lds_base_n + lds_width_n//2 ))
            self._emit(ds_read_a(local_a(sub_tile_m), 'v_sld_a_os', lds_base_m + lds_width_m//2 ))
            self._emit('.itr_k = 0')
            self._emit('.rept {}'.format(unroll_k-1))
            with self._indent_context():
                # 1st fma
                self._emit('s_waitcnt lgkmcnt(2)')
                self._emit(fma_sub_tile(local_c(), local_a(), local_b()))
                self._emit_empty_line()

                # 2nd fma
                self._emit('s_waitcnt lgkmcnt(1)')
                self._emit(fma_sub_tile(local_c(sub_tile_n), local_a(), local_b(sub_tile_n)))
                self._emit_empty_line()

                # 3rd fma
                self._emit(ds_read_a(local_a(), 'v_sld_a_os', '{}+(.itr_k+1)*{}'.format(lds_base_m, lds_width_m)))
                self._emit('s_waitcnt lgkmcnt(1)')
                self._emit(fma_sub_tile(local_c(sub_tile_m*tile_n), local_a(sub_tile_m), local_b()))
                self._emit_empty_line()

                # 4th fma
                self._emit(ds_read_b(local_b(), 'v_sld_b_os', '{}+(.itr_k+1)*{}'.format(lds_base_n, lds_width_n)))
                self._emit(fma_sub_tile(local_c(sub_tile_m*tile_n+sub_tile_n), local_a(sub_tile_m), local_b(sub_tile_n)))
                self._emit_empty_line()

                # last
                self._emit(ds_read_b(local_b(sub_tile_n), 'v_sld_b_os', '{}+(.itr_k+1)*{}+{}'.format(lds_base_n, lds_width_n, lds_width_n//2)))
                self._emit(ds_read_a(local_a(sub_tile_m), 'v_sld_a_os', '{}+(.itr_k+1)*{}+{}'.format(lds_base_m, lds_width_m, lds_width_m//2)))
                self._emit('.itr_k = .itr_k + 1')
            self._emit('.endr')
            self._emit_empty_line()
            self._emit('; last unroll')
            self._emit('v_xor_b32 v[v_sld_b_os], {}, v[v_sld_b_os] ; switch double buffer b load'.format(hex(lds_single)))
            self._emit('v_xor_b32 v[v_sld_a_os], {}, v[v_sld_a_os] ; switch double buffer a load'.format(hex(lds_single)))
            # 1st fma
            self._emit('s_waitcnt lgkmcnt(2)')
            self._emit(fma_sub_tile(local_c(), local_a(), local_b()))
            self._emit_empty_line()

            # 2nd fma
            self._emit('s_waitcnt lgkmcnt(1)')
            self._emit(fma_sub_tile(local_c(sub_tile_n), local_a(), local_b(sub_tile_n)))
            self._emit_empty_line()

            #       wait global and store to LDS
            self._emit('s_waitcnt vmcnt({})'.format(wei_issues))
            self._emit(in_sst('v_gld_b', 'v_sst_b_os'))
            self._emit('s_waitcnt vmcnt(0)')
            self._emit(wei_sst('v_gld_a', 'v_sst_a_os'))

            #       iteration--
            self._emit('s_sub_i32 s[s_kitr], s[s_kitr], {}'.format(unroll_k))
            self._emit('s_cmp_gt_i32 s[s_kitr], 0')
            self._emit('s_cbranch_scc0 {}'.format(label_fma_finishing))

            #       move slice window
            self._emit(in_move_slice_window('v_in_os', 'v_in_ic', 'v_in_iy', 'v_in_ix', 'v_in_ihi', 'v_in_iwi', 'v_flag',
                        's_hi', 's_wi', 's_y', 's_x', 's_in_stride_c', 's_dilation_h', 's_dilation_w', 's_in_ic', 's_in_iy', 's_in_ix', 'v_idc', 'v_idy', 'v_idx', 's_tmp'))
            self._emit(wei_move_slice_window('v_wei_os', 'v_wei_ic', 'v_wei_iy', 'v_wei_ix',
                        's_y', 's_x', 's_wei_stride_c', 's_wei_ic', 's_wei_iy', 's_wei_ix', 'v_idc', 'v_idy', 'v_idx', 's_tmp'))

            # 3rd fma
            self._emit('s_waitcnt lgkmcnt({})'.format(in_sst.get_issues() + wei_sst.get_issues()))
            self._emit(fma_sub_tile(local_c(sub_tile_m*tile_n), local_a(sub_tile_m), local_b()))
            self._emit_empty_line()

            self._emit('v_xor_b32 v[v_sst_b_os], {}, v[v_sst_b_os] ; switch double buffer b store'.format(hex(lds_single)))
            self._emit('v_xor_b32 v[v_sst_a_os], {}, v[v_sst_a_os] ; switch double buffer a store'.format(hex(lds_single)))
            #       barrier here!
            self._emit('s_waitcnt lgkmcnt(0)')
            self._emit('s_barrier')

            #       load next from global
            self._emit(in_load('v_gld_b', 's_p_buf_in', 'v_in_os', 's_in_stride_n1', 's_in_stride_n2', 'v_flag', 's_tmp'))
            self._emit(wei_load('v_gld_a', 's_p_buf_wei', 'v_wei_os', 's_wei_stride_k', 's_tmp'))

            # 4th fma
            self._emit(fma_sub_tile(local_c(sub_tile_m*tile_n+sub_tile_n), local_a(sub_tile_m), local_b(sub_tile_n)))
            self._emit_empty_line()
            self._emit('s_branch {}'.format(label_fma_body))

            # Label: finishing of fma body
            self._emit_front('{}:'.format(label_fma_finishing))
            self._emit('s_waitcnt lgkmcnt({})'.format(in_sst.get_issues() + wei_sst.get_issues()))
            self._emit(fma_sub_tile(local_c(sub_tile_m*tile_n), local_a(sub_tile_m), local_b()))
            self._emit(fma_sub_tile(local_c(sub_tile_m*tile_n+sub_tile_n), local_a(sub_tile_m), local_b(sub_tile_n)))

            # Label: end of fma body
            self._emit_front('{}:'.format(label_fma_end))
            self._emit('s_waitcnt lgkmcnt(0)')
            self._emit('s_barrier')
            self._emit(ds_read_a(local_a(), 'v_sld_a_os', lds_base_m))
            self._emit(ds_read_b(local_b(), 'v_sld_b_os', lds_base_n))
            self._emit(ds_read_b(local_b(sub_tile_n), 'v_sld_b_os', lds_base_n + lds_width_n//2 ))
            self._emit(ds_read_a(local_a(sub_tile_m), 'v_sld_a_os', lds_base_m + lds_width_m//2 ))
            self._emit('.itr_k = 0')
            self._emit('.rept {}'.format(unroll_k - 1))
            with self._indent_context():
                # 1st fma
                self._emit('s_waitcnt lgkmcnt(2)')
                self._emit(fma_sub_tile(local_c(), local_a(), local_b()))
                self._emit_empty_line()

                # 2nd fma
                self._emit('s_waitcnt lgkmcnt(1)')
                self._emit(fma_sub_tile(local_c(sub_tile_n), local_a(), local_b(sub_tile_n)))
                self._emit_empty_line()

                # 3rd fma
                self._emit(ds_read_a(local_a(), 'v_sld_a_os', '{}+(.itr_k+1)*{}'.format(lds_base_m, lds_width_m)))
                self._emit('s_waitcnt lgkmcnt(1)')
                self._emit(fma_sub_tile(local_c(sub_tile_m*tile_n), local_a(sub_tile_m), local_b()))
                self._emit_empty_line()

                # 4th fma
                self._emit(ds_read_b(local_b(), 'v_sld_b_os', '{}+(.itr_k+1)*{}'.format(lds_base_n, lds_width_n)))
                self._emit(fma_sub_tile(local_c(sub_tile_m*tile_n+sub_tile_n), local_a(sub_tile_m), local_b(sub_tile_n)))
                self._emit_empty_line()

                # last
                self._emit(ds_read_b(local_b(sub_tile_n), 'v_sld_b_os', '{}+(.itr_k+1)*{}+{}'.format(lds_base_n, lds_width_n, lds_width_n//2)))
                self._emit(ds_read_a(local_a(sub_tile_m), 'v_sld_a_os', '{}+(.itr_k+1)*{}+{}'.format(lds_base_m, lds_width_m, lds_width_m//2)))
                self._emit('.itr_k = .itr_k + 1')
            self._emit('.endr')
            self._emit_empty_line()
            self._emit('; last unroll')
            # 1st fma
            self._emit('s_waitcnt lgkmcnt(2)')
            self._emit(fma_sub_tile(local_c(), local_a(), local_b()))
            self._emit_empty_line()

            # 2nd fma
            self._emit('s_waitcnt lgkmcnt(1)')
            self._emit(fma_sub_tile(local_c(sub_tile_n), local_a(), local_b(sub_tile_n)))
            self._emit_empty_line()

            # 3rd fma
            self._emit('s_waitcnt lgkmcnt(0)')
            self._emit(fma_sub_tile(local_c(sub_tile_m*tile_n), local_a(sub_tile_m), local_b()))
            self._emit_empty_line()

            # 4th fma
            self._emit(fma_sub_tile(local_c(sub_tile_m*tile_n+sub_tile_n), local_a(sub_tile_m), local_b(sub_tile_n)))
            self._emit_empty_line()

        fma_main_loop_sub_2x2_double_buffer()

    def emit_kernel_writeout(self):
        out_write = emit_out_write_k0_k1_n1_b_n2_t(self.mc, self.tunable)
        self._emit('s_mov_b32 s[s_tmp], 0')
        self._emit('s_mov_b32 s[s_tmp+1], 0')
        self._emit('s_mov_b32 s[s_tmp+2], 0')
        self._emit('s_mov_b32 s[s_tmp+3], 0')
        self._emit(out_write('v_c', 's_p_buf_out', 'v_out_os', 's_out_stride_k0', 's_out_stride_k1', 's_out_stride_n1', 's_out_stride_n2', 's_tmp'))

    def emit(self):
        self._emit(';----------------------------------------------------------')
        self._emit('; starting of kernel {}'.format(self.name()))
        self._emit(self.tunable.serialize())
        if self.emit_unified_define:
            #self.emit_kernel_define_karg()
            self.kernel_karg.emit()
            #self.emit_kernel_define_sgpr()  # TODO: for every v4r1, sgpr, karg is the same
            self.kernel_sgpr.emit()
        #self.emit_kernel_define_vgpr()
        self.kernel_vgpr.emit()
        self.emit_kernel_header()
        with self._indent_context():
            if self.mc.arch_config.code_object == AMDGPU_CODEOBJECT_V2:
                self.emit_kernel_amd_kernel_code_t()
            self.emit_kernel_prepare_phase()
            self.emit_kernel_fma_body()
            self.emit_kernel_writeout()
            self.emit_kernel_end()
        if self.mc.arch_config.code_object == AMDGPU_CODEOBJECT_V3:
            self.emit_kernel_amd_kernel_code_t()
        self.emit_kernel_footer()

class v4r1_dynamic_index_t(object):
    def __init__(self):
        self.tid            = 0
        self.bid            = 0
        self.v_in_ie        = 0
        self.v_in_in1       = 0
        self.v_in_ib        = 0
        self.v_in_in2       = 0
        self.v_wei_ie       = 0
        self.v_wei_ik       = 0

        self.v_gemm_im      = 0
        self.v_gemm_in      = 0

        # conv_param dependent
        self.s_block_ik     = 0
        self.s_block_ib     = 0
        self.v_in_in0       = 0
        self.v_in_iho       = 0
        self.v_in_iwo       = 0

        self.v_in_ic        = 0
        self.v_in_iy        = 0
        self.v_in_ix        = 0
        self.v_in_ihi       = 0
        self.v_in_iwi       = 0
        self.v_wei_ic       = 0
        self.v_wei_iy       = 0
        self.v_wei_ix       = 0

        self.v_out_ik0      = 0
        self.v_out_ik1      = 0
        self.v_out_ib       = 0

        self.v_out_in0      = 0
        self.v_out_iho      = 0
        self.v_out_iwo      = 0

        self.v_in_os        = 0
        self.v_wei_os       = 0
        self.v_out_os       = 0
        self.v_sst_a_os     = 0
        self.v_sst_b_os     = 0
        self.v_sld_a_os     = 0
        self.v_sld_b_os     = 0

def v4r1_dynamic_get_block_size(tunable):
    return tunable.gemm_m_level0_cluster * tunable.gemm_n_level0_cluster * \
                    tunable.gemm_m_level1_cluster * tunable.gemm_n_level1_cluster

def v4r1_dynamic_get_dynamic_index(tunable, conv_param, tid, bid):
    # For simplicity following calculation use divide/mod insteat of and/shift, since we are in cpu world
    dynamic_index = v4r1_dynamic_index_t()
    # in e_n1_b_n2, order:{0,1,3,2}
    dynamic_index.v_in_ib = tid % tunable.in_block_copy_cluster_lengths_b
    dynamic_index.v_in_ib *= tunable.in_block_copy_sub_lengths_b
    tmp = tid // tunable.in_block_copy_cluster_lengths_b

    dynamic_index.v_in_in2 = tmp % tunable.in_block_copy_cluster_lengths_n2
    dynamic_index.v_in_in2 *= tunable.in_block_copy_sub_lengths_n2
    tmp = tmp // tunable.in_block_copy_cluster_lengths_n2

    dynamic_index.v_in_in1 = tmp % tunable.in_block_copy_cluster_lengths_n1
    dynamic_index.v_in_in1 *= tunable.in_block_copy_sub_lengths_n1
    tmp = tmp // tunable.in_block_copy_cluster_lengths_n1

    dynamic_index.v_in_ie = tmp % tunable.in_block_copy_cluster_lengths_e
    dynamic_index.v_in_ie *= tunable.in_block_copy_sub_lengths_e

    # wei e_k, order:{1,0}
    dynamic_index.v_wei_ie = tid % tunable.wei_block_copy_cluster_lengths_e
    dynamic_index.v_wei_ie *= tunable.wei_block_copy_sub_lengths_e
    tmp = tid // tunable.wei_block_copy_cluster_lengths_e

    dynamic_index.v_wei_ik = tmp % tunable.wei_block_copy_cluster_lengths_k
    dynamic_index.v_wei_ik *= tunable.wei_block_copy_sub_lengths_k

    # gemm im, in
    level_0_cluster = tunable.gemm_m_level0_cluster * tunable.gemm_n_level0_cluster
    level_0_id = tid % level_0_cluster
    level_0_id_m = level_0_id // tunable.gemm_n_level0_cluster
    level_0_id_n = level_0_id % tunable.gemm_n_level0_cluster

    level_1_id = tid // level_0_cluster
    level_1_id_m = level_1_id // tunable.gemm_n_level1_cluster
    level_1_id_n = level_1_id % tunable.gemm_n_level1_cluster

    dynamic_index.v_gemm_im = level_1_id_m * tunable.gemm_m_level0_cluster + level_0_id_m
    dynamic_index.v_gemm_in = level_1_id_n * tunable.gemm_n_level0_cluster + level_0_id_n

    # conv_param dependent
    n1 = tunable.gemm_n_repeat
    n2 = tunable.gemm_n_per_thread_subc
    n0 = conv_param.n // (n1 * n2)
    b = n0 * conv_param.ho * conv_param.wo
    e = conv_param.c * conv_param.y * conv_param.x

    # k_block_work = conv_param.k // tunable.k_per_block
    b_block_work = b // tunable.b_per_block
    block_id_b = bid % b_block_work
    block_id_k = bid // b_block_work
    dynamic_index.s_block_ib = block_id_b * tunable.b_per_block
    dynamic_index.s_block_ik = block_id_k * tunable.k_per_block

    # e_n1_b_n2:b, transform: b -> n0*ho*wo
    dynamic_index.v_in_in0 = (dynamic_index.v_in_ib + dynamic_index.s_block_ib) // (conv_param.ho * conv_param.wo)
    tmp = (dynamic_index.v_in_ib + dynamic_index.s_block_ib) % (conv_param.ho * conv_param.wo)
    dynamic_index.v_in_iho = tmp // conv_param.wo
    dynamic_index.v_in_iwo = tmp % conv_param.wo
    # e_n1_b_n2:e, 1) transform e -> c*y*x
    dynamic_index.v_in_ic = dynamic_index.v_in_ie // (conv_param.y * conv_param.x)
    tmp = dynamic_index.v_in_ie % (conv_param.y * conv_param.x)
    dynamic_index.v_in_iy = tmp // conv_param.x
    dynamic_index.v_in_ix = tmp % conv_param.x
    # 2) transform iho, iwo, iy, ix -> hip, wip
    dynamic_index.v_in_ihi = conv_param.sy * dynamic_index.v_in_iho + conv_param.dy * dynamic_index.v_in_iy - conv_param.py
    dynamic_index.v_in_iwi = conv_param.sx * dynamic_index.v_in_iwo + conv_param.dx * dynamic_index.v_in_ix - conv_param.px

    # calculate input offset, from ihi, iwi, ic, in, calculate v_in_os
    dynamic_index.v_in_os = (dynamic_index.v_in_in0 * tunable.gemm_n_repeat * tunable.gemm_n_per_thread_subc + \
                             dynamic_index.v_in_in1 * tunable.gemm_n_per_thread_subc + \
                             dynamic_index.v_in_in2 ) * (conv_param.c * conv_param.hi * conv_param.wi) +\
                             dynamic_index.v_in_ic * (conv_param.hi * conv_param.wi) + \
                             dynamic_index.v_in_ihi * conv_param.wi + \
                             dynamic_index.v_in_iwi
    dynamic_index.v_in_os *= 4 # sizeof(float)

    # calculate weight transform, e_k: e->c*y*x
    dynamic_index.v_wei_ic = dynamic_index.v_wei_ie // (conv_param.y * conv_param.x)
    tmp = dynamic_index.v_wei_ie % (conv_param.y * conv_param.x)
    dynamic_index.v_wei_iy = tmp // conv_param.x
    dynamic_index.v_wei_ix = tmp % conv_param.x
    # wei offset: from ic, iy, ix, ik, calculate v_wei_os
    dynamic_index.v_wei_os = dynamic_index.v_wei_ik * (conv_param.c * conv_param.y * conv_param.x) + \
                             dynamic_index.v_wei_ic * (conv_param.y * conv_param.x) + \
                             dynamic_index.v_wei_iy * conv_param.x + \
                             dynamic_index.v_wei_ix
    dynamic_index.v_wei_os *= 4 # sizeof(float)

    # output
    k1 = tunable.gemm_m_per_thread_subc * tunable.gemm_m_level0_cluster * tunable.gemm_m_level1_cluster
    k_thread_data_on_global = dynamic_index.s_block_ik + (dynamic_index.v_gemm_im * tunable.gemm_n_per_thread_subc)
    dynamic_index.v_out_ik0 = k_thread_data_on_global // k1
    dynamic_index.v_out_ik1 = k_thread_data_on_global % k1

    b_thread_data_on_global = dynamic_index.s_block_ib + dynamic_index.v_gemm_in
    dynamic_index.v_out_ib = b_thread_data_on_global

    dynamic_index.v_out_in0 = dynamic_index.v_out_ib // (conv_param.ho * conv_param.wo)
    tmp = dynamic_index.v_out_ib % (conv_param.ho * conv_param.wo)
    dynamic_index.v_out_iho = tmp // conv_param.wo
    dynamic_index.v_out_iwo = tmp % conv_param.wo

    dynamic_index.v_out_os = (dynamic_index.v_out_in0 * tunable.gemm_n_repeat * tunable.gemm_n_per_thread_subc) * (conv_param.k * conv_param.ho * conv_param.wo) + \
                              dynamic_index.v_out_ik0 * k1 * (conv_param.ho * conv_param.wo) + \
                              dynamic_index.v_out_ik1 * (conv_param.ho * conv_param.wo) + \
                              dynamic_index.v_out_iho * conv_param.wo + \
                              dynamic_index.v_out_iwo
    dynamic_index.v_out_os *= 4 # sizeof(float)
    # print(' [v_out_in0:{}, v_out_iho:{}, v_out_iwo:{}, m:{}, n:{}]'.format(
    #         dynamic_index.v_out_in0,dynamic_index.v_out_iho,dynamic_index.v_out_iwo,
    #         dynamic_index.v_gemm_im, dynamic_index.v_gemm_in
    # ))

    # input sst, e_n1_b_n2
    dynamic_index.v_sst_b_os = dynamic_index.v_in_ie * (tunable.gemm_n_repeat * tunable.b_per_block * tunable.gemm_n_per_thread_subc) + \
                               dynamic_index.v_in_in1 * (tunable.b_per_block * tunable.gemm_n_per_thread_subc) + \
                               dynamic_index.v_in_ib * tunable.gemm_n_per_thread_subc + \
                               dynamic_index.v_in_in2
    dynamic_index.v_sst_b_os *= 4 # sizeof(float)

    dynamic_index.v_sld_b_os = dynamic_index.v_gemm_in * tunable.gemm_n_per_thread_subc
    dynamic_index.v_sld_b_os *= 4 # sizeof(float)

    # wei sst e_k
    dynamic_index.v_sst_a_os = dynamic_index.v_wei_ie * tunable.k_per_block +\
                               dynamic_index.v_wei_ik
    dynamic_index.v_sst_a_os *= 4 # sizeof(float)
    dynamic_index.v_sst_a_os += tunable.byte_lds_b_np2

    dynamic_index.v_sld_a_os = dynamic_index.v_gemm_im * tunable.gemm_m_per_thread_subc
    dynamic_index.v_sld_a_os *= 4 # sizeof(float)
    dynamic_index.v_sld_a_os += tunable.byte_lds_b_np2
    return dynamic_index

def emit_v4r1_dynamic_macros(mc, tunable_dicts):
    def emit_per_macro(m):
        for tunable_dict in tunable_dicts:
            m(mc, igemm_tunable_parameter_t(tunable_dict))._emit_unique_macro()
    emit_per_macro(emit_fma_subtile_t)
    emit_per_macro(emit_in_set_flag_t)
    emit_per_macro(emit_in_load_e_n1_b_n2_t)
    emit_per_macro(emit_wei_load_e_k_t)
    emit_per_macro(emit_in_sst_e_n1_b_n2_t)
    emit_per_macro(emit_wei_sst_e_k_t)
    emit_per_macro(emit_out_write_k0_k1_n1_b_n2_t)
    emit_per_macro(emit_in_move_slice_window_t)
    emit_per_macro(emit_wei_move_slice_window_t)

def emit_v4r1_dynamic_kernel(mc, tunable_dicts):
    first_kernel = True
    kernel_info_list = []
    for tunable_dict in tunable_dicts:
        kernel = emit_v4r1_dynamic_kernel_t(mc, igemm_tunable_parameter_t(tunable_dict), True if first_kernel else False)
        kernel._emit_unique_macro()
        first_kernel = False

        kernel_info_list.append(kernel.get_kernel_info())

    emit_amd_metadata_t(mc, kernel_info_list).emit()

        # if emit_v4r1_dynamic_kernel_t(mc, igemm_tunable_parameter_t(tunable_dict)).name() == 'igemm_v4r1_dynamic_64x64x8_8x8_4x4x2x4x2x4_8x1x8x1_4x16':
        #     #                         n,  g,  c, hi, wi,   k, y, x, py, px, sy, sx, dy, dx, ho, wo, direction
        #     conv_param = conv_param_t(64, 1, 64, 56, 56, 256, 1, 1,  0,  0,  1,  1,  1,  1,  0,  0, CONV_DIRECTION_FWD)
        #     conv_param.dump()
        #     for tid in range(64):
        #         dynamic_index = v4r1_dynamic_get_dynamic_index(igemm_tunable_parameter_t(tunable_dict), conv_param, tid, 0)
        #         print('tid:{}, v_in_os:{}, v_wei_os:{}, v_out_os:{}, v_sst_a_os:{}, v_sst_b_os:{}, v_sld_a_os:{}, v_sld_b_os:{}'.format(
        #             tid, dynamic_index.v_in_os, dynamic_index.v_wei_os, dynamic_index.v_out_os, dynamic_index.v_sst_a_os, dynamic_index.v_sst_b_os,
        #             dynamic_index.v_sld_a_os, dynamic_index.v_sld_b_os
        #         ))