from python import *
from python.cellfft import *
import sys, os, shutil
import subprocess

OUT_DIR='out'
CPP_DIR=os.path.join('test', 'twiddle')
ARCH="gfx908"

def run_cmd(cmd):
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr = subprocess.STDOUT)
    try:
        (out, _) = p.communicate()
        if p.returncode != 0:
            print('run fail:{}'.format(" ".join(cmd)))
            print('{}'.format(out.decode('utf-8')))
            return False
        print('{}'.format(out.decode('utf-8')), end='')
        return True
    except Exception as e:
        print('fail to run cmd:{}'.format(" ".join(cmd)))
        print('err:{}'.format(e))
        return False


def emit_kernel_header(mc, kernel_name, covx):
    mc.emit('.text')
    if covx == 'cov3':
        mc.emit('.globl {}'.format(kernel_name))
    mc.emit('.p2align 8')
    if covx == 'cov3':
        mc.emit('.type {},@function'.format(kernel_name))
    if covx == 'cov2':
        mc.emit('.amdgpu_hsa_kernel {}'.format(kernel_name))
    mc.emit('{}:'.format(kernel_name))

def test_fft():
    asm_target = os.path.join(OUT_DIR, "twiddle.s")
    emitter = mc_emit_to_file_t(asm_target)

    arch = amdgpu_arch_config_t({'arch' :   amdgpu_string_to_arch(ARCH) })

    # create mc
    mc = mc_asm_printer_t(emitter, arch)
    mc_set_current(mc)
    hsa_header_t(mc).emit() 

    kernel_info_list = []

    def emit_fft(n, is_fwd):
        kernel_func = 'twiddle_fft{}_{}'.format(n, 'fwd' if is_fwd else 'bwd')
        fft = fft_t(mc, ctrl_fft_t(n, 0, BUTTERFLY_DIRECTION_FORWARD if is_fwd else BUTTERFLY_DIRECTION_BACKWARD), True)
        emit_kernel_header(mc, kernel_func, 'cov3')

        def get_kernel_code():
            kernel_code = amdgpu_kernel_code_t({
                    'enable_sgpr_kernarg_segment_ptr'   :   1,
                    'enable_sgpr_workgroup_id_x'        :   1,
                    'enable_vgpr_workitem_id'           :   0,
                    'workgroup_group_segment_byte_size' :   0,
                    'kernarg_segment_byte_size'         :   16,
                    'wavefront_sgpr_count'              :   100,
                    'workitem_vgpr_count'               :   100})   # this is test kernel so just let this value big enough
            return kernel_code

        def get_kernel_args():
            '''
            float *p_in;
            float *p_out;
            '''
            kas = []
            # name: {}, .size: {}, .offset: {}, .value_kind: {}, .value_type
            kas.append(amdgpu_kernel_arg_t('p_in'           , 8,   0, 'global_buffer','f32',address_space='global',is_const='true'))
            kas.append(amdgpu_kernel_arg_t('p_out'          , 8,   8, 'global_buffer','f32',address_space='global',is_const='false'))
            return kas

        def get_kernel_info():
            kernel_code = get_kernel_code()
            kernel_args = get_kernel_args()
            kernel_info = amdgpu_kernel_info_t(kernel_code, kernel_func, 256, kernel_args)
            return kernel_info
        
        kernel_info_list.append(get_kernel_info())
        
        label_end = f"{kernel_func}_end"
        mc.emit(f".set s_ka,        0")
        mc.emit(f".set s_bx,        2")
        mc.emit(f".set s_in,        4")
        mc.emit(f".set s_out,       8")
        mc.emit(f".set v_tid,       0")
        mc.emit(f".set v_pt,        0   ; for simplicity, give 64 vgpr for this twiddle")
        mc.emit(f".set v_tmp,       64   ; for simplicity, give 64 vgpr for this twiddle")
        mc.emit(f"")
        mc.emit(f"s_load_dwordx2    s[s_in:s_in+1],     s[s_ka:s_ka+1],    0")
        mc.emit(f"s_load_dwordx2    s[s_out:s_out+1],   s[s_ka:s_ka+1],    8")
        mc.emit(f"s_mov_b32         s[s_in+2],          0xffffffff")
        mc.emit(f"s_mov_b32         s[s_in+3],          0x27000")
        mc.emit(f"s_mov_b32         s[s_out+2],         0xffffffff")
        mc.emit(f"s_mov_b32         s[s_out+3],         0x27000")
        mc.emit(f"s_waitcnt         lgkmcnt(0)")
        mc.emit(f"")
        mc.emit(f"s_cmp_eq_u32      0,   s[s_bx]")
        mc.emit(f"s_cbranch_scc0    {label_end}")
        mc.emit(v_cmpx_eq_u32("vcc",    0,  "v_tid"))
        mc.emit(f"v_mov_b32         v[v_tmp],   0")
        for i in range(n * 2):
            # mc.emit(f"buffer_load_dword v[v_pt + {i}], v[v_tmp], s[s_in:s_in+3], 0, offen offset:0")
            # mc.emit(v_add_nc_u32("v_tmp",  "v_tmp",   4))
            # mc.emit(f"buffer_load_dword v[v_pt + {i}], v[v_tmp], s[s_in:s_in+3], 0, offen offset:{i * 4}")
            mc.emit(f"global_load_dword v[v_pt + {i}], v[v_tmp], s[s_in:s_in+1] offset:{i * 4}")
        mc.emit(f"s_waitcnt vmcnt(0)")
        mc.emit(f";----------------")
        mc.emit(fft("v_pt", "v_tmp"))

        mc.emit(f";----------------")
        mc.emit(f"v_mov_b32         v[v_tmp],   0")
        for i in range(n * 2):
            # mc.emit(f"buffer_store_dword v[v_pt + {i}], v[v_tmp], s[s_out:s_out+3], 0, offen offset:0")
            # mc.emit(v_add_nc_u32("v_tmp",  "v_tmp",   4))
            # mc.emit(f"buffer_store_dword v[v_pt + {i}], v[v_tmp], s[s_out:s_out+3], 0, offen offset:{i * 4}")
            mc.emit(f"global_store_dword  v[v_tmp],  v[v_pt + {i}],s[s_out:s_out+1] offset:{i * 4}")
        mc.emit(f"s_waitcnt vmcnt(0)")
        mc.emit(f"s_mov_b64         exec,   -1")
        mc.emit(f"{label_end}:")
        mc.emit(f"s_endpgm")
        mc.emit(f"")

        amd_kernel_code_t(mc, get_kernel_info()).emit()
        mc.emit(f"")
        mc.emit(f"")

    radix_list =  [4, 8, 16, 32]
    for radix in radix_list:
        emit_fft(radix, True)
        emit_fft(radix, False)

    amdgpu_metadata_t(mc, kernel_info_list).emit()

    # compile device code
    ass = compile_asm_t(mc, mc.emitter.file_name)
    rtn = ass.compile()
    if not rtn:
        assert False

    disass = compile_disass_t(mc, ass.target_hsaco)
    rtn = disass.compile()
    if not rtn:
        assert False

    # compile host code
    cpp_src = os.path.join(CPP_DIR, "twiddle_test.cpp")
    target_exe = os.path.join(OUT_DIR, 'twiddle_test.exe')
    builder = compile_host_t(arch, cpp_src, target_exe)

    rtn = builder.compile(cxxflags=['-DHSACO=\"{}\"'.format(ass.target_hsaco),
        '-I{}'.format(os.path.join('test', 'common')) ])
    if not rtn:
        assert False

    while True:
        for radix in radix_list:
            # run this exe
            cmd = [target_exe, f"{radix}", "fwd"]
            run_cmd(cmd)
            cmd = [target_exe, f"{radix}", "bwd"]
            run_cmd(cmd)
        break

if __name__ == '__main__':
    if os.path.exists(OUT_DIR):
        shutil.rmtree(OUT_DIR)
    os.mkdir(OUT_DIR)
    test_fft()
 