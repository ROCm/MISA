.include "metadata.inc"

KERNEL_PROLOG add_1

kernarg = 0
gid_x = 2

    workgroup_size = 128

    
    s_in_addr = 4
    s_out_addr = 8
    s_grid_ptr = 12

    v_tid = 0
    v_acc = 1
    v_ptr = 4

    s_load_dwordx2 s[s_in_addr:s_in_addr+1], s[kernarg:kernarg+1], 0x0
    s_load_dwordx2 s[s_out_addr:s_out_addr+1], s[kernarg:kernarg+1], 0x4 * 2

    .macro const_buffer_res dst_3, Num_format, Data_forma, tid_enable, el_size, index_stride, non_volatile=0
    //0..47  base addres 48b
    //48..61 Stride of the record in bytes V#.const_stride[13:0] 14b
    //62     Cache swizzle
    //63     Swizzle enable
    //2____
    //64..95 Number of records in the buffer 32b
    //3______
    //96..98 Dst_sel_x   //0
    //99.101 Dst_sel_y   //3
    //102.104 Dst_sel_z  //6
    //105.107 Dst_sel_w  //9
    //108.110 Num format Numeric format of data in memory:0 unorm, 1 snorm, 2 uscaled, 3 sscaled,
        //4 uint, 5 sint, 6 reserved, 7 float
        // -- 12
    //111.114 Data forma.   For MUBUF instructions with ADD_TID_EN = 1. This fieldholds Stride [17:14]
        // 1 8, 2 16, 3 8_8, 4 32, ... -- 15
    //115     User VM Enable  -- 19
    //116     User VM mode    -- 20
    //117-118  index_stride   -- 21
    //119     const_add_tid_enable -- 23
    //120-122 RSVD zero     -- 24
    //123     Non-volatile (0=volatile) -- 27
    //124.125 RSVD zero
    //126.127 Type  Value == 0 for buffer. Overlaps upper two bits of four-bit TYPE field in128-bit T# resource.
    
    \dst_3 = \Num_format * (1 << 12) + \Data_forma * (1 << 15) + \index_stride * (1<<21)  + \tid_enable * (1<<23) + \non_volatile * (1<<27)
    .endm

    s_mov_b32 s[s_in_addr+3], 0x00020000     //32bit type
    s_mov_b32 s[s_out_addr+3], 0x00020000     //32bit type
    
    s_mov_b32 s[s_in_addr+2], 1024
    s_mov_b32 s[s_out_addr+2], 1024

    s_waitcnt 0
    .macro vmem_loads vdst, addr, d_desc, num, elem_size
        i = 0
        .rept \num
            .if \elem_size == 2
                buffer_load_ushort v[\vdst + i], v[\addr + i], s[\d_desc : \d_desc +3], 0, offen
            .else
                buffer_load_dword v[\vdst + i], v[\addr + i], s[\d_desc : \d_desc +3], 0, offen
            .endif
            i = i + 1
        .endr
    .endm

    .macro vmem_stores vsrc, addr, o_desc, num, elem_size
        i = 0
        .rept \num
            .if \elem_size == 2
                buffer_store_short v[\vsrc+i], v[\addr + i], s[\o_desc : \o_desc +3], 0, offen
            .else
                buffer_store_dword v[\vsrc+i], v[\addr + i], s[\o_desc : \o_desc +3], 0, offen
            .endif
            i = i + 1
        .endr
    .endm

    v_mov_b32 v[v_ptr], v[v_tid]
    v_mul_u32_u24 v[v_ptr], v[v_tid], 4

    s_mul_i32 s[s_grid_ptr], 4 * workgroup_size, s[gid_x]
    v_add_u32 v[v_ptr], s[s_grid_ptr], v[v_ptr]
    
    v_add_u32 v[v_acc], s[gid_x], 1
    v_mul_u32_u24 v[v_acc], 5, v[v_acc]

    vmem_loads v_acc+1, v_ptr, s_in_addr, 1, 4

    s_waitcnt 0

    v_add_u32 v[v_acc], v[v_acc], v[v_acc+1]

    vmem_stores v_acc, v_ptr, s_out_addr, 1, 4

.macro METADATA sc,wc,wg_x, kernel_name
.amdgpu_metadata
---
amdhsa.version: [ 1, 0 ]
amdhsa.kernels:
  - .name: \kernel_name
    .symbol: \kernel_name\().kd
    .language: "OpenCL C"
    .language_version: [ 1, 2 ]
    .sgpr_count: \sc
    .vgpr_count: \wc
    .group_segment_fixed_size: 0
    .private_segment_fixed_size: 0
    .kernarg_segment_size: 16
    .kernarg_segment_align: 8
    .reqd_workgroup_size: [ \wg_x, 1, 1 ]
    .max_flat_workgroup_size: \wg_x
    .wavefront_size: 64
    .args:
    - { .size: 8, .offset:  0, .value_kind: global_buffer, .value_type: f32, .name: in,  .address_space: global, .is_const: true }
    - { .size: 8, .offset:  8, .value_kind: global_buffer, .value_type: f32, .name: out, .address_space: global, .is_const: false }
...
.end_amdgpu_metadata
.endm // METADATA

.macro KERNEL_PARAMS

        vgpr_size = 128
        workgroup_size_x = 128

    .amdgcn.next_free_sgpr = 101
    .amdgcn.next_free_vgpr = vgpr_size

    //xnack disabled by default for asm kernels
    __sgpr_reserve_vcc_default = 1
    __sgpr_reserve_xnack_default = 0
    __sgpr_reserve_flatscr_default = 0

    __group_segment_fixed_size = 0
    __sgpr_private_segment_buffer = 0
    __sgpr_dispatch_ptr = 0
    __sgpr_kernarg_segment_ptr = 1
    __sgpr_workgroup_id_x = 1
    __sgpr_workgroup_id_y = 0
    __sgpr_workgroup_id_z = 0
    __vgpr_workitem_id = 1
    __ieee_mode = 0
    __dx10_clamp = 0
.endm

KERNEL_EPILOG add_1

//$clang -x assembler -target amdgcn--amdhsa -mcpu=$mcpu $gas_include $gas_defs $gas_src -o $out_path
///opt/rocm/llvm/bin/clang++ -x assembler -target amdgcn--amdhsa -mcpu=gfx906 -I../ ../add_1.s -o ./add_1.o