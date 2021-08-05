import runtime_py

runtime_py.SetSeverity(5)

rt = runtime_py.Runtime()

st = input('0--->')
mem1 = runtime_py.memtag("some in 1", 1024*1024*1024, runtime_py.mem_usage.IN, True)
in1_buffer = rt.create_buf(mem1)
in1 = in1_buffer.make_view()
out1_buffer = rt.create_buf(runtime_py.memtag("some out 1", 1024*1024, runtime_py.mem_usage.OUT, bool(True)))
out1 = out1_buffer.make_view()
st = input('Buffers created. send some key--->')

rt.memset_buf(in1, 0, 256 * 4);

path = "/home/kamil/igemm/iGEMMgen/runtimes_4python/build/add_1.o"
kernel_name = "add_1_gfx9"

kernel = rt.get_kernel_from_binary(path, kernel_name)
print(f'kernel: {kernel.name} loaded')

params = runtime_py.dispatch_params()

params.wg_size = [128, 1, 1]
params.grid_size = [2, 1, 1]
params.dynamic_lds = 0

kac = runtime_py.kernel_args_constructor()

kac.pushb_ptr64(in1.ptr())
kac.pushb_ptr64(out1.ptr())

params.kernarg_size = kac.get_size()
params.kernarg = kac.create_array()

rt.run_kernel(kernel, params)

out = out1.get_vector_int( 256*4)

print(out)

rt.delete_buf(in1.buf)
st = input('5--->')
#rt.delete_buf(in2.buf)


