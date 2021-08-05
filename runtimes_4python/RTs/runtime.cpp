#include "runtime.hpp"
#include <vector>
#include <fstream>

#define DISABLE_PAL_RUNTIME

//#ifndef DISABLE_HIP_RUNTIME
	#include "runtime_hip.hpp"
//#endif


#include "utils_math.hpp"
#include "utils.hpp"
#include "log.hpp"
#include <sstream>
#include <iostream>

using std::endl;

#define RT_CALL(call)                                                   \
    do {                                                                \
        if (!(call)) {                                                    \
            throw std::runtime_error(                                   \
				std::string("[Runtime] failed at call") + #call);   \
        }                                                               \
    } while (0)


ostream& operator<<(ostream& os, const memtag& t)
{
    static const char* tnames[] = { " IN", "OUT", "AUX" };
    return os << (t.gpu ? "gpu  " : "host ") << tnames[(int)t.use] << ", " << t.name << ",\t" << t.size;
}

ostream& operator<<(ostream& os, const bufview& v)
{
	return os << "ptr=" << v.ptr() << " rtbe="<< v.buf->rtbe << " (" << v.tag << ")";
}

ostream& operator<<(ostream& os, const gpu_info& v)
{
	int gops = (v.cu_count * 64 * v.sclk * 2) / 1000;
	int gops_fp32  = gops * v.rate_fp32;
	int gops_bfp16 = gops * v.rate_bfp16;
	int gops_fp16  = gops * v.rate_fp16;

	return os << "GPU info: " << v.rt_name << ' ' << v.agent_name << ' '
		<< v.cu_count << "CUs " << v.sclk << "s " << v.mclk << "m " << v.mem_width
		<< "b (" << v.mem_gbps << " GB/s, " << gops_fp32 << " Gflops, " << gops_bfp16
		<< " Gbfp16, " << gops_fp16 << " Gfp16) " << v.full_name << "\nMem,MaxAlloc,Gran: "
		<< (v.mem_size >> 20) << " MB, " << (v.max_alloc >> 20) << " MB, " << v.alloc_gran;
}

inline void rt_copy_mem(void* dst, const bufview* src, size_t byte_size)
{
    if (src->tag.gpu)
        RT_CALL(src->buf->rtbe->memcpyDtoH(dst, src->ptr(), byte_size));
    else
        memcpy(dst, src->ptr(), byte_size);
}

inline void rt_copy_mem(bufview* dst, const void* src, size_t byte_size)
{
    if (dst->tag.gpu)
        RT_CALL(dst->buf->rtbe->memcpyHtoD(dst->ptr(), src, byte_size));
    else
        memcpy(dst->ptr(), src, byte_size);
}

void * bufview::ptr() const { return (char*)buf->ptr + off; }
void bufview::map(membuf* b, size_t offset) { buf = b; off = offset; }
bufview::bufview(membuf& b) : tag(b.tag) { map(&b, 0); };

void bufview::copy_mem_out(void* dst, size_t byte_size)
{
    rt_copy_mem(dst, this, byte_size);
}
void bufview::copy_mem_in(const void* src, size_t byte_size)
{
    rt_copy_mem(this, src, byte_size);
}

Runtime::Runtime() : rtbe(nullptr)
{

}

Runtime::~Runtime()
{
	shutdown();
}

void Runtime::init(string rt, bool profiling, bool counters, uint gpu_id)
{

	if (rt == "hip" && !rtbe) { rtbe = new RTBackendHIP; }
//	if (rt == "hsa" && !rtbe) { rtbe = new RTBackendHSA; }

	if (!rtbe)
	{
		rtbe = new RTBackend;
		LOG(severity::WARNING) << "Unknown runtime: \"" << rt << "\". Running with no gpu support.\n";
	}

	rtbe->init(profiling, counters, gpu_id, &gi);

	gi.rt_name = rt;
	gi.mem_gbps = gi.mclk * gi.mem_width / 4000;
	gi.rate_fp32 = 1;
	gi.rate_bfp16 = 0;
	gi.rate_fp16 = 2;
	if (!gi.agent_name.compare(0, 6, "gfx908")
		|| !gi.agent_name.compare(0, 6, "gfx90a"))
	{
		gi.rate_fp32 = 2;
		gi.rate_bfp16 = 4;
		gi.rate_fp16 = 8;
	}

	if (!rtbe->meminfo(&memfree))
		memfree = gi.mem_size;

}

void Runtime::overwrite_clock_info(int sclk, int mclk)
{
	gi.sclk = sclk;
	gi.mclk = mclk;
	gi.mem_gbps = gi.mclk * gi.mem_width / 4000;
}

void Runtime::shutdown()
{
	RT_CALL(rtbe->shutdown());
}

void Runtime::delete_buf(membuf& a)
{
    if (!a.tag.gpu)
        RT_CALL(rtbe->free_cpumem(a.ptr));
    else{
        RT_CALL(rtbe->free_gpumem(a.ptr));
        track_gpumemfree(a.tag.size);
    }
}

inline void Runtime::track_gpumemfree(size_t size)
{
	if (!rtbe->meminfo(&memfree))
		memfree += ceil(size, gi.alloc_gran);
}

inline void Runtime::track_gpumemalloc(size_t size)
{
	if (!rtbe->meminfo(&memfree))
	{
		size_t gran_size = ceil(size, gi.alloc_gran);
		if (gran_size > memfree)
		{
			memfree = 0;
			LOG(severity::WARNING) << "Estimated gpu memfree size is negative, reset to 0\n";
		}
		else
			memfree -= gran_size;
	}
}

membuf Runtime::create_buf(const memtag& tag)
{
	membuf a = { tag, nullptr, rtbe};

	if (tag.gpu){
		a.ptr = rtbe->allocate_gpumem(tag.size);
    }
	else
		a.ptr = rtbe->allocate_cpumem(tag.size);

	LOG(severity::DEBUG) << "Allocate buf " << a << endl;

	if (tag.gpu && a.ptr)
		track_gpumemalloc(tag.size);

	return a;
}

void Runtime::memset_buf(const bufview& dst, uint8_t val, size_t byte_size)
{
    if (dst.tag.gpu)
        RT_CALL(rtbe->memsetD8(dst, val, byte_size));
    else
        memset(dst.ptr(), val, byte_size);
}

void Runtime::copy_mem(void* dst, const bufview* src, size_t byte_size) const
{
    if (src->tag.gpu)
        RT_CALL(rtbe->memcpyDtoH(dst, src->ptr(), byte_size));
    else
        memcpy(dst, src->ptr(), byte_size);
}

void Runtime::copy_mem(const bufview* dst, const void* src, size_t byte_size) const
{
    if (dst->tag.gpu)
        RT_CALL(rtbe->memcpyHtoD(dst->ptr(), src, byte_size));
    else
        memcpy(dst->ptr(), src, byte_size);
}

//void Runtime::load_code_object(const string& src_path)
//{
//
//}

void Runtime::load_kernel_from_source(kernel* kern, const string& src_path, const string& asmpl_path, const string& params, const string& name)
{
    TempFile outfile("conv-asm-out-XXXXXX");

    vector<string> args({
        "-p", src_path,
        "-o", outfile,
        "-mcpu", gi.agent_name,
        });

    {
        std::istringstream iss(params);
        string param;
        while (iss >> param) {
            args.push_back(param);
        };
    }

    LOG(severity::DEBUG) << "Assembly options:\n";
    for (auto arg : args)
        LOG(severity::DEBUG, 0) << "\t" << arg << "\n";

    std::ostringstream asmsh_stderr;
    const auto asm_rc = ExecuteProcess(asmpl_path, args, nullptr, nullptr, &asmsh_stderr);
    if (asm_rc != 0){
        LOG(severity::ERROR) << "Asm stderr:\n" << asmsh_stderr.str().substr(0, 2048)
        << "\nAssembly error(" << asm_rc << ")\n";
        RT_CALL(asm_rc != 0);
    }
    load_kernel_from_binary(kern, outfile.path(), name);
}

void Runtime::load_kernel(kernel* kern, const string& src_path, const string& asmpl_path, const string& params, const string& name)
{
        //if(is_binary_kernel(src_path))
                load_kernel_from_binary(kern, src_path, name);
        //else 
        //      load_kernel_from_source(kern, src_path, asmpl_path, params);
}

void Runtime::load_kernel_from_binary(kernel* kern, const string& path, const string& name)
{
        vector<char> bin;
        RT_CALL(read_file(path.c_str(), bin));
                //throw std::runtime_error(std::string("[Runtime] failed to read the file:") + path);

        RT_CALL(rtbe->load_kernel_from_memory(kern, bin.data(), bin.size(), name));
}

