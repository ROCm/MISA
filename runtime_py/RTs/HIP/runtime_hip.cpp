#include <iostream>
#include "runtime_hip.hpp"
#include "log.hpp"
using std::endl;

//struct CodeObject
//{
//    kernel kern;
//    hipModule_t mod;
//    hipFunction_t func;
//};

static inline bool check(hipError_t status, const char* msg)
{
    if (status == hipSuccess)
        return true;

    LOG(severity::ERROR) << msg << " failed with code " << status << ' ' << hipGetErrorString(status) << endl;
    return false;
}

#define CHECK(x) check(x, #x)

bool RTBackendHIP::init(bool profiling, bool counters, uint gpu_id, base_gpu_info* gi)
{
    enable_profiling = profiling;
    cobjects.reserve(30);

    int ndevs;
    if (!CHECK(hipInit(0)) || !CHECK(hipGetDeviceCount(&ndevs)))
        return false;

    if (gpu_id < 0 || (int)gpu_id >= ndevs) {
        //LOG(severity::FATAL) << "Invalid HIP device id. " << ndevs << " gpu devices available, requested gpu_id==" << gpu_id << endl;
        return false;
    }

    size_t memfree, memtotal;
    if (!CHECK(hipDeviceGet(&device, gpu_id))
        || !CHECK(hipGetDeviceProperties(&props, gpu_id))
        || !CHECK(hipSetDevice(gpu_id))
        || !CHECK(hipSetDeviceFlags(hipDeviceScheduleSpin))
        || !CHECK(hipEventCreate(&start))
        || !CHECK(hipEventCreate(&stop))
        || !CHECK(hipMemGetInfo(&memfree, &memtotal)))
        return false;

    gi->agent_name = props.gcnArchName;
    gi->cu_count = props.multiProcessorCount;
    gi->full_name = props.name;
    gi->mclk = props.memoryClockRate / 1000;
    gi->mem_size = props.totalGlobalMem;
    gi->mem_width = props.memoryBusWidth;
    gi->sclk = props.clockRate / 1000;
    gi->max_alloc = memfree;
    gi->alloc_gran = 4096; // wild guess as cuMemGetAllocationGranularity is not supported

    return true;
}

bool RTBackendHIP::meminfo(size_t* memfree)
{
    size_t memtotal;
    return CHECK(hipMemGetInfo(memfree, &memtotal));
}

bool RTBackendHIP::shutdown()
{
    return CHECK(hipEventDestroy(stop))
        && CHECK(hipEventDestroy(start));
}

bool RTBackendHIP::memcpyDtoH(void* dst, const void* src, size_t size) const
{
    return CHECK(hipMemcpyDtoH(dst, (void*)src, size));
}

bool RTBackendHIP::memcpyHtoD(void* dst, const void* src, size_t size) const
{
    return CHECK(hipMemcpyHtoD(dst, (void*)src, size));
}

bool RTBackendHIP::memsetD8(const bufview& dst, uint8_t val, size_t size) const
{
    return CHECK(hipMemsetD8(dst.ptr(), val, size));
}

void* RTBackendHIP::allocate_gpumem(size_t size)
{
    void* ptr;
    bool status = CHECK(hipMalloc(&ptr, size));

    return status ? ptr : 0;
}

bool RTBackendHIP::free_gpumem(void* ptr)
{
    return CHECK(hipFree(ptr));
}

void* RTBackendHIP::allocate_cpumem(size_t size)
{
    void* ptr;
    bool status = CHECK(hipHostMalloc(&ptr, size));

    return status ? ptr : 0;
}

bool RTBackendHIP::free_cpumem(void* ptr)
{
    return CHECK(hipHostFree(ptr));
}

//bool RTBackendHIP::load_kernel_from_memory(kernel* kern, void* bin, size_t size)
//{
//    CodeObjectHIP co;
//    co.mod = 0;
//    co.func = 0;
//    co.name = "sp3AsmKernel"; // todo: query function names
//    if (!CHECK(hipModuleLoadData(&co.mod, bin)) ||
//        !CHECK(hipModuleGetFunction(&co.func, co.mod, co.name.c_str())))
//    {
//        if (co.mod)
//            CHECK(hipModuleUnload(co.mod));
//        return false;
//    }
//
//    co.lds_size = 0; // todo: add lds_size
//    kern->name = co.name;
//    kern->handle = cobjects.size();
//    cobjects.push_back(co);
//
//    return true;
//}

bool RTBackendHIP::run_kernel(const kernel* kern, const dispatch_params* params, uint64_t timeout, int64_t* time, int64_t* clocks)
{
    const CodeObjectHIP* co = &cobjects[kern->handle];
    const void* config[] = { HIP_LAUNCH_PARAM_BUFFER_POINTER, params->kernarg, HIP_LAUNCH_PARAM_BUFFER_SIZE, &params->kernarg_size,
                      HIP_LAUNCH_PARAM_END };

    return CHECK(hipModuleLaunchKernel(co->func, params->grid_size[0], params->grid_size[1], params->grid_size[2],
        params->wg_size[0], params->wg_size[1], params->wg_size[2], params->dynamic_lds,
        hipStreamDefault, NULL, (void**)&config));
}
