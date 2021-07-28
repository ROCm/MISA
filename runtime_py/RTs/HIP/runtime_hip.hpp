#ifndef RUNTIME_HIP_HPP__
#define RUNTIME_HIP_HPP__ 1

#include "runtime.hpp"

#define __HIP_PLATFORM_HCC__ 1
#include <hip/hip_runtime.h>

struct CodeObjectHIP
{
	string name;
	hipModule_t mod;
	hipFunction_t func;
	uint32_t lds_size;
};

class RTBackendHIP : RTBackend
{
    friend class Runtime;
private:
	hipDeviceProp_t props;
	hipDevice_t device;
	hipEvent_t start, stop;
	//size_t memfree, memtotal;
	bool enable_profiling;
	vector<CodeObjectHIP> cobjects;

public:
	bool memcpyDtoH(void* dst, const void* src, size_t size) const override;
	bool memcpyHtoD(void* dst, const void* src, size_t size) const override;
	bool memsetD8(const bufview& dst, uint8_t val, size_t size) const override;

protected:
	bool init(bool profiling, bool counters, uint gpu_id, base_gpu_info* gi) override;
	bool shutdown() override;

	void* allocate_gpumem(size_t size) override;
	bool free_gpumem(void* ptr) override;
	bool meminfo(size_t* memfree) override;

	void* allocate_cpumem(size_t size) override;
	bool free_cpumem(void* ptr) override;
	
	bool load_kernel_from_memory(kernel* kern, void* bin, size_t size, const string& name) override;
	bool run_kernel(const kernel* kern, const dispatch_params* params, uint64_t timeout, int64_t* time, int64_t* clocks) override;
};

#endif