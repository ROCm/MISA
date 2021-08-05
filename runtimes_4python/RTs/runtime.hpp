#ifndef RUNTIME_HPP__
#define RUNTIME_HPP__ 1

#include <vector>
#include <array>
#include <string>
using std::vector;
using std::string;
using std::size_t;
using std::ostream;
using std::istream;

enum class usage
{
	IN, OUT, AUX
};

struct memtag
{
    memtag(){}
    memtag(string name, size_t size, usage use, bool gpu)
        : name(name), size(size), use(use), gpu(gpu) { }
    string name;
    size_t size;
    usage use;
    bool gpu;
};

struct membuf;
class RTBackend;

struct bufview
{
    memtag tag;

    membuf* buf;//TODO shared ptr for buf
    size_t off;
    void* ptr() const ;

    void map(membuf* b, size_t offset);
	bufview() = default;
	//bufview(membuf& b) : tag(b.tag) { map(&b, 0); };
    bufview(membuf& b);

    void copy_mem_out(void* dst, size_t byte_size);
    void copy_mem_in(const void* src, size_t byte_size);

    template <typename T>
    std::vector<T> get_vector() { return get_vector<T>(tag.size); }
    
    template <typename T>
    std::vector<T> get_vector(size_t bytes)
    {
        int vector_size = (bytes + sizeof(T)) / sizeof(T);
        std::vector<T> val(vector_size);
        copy_mem_out(val.data(), bytes);
        return val;
    }

    template <typename T>
    void set_vector(const std::vector<T>& val) { set_vector<T>(val, tag.size); }

    template <typename T>
    void set_vector(const std::vector<T>& val, size_t bytes) { copy_mem_in(val.data(), bytes);}

};

struct membuf
{
    memtag tag;
    void* ptr;
    RTBackend* rtbe; //TODO weak_ptr for rtbe

    bufview make_view(){ return bufview(*this); }
};

typedef void* gpu_ptr;

ostream& operator<<(ostream& os, const memtag& v);
ostream& operator<<(ostream& os, const bufview& v);

struct dispatch_params
{
    dispatch_params(){};
    std::array<uint32_t, 3> wg_size = {};
    std::array<uint32_t, 3> grid_size = {};
    
    const void* kernarg;
    size_t kernarg_size;
    uint32_t dynamic_lds;
};

struct kernel
{
	uint64_t handle;
	string name;
};

struct base_gpu_info
{
	string agent_name;
	string full_name;
	size_t mem_size;
	size_t max_alloc; // max allocation size
	size_t alloc_gran; // allocation granularity
	int sclk; // in MHz
	int mclk; // in MHz
	int cu_count;
	int mem_width;
};

struct gpu_info : base_gpu_info
{
	string rt_name;
	int mem_gbps;
	int rate_fp32;
	int rate_bfp16;
	int rate_fp16;
};
ostream& operator<<(ostream& os, const gpu_info& v);

class RTBackend
{
	friend class Runtime;

// subset of functions available for solvers
public:
	virtual bool memcpyDtoH(void* dst, const void* src, size_t size) const { return false; }
	virtual bool memcpyHtoD(void* dst, const void* src, size_t size) const { return false; }
	virtual bool memsetD8(const bufview& dst, uint8_t val, size_t size) const { return false; }
	virtual void* stream() const { return nullptr; };

protected:
	RTBackend() {};
	virtual ~RTBackend() {};
	
	virtual bool init(bool profiling, bool counters, uint gpu_id, base_gpu_info* gi) { *gi = {};  return true; }
	virtual bool shutdown() { return true; }

	virtual void* allocate_gpumem(size_t size) { return nullptr; }
	virtual bool free_gpumem(void* ptr) { return false; }
	virtual bool meminfo(size_t* memfree) { return false; } // Backend could leave it unoverride. Runtime then will provide estimation.

	virtual void* allocate_cpumem(size_t size) { return new char[size]; }
	virtual bool free_cpumem(void* ptr) { delete[] static_cast<char*>(ptr); return true; }

    virtual bool load_kernel_from_memory(kernel* kern, void* bin, size_t size, const string& name) { return false; }
	virtual bool run_kernel(const kernel* kern, const dispatch_params* params, uint64_t timeout, int64_t* time, int64_t* clocks) { return false; }
};

class Runtime
{
private:
	RTBackend* rtbe;
	gpu_info gi;
	size_t memfree; // free gpu memory

	inline void track_gpumemfree(size_t size);
	inline void track_gpumemalloc(size_t size);

public:
	Runtime();
	~Runtime();

	void init(string rt, bool profiling, bool counters, uint gpu_id);
	void init(string rt) { init(rt, false, false, 0);}
	void init_py() { init("hip", false, false, 0);}

	void overwrite_clock_info(int sclk, int mclk);
	inline gpu_info const* get_gpu_info() const { return &gi; }
	const RTBackend* BEptr() const { return rtbe; }
	void shutdown();

	membuf create_buf(const memtag& tag);
	void delete_buf(membuf& a);
	size_t get_freegpumem() const { return memfree; } // it is not precies and might be optimistic estimation depending on actual runtime api
	void memset_buf(const bufview& dst, uint8_t val, size_t byte_size);
	void copy_mem(void* dst, const bufview* src, size_t byte_size) const;
	void copy_mem(const bufview* dst, const void* src, size_t byte_size) const;

	void load_kernel_from_binary(kernel* kern, const string& src_path, const string& name);
	void load_kernel_from_source(kernel* kern, const string& src_path, const string& asmpl_path, const string& params, const string& name);
	void load_kernel(kernel* kern, const string& src_path, const string& asmpl_path, const string& params, const string& name);
	void run_kernel(const kernel* kern, const dispatch_params* params, uint64_t timeout = 0, int64_t* time = nullptr, int64_t* clocks = nullptr)
		const { rtbe->run_kernel(kern, params, timeout, time, clocks); }
    void short_kernel_run(const kernel* kern, const dispatch_params* params)
        const { run_kernel(kern, params);}
};

bool is_binary_kernel(const string& src_path);

#endif