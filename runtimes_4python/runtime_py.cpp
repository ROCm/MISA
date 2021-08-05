#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

#include "runtime.hpp"
#include "log.hpp"
#include "op_kernel_args.hpp"

PYBIND11_MAKE_OPAQUE(std::vector<int>);



void SetSeverity(uint s){
    LOG.SetSeverity(s);
}

class  PyRuntime : public Runtime
{
public:
    PyRuntime() : Runtime() { init_py();};
    ~PyRuntime(){};

    kernel get_kernel_from_binary(const string& src_path, const string& name)
    {
        kernel k;
        load_kernel_from_binary(&k, src_path, name);
        return k;
    }

private:

};

namespace py = pybind11;

PYBIND11_MODULE(runtime_py, m) {
    // Optional docstring

    m.def("SetSeverity", &SetSeverity);

    py::enum_<usage>(m, "mem_usage",  py::module_local())
        .value("IN", usage::IN)
        .value("OUT", usage::OUT)
        .value("AUX", usage::AUX)
        .export_values();

    py::class_<memtag>(m, "memtag",  py::module_local())
        .def(py::init<std::string, std::size_t, usage, bool>(),
            py::arg("name"), py::arg("size"), py::arg("use"), py::arg("gpu"))
        .def_readonly("name", &memtag::name)
        .def_readonly("size", &memtag::size);
    
    py::class_<bufview>(m, "bufview",  py::module_local())
        .def_readonly("buf", &bufview::buf)
        .def_readonly("tag", &bufview::tag)
        .def_readwrite("off", &bufview::off)
        .def("ptr", &bufview::ptr, "return buffer ptr with view offset")
        .def("get_vector_int", static_cast<std::vector<int> (bufview::*)(size_t)>(&bufview::get_vector),
            "return vector<int> with data from buffer, limitted by byte_size")
        .def("set_vector_int", static_cast<void (bufview::*)(const std::vector<int>&, size_t)>(&bufview::set_vector),
            "copy vector<int> to data in buffer, limitted by byte_size")
        .def(py::init<>());

    py::class_<membuf>(m, "membuf",  py::module_local())
        .def(py::init<>())
        .def("make_view", &membuf::make_view, "create default view for this buffer")
        .def_readonly("tag", &membuf::tag);

    py::class_<kernel>(m, "kernel",  py::module_local())
        .def_readonly("name", &kernel::name)
        .def(py::init<>());

    py::class_<dispatch_params>(m, "dispatch_params",  py::module_local())
        .def_readwrite("wg_size", &dispatch_params::wg_size)
        .def_readwrite("grid_size", &dispatch_params::grid_size)
        .def_readwrite("kernarg", &dispatch_params::kernarg)
        .def_readwrite("kernarg_size", &dispatch_params::kernarg_size)
        .def_readwrite("dynamic_lds", &dispatch_params::dynamic_lds)
        .def(py::init<>());

    py::class_<kernel_args_constructor>(m, "kernel_args_constructor",  py::module_local())
        .def("get_size", &kernel_args_constructor::get_size)
        .def("create_array", &kernel_args_constructor::create_array, py::return_value_policy::take_ownership)
        .def("pushb_ptr64", &kernel_args_constructor::push_back_ptr)
        .def("pushb_int32", &kernel_args_constructor::push_back_int_32)
        .def("pushb_int64", &kernel_args_constructor::push_back_sizet_64)
        .def("pushb_fp32", &kernel_args_constructor::push_back_float_32)
        .def(py::init<>());

    py::class_<PyRuntime>(m, "Runtime")
        .def(py::init<>())
        //.def("init_rt", &PyRuntime::init_py)
        .def("create_buf", &PyRuntime::create_buf, "Allocate memory in bytes on the default accelerator")
        .def("delete_buf", &PyRuntime::delete_buf, "remove memory buffer on the default accelerator")
        .def("memset_buf", &PyRuntime::memset_buf, "set vall for all buffer elements")
        .def("load_kernel_from_binary", &PyRuntime::load_kernel_from_binary,
            py::arg("dest_kernel"), py::arg("src_path"), py::arg("name") )
        .def("load_kernel_from_source", &PyRuntime::load_kernel_from_source,
            py::arg("dest_kernel"), py::arg("src_path"), py::arg("asembler_path"), py::arg("asm_params"),
            py::arg("name") )
        .def("get_kernel_from_binary", &PyRuntime::get_kernel_from_binary,
            py::arg("src_path"), py::arg("name") )
        .def("run_kernel", &PyRuntime::short_kernel_run,
            py::arg("kernel"), py::arg("arguments") )
        ;

    py::bind_vector<std::vector<int>>(m, "VectorInt");

}
