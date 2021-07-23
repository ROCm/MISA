#include <pybind11/pybind11.h>

#include "runtime.hpp"

namespace py = pybind11;

int add(int i, int j) {
    return i + j;
}


PYBIND11_MODULE(runtime_py, m) {
    // Optional docstring

    py::class_<RTCodeObject>(m, "RTCodeObject")
        .def(py::init<>())
        .def("init_rt", &Runtime::init_py)
        .def("create_buf", &Runtime::create_buf, "Allocate memory on the default accelerator")
        //.def("delete_buf", &Runtime::delete_buf, "remove memory buffer on the default accelerator")
        //.def("memset_buf", &Runtime::memset_buf, "set vall for all buffer elements")
        //.def("get_kernel_from_file", &Runtime::load_kernel_from_binary)
        //.def("run_kernel", &Runtime::run_kernel)
        ;
    //def("loadModuleFromFile", GetModuleFromfile, "return hipModule_t object from src file");
    //def("loadModuleFromFile", ModuleUnload, "return hipModule_t object from src file");
    //def("mallocGPU", gpuMalloc, return_value_policy<copy_non_const_reference>(), "Allocate memory on the default accelerator");
    //def("mallocHost", hostMalloc, return_value_policy<copy_non_const_reference>(), "Allocate device accessible page locked host memory");
    //def("moduleGetFunction", moduleGetFunction, "Function with kname will be extracted if present in module");
}
