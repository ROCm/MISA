cmake_minimum_required(VERSION 3.5)
project(runtime_py VERSION 1.0)
#CXX=/opt/rocm/llvm/bin/clang++ cmake -DPYBIND11_PYTHON_VERSION=3.6 ../

# specify the C++ standard
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)


# Without this, any build libraries automatically have names "lib{x}.so"
set(CMAKE_SHARED_MODULE_PREFIX "")


#debug info
#set(CMAKE_DEBUG_TARGET_PROPERTIES
#  INCLUDE_DIRECTORIES
#  COMPILE_DEFINITIONS
#  POSITION_INDEPENDENT_CODE
#  CONTAINER_SIZE_REQUIRED
#  LIB_VERSION
#)

add_library(runtime_py MODULE runtime_py.cpp)
add_executable(test_libs main.cpp)

add_subdirectory(RTs)



# very problematic to use package installed by pip
# so used git submodule
add_subdirectory(pybind11)

#g++ -fPIC -I/usr/include/python3.6m extending.cpp -shared -o extending.so -lboost_python3

target_link_libraries(runtime_py 
    PUBLIC
        pybind11::module
    #PRIVATE  
        general_runtime_cxx
        ${PYTHON_LIBRARIES}
    )
target_include_directories(runtime_py
    #    PRIVATE  
    #        ${PYTHON_INCLUDE_DIRS} 
        PUBLIC
            ${CMAKE_CURRENT_SOURCE_DIR}
    #        ${PYBIND11_INC}
        )
    

target_link_libraries(test_libs 
    PUBLIC
        pybind11::module
    #PRIVATE  
        general_runtime_cxx
        ${PYTHON_LIBRARIES}
    )
target_include_directories(test_libs
    #    PRIVATE  
    #        ${PYTHON_INCLUDE_DIRS} 
        PUBLIC
            ${CMAKE_CURRENT_SOURCE_DIR}
    #        ${PYBIND11_INC}
        )
    


    
#get_target_property(LIBA_INCLUDES runtime_py INCLUDE_DIRECTORIES)
#
#message(STATUS "main runtime: include")
## Construct the compiler string for the include directories.
#foreach(dir ${LIBA_INCLUDES})
#    string(APPEND INCLUDE_COMPILER_STRING "/I${dir} ")
#    message(STATUS "/I${dir} ")
#endforeach()