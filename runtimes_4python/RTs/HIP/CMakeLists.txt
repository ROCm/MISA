set(ROCM_PATH "/opt/rocm" CACHE PATH "")
find_package(ROCM REQUIRED PATHS /opt/rocm)
list(APPEND CMAKE_PREFIX_PATH ${ROCM_PATH} ${ROCM_PATH}/llvm ${ROCM_PATH}/hip)
find_package(hip)

message(STATUS " HIP CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH}")
message(STATUS " HIP_INCLUDE_DIRS ${HIP_INCLUDE_DIRS}")


add_library(runtime_hip runtime_hip.cpp )
set_target_properties(runtime_hip PROPERTIES POSITION_INDEPENDENT_CODE ON)
#set_target_properties(runtime_hip PROPERTIES CXX_VISIBILITY_PRESET hidden)

# Workaround hip not setting its usage requirements correctly
target_compile_definitions(runtime_hip PUBLIC -D__HIP_PLATFORM_HCC__=1 )

target_link_libraries(runtime_hip PUBLIC hip::device logger)

target_include_directories(runtime_hip 
    PUBLIC 
        ${HIP_INCLUDE_DIRS}
        "../" #"../include" 
		${CMAKE_CURRENT_SOURCE_DIR}
    )

##end

#get_target_property(LIBA_INCLUDES runtime_hip INCLUDE_DIRECTORIES)
#
## Construct the compiler string for the include directories.
#message(STATUS "hip runtime: include")
#foreach(dir ${LIBA_INCLUDES})
#    string(APPEND INCLUDE_COMPILER_STRING "/I${dir} ")
#    message(STATUS "/I${dir} ")
#endforeach()