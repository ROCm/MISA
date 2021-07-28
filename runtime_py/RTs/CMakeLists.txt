cmake_minimum_required(VERSION 3.5)
project(general_runtime_cxx VERSION 1.0)

#set(CMAKE_DEBUG_TARGET_PROPERTIES
#  INCLUDE_DIRECTORIES
#  COMPILE_DEFINITIONS
#  POSITION_INDEPENDENT_CODE
#  CONTAINER_SIZE_REQUIRED
#  LIB_VERSION
#)

add_library(general_runtime_cxx runtime.cpp)
set_target_properties(general_runtime_cxx PROPERTIES POSITION_INDEPENDENT_CODE ON)
#set_target_properties(general_runtime_cxx PROPERTIES CXX_VISIBILITY_PRESET hidden)

add_subdirectory(Logger)

if(WIN32)
    add_subdirectory(PAL)
else()
    add_subdirectory(HIP)
    set(RUNTIME_LIBS runtime_hip)
endif()

target_include_directories(general_runtime_cxx
    PUBLIC 
        ${CMAKE_CURRENT_SOURCE_DIR}/include
    INTERFACE
        ${CMAKE_CURRENT_SOURCE_DIR}
    )

target_link_libraries(general_runtime_cxx PRIVATE ${RUNTIME_LIBS} logger)

#get_target_property(LIBA_INCLUDES general_runtime_cxx INCLUDE_DIRECTORIES)
## Construct the compiler string for the include directories.
#message(STATUS "general_runtime_cxx: include:")
#foreach(dir ${LIBA_INCLUDES})
#    string(APPEND INCLUDE_COMPILER_STRING "/I${dir} ")
#    message(STATUS "+ /I${dir} ")
#endforeach()