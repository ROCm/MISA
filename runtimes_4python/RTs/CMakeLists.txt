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
add_subdirectory(Utils)

option( NO_RUNTIME_INTERFACE "" OFF)

if(NO_RUNTIME_INTERFACE)

elseif(WIN32)
    option( ENABLE_PAL_RUNTIME "" OFF)
    
    if(ENABLE_PAL_RUNTIME)
        #add_subdirectory(PAL)
        #set(RUNTIME_LIBS runtime_hip)
        #target_compile_definitions(general_runtime_cxx 
        #PRIVATE 
        #ENABLE_PAL_RUNTIME)
    endif()
else()
    option( ENABLE_HIP_RUNTIME "" ON)

    if(ENABLE_HIP_RUNTIME)
        add_subdirectory(HIP)
        set(RUNTIME_LIBS runtime_hip)
        target_compile_definitions(general_runtime_cxx 
            PRIVATE 
                ENABLE_HIP_RUNTIME
        )
    endif()
endif()

target_include_directories(general_runtime_cxx
    PUBLIC 
        ${CMAKE_CURRENT_SOURCE_DIR}/include
    INTERFACE
        ${CMAKE_CURRENT_SOURCE_DIR}
    )

target_link_libraries(general_runtime_cxx PRIVATE ${RUNTIME_LIBS} logger System_Utils)

#get_target_property(LIBA_INCLUDES general_runtime_cxx INCLUDE_DIRECTORIES)
## Construct the compiler string for the include directories.
#message(STATUS "general_runtime_cxx: include:")
#foreach(dir ${LIBA_INCLUDES})
#    string(APPEND INCLUDE_COMPILER_STRING "/I${dir} ")
#    message(STATUS "+ /I${dir} ")
#endforeach()