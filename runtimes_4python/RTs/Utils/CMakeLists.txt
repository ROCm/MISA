if(WIN32)
    set(OC_based_src utils_win32.cpp)
else()
    set(OC_based_src utils_posix.cpp)
endif()

add_library(System_Utils utils.cpp ${OC_based_src})
set_target_properties(System_Utils PROPERTIES POSITION_INDEPENDENT_CODE ON)

target_include_directories(System_Utils 
    PUBLIC 
        "../include"
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