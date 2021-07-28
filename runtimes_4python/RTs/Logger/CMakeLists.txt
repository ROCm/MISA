
add_library(logger log.cpp )
set_target_properties(logger PROPERTIES POSITION_INDEPENDENT_CODE ON)

target_include_directories(logger 
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