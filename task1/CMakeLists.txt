option(USE_DOUBLE OFF)

add_executable(a main.cpp)

if(USE_DOUBLE)
    target_compile_definitions(a PRIVATE USE_DOUBLE)
endif()
