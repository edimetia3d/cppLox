
set(UPSTREAM_STYLE_ERROR_MSG "OFF" CACHE BOOL "If set to ON, the project will synchronize as original upstream style.")
set(ENABLE_JIT_BACKEND "OFF" CACHE BOOL "If set to ON, the project will build with MLIR JIT backend.")

if (${UPSTREAM_STYLE_ERROR_MSG} STREQUAL "ON")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DUPSTREAM_STYLE_ERROR_MSG")
endif ()

if (${ENABLE_JIT_BACKEND} STREQUAL "ON")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DENABLE_JIT_BACKEND")
endif ()

