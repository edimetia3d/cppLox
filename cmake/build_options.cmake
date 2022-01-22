
set(UPSTREAM_STYLE_SYNCHRONIZE "OFF" CACHE BOOL "If set to ON, the project will synchronize as original upstream style.")

if (${UPSTREAM_STYLE_SYNCHRONIZE} STREQUAL "ON")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DUPSTREAM_STYLE_SYNCHRONIZE")
endif ()