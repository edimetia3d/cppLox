
set(UPSTREAM_STYLE_ERROR_MSG "OFF" CACHE BOOL "If set to ON, the project will synchronize as original upstream style.")

if (${UPSTREAM_STYLE_ERROR_MSG} STREQUAL "ON")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DUPSTREAM_STYLE_ERROR_MSG")
endif ()