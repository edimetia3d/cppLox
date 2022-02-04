cmake_minimum_required(VERSION 3.13)
project(Lox)

set(CMAKE_CXX_STANDARD 20)

if (CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    # prefer libc++
    set(CMAKE_CXX_FLAGS "-stdlib=libc++ ${CMAKE_CXX_FLAGS}")
endif ()

set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -DSPDLOG_ACTIVE_LEVEL=1")

include(cmake/build_options.cmake)
include(cmake/win32_compat.cmake)

add_subdirectory(third_party)
include_directories(${CLI11_DIR}/include)
include_directories(${SPDLOG_DIR}/include)

include_directories(${CMAKE_CURRENT_LIST_DIR}/includes)
include_directories(${CMAKE_CURRENT_LIST_DIR}/srcs)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/includes)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/srcs)

add_subdirectory(srcs)
add_subdirectory(bin)
