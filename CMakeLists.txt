cmake_minimum_required(VERSION 3.17)
project(Lox)

set(CMAKE_CXX_STANDARD 20)

include_directories(${CMAKE_CURRENT_LIST_DIR}/includes)
include_directories(${CMAKE_CURRENT_LIST_DIR}/srcs)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/includes)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/srcs)

if (NOT DEFINED CLI11_DIR)
    set(CLI11_DIR "${CMAKE_CURRENT_LIST_DIR}/third_party/CLI11" CACHE STRING "Bundled CLI11 repo")
endif ()
include_directories(${CLI11_DIR}/include)


add_subdirectory(srcs)
add_subdirectory(bin)
add_subdirectory(test)
