cmake_minimum_required(VERSION 3.17)
project(Lox)

set(CMAKE_CXX_STANDARD 20)

include_directories(${CMAKE_CURRENT_LIST_DIR}/includes)
include_directories(${CMAKE_CURRENT_LIST_DIR}/srcs)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/includes)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/srcs)

add_subdirectory(srcs)
add_subdirectory(bin)
add_subdirectory(test)
