cmake_minimum_required(VERSION 3.17)
project(Lox)

set(CMAKE_CXX_STANDARD 20)

add_subdirectory(third_party)
include_directories(${CLI11_DIR}/include)
include_directories(${SPDLOG_DIR}/include)

include_directories(${CMAKE_CURRENT_LIST_DIR}/includes)
include_directories(${CMAKE_CURRENT_LIST_DIR}/srcs)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/includes)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/srcs)

add_subdirectory(srcs)
add_subdirectory(bin)
add_subdirectory(test)
