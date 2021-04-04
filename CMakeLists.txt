cmake_minimum_required(VERSION 3.17)
project(cppLox)

set(CMAKE_CXX_STANDARD 14)

include_directories(${CMAKE_CURRENT_LIST_DIR}/includes)

add_subdirectory(srcs)

add_executable(cppLox main.cpp)
target_link_libraries(cppLox lox version_info)
