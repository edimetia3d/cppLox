
include_directories(${CMAKE_CURRENT_LIST_DIR})
file(GLOB_RECURSE LOX_SRCS ${CMAKE_CURRENT_LIST_DIR}/*.c*)
add_library(lox ${LOX_SRCS})