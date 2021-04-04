
include_directories(${CMAKE_CURRENT_LIST_DIR})
add_subdirectory(version)
file(GLOB LOX_SRCS ${CMAKE_CURRENT_LIST_DIR}/lox/*.c*)
add_library(lox ${LOX_SRCS})