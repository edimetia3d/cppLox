
file(GLOB LOX_SRCS ${CMAKE_CURRENT_LIST_DIR}/*.c*)
add_library(lox ${LOX_SRCS})

add_subdirectory(version)
