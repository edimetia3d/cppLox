
add_library(bytecode chunk.cc debug.cc)
target_link_libraries(bytecode PUBLIC common)
